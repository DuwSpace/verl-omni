# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One real vllm-omni MP worker per rank, with bounded full-policy export."""

import json
import multiprocessing
import os
import socket
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from verl.utils.device import get_device_id, get_device_name

from verl_omni.pipelines.bagel_unigrpo_nft.replay import replay_from_metadata


@contextmanager
def independent_rollout_port():
    """Prevent the native config from reusing Ray's shared training MASTER_PORT."""
    training_port = os.environ.pop("MASTER_PORT", None)
    try:
        yield
    finally:
        if training_port is not None:
            os.environ["MASTER_PORT"] = training_port


def original_prompt_text(value):
    """Accept original T2I text; never silently decode IDs or drop chat turns."""
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], dict)
        or value[0].get("role") != "user"
    ):
        raise ValueError("Joint BAGEL requires one original user prompt")
    content = value[0].get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, np.ndarray):
        content = content.tolist()
    if isinstance(content, list) and content and all(isinstance(p, dict) and p.get("type") == "text" for p in content):
        return "".join(p["text"] for p in content)
    raise ValueError("Joint BAGEL requires original text-only prompt content")


class BagelJointVllmBackend:
    """Independent TP=1 engines concurrently serve each actor rank's prompt slice."""

    def __init__(self, model_config):
        from vllm_omni.diffusion.data import AttentionConfig, OmniDiffusionConfig
        from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

        self.config = model_config.bagel_joint
        self.geometry = model_config.pipeline
        self.rank = dist.get_rank()
        self.group = dist.new_group(backend="gloo")
        hosts = [None] * dist.get_world_size()
        dist.all_gather_object(hosts, socket.gethostname(), group=self.group)
        if len(set(hosts)) != 1:
            raise ValueError("Joint BAGEL shared-file policy transport currently requires one node")
        # Native MP queues and children must use the same context, including under Ray.
        multiprocessing.set_start_method("spawn", force=True)
        with independent_rollout_port():
            config = OmniDiffusionConfig(
                model=model_config.local_path,
                model_class_name="BagelPipeline",
                num_gpus=1,
                distributed_executor_backend="mp",
                diffusion_load_format="custom_pipeline",
                custom_pipeline_args={
                    "pipeline_class": (
                        "verl_omni.pipelines.bagel_unigrpo_nft.vllm_omni_rollout_adapter.BagelUniGRPONFTPipeline"
                    )
                },
                worker_extension_cls="verl_omni.workers.rollout.bagel_joint_worker_extension.BagelJointWorkerExtension",
                diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
                enforce_eager=True,
            )
        self.engine = DiffusionEngine(config)
        info = self.engine.collective_rpc("joint_model_info", timeout=self.config.sync_timeout_seconds)[0]
        print("BAGEL_JOINT_WORKER_READY " + json.dumps({"rank": self.rank, **info}), flush=True)

    @torch.no_grad()
    def load_actor_policy(self, module, *, version):
        """All ranks gather BF16 weights; only rank zero writes one bounded bucket."""
        state = module.state_dict()
        names = ["transformer." + name for name in state]
        timeout = self.config.sync_timeout_seconds
        self.engine.collective_rpc("begin_joint_policy_sync", args=(version, names), timeout=timeout)
        directory = [tempfile.mkdtemp(prefix="bagel-joint-policy-") if self.rank == 0 else None]
        dist.broadcast_object_list(directory, src=0, group=self.group)
        device = torch.device(get_device_name(), get_device_id())
        bucket, bucket_bytes, index, total_bytes = {}, 0, 0, 0
        limit = self.config.sync_bucket_size_mb * 1024 * 1024

        def flush():
            nonlocal bucket, bucket_bytes, index
            path = Path(directory[0]) / f"bucket-{index}.pt"
            if self.rank == 0:
                torch.save(bucket, path)
            dist.barrier(group=self.group)
            self.engine.collective_rpc("load_joint_weight_file", args=(str(path),), timeout=timeout)
            dist.barrier(group=self.group)
            if self.rank == 0:
                path.unlink()
            bucket, bucket_bytes = {}, 0
            index += 1

        for name, parameter in state.items():
            size = parameter.numel() * 2
            if bucket_bytes and bucket_bytes + size > limit:
                flush()
            # Offloaded CPU DTensors cannot be gathered by HCCL. Cast/move the
            # local shard first, avoiding a full FP32 transient on any device.
            tensor = parameter.detach().to(device=device, dtype=torch.bfloat16)
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            if self.rank == 0:
                bucket["transformer." + name] = tensor.cpu()
            del tensor
            bucket_bytes += size
            total_bytes += size
        if bucket_bytes:
            flush()
        committed = self.engine.collective_rpc("commit_joint_policy_sync", timeout=timeout)[0]
        if committed["version"] != version:
            raise RuntimeError("Native BAGEL worker committed an unexpected policy version")
        dist.barrier(group=self.group)
        if self.rank == 0:
            Path(directory[0]).rmdir()
            print(
                "BAGEL_JOINT_POLICY_SYNC "
                + json.dumps({"version": version, "tensors": len(names), "buckets": index, "bytes": total_bytes}),
                flush=True,
            )

    def generate(self, prompts, seeds, *, version):
        """Use the engine's synchronous queue API inside Ray's already-running loop.

        BAGEL admits one request per replica. The engine owns worker execution
        and postprocessing; no nested event loop or actor-side sampling is used.
        """
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        images, records = [], []
        for index, (prompt, seed) in enumerate(zip(prompts, seeds, strict=True)):
            params = OmniDiffusionSamplingParams(
                height=self.geometry.height,
                width=self.geometry.width,
                num_inference_steps=self.geometry.num_inference_steps,
                seed=int(seed),
                extra_args={
                    "think": True,
                    "do_sample": True,
                    "max_think_tokens": self.config.max_think_tokens,
                    "text_temperature": self.config.text_temperature,
                    "cfg_text_scale": self.config.cfg_text_scale,
                    "cfg_img_scale": self.config.cfg_img_scale,
                    "policy_version": version,
                },
            )
            request = OmniDiffusionRequest(
                prompt={"prompt": prompt, "modalities": ["image"]},
                sampling_params=params,
                request_id=f"joint-{version}-{self.rank}-{index}",
            )
            native_output = self.engine.add_req_and_wait_for_response(request)
            final = self.engine.postprocess_output(request, native_output)
            if not final or len(final) != 1 or len(final[0].images) != 1:
                raise RuntimeError("Native BAGEL worker must return exactly one image per request")
            output = final[0]
            record = replay_from_metadata(output.multimodal_output["metadata"]["rl"])
            if record.policy_version != version:
                raise RuntimeError("Native BAGEL returned a stale policy replay")
            print(
                "BAGEL_JOINT_SAMPLE "
                + json.dumps(
                    {
                        "version": version,
                        "rank": self.rank,
                        "index": index,
                        "seed": int(seed),
                        "thinking_tokens": record.response_token_ids.numel(),
                        "thinking_ended_with_eos": int(record.response_token_ids[-1]) == record.eos_token_id,
                    }
                ),
                flush=True,
            )
            pixels = np.array(output.images[0].convert("RGB"), dtype=np.uint8, copy=True)
            images.append(torch.from_numpy(pixels).permute(2, 0, 1))
            records.append(record)
        return torch.stack(images), records

    def close(self):
        self.engine.close()
