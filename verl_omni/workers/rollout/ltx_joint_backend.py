# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synchronous native vLLM-Omni transport for a paired LTX rollout policy."""

import multiprocessing
import shutil
import tempfile
from pathlib import Path

from verl_omni.pipelines.ltx2_unigrpo_nft.replay import LTXJointReplay
from verl_omni.pipelines.ltx2_unigrpo_nft.weight_sync import validate_joint_lora_bundle
from verl_omni.workers.rollout.bagel_joint_backend import independent_rollout_port


class LTXJointVllmBackend:
    """One worker executes Gemma recaption, shared-Gemma encoding, and native AV DiT."""

    def __init__(
        self,
        *,
        model_path,
        geometry,
        recaption_temperature,
        recaption_max_new_tokens,
        sync_timeout_seconds,
        enable_cpu_offload=True,
        vae_use_tiling=False,
        text_diagnostics=None,
    ):
        from vllm_omni.diffusion.data import AttentionConfig, OmniDiffusionConfig
        from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

        self.geometry = dict(geometry)
        self.temperature = recaption_temperature
        self.max_new_tokens = recaption_max_new_tokens
        self.timeout = sync_timeout_seconds
        self.policy_version = None
        self.lora_request = None
        self._actor_policy_root = None
        self._coordination_group = None
        self.text_diagnostics = dict(text_diagnostics or {})
        multiprocessing.set_start_method("spawn", force=True)
        with independent_rollout_port():
            config = OmniDiffusionConfig(
                model=str(model_path),
                model_class_name="LTX2Pipeline",
                num_gpus=1,
                distributed_executor_backend="mp",
                diffusion_load_format="custom_pipeline",
                custom_pipeline_args={
                    "pipeline_class": "verl_omni.pipelines.ltx2_unigrpo_nft.vllm_omni_rollout_adapter.LTXJointPipeline"
                },
                worker_extension_cls="verl_omni.workers.rollout.ltx_joint_worker_extension.LTXJointWorkerExtension",
                diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
                enforce_eager=True,
                enable_cpu_offload=enable_cpu_offload,
                vae_use_tiling=vae_use_tiling,
                # LTX reads extras during construction; pass a concrete mapping
                # even on backends whose dataclass default is a Pydantic Field.
                extras={},
            )
        self.engine = DiffusionEngine(config)

    def load_actor_policy(self, module, *, version):
        """Collectively export paired actor LoRA, then commit it on every worker."""
        import torch
        import torch.distributed as dist
        from verl.utils.device import get_device_id, get_device_name

        from verl_omni.pipelines.ltx2_unigrpo_nft.weight_sync import export_joint_actor_lora

        distributed = dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        if self._actor_policy_root is None:
            if distributed:
                self._coordination_group = dist.new_group(backend="gloo")
            root = [tempfile.mkdtemp(prefix="ltx-joint-policy-") if rank == 0 else None]
            if distributed:
                dist.broadcast_object_list(root, src=0, group=self._coordination_group)
            self._actor_policy_root = Path(root[0])
        name = get_device_name()
        device = torch.device(name, get_device_id()) if name != "cpu" else torch.device("cpu")
        directory = export_joint_actor_lora(
            module,
            self._actor_policy_root / f"policy-{version}",
            version=version,
            base_model=self.engine.od_config.model,
            device=device,
            coordination_group=self._coordination_group,
        )
        failure = None
        try:
            manifest = self.load_policy(directory)
        except Exception as error:
            failure = f"rank {rank}: {type(error).__name__}: {error}"
        failures = [failure]
        if distributed:
            failures = [None] * dist.get_world_size()
            dist.all_gather_object(failures, failure, group=self._coordination_group)
        if any(failures):
            self.policy_version = None
            raise RuntimeError(f"Joint policy was not committed by all rollout replicas: {failures}")
        # Every native worker has loaded the new pair. Retain its files while
        # active; release previous versions only after this collective commit.
        if rank == 0:
            for previous in self._actor_policy_root.glob("policy-*"):
                if previous != directory:
                    shutil.rmtree(previous)
        return manifest

    def load_policy(self, directory):
        from vllm.lora.request import LoRARequest

        directory = Path(directory).resolve()
        manifest = validate_joint_lora_bundle(directory)
        version = manifest["policy_version"]
        # Disable local admission until every native worker acknowledges the pair.
        self.policy_version = None
        args = (str(directory),)
        diagnostics = getattr(self, "text_diagnostics", {})
        if diagnostics.get("enabled", False):
            import torch.distributed as dist

            args += (diagnostics, dist.get_rank() if dist.is_initialized() else 0)
        replies = self.engine.collective_rpc("load_joint_lora_bundle", args=args, timeout=self.timeout)
        if not replies or any(
            reply["policy_version"] != version or reply["components"] != manifest["components"] for reply in replies
        ):
            raise RuntimeError("Native workers did not acknowledge the same Gemma/DiT policy manifest")
        self.lora_request = LoRARequest(
            lora_name=f"ltx-joint-{version}", lora_int_id=version + 1, lora_path=str(directory / "dit")
        )
        self.policy_version = version
        return manifest

    def generate(self, prompts, seeds, *, version, sampling_params=None):
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        if self.policy_version is None or version != self.policy_version:
            raise RuntimeError("Load the complete paired policy before rollout")
        results = []
        for index, (prompt, seed) in enumerate(zip(prompts, seeds, strict=True)):
            params = dict(self.geometry if sampling_params is None else sampling_params)
            extra = dict(params.pop("extra_args", {}))
            extra.update(
                policy_version=version,
                recaption_temperature=self.temperature,
                recaption_max_new_tokens=self.max_new_tokens,
            )
            request = OmniDiffusionRequest(
                prompt={"prompt": prompt, "modalities": ["video", "audio"]},
                sampling_params=OmniDiffusionSamplingParams(
                    **params, seed=int(seed), lora_request=self.lora_request, extra_args=extra
                ),
                request_id=f"ltx-joint-{version}-{index}",
            )
            output = self.engine.add_req_and_wait_for_response(request)
            if output.error is not None:
                raise RuntimeError(f"Native joint LTX rollout failed: {output.error}")
            envelope = output.output
            if not isinstance(envelope, dict) or "metadata" not in envelope or "payload" not in envelope:
                raise RuntimeError("Native joint LTX worker lost the replay/media envelope")
            metadata = envelope["metadata"]["rl"]
            record = LTXJointReplay(**metadata["ltx_joint_replay"])
            record.validate()
            if record.policy_version != version or record.original_prompt != prompt or record.seed != int(seed):
                raise RuntimeError("Native worker returned stale or mismatched replay")
            video, audio = envelope["payload"]["video"]
            if audio.ndim == 3 and audio.shape[0] == 1:
                audio = audio[0]
            if audio.ndim != 2:
                raise RuntimeError("Native joint LTX audio must have channels/time layout")
            results.append(
                {
                    "video": video,
                    "audio": audio,
                    "replay": record,
                    "video_latents_clean": metadata["video_latents_clean"],
                    "audio_latents_clean": metadata["audio_latents_clean"],
                    "train_timesteps": metadata["train_timesteps"],
                    "recaption_time": metadata["recaption_time"],
                    "generation_time": metadata["generation_time"],
                    "memory_metrics": metadata.get("memory_metrics", {}),
                    "fps": float(metadata["fps"].item()),
                    "audio_sample_rate": int(metadata["audio_sample_rate"].item()),
                }
            )
        self.engine.collective_rpc("release_joint_device_memory", timeout=self.timeout)
        return results

    def close(self):
        try:
            self.engine.close()
        finally:
            import torch.distributed as dist

            if self._actor_policy_root is not None and (not dist.is_initialized() or dist.get_rank() == 0):
                shutil.rmtree(self._actor_policy_root, ignore_errors=True)
            if self._coordination_group is not None and dist.is_initialized():
                dist.destroy_process_group(self._coordination_group)
                self._coordination_group = None
