# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Actual MP vllm-omni BAGEL worker, rollout replay and actor-weight sync on NPU."""

import argparse
import asyncio
import json
import multiprocessing
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from vllm_omni.diffusion.data import AttentionConfig, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from verl_omni.pipelines.bagel_flow_grpo.bagel_sft_model import BagelForSFT
from verl_omni.pipelines.bagel_unigrpo.bagel_ar_thinking import replay_thinking_logprobs
from verl_omni.pipelines.bagel_unigrpo_nft.replay import replay_from_metadata


async def _generate(engine, *, version, request_id):
    params = OmniDiffusionSamplingParams(
        height=64,
        width=64,
        num_inference_steps=5,
        seed=42,
        extra_args={
            "think": True,
            "do_sample": True,
            "max_think_tokens": 8,
            # Tiny random checkpoints can put nearly all mass on BOS at T=1;
            # use a softer distribution for meaningful backend comparisons.
            "text_temperature": 10.0,
            "policy_version": version,
            "cfg_text_scale": 1.0,
            "cfg_img_scale": 1.0,
        },
    )
    request = OmniDiffusionRequest(
        prompt={"prompt": "a red circle", "modalities": ["image"]}, sampling_params=params, request_id=request_id
    )
    final = None
    started = time.perf_counter()
    async for outputs in engine.step_streaming(request):
        final = outputs
    elapsed = time.perf_counter() - started
    if not final or len(final) != 1:
        raise RuntimeError("Native vllm-omni worker returned no unique image output")
    output = final[0]
    metadata = output.multimodal_output["metadata"]["rl"]
    replay = replay_from_metadata(metadata)
    if replay.policy_version != version:
        raise RuntimeError("Native replay policy version differs from synced version")
    return {
        "policy_version": version,
        "prompt_ids": replay.prompt_token_ids.tolist(),
        "cached_thinking_ids": replay.cached_thinking_token_ids.tolist(),
        "response_ids": replay.response_token_ids.tolist(),
        "latent_shape": list(replay.latents_clean.shape),
        "rollout_log_probs": replay.rollout_log_probs.tolist(),
        "output_type": type(output).__name__,
        "request_seconds": elapsed,
    }, replay


async def _run(args):
    config = OmniDiffusionConfig(
        model=args.model,
        model_class_name="BagelPipeline",
        num_gpus=1,
        distributed_executor_backend="mp",
        diffusion_load_format="custom_pipeline",
        custom_pipeline_args={
            "pipeline_class": "verl_omni.pipelines.bagel_unigrpo_nft.vllm_omni_rollout_adapter.BagelUniGRPONFTPipeline"
        },
        worker_extension_cls="verl_omni.workers.rollout.bagel_joint_worker_extension.BagelJointWorkerExtension",
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
        enforce_eager=True,
    )
    engine = DiffusionEngine(config)
    try:
        before = engine.collective_rpc("joint_model_info", timeout=120)[0]
        print("JOINT_WORKER_READY " + json.dumps(before), flush=True)
        first, first_replay = await _generate(engine, version=0, request_id="joint-smoke-before")
        model = BagelForSFT.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        model.to("npu:0").eval()
        consistency = []

        def compare_log_probs(record):
            with torch.no_grad():
                actor = (
                    replay_thinking_logprobs(
                        model,
                        record.ar_prompt_token_ids.tolist(),
                        record.response_token_ids.tolist(),
                        temperature=record.temperature,
                    )
                    .float()
                    .cpu()
                )
            delta = (actor - record.rollout_log_probs).abs()
            if not bool(torch.isfinite(delta).all()):
                raise RuntimeError("Nonfinite rollout/actor logprob difference")
            return {"mae": float(delta.mean()), "max_abs": float(delta.max())}

        consistency.append(compare_log_probs(first_replay))
        # Change both MoT paths and image/time projections so a version-only
        # acknowledgement cannot accidentally pass this real worker sync test.
        attention_prefix = "language_model.model.layers.0.self_attn"
        actor_probe_names = {
            f"{attention_prefix}.qkv_proj.weight": "layers.0.self_attn.q_proj.weight",
            f"{attention_prefix}.qkv_proj.gen_exp.weight": "layers.0.self_attn.q_proj_moe_gen.weight",
            "bagel.vae2llm.weight": "vae2llm.weight",
            "bagel.llm2vae.weight": "llm2vae.weight",
            "bagel.time_embedder.mlp.0.weight": "time_embedder.mlp.0.weight",
        }
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name in actor_probe_names.values():
                parameters[name].flatten()[:8].add_(0.25)
        weights = {"transformer." + name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        bucket = Path(args.output) / "actor-weight-bucket.pt"
        bucket.parent.mkdir(parents=True, exist_ok=True)
        torch.save(weights, bucket)
        engine.collective_rpc("begin_joint_policy_sync", args=(1, list(weights)), timeout=120)
        engine.collective_rpc("load_joint_weight_file", args=(str(bucket.resolve()),), timeout=120)
        synced = engine.collective_rpc("commit_joint_policy_sync", timeout=120)[0]
        after_sync = engine.collective_rpc("joint_model_info", timeout=120)[0]
        for native_name, actor_name in actor_probe_names.items():
            observed = torch.tensor(after_sync["probes"][native_name]["values"])
            expected = weights["transformer." + actor_name].flatten()[:8].float()
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            if after_sync["probes"][native_name]["values"] == before["probes"][native_name]["values"]:
                raise RuntimeError(f"Policy sync did not change native parameter {native_name}")
        second, second_replay = await _generate(engine, version=1, request_id="joint-smoke-after")
        consistency.append(compare_log_probs(second_replay))
        torch.save({"before": first_replay, "after": second_replay}, Path(args.output) / "replays.pt")
        result = {
            "backend": "vllm-omni DiffusionEngine with MultiprocDiffusionExecutor",
            "worker": before,
            "sync": synced,
            "synced_worker": after_sync,
            "changed_parameter_probes_verified": len(actor_probe_names),
            "rollout_actor_logprob_difference": consistency,
            "before": first,
            "after": second,
            "scope": "tiny random BAGEL; not full-weight training or convergence",
        }
        (Path(args.output) / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("BAGEL_JOINT_VLLM_NPU_OK " + json.dumps(result), flush=True)
    finally:
        engine.close()


def main():
    """Run one actual worker before and after complete actor policy synchronization."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="reproduction/models/tiny-bagel")
    parser.add_argument("--output", default="reproduction/joint-vllm-smoke")
    args = parser.parse_args()
    multiprocessing.set_start_method("spawn", force=True)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
