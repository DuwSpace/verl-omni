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

"""Real native-worker inference probe; this is not a joint-training acceptance run."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch
from peft import LoraConfig

from verl_omni.pipelines.ltx2_unigrpo_nft.weight_sync import validate_joint_lora_bundle, write_joint_lora_bundle
from verl_omni.workers.rollout.ltx_joint_backend import LTXJointVllmBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="A small bell rings on a wooden table in a quiet room.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--recaption-tokens", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-frozen-components", action="store_true")
    parser.add_argument(
        "--policy-bundle", type=Path, help="Existing complete paired LoRA export; defaults to zero-B probe"
    )
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = Path(args.model)
    if args.policy_bundle is not None:
        policy_directory = args.policy_bundle.resolve()
        version = validate_joint_lora_bundle(policy_directory)["policy_version"]
    else:
        policy_directory = output / "policy-0"
        version = 0
        gemma = json.loads((model / "text_encoder/config.json").read_text())["text_config"]
        dit = json.loads((model / "transformer/config.json").read_text())
        # Zero-B adapters are base-equivalent and only probe transport/native kernels.
        # Real training must export all selected actor adapters, never use this probe bundle.
        dimensions = {
            "gemma.model.language_model.layers.0.self_attn.q_proj": (
                gemma["hidden_size"],
                gemma["num_attention_heads"] * gemma["head_dim"],
            ),
            "transformer.transformer_blocks.0.attn1.to_q": (dit["num_attention_heads"] * dit["attention_head_dim"],)
            * 2,
        }
        torch.manual_seed(42)
        weights = {}
        for name, (inputs, outputs) in dimensions.items():
            weights[name + ".lora_A.default.weight"] = torch.randn(8, inputs) * 0.01
            weights[name + ".lora_B.default.weight"] = torch.zeros(outputs, 8)
        write_joint_lora_bundle(
            output / "policy-0", weights, LoraConfig(r=8, lora_alpha=8), version=0, base_model=str(model)
        )
    start = perf_counter()
    backend = LTXJointVllmBackend(
        model_path=model,
        geometry={
            "height": args.height,
            "width": args.width,
            "num_frames": args.frames,
            "frame_rate": 24.0,
            "num_inference_steps": args.steps,
            "max_sequence_length": 1024,
            "extra_args": {
                "video_cfg_scale": 1.0,
                "audio_cfg_scale": 1.0,
                "video_modality_scale": 1.0,
                "audio_modality_scale": 1.0,
                "video_rescale_scale": 0.0,
                "audio_rescale_scale": 0.0,
            },
        },
        recaption_temperature=1.0,
        recaption_max_new_tokens=args.recaption_tokens,
        sync_timeout_seconds=300,
    )
    loading = perf_counter() - start
    try:
        start = perf_counter()
        backend.load_policy(policy_directory)
        sync = perf_counter() - start
        frozen_before = None
        if args.audit_frozen_components:
            frozen_before = backend.engine.collective_rpc("snapshot_joint_frozen_components", timeout=300)
            (output / "frozen_before.json").write_text(json.dumps(frozen_before, indent=2))
        rollout_start = perf_counter()
        results = backend.generate(
            [args.prompt] * args.num_samples, list(range(args.seed, args.seed + args.num_samples)), version=version
        )
        rollout_time = perf_counter() - rollout_start
        if args.audit_frozen_components:
            frozen_after = backend.engine.collective_rpc("snapshot_joint_frozen_components", timeout=300)
            (output / "frozen_after.json").write_text(json.dumps(frozen_after, indent=2))
            if not frozen_before or frozen_before != frozen_after:
                raise RuntimeError("Native frozen component parameters changed during generation")
        summaries = []
        for index, sample in enumerate(results):
            record = sample["replay"]
            sample["uid"] = "prompt-0"
            sample["replay"] = asdict(record)
            stem = "sample" if index == 0 else f"sample-{index:03d}"
            torch.save(sample, output / f"{stem}.pt")
            import imageio.v2 as imageio
            import soundfile as sf

            frames = sample["video"].float().clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).numpy()
            imageio.mimwrite(output / f"{stem}.mp4", frames, fps=sample["fps"], codec="libx264")
            sf.write(output / f"{stem}.wav", sample["audio"].float().T.numpy(), sample["audio_sample_rate"])
            summaries.append(
                {
                    "sample_file": f"{stem}.pt",
                    "uid": sample["uid"],
                    "seed": record.seed,
                    "original_prompt": record.original_prompt,
                    "recaption": record.recaption,
                    "recaption_tokens": record.response_token_ids.numel(),
                    "recaption_ended_with_eos": int(record.response_token_ids[-1]) in record.eos_token_ids,
                    "policy_version": record.policy_version,
                    "recaption_time": sample["recaption_time"],
                    "generation_time": sample["generation_time"],
                    "video_shape": list(sample["video"].shape),
                    "audio_shape": list(sample["audio"].shape),
                    "fps": sample["fps"],
                    "audio_sample_rate": sample["audio_sample_rate"],
                }
            )
        summary = {
            "policy_bundle": str(policy_directory),
            "policy_version": version,
            "model_load_time": loading,
            "sync_time": sync,
            "rollout_time": rollout_time,
            "samples_per_second": len(results) / rollout_time,
            "frozen_components_audited": args.audit_frozen_components,
            "samples": summaries,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
