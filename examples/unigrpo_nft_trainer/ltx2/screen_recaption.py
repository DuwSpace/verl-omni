# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Screen saved native recaption logprobs using the actual actor computation.

This diagnostic does not sample, update parameters, or replace paired MSProbe
collection. It preserves per-token evidence rather than declaring a tolerance.
Only load trusted, locally generated torch sample files.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import Gemma3ForConditionalGeneration

from verl_omni.pipelines.ltx2_unigrpo_nft.model import LTXJointModel
from verl_omni.pipelines.ltx2_unigrpo_nft.replay import LTXJointReplay
from verl_omni.pipelines.ltx2_unigrpo_nft.weight_sync import validate_joint_lora_bundle
from verl_omni.trainer.diffusion.recaption_grpo import old_policy_logprob_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--sample", required=True, action="append")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--clip-ratio", type=float, default=0.01)
    parser.add_argument("--cached-comparison", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    policy = Path(args.policy)
    manifest = validate_joint_lora_bundle(policy)
    if Path(manifest["base_model"]).resolve() != Path(args.model).resolve():
        raise ValueError("Screening base model differs from the paired policy manifest")
    records = []
    for path in args.sample:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        record = saved["replay"]
        if isinstance(record, dict):
            record = LTXJointReplay(**record)
        record.validate()
        if record.policy_version != manifest["policy_version"]:
            raise ValueError("Saved rollout and supplied old policy versions differ")
        records.append(record)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "npu":
        import torch_npu  # noqa: F401

    start = perf_counter()
    gemma = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, subfolder="text_encoder", torch_dtype=getattr(torch, args.dtype), attn_implementation="sdpa"
    )
    inject_adapter_in_model(LoraConfig.from_pretrained(policy / "gemma"), gemma, adapter_name="joint")
    loaded = set_peft_model_state_dict(
        gemma, load_file(policy / "gemma/adapter_model.safetensors"), adapter_name="joint"
    )
    if loaded.unexpected_keys or any("lora_" in name for name in loaded.missing_keys):
        raise ValueError("Screening Gemma LoRA did not load exactly")
    gemma.requires_grad_(False).eval().to(device)
    loading = perf_counter() - start
    results = []
    with torch.no_grad():
        for index, (path, record) in enumerate(zip(args.sample, records, strict=True)):
            start = perf_counter()
            inputs = record.ar_inputs(device)
            # Invoke precisely the training method, without loading the AV weights.
            full = LTXJointModel.recaption_logprobs(SimpleNamespace(gemma=gemma), **inputs)
            train = full[inputs["response_mask"][:, 1:]].float().cpu()
            rollout = record.rollout_logprobs
            metrics = old_policy_logprob_metrics(
                train[None], rollout[None], torch.ones_like(train[None], dtype=torch.bool), clip_ratio=args.clip_ratio
            )
            cached = None
            if args.cached_comparison:
                # Teacher-force the saved sequence one token at a time. This
                # diagnoses cache/shape effects; it never samples a new rollout.
                cache, values = None, []
                ids = record.prompt_token_ids.to(device)[None]
                length = ids.shape[1]
                for token in record.response_token_ids:
                    prediction = gemma(
                        input_ids=ids,
                        attention_mask=torch.ones((1, length), dtype=torch.long, device=device),
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                        return_dict=True,
                    )
                    values.append((prediction.logits[0, -1].float() / record.temperature).log_softmax(-1)[token])
                    cache = prediction.past_key_values
                    ids = token.to(device).reshape(1, 1)
                    length += 1
                cached = torch.stack(values).cpu()
                metrics["cached_vs_rollout"] = old_policy_logprob_metrics(
                    cached[None],
                    rollout[None],
                    torch.ones_like(cached[None], dtype=torch.bool),
                    clip_ratio=args.clip_ratio,
                )
                metrics["full_vs_cached"] = old_policy_logprob_metrics(
                    train[None],
                    cached[None],
                    torch.ones_like(cached[None], dtype=torch.bool),
                    clip_ratio=args.clip_ratio,
                )
                del cache, prediction
            torch.save(
                {
                    "response_token_ids": record.response_token_ids,
                    "train_old_logprobs": train,
                    "rollout_logprobs": rollout,
                    "prompt_token_ids": record.prompt_token_ids,
                    "cached_logprobs": cached,
                },
                output / f"tokens-{index:03d}.pt",
            )
            results.append(
                {
                    "sample": str(path),
                    "seed": record.seed,
                    "policy_version": record.policy_version,
                    "temperature": record.temperature,
                    "tokens": train.numel(),
                    "replay_time": perf_counter() - start,
                    "metrics": metrics,
                }
            )
            print(json.dumps(results[-1]), flush=True)
    report = {
        "kind": "screening_only",
        "manifest": manifest,
        "model_load_time": loading,
        "clip_ratio": args.clip_ratio,
        "dtype": args.dtype,
        "attention": "sdpa",
        "samples": results,
    }
    (output / "screening.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
