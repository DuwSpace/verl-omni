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

"""Score saved native joint rollouts with all five pinned OmniNFT rewards."""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
from verl_omni.pipelines.ltx2_unigrpo_nft.replay import LTXJointReplay
from verl_omni.pipelines.ltx2_unigrpo_nft.rewards import LTXJointRewardRunner


def reward_config(root, fps):
    config = OmegaConf.merge(
        OmegaConf.load(Path(__file__).with_name("rewards.yaml")),
        {
            "actor_rollout_ref": {
                "rollout": {"pipeline": {"frame_rate": fps, "output_type": "video"}},
                "actor": {
                    "data_loader_seed": 42,
                    "diffusion_loss": {
                        "adv_clip_max": 5.0,
                        "text_probability_reduction": "mean",
                        "text_advantage_epsilon": 1e-6,
                    },
                },
            },
            "algorithm": {
                "norm_adv_by_std_in_grpo": True,
                "global_std": True,
                "adv_mode": "continuous",
                "timestep_fraction": 1.0,
            },
        },
    )
    paths = {
        "video_align": "VideoReward/checkpoint-11352/model.pth",
        "hpsv3": "HPSv3/HPSv3.safetensors",
        "audiobox": "audiobox-aesthetics",
        "clap": "checkpoints/clap-htsat-unfused",
        "desync": "synchformer/synchformer_state_dict.pth",
    }
    for name, path in paths.items():
        config.reward.models[name].model_path = str(root / path)
    config.reward.models.video_align.executor.kwargs = {"base_model_path": str(root / "Qwen2-VL-2B-Instruct")}
    config.reward.models.hpsv3.executor.kwargs.base_model_path = str(root / "Qwen2-VL-7B-Instruct")
    config.reward.models.desync.executor.kwargs = {"source_root": str(root / "OmniNFT-reference")}
    return config


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True, help="Original LTX model, for legacy sample-rate metadata")
    parser.add_argument("--reward-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    samples = []
    for path in args.sample:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        record = LTXJointReplay(**sample["replay"])
        sample["replay"] = record
        if sample["audio"].ndim == 3 and sample["audio"].shape[0] == 1:
            sample["audio"] = sample["audio"][0]
        sample.setdefault("fps", record.generation_params["frame_rate"])
        sample.setdefault(
            "audio_sample_rate", json.loads((args.model / "vocoder/config.json").read_text())["output_sampling_rate"]
        )
        samples.append(sample)
    config = reward_config(args.reward_root, samples[0]["fps"])
    OmegaConf.save(config, args.output / "resolved_config.yaml", resolve=True)
    scores, names, infos, metrics = await LTXJointRewardRunner(config).score(samples)
    batch = DataProto.from_dict(
        tensors={
            **{
                key: torch.cat([s[key] for s in samples])
                for key in (
                    "video_latents_clean",
                    "audio_latents_clean",
                    "train_timesteps",
                )
            },
            "rm_scores": scores,
        },
        non_tensors={
            "ltx_joint_replay": np.array([s["replay"] for s in samples], dtype=object),
            "uid": np.array([s.get("uid", s["replay"].original_prompt) for s in samples], dtype=object),
        },
        meta_info={"reward_names": names},
    )
    batch = LTXJointLoss.prepare_actor_batch(batch, scores, config)
    report = {
        "source_commit": "66595248b0407749eb0a7cccc5107f8642519d2a",
        "sample_files": [str(path) for path in args.sample],
        "original_prompts": [s["replay"].original_prompt for s in samples],
        "reward_names": names,
        "scores": scores.tolist(),
        "scorer_details": infos,
        "reward_probability_video_audio": batch.batch["reward_prob"][:, 0].tolist(),
        "text_reward_probability": batch.batch["text_reward_prob"].tolist(),
        "text_advantages": batch.batch["text_advantages"].tolist(),
        "metrics": {**metrics, **batch.meta_info["text_reward_metrics"]},
    }
    (args.output / "scores.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
