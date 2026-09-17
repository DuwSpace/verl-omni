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
"""
Metrics for diffusion (image generation) training.
"""

from typing import Any, Literal

import numpy as np
import torch
from verl import DataProto


def compute_data_metrics_diffusion(batch: DataProto) -> dict[str, Any]:
    """
    Computes various metrics from a diffusion training batch.

    For diffusion (image generation) models, rewards and advantages are
    indexed over denoising timesteps rather than output tokens.

    Args:
        batch: A DataProto object containing diffusion batch data. GRPO-style
            batches include sample_level_rewards [B, T], advantages [B, T], and
            returns [B, T]. DPO-style batches may only include sample_level_rewards [B].

    Returns:
        A dictionary of metrics including:
            - critic/rewards/mean, max, min: Per-image reward statistics
            - critic/rewards/zero_std_ratio: Fraction of prompt groups whose reward std is zero
            - critic/rewards/std_mean: Mean per-prompt reward standard deviation
            - critic/rewards/group_size: Average number of images sampled per unique prompt
            - critic/advantages/mean, max, min: Element-wise advantage statistics over B*T, when available
            - critic/returns/mean, max, min: Element-wise return statistics over B*T, when available
    """
    sample_level_rewards = batch.batch["sample_level_rewards"]
    if sample_level_rewards.ndim > 1:
        sequence_reward = sample_level_rewards.mean(dim=1)  # [B]
    else:
        sequence_reward = sample_level_rewards  # [B]

    reward_mean = torch.mean(sequence_reward).detach().item()
    reward_max = torch.max(sequence_reward).detach().item()
    reward_min = torch.min(sequence_reward).detach().item()

    metrics = {
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
    }

    if "advantages" in batch.batch:
        # Flatten [B, T] tensors for aggregate statistics across timesteps.
        advantages = batch.batch["advantages"].flatten()  # [B*T]
        metrics.update(
            {
                "critic/advantages/mean": torch.mean(advantages).detach().item(),
                "critic/advantages/max": torch.max(advantages).detach().item(),
                "critic/advantages/min": torch.min(advantages).detach().item(),
            }
        )

    if "returns" in batch.batch:
        returns = batch.batch["returns"].flatten()  # [B*T]
        metrics.update(
            {
                "critic/returns/mean": torch.mean(returns).detach().item(),
                "critic/returns/max": torch.max(returns).detach().item(),
                "critic/returns/min": torch.min(returns).detach().item(),
            }
        )

    if "uid" in batch.non_tensor_batch:
        rewards_np = sequence_reward.cpu().float().numpy()
        uid_array = np.array(batch.non_tensor_batch["uid"])
        unique_uids = np.unique(uid_array)

        per_prompt_stds = np.array([np.std(rewards_np[uid_array == uid]) for uid in unique_uids])

        metrics["critic/rewards/zero_std_ratio"] = float(np.mean(per_prompt_stds == 0))
        metrics["critic/rewards/std_mean"] = float(np.mean(per_prompt_stds))
        metrics["critic/rewards/group_size"] = float(len(rewards_np) / len(unique_uids))

    return metrics


def compute_component_reward_metrics_diffusion(batch: DataProto) -> dict[str, float]:
    """Summarize ``rm_scores[B, K]`` in reward_names order without routing weights.

    Return per-column mean/population std/min/max and row-sum mean/population
    std as Python floats. Return an empty dict when names or scores are absent;
    raise ValueError when score shape and column names disagree.
    """
    reward_names = list(batch.meta_info.get("reward_names") or [])
    scores = batch.batch.get("rm_scores")
    if not reward_names or scores is None:
        return {}
    if scores.ndim != 2 or scores.shape[1] != len(reward_names):
        raise ValueError(
            f"Component reward scores must have shape (B, {len(reward_names)}), got {tuple(scores.shape)}."
        )

    scores = scores.detach().float()
    combined = scores.sum(dim=1)
    metrics = {
        "train/reward/sum/mean": float(combined.mean()),
        "train/reward/sum/std": float(combined.std(unbiased=False)),
    }
    for index, name in enumerate(reward_names):
        column = scores[:, index]
        metrics.update(
            {
                f"train/reward/{name}/mean": float(column.mean()),
                f"train/reward/{name}/std": float(column.std(unbiased=False)),
                f"train/reward/{name}/min": float(column.min()),
                f"train/reward/{name}/max": float(column.max()),
            }
        )
    return metrics


def compute_old_policy_metrics(
    update_result: tuple[bool, float, Literal["none", "copy", "ema"]],
) -> dict[str, Any]:
    """Build metrics for old-policy adapter refreshes."""
    update_applied, decay, update_type = update_result
    return {
        "old_policy/update_applied": float(update_applied),
        "old_policy/copy_update": float(update_type == "copy"),
        "old_policy/ema_update": float(update_type == "ema"),
        "old_policy/decay": float(decay),
    }


def compute_timing_metrics_diffusion(timing_raw: dict[str, float], num_images: int) -> dict[str, Any]:
    """
    Computes timing metrics for diffusion training.

    Args:
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
        num_images: Total number of images processed in the batch, used to compute per-image timing.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_image_ms/{name}: Per-image timing in milliseconds for core compute stages
              (gen, ref, old_log_prob, adv, update_actor). Non-compute stages such as
              save_checkpoint, update_weights, and testing are excluded.
    """
    num_images_of_section = {name: num_images for name in ["gen", "ref", "old_log_prob", "adv", "update_actor"]}

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_image_ms/{name}": timing_raw[name] * 1000 / num_images_of_section[name]
            for name in set(num_images_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughput_metrics_diffusion(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for diffusion (image/video generation) training.

    Unlike language model training where throughput is measured in tokens/sec,
    diffusion training generates images, so throughput is reported as images
    per second.

    Args:
        batch: A DataProto object containing diffusion batch data.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_images: Number of images processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Images generated per second per GPU
    """
    if "advantages" in batch.batch:
        batch_size = batch.batch["advantages"].shape[0]
    else:
        batch_size = batch.batch["sample_level_rewards"].shape[0]
    time = timing_raw["step"]
    return {
        "perf/total_num_images": batch_size,
        "perf/time_per_step": time,
        "perf/throughput": batch_size / (time * n_gpus),
    }


def compute_reward_extra_metrics_diffusion(reward_extra_infos_dict: dict) -> dict[str, Any]:
    """Computes per-sub-reward mean metrics for multi-reward tracking."""
    metrics = {}
    if not reward_extra_infos_dict:
        return metrics
    for key, values in reward_extra_infos_dict.items():
        if isinstance(values, np.ndarray):
            if not np.issubdtype(values.dtype, np.number):
                continue
            metrics[f"critic/{key}/mean"] = float(values.mean())
        elif isinstance(values, list) and len(values) > 0:
            if not isinstance(values[0], int | float):
                continue
            metrics[f"critic/{key}/mean"] = float(np.mean(values))
    return metrics
