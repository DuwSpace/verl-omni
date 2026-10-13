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

"""Token GRPO and same-policy diagnostics for recaption-enabled diffusion actors."""

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch


def probability_group_advantages(
    probabilities: torch.Tensor, group_ids: Sequence[Any], *, epsilon: float = 1e-6
) -> tuple[torch.Tensor, dict[str, float]]:
    """Normalize sample reward probabilities within each original-prompt group.

    The caller selects its scalar probability from the reward pipeline; this
    helper never aggregates or reroutes the component scorers. Population std
    keeps singleton and equal-reward groups well-defined with zero advantage.
    """
    if probabilities.ndim != 1 or probabilities.numel() == 0 or len(group_ids) != probabilities.numel():
        raise ValueError("Recaption requires one scalar probability and group ID per sample")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Recaption advantage epsilon must be finite and positive")
    probabilities = probabilities.detach().float()
    if not bool(torch.isfinite(probabilities).all()) or bool(((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Recaption rewards must be finite probabilities in [0, 1]")
    groups = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        groups[group_id].append(index)
    advantages = torch.empty_like(probabilities)
    constant_groups = 0
    for indices in groups.values():
        values = probabilities[indices]
        constant = bool((values == values[0]).all())
        constant_groups += int(constant)
        if constant:
            advantages[indices] = 0
        else:
            advantages[indices] = (values - values.mean()) / (values.std(correction=0) + epsilon)
    return advantages, {
        "text/probability_mean": probabilities.mean().item(),
        "text/probability_std": probabilities.std(correction=0).item(),
        "text/constant_group_fraction": constant_groups / len(groups),
    }


def _validate_token_logprobs(current, old, mask):
    if current.ndim != 2 or current.shape != old.shape or current.shape != mask.shape:
        raise ValueError("Recaption logprobs and response mask must have identical [batch, tokens] shapes")
    if mask.dtype != torch.bool:
        raise ValueError("Recaption response mask must be boolean")
    if mask.shape[0] == 0 or not bool(mask.any(dim=-1).all()):
        raise ValueError("Every recaption must contain at least one valid generated token")
    if not bool(torch.isfinite(current[mask]).all() and torch.isfinite(old[mask]).all()):
        raise ValueError("Valid recaption token logprobs must be finite")


def clipped_recaption_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    clip_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average clipped token GRPO over tokens per response, then over samples.

    Frozen old probabilities and outcome advantages never receive gradients.
    Prompt and padding tokens are excluded, including from diagnostics. The
    caller scales this sample mean by its micro-batch fraction before backward.
    """
    _validate_token_logprobs(current_logprobs, old_logprobs, response_mask)
    if advantages.shape != current_logprobs.shape[:1] or not bool(torch.isfinite(advantages).all()):
        raise ValueError("Recaption requires one finite group-relative advantage per sample")
    if not math.isfinite(clip_ratio) or not 0 < clip_ratio < 1:
        raise ValueError("Recaption clip ratio must be finite and in (0, 1)")
    delta = torch.where(response_mask, current_logprobs.float() - old_logprobs.detach().float(), 0.0)
    ratio = delta.exp()
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError("Recaption current/old probability ratio overflowed")
    advantage = advantages.detach().float()[:, None]
    clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio)
    token_loss = -torch.minimum(ratio * advantage, clipped * advantage)
    sample_loss = torch.where(response_mask, token_loss, 0.0).sum(dim=-1) / response_mask.sum(dim=-1)
    loss = sample_loss.mean()
    valid_ratios = ratio.detach()[response_mask]
    return loss, {
        "text/grpo_loss": loss.detach().item(),
        "text/ratio_mean": valid_ratios.mean().item(),
        "text/clip_fraction": ((valid_ratios - 1).abs() > clip_ratio).float().mean().item(),
    }


def old_policy_logprob_metrics(
    train_old_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    clip_ratio: float,
) -> dict[str, float]:
    """Measure identical-old-policy replay differences, without inventing a pass threshold."""
    _validate_token_logprobs(train_old_logprobs, rollout_logprobs, response_mask)
    if not math.isfinite(clip_ratio) or not 0 < clip_ratio < 1:
        raise ValueError("Recaption clip ratio must be finite and in (0, 1)")
    delta = (train_old_logprobs.detach().float() - rollout_logprobs.detach().float())[response_mask]
    absolute = delta.abs()
    ratios = delta.exp()
    return {
        "consistency/old_logprob_abs_mean": absolute.mean().item(),
        "consistency/old_logprob_abs_p95": torch.quantile(absolute, 0.95).item(),
        "consistency/old_logprob_abs_p99": torch.quantile(absolute, 0.99).item(),
        "consistency/old_logprob_abs_max": absolute.max().item(),
        "consistency/old_ratio_mean": ratios.mean().item(),
        "consistency/old_ratio_abs_error_mean": (ratios - 1).abs().mean().item(),
        "consistency/old_ratio_abs_error_max": (ratios - 1).abs().max().item(),
        "consistency/old_ratio_clip_fraction": ((ratios - 1).abs() > clip_ratio).float().mean().item(),
    }
