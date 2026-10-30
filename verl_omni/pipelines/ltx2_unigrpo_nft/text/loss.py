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

"""Token-level GRPO and sampled reference KL for recaptioning."""

import math

import torch


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
    """Sample-mean token GRPO.

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
        "text/grpo_abs_sample_loss": sample_loss.detach().abs().mean().item(),
        "text/ratio_mean": valid_ratios.mean().item(),
        "text/clip_fraction": ((valid_ratios - 1).abs() > clip_ratio).float().mean().item(),
    }


def recaption_reference_kl(current_logprobs, reference_logprobs, response_mask):
    """Sampled-token k3 penalty, averaged over response tokens then samples.

    This is an estimate on rollout tokens, not a full-vocabulary KL. Only the
    current policy receives gradients; padding is removed before exponentiation.
    """
    _validate_token_logprobs(current_logprobs, reference_logprobs, response_mask)
    delta = torch.where(response_mask, reference_logprobs.detach().float() - current_logprobs.float(), 0.0)
    penalty = delta.expm1() - delta
    if not bool(torch.isfinite(penalty).all()):
        raise ValueError("Recaption reference KL overflowed")
    return (penalty.sum(dim=-1) / response_mask.sum(dim=-1)).mean()
