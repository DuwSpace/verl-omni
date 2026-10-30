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

"""Sampling versus old-policy replay statistics, without additional forwards."""

import torch


@torch.no_grad()
def recaption_logprob_diff(
    recomputed: torch.Tensor, rollout_logprobs: list[torch.Tensor], response_mask: torch.Tensor
) -> dict[str, float]:
    """Compare temperature-scaled logprobs on sampled response tokens, including EOS.

    `recomputed` and `response_mask` have shape [B, L-1]; replay vectors contain
    only each response's tokens. Means first average tokens within a response,
    then average responses, matching the GRPO loss weighting. The maximum is
    over all valid tokens. Prompt and padding positions are excluded.
    """
    sampled = torch.cat(rollout_logprobs).to(device=recomputed.device, dtype=torch.float32)
    valid_diff = recomputed.detach()[response_mask].float() - sampled
    diff = torch.zeros_like(recomputed, dtype=torch.float32).masked_scatter_(response_mask, valid_diff)
    lengths = response_mask.sum(dim=-1)
    return {
        "text/logprob_diff_mean": (diff.sum(dim=-1) / lengths).mean().item(),
        "text/logprob_diff_abs_mean": (diff.abs().sum(dim=-1) / lengths).mean().item(),
        "text/logprob_diff_abs_max": valid_diff.abs().max().item(),
    }
