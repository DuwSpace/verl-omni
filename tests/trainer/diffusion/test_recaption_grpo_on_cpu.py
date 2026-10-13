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

"""Recaption GRPO credit, masking, clipping, and replay diagnostics."""

import pytest
import torch

from verl_omni.trainer.diffusion.recaption_grpo import (
    clipped_recaption_loss,
    old_policy_logprob_metrics,
    probability_group_advantages,
)


def test_probabilities_are_centered_by_original_prompt_and_constant_groups_are_zero():
    p = torch.tensor([0.2, 0.9, 0.8, 0.9, 0.4], requires_grad=True)
    advantage, metrics = probability_group_advantages(p, ["a", "b", "a", "b", "c"])
    torch.testing.assert_close(advantage, torch.tensor([-1.0, 0.0, 1.0, 0.0, 0.0]), atol=4e-6, rtol=0)
    assert not advantage.requires_grad
    assert metrics["text/constant_group_fraction"] == pytest.approx(2 / 3)


def test_clipped_loss_excludes_padding_and_freezes_denominator_and_advantage():
    ratios = torch.tensor([[1.4, 1.0, 1.0], [0.6, 1.0, 1.0]])
    current = ratios.log().requires_grad_()
    old = torch.zeros_like(current, requires_grad=True)
    advantage = torch.tensor([1.0, -1.0], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    # A masked nonfinite rollout value must neither enter the loss nor its metrics.
    with torch.no_grad():
        old[0, 2] = float("nan")
    loss, metrics = clipped_recaption_loss(current, old, advantage, mask, clip_ratio=0.2)
    # First sample: -(1.2 + 1)/2. Second sample: max(.6, .8).
    assert loss.item() == pytest.approx((-(1.2 + 1) / 2 + 0.8) / 2)
    assert metrics["text/clip_fraction"] == pytest.approx(2 / 3)
    loss.backward()
    assert old.grad is None and advantage.grad is None
    torch.testing.assert_close(current.grad, torch.tensor([[0.0, -0.25, 0.0], [0.0, 0.0, 0.0]]))


def test_microbatch_sample_weighting_matches_full_batch_gradient():
    mask = torch.tensor([[True, True, True], [True, False, False], [True, True, False]])
    advantage = torch.tensor([-1.0, 0.5, 1.0])
    old = torch.zeros(3, 3)
    full = torch.nn.Parameter(torch.tensor(0.02))
    loss, _ = clipped_recaption_loss(full.expand(3, 3), old, advantage, mask, clip_ratio=0.2)
    loss.backward()
    micro = torch.nn.Parameter(torch.tensor(0.02))
    for index in (slice(0, 2), slice(2, 3)):
        local_mask = mask[index]
        loss, _ = clipped_recaption_loss(
            micro.expand(local_mask.shape), old[index], advantage[index], local_mask, clip_ratio=0.2
        )
        (loss * local_mask.shape[0] / 3).backward()
    torch.testing.assert_close(micro.grad, full.grad)


def test_same_old_replay_metrics_ignore_padding_and_expose_clipping_scale():
    train = torch.tensor([[0.0, 0.01, 100.0], [0.3, 0.0, 0.0]])
    rollout = torch.zeros_like(train)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    metrics = old_policy_logprob_metrics(train, rollout, mask, clip_ratio=0.2)
    assert metrics["consistency/old_logprob_abs_mean"] == pytest.approx(0.31 / 3)
    assert metrics["consistency/old_logprob_abs_max"] == pytest.approx(0.3)
    assert metrics["consistency/old_ratio_clip_fraction"] == pytest.approx(1 / 3)
    identical = old_policy_logprob_metrics(rollout, rollout, mask, clip_ratio=0.2)
    assert identical["consistency/old_ratio_mean"] == 1.0
    assert identical["consistency/old_ratio_abs_error_max"] == 0.0


@pytest.mark.parametrize("values", [[float("nan")], [-0.1], [1.1], []])
def test_invalid_probability_rewards_fail(values):
    with pytest.raises(ValueError):
        probability_group_advantages(torch.tensor(values), ["a"] * len(values))


def test_empty_generated_response_fails():
    with pytest.raises(ValueError, match="at least one"):
        clipped_recaption_loss(
            torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1), torch.zeros(1, 2, dtype=torch.bool), clip_ratio=0.2
        )
