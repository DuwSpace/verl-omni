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


@pytest.mark.parametrize("policy_loss", ["grpo", "gspo"])
def test_microbatch_sample_weighting_matches_full_batch_gradient(policy_loss):
    mask = torch.tensor([[True, True, True], [True, False, False], [True, True, False]])
    advantage = torch.tensor([-1.0, 0.5, 1.0])
    old = torch.zeros(3, 3)
    full = torch.nn.Parameter(torch.tensor(0.02))
    loss, _ = clipped_recaption_loss(full.expand(3, 3), old, advantage, mask, clip_ratio=0.2, policy_loss=policy_loss)
    loss.backward()
    micro = torch.nn.Parameter(torch.tensor(0.02))
    for index in (slice(0, 2), slice(2, 3)):
        local_mask = mask[index]
        loss, _ = clipped_recaption_loss(
            micro.expand(local_mask.shape),
            old[index],
            advantage[index],
            local_mask,
            clip_ratio=0.2,
            policy_loss=policy_loss,
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


def test_initial_zero_mean_loss_still_has_policy_gradient():
    current = torch.zeros(2, 3, requires_grad=True)
    advantages = torch.tensor([-1.0, 1.0])
    loss, metrics = clipped_recaption_loss(
        current, current.detach().clone(), advantages, torch.ones(2, 3, dtype=torch.bool), clip_ratio=0.01
    )
    assert loss.item() == pytest.approx(0.0)
    assert metrics["text/grpo_abs_sample_loss"] == pytest.approx(1.0)
    loss.backward()
    assert current.grad[0].gt(0).all() and current.grad[1].lt(0).all()


def test_reference_kl_masks_padding_and_only_differentiates_current():
    from verl_omni.trainer.diffusion.recaption_grpo import recaption_reference_kl

    current = torch.tensor([[0.2, float("nan")], [-0.3, 0.1]], requires_grad=True)
    reference = torch.zeros(2, 2, requires_grad=True)
    mask = torch.tensor([[True, False], [True, True]])
    loss = recaption_reference_kl(current, reference, mask)
    expected = (
        (torch.expm1(torch.tensor(-0.2)) + 0.2)
        + (torch.expm1(torch.tensor(0.3)) - 0.3 + torch.expm1(torch.tensor(-0.1)) + 0.1) / 2
    ) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert reference.grad is None
    assert current.grad[0, 1] == 0
    assert torch.isfinite(current.grad).all()
    assert current.grad[0, 0] > 0 and current.grad[1, 0] < 0


def test_text_kl_coefficient_and_zero_at_reference():
    from types import SimpleNamespace

    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    config = SimpleNamespace(diffusion_loss=LTXJointLossConfig(text_kl_coef=0.001))
    current = torch.tensor([[0.2]], requires_grad=True)
    arguments = dict(
        current=current,
        old=current.detach(),
        advantages=torch.zeros(1),
        mask=torch.ones(1, 1, dtype=torch.bool),
        config=config,
    )
    loss, metrics = LTXJointLoss.text_loss(**arguments, reference=torch.zeros_like(current))
    assert loss.item() == pytest.approx(0.001 * metrics["text/kl_loss"])
    assert metrics["text/kl_coef"] == 0.001
    loss.backward()
    assert current.grad.item() > 0
    current.grad = None
    loss, metrics = LTXJointLoss.text_loss(**arguments, reference=current.detach().clone())
    loss.backward()
    assert loss.item() == 0 and current.grad.item() == 0
    with pytest.raises(ValueError, match="reference logprobs"):
        LTXJointLoss.text_loss(**arguments)


@pytest.mark.parametrize("coefficient", [-1, float("nan"), float("inf")])
def test_invalid_text_kl_coefficient_rejected(coefficient):
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    with pytest.raises(ValueError, match="text_kl_coef"):
        LTXJointLossConfig(text_kl_coef=coefficient)


def test_gspo_uses_geometric_sequence_ratio_and_masks_padding():
    # Token ratios 2 and 1/2 cancel at sequence level, but clip differently in GRPO.
    current = torch.tensor([[2.0, 0.5, 1.0], [1.0, 1.0, 1.0]]).log().requires_grad_()
    old = torch.zeros_like(current, requires_grad=True)
    advantage = torch.tensor([1.0, -1.0], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    with torch.no_grad():
        current[0, 2] = float("nan")
    loss, metrics = clipped_recaption_loss(current, old, advantage, mask, clip_ratio=0.1, policy_loss="gspo")
    assert loss.item() == pytest.approx(0.0)
    assert metrics["text/ratio_mean"] == pytest.approx(1.0)
    assert metrics["text/clip_fraction"] == 0
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([[-0.25, -0.25, 0.0], [0.5, 0.0, 0.0]]))
    assert old.grad is None and advantage.grad is None


def test_gspo_clips_whole_sequences_for_both_advantage_signs():
    current = torch.tensor([[1.4, 1.4], [0.6, 0.6]]).log().requires_grad_()
    loss, metrics = clipped_recaption_loss(
        current,
        torch.zeros_like(current),
        torch.tensor([1.0, -1.0]),
        torch.ones_like(current, dtype=torch.bool),
        clip_ratio=0.2,
        policy_loss="gspo",
    )
    assert loss.item() == pytest.approx(-0.2)
    assert metrics["text/clip_fraction"] == 1
    loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current))


def test_gspo_and_grpo_gradients_match_at_behavior_policy():
    mask = torch.tensor([[True, True], [True, False]])
    gradients = []
    for algorithm in ("grpo", "gspo"):
        current = torch.zeros(2, 2, requires_grad=True)
        loss, _ = clipped_recaption_loss(
            current, current.detach(), torch.tensor([1.0, -1.0]), mask, clip_ratio=0.01, policy_loss=algorithm
        )
        loss.backward()
        gradients.append(current.grad)
    torch.testing.assert_close(*gradients)


def test_joint_loss_dispatches_gspo_and_rejects_unknown_algorithm():
    from types import SimpleNamespace

    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    config = SimpleNamespace(diffusion_loss=LTXJointLossConfig(text_policy_loss="gspo"))
    current = torch.tensor([[2.0, 0.5]]).log().requires_grad_()
    loss, metrics = LTXJointLoss.text_loss(
        current=current,
        old=torch.zeros_like(current),
        advantages=torch.ones(1),
        mask=torch.ones_like(current, dtype=torch.bool),
        config=config,
    )
    assert loss.item() == pytest.approx(-1.0)
    assert "text/gspo_loss" in metrics and "text/grpo_loss" not in metrics
    with pytest.raises(ValueError, match="text_policy_loss"):
        LTXJointLossConfig(text_policy_loss="invalid")


def test_scalar_credit_handles_signed_interleaved_and_constant_groups():
    from verl_omni.trainer.diffusion.recaption_grpo import reward_group_advantages

    rewards = torch.tensor([-1.0, 9.0, 1.0, 9.0, 5.0], requires_grad=True)
    advantage, metrics = reward_group_advantages(rewards, ["a", "b", "a", "b", "c"])
    torch.testing.assert_close(advantage, torch.tensor([-1.0, 0.0, 1.0, 0.0, 0.0]), atol=2e-6, rtol=0)
    assert not advantage.requires_grad
    assert metrics["text/constant_group_fraction"] == pytest.approx(2 / 3)
    assert metrics["text/reward_mean"] == pytest.approx(4.6)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_scalar_credit_rejects_nonfinite_rewards(value):
    from verl_omni.trainer.diffusion.recaption_grpo import reward_group_advantages

    with pytest.raises(ValueError, match="finite"):
        reward_group_advantages(torch.tensor([value]), ["a"])


def test_text_credit_source_config_rejects_unknown_value():
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    assert LTXJointLossConfig().text_advantage_source == "modality_probability"
    assert LTXJointLossConfig(text_advantage_source="combined_reward").text_advantage_source == "combined_reward"
    with pytest.raises(ValueError, match="text_advantage_source"):
        LTXJointLossConfig(text_advantage_source="invalid")
