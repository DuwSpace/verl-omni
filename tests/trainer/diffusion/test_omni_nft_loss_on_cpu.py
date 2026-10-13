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

"""Numerical contracts for the pinned LTX OmniNFT reward routing and objective."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl import DataProto

from verl_omni.trainer.diffusion.diffusion_algos import OmniNFTLoss, get_diffusion_loss_fn
from verl_omni.workers.config.diffusion.actor import OmniNFTLossConfig


def _config(**kwargs):
    return SimpleNamespace(diffusion_loss=OmniNFTLossConfig(**kwargs), data_loader_seed=42)


def test_probability_mapping_keeps_components_and_modality_routing():
    # Same two outcomes, interleaved across two prompt groups. Two scorers are
    # constant; the remaining scorers disagree across modalities.
    scores = torch.tensor(
        [[-1.0, 0.0, 2.0, 3.0, 0.0], [1.0, 0.0, 0.0, 3.0, 2.0], [1.0, 0.0, 0.0, 3.0, 2.0], [-1.0, 0.0, 2.0, 3.0, 0.0]]
    )
    names = ["video_align", "hpsv3", "audiobox", "clap", "desync"]
    routes = [(1.0, 0.0), (1.5, 0.0), (0.0, 0.5), (0.0, 1.0), (1.0, 1.0)]
    config = SimpleNamespace(
        algorithm=SimpleNamespace(
            norm_adv_by_std_in_grpo=True, global_std=True, adv_mode="continuous", timestep_fraction=1.0
        ),
        actor_rollout_ref=SimpleNamespace(actor=_config()),
        reward=SimpleNamespace(
            reward_functions={
                name: {"routing_weights": {"video": v, "audio": a}} for name, (v, a) in zip(names, routes, strict=True)
            }
        ),
    )
    batch = DataProto(
        batch=TensorDict(
            {
                "rm_scores": scores,
                "reward_valid_mask": torch.ones_like(scores, dtype=torch.bool),
                "video_latents_clean": torch.zeros(4, 2, 3),
                "audio_latents_clean": torch.zeros(4, 1, 3),
                "train_timesteps": torch.tensor([[100.0, 500.0]]).repeat(4, 1),
            },
            batch_size=[4],
        ),
        non_tensor_batch={"uid": np.array(["a", "b", "a", "b"])},
        meta_info={"reward_names": names},
    )
    result = OmniNFTLoss.prepare_actor_batch(batch, scores, config)
    expected_adv = torch.tensor([[-2.0, -0.5], [2.0, 0.5], [2.0, 0.5], [-2.0, -0.5]]) / 1.0001
    expected_prob = (expected_adv / 10.0 + 0.5)[:, None, :].expand(4, 2, 2)
    torch.testing.assert_close(result.batch["reward_prob"], expected_prob)
    torch.testing.assert_close(result.batch["sample_level_rewards"], scores)
    assert result.batch["reward_prob"].shape == (4, 2, 2)
    assert isinstance(get_diffusion_loss_fn("omni_nft"), OmniNFTLoss)
    batch.batch["reward_valid_mask"][0, 0] = False
    with pytest.raises(ValueError, match="fully valid"):
        OmniNFTLoss.prepare_actor_batch(batch, scores, config)


def test_modality_objective_matches_hand_calculation_and_detaches_frozen_policies():
    config = _config(
        mix_beta=1.0, adv_clip_max=5.0, video_weight=2.0, audio_weight=1.0, video_ref_kl_coef=0.1, audio_ref_kl_coef=0.2
    )
    output = {}
    for modality in ("video", "audio"):
        output.update(
            {
                f"{modality}_forward_prediction": torch.tensor([[[0.4]]], requires_grad=True),
                f"{modality}_old_prediction": torch.tensor([[[0.2]]], requires_grad=True),
                f"{modality}_ref_forward_prediction": torch.tensor([[[0.1]]], requires_grad=True),
                f"{modality}_x0": torch.tensor([[[0.3]]]),
                f"{modality}_xt": torch.tensor([[[0.8]]]),
                f"{modality}_t_expanded": torch.tensor([[[0.5]]]),
            }
        )
    result = OmniNFTLoss()(
        config=config, model_output=output, data=TensorDict({"reward_prob": torch.tensor([[0.7, 0.2]])}, batch_size=[1])
    )
    # beta=1: positive x0=.6 (error=.3), negative x0=.8 (error=.5).
    # Adaptive scalar weights equal the absolute errors; policy is weighted MAE * 5.
    video = 5 * (0.7 * 0.3 + 0.3 * 0.5) + 0.1 * 0.3**2
    audio = 5 * (0.2 * 0.3 + 0.8 * 0.5) + 0.2 * 0.3**2
    assert result.loss.item() == pytest.approx((2 * video + audio) / 3)
    result.loss.backward()
    for modality in ("video", "audio"):
        assert output[f"{modality}_old_prediction"].grad is None
        assert output[f"{modality}_ref_forward_prediction"].grad is None
        gradient = output[f"{modality}_forward_prediction"].grad
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"video_weight": -1},
        {"audio_weight": float("nan")},
        {"video_ref_kl_coef": float("inf")},
        {"audio_ref_kl_coef": -1},
        {"video_weight": 0, "audio_weight": 0},
    ],
)
def test_invalid_modality_coefficients_are_rejected(kwargs):
    with pytest.raises(ValueError):
        OmniNFTLossConfig(**kwargs)


def test_joint_registered_loss_is_unnormalized_weighted_sum_with_both_gradients():
    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.trainer.diffusion.recaption_grpo import clipped_recaption_loss
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    config = SimpleNamespace(
        diffusion_loss=LTXJointLossConfig(text_loss_weight=2.0, av_loss_weight=3.0, ar_clip_ratio=0.2)
    )
    parameter = torch.nn.Parameter(torch.tensor(0.04))
    output = {"ar_logprobs": parameter.expand(1, 2)}
    for modality in ("video", "audio"):
        output.update(
            {
                f"{modality}_forward_prediction": parameter.expand(1, 2, 3),
                f"{modality}_old_prediction": torch.ones(1, 2, 3),
                f"{modality}_ref_forward_prediction": torch.zeros(1, 2, 3),
                f"{modality}_x0": torch.full((1, 2, 3), 0.3),
                f"{modality}_xt": torch.ones(1, 2, 3),
                f"{modality}_t_expanded": torch.full((1, 1, 1), 0.4),
            }
        )
    data = TensorDict(
        {
            "ar_old_logprobs": torch.zeros(1, 2),
            "text_advantages": torch.tensor([0.7]),
            "ar_response_mask": torch.ones(1, 2, dtype=torch.bool),
            "reward_prob": torch.tensor([[0.8, 0.6]]),
        },
        batch_size=[1],
    )
    text, _ = clipped_recaption_loss(
        output["ar_logprobs"],
        data["ar_old_logprobs"],
        data["text_advantages"],
        data["ar_response_mask"],
        clip_ratio=0.2,
    )
    av = OmniNFTLoss()(config=config, model_output=output, data=data).loss
    expected = 2 * text + 3 * av
    actual = LTXJointLoss()(config=config, model_output=output, data=data).loss
    torch.testing.assert_close(actual, expected)
    expected_gradient = torch.autograd.grad(expected, parameter, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, parameter)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert actual_gradient.abs() > 0
