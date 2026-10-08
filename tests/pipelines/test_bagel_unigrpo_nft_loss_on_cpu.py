# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hybrid objective algebra, gradient ownership, and continuous-time preparation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl import DataProto

from verl_omni.pipelines.bagel_unigrpo_nft.loss import BagelUniGRPONFTLoss
from verl_omni.pipelines.bagel_unigrpo_nft.replay import BagelJointReplay
from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig
from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn
from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig


def _cfg(**kwargs):
    return SimpleNamespace(diffusion_loss=DiffusionLossConfig(loss_mode="unigrpo_nft", ref_kl_coef=1.5e-5, **kwargs))


def _inputs(parameter):
    output = {
        "ar_log_probs": parameter * torch.tensor([0.2, -0.1, 0.3]),
        "forward_prediction": parameter.expand(1, 2, 3),
        "old_prediction": torch.ones(1, 2, 3, requires_grad=True),
        "ref_forward_prediction": torch.zeros(1, 2, 3, requires_grad=True),
        "x0": torch.arange(6).reshape(1, 2, 3).float(),
        "xt": torch.ones(1, 2, 3),
        "t_expanded": torch.full((1, 1, 1), 0.4),
    }
    data = {
        "ar_old_log_probs": torch.zeros(3, requires_grad=True),
        "ar_advantages": torch.tensor([0.8, -0.4, 0.8]),
        "reward_prob": torch.tensor([0.7]),
    }
    return output, data


def test_hybrid_loss_has_distinct_registry_and_config_mode():
    assert isinstance(get_diffusion_loss_fn("unigrpo_nft"), BagelUniGRPONFTLoss)
    assert _cfg().diffusion_loss.ar_clip_ratio == pytest.approx(0.01)


def test_two_backwards_equal_joint_backward_and_only_current_policy_gets_gradients():
    cfg = _cfg(text_loss_weight=2, image_loss_weight=3)
    joint_parameter = torch.nn.Parameter(torch.tensor(0.7))
    output, data = _inputs(joint_parameter)
    joint_loss, _ = BagelUniGRPONFTLoss.compute_loss(model_output=output, data=data, config=cfg)
    joint_loss.backward()
    assert output["old_prediction"].grad is None
    assert output["ref_forward_prediction"].grad is None
    assert data["ar_old_log_probs"].grad is None
    separate_parameter = torch.nn.Parameter(torch.tensor(0.7))
    output, data = _inputs(separate_parameter)
    ar_loss, _ = BagelUniGRPONFTLoss.compute_ar_loss(
        new_log_probs=output["ar_log_probs"],
        old_log_probs=data["ar_old_log_probs"],
        advantages=data["ar_advantages"],
        config=cfg,
    )
    ar_loss.backward()
    image_loss, _ = BagelUniGRPONFTLoss.compute_image_loss(
        model_output=output, reward_prob=data["reward_prob"], config=cfg
    )
    image_loss.backward()
    torch.testing.assert_close(separate_parameter.grad, joint_parameter.grad)
    assert torch.isfinite(joint_parameter.grad) and float(joint_parameter.grad.abs()) > 0
    torch.testing.assert_close(ar_loss.detach() + image_loss.detach(), joint_loss.detach())


def test_missing_exact_old_policy_inputs_fail_closed():
    output, data = _inputs(torch.nn.Parameter(torch.tensor(0.7)))
    del data["ar_old_log_probs"]
    with pytest.raises(KeyError, match="ar_old_log_probs"):
        BagelUniGRPONFTLoss()(config=_cfg(), model_output=output, data=data)


def _record():
    return BagelJointReplay(
        prompt_token_ids=torch.tensor([1, 3, 2]),
        cached_thinking_token_ids=torch.tensor([1, 4]),
        response_token_ids=torch.tensor([4, 2]),
        rollout_log_probs=torch.tensor([-0.3, -0.4]),
        latents_clean=torch.ones(2, 3),
        latent_pos_ids=torch.tensor([0, 1]),
        image_position_ids=torch.full((4,), 5),
        image_boundary_token_ids=torch.tensor([10, 11]),
        temperature=1,
        policy_version=0,
        eos_token_id=2,
    )


def test_actor_preparation_routes_same_outcome_credit_without_integer_sigma_cast():
    cfg = SimpleNamespace(
        algorithm=DiffusionAlgoConfig(global_std=False),
        actor_rollout_ref=SimpleNamespace(actor=_cfg(nft_timesteps_per_sample=3)),
    )
    batch = DataProto(
        batch=TensorDict({}, batch_size=[4]),
        non_tensor_batch={
            "uid": np.array(["a", "a", "b", "b"]),
            "bagel_joint_replay": np.array([_record() for _ in range(4)], dtype=object),
        },
    )
    torch.manual_seed(42)
    result = BagelUniGRPONFTLoss.prepare_actor_batch(batch, torch.tensor([0.0, 2.0, 5.0, 1.0]), cfg)
    assert result is batch
    assert result.batch["train_timesteps"].shape == (4, 3)
    assert result.batch["train_timesteps"].dtype == torch.float32
    assert bool(((result.batch["train_timesteps"] > 0) & (result.batch["train_timesteps"] < 1)).all())
    assert result.batch["nft_noise"].shape == (4, 3, 2, 3)
    assert result.batch["advantages"][0] < 0 < result.batch["advantages"][1]
    expected_prob = result.batch["advantages"].clamp(-5, 5) / 10 + 0.5
    torch.testing.assert_close(result.batch["reward_prob"], expected_prob)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text_loss_weight": -1},
        {"image_loss_weight": float("nan")},
        {"text_loss_weight": 0, "image_loss_weight": 0},
        {"ar_clip_ratio": 0},
        {"ar_clip_ratio": 1},
        {"nft_timesteps_per_sample": 0},
        {"nft_timesteps_per_sample": 1.5},
    ],
)
def test_invalid_hybrid_loss_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        _cfg(**kwargs)
