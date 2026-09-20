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
"""CPU mathematics and routing tests for the OmniNFT objective."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl.protocol import DataProto

from verl_omni.pipelines.ltx2_omni_nft.config import OmniNFTLossConfig
from verl_omni.trainer.diffusion.diffusion_algos import OmniNFTLoss


def _loss_config(**kwargs):
    return SimpleNamespace(diffusion_loss=OmniNFTLossConfig(**kwargs))


def test_component_advantages_normalize_columns_by_prompt_group():
    scores = torch.tensor([[1.0, 10.0], [3.0, 14.0], [2.0, 9.0], [6.0, 17.0]])
    uid = np.array(["p0", "p0", "p1", "p1"], dtype=object)

    group = OmniNFTLoss._compute_component_advantages(scores, uid, norm_by_std=True, global_std=False)
    global_ = OmniNFTLoss._compute_component_advantages(scores, uid, norm_by_std=True, global_std=True)

    torch.testing.assert_close(group.sum(dim=0), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(global_.sum(dim=0), torch.zeros(2), atol=1e-6, rtol=0)
    assert not torch.allclose(group, global_)


def test_component_advantages_handle_single_sample_and_zero_variance():
    scores = torch.ones(3, 2)
    result = OmniNFTLoss._compute_component_advantages(
        scores,
        np.array(["p0", "p1", "p1"], dtype=object),
        norm_by_std=True,
        global_std=False,
    )

    torch.testing.assert_close(result, torch.zeros_like(result))
    assert torch.isfinite(result).all()


def test_prepare_actor_batch_routes_reward_columns_in_declared_order():
    scores = torch.tensor([[1.0, 2.0], [3.0, 0.0], [2.0, 1.0], [0.0, 3.0]])
    batch = DataProto.from_dict(
        tensors={
            "video_latents_clean": torch.zeros(4, 1),
            "audio_latents_clean": torch.zeros(4, 1),
            "train_timesteps": torch.arange(16).reshape(4, 4),
            "rm_scores": scores,
            "reward_valid_mask": torch.ones_like(scores, dtype=torch.bool),
        },
        non_tensors={"uid": np.array(["p0", "p0", "p1", "p1"], dtype=object)},
        meta_info={"reward_names": ["audio", "video"]},
    )
    config = SimpleNamespace(
        algorithm=SimpleNamespace(
            norm_adv_by_std_in_grpo=False,
            global_std=False,
            adv_mode="continuous",
            timestep_fraction=0.5,
        ),
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(diffusion_loss=OmniNFTLossConfig(), data_loader_seed=7)
        ),
        reward=SimpleNamespace(
            reward_functions={
                "video": {"routing_weights": {"video": 2.0, "audio": 0.0}},
                "audio": {"routing_weights": {"video": 0.0, "audio": 3.0}},
            }
        ),
    )

    result = OmniNFTLoss.prepare_actor_batch(batch, scores, config)

    expected = result.batch["reward_advantages"] @ torch.tensor([[0.0, 3.0], [2.0, 0.0]])
    torch.testing.assert_close(result.batch["modality_advantages"], expected)
    assert result.batch["reward_prob"].shape == (4, 2, 2)
    assert ((result.batch["reward_prob"] >= 0) & (result.batch["reward_prob"] <= 1)).all()


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf")])
def test_reward_routing_rejects_missing_or_nonfinite_weights(value):
    routing_weights = {"video": 1.0}
    if value is not None:
        routing_weights["audio"] = value

    with pytest.raises(ValueError, match="routing weight|routing_weights.audio"):
        OmniNFTLoss._build_reward_routing_matrix(
            reward_names=["score"],
            reward_functions={"score": {"routing_weights": routing_weights}},
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_modality_loss_detaches_old_and_reference_and_normalizes_weights():
    video_current = torch.tensor([[0.2], [0.8]], requires_grad=True)
    audio_current = torch.tensor([[0.4], [0.6]], requires_grad=True)
    video_old = torch.zeros(2, 1, requires_grad=True)
    audio_old = torch.zeros(2, 1, requires_grad=True)
    video_ref = torch.ones(2, 1, requires_grad=True)
    audio_ref = torch.ones(2, 1, requires_grad=True)
    common = {
        "video_x0": torch.zeros(2, 1),
        "video_xt": torch.ones(2, 1),
        "video_t_expanded": torch.ones(2, 1),
        "video_reward_prob": torch.tensor([1.0, 0.0]),
        "audio_x0": torch.zeros(2, 1),
        "audio_xt": torch.ones(2, 1),
        "audio_t_expanded": torch.ones(2, 1),
        "audio_reward_prob": torch.tensor([0.0, 1.0]),
    }
    config = _loss_config(
        video_weight=2.0,
        audio_weight=1.0,
        video_ref_kl_coef=0.25,
        audio_ref_kl_coef=0.5,
    )

    loss, metrics = OmniNFTLoss.compute_loss(
        video_forward_prediction=video_current,
        video_old_prediction=video_old,
        video_ref_forward_prediction=video_ref,
        audio_forward_prediction=audio_current,
        audio_old_prediction=audio_old,
        audio_ref_forward_prediction=audio_ref,
        config=config,
        **common,
    )
    expected = (
        2.0 * (metrics["actor/video/policy_loss"] + 0.25 * metrics["actor/video/ref_kl_loss"])
        + metrics["actor/audio/policy_loss"]
        + 0.5 * metrics["actor/audio/ref_kl_loss"]
    ) / 3.0
    assert loss.item() == pytest.approx(expected)

    loss.backward()
    assert video_current.grad is not None and video_current.grad.abs().sum() > 0
    assert audio_current.grad is not None and audio_current.grad.abs().sum() > 0
    assert video_old.grad is None
    assert audio_old.grad is None
    assert video_ref.grad is None
    assert audio_ref.grad is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"video_weight": -1.0}, "video_weight"),
        ({"video_weight": float("nan")}, "video_weight"),
        ({"audio_weight": float("inf")}, "audio_weight"),
        ({"video_ref_kl_coef": -float("inf")}, "video_ref_kl_coef"),
        ({"video_weight": 0.0, "audio_weight": 0.0}, "At least one"),
        ({"loss_mode": "diffusion_nft"}, "loss_mode"),
    ],
)
def test_omni_nft_loss_config_fails_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OmniNFTLossConfig(**kwargs)
