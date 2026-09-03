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

import os

import numpy as np
import pytest
import torch

from verl_omni.trainer.diffusion import diffusion_algos


@pytest.mark.parametrize("norm_adv_by_std_in_grpo", [True, False])
@pytest.mark.parametrize("global_std", [True, False])
def test_flow_grpo_advantage_return(norm_adv_by_std_in_grpo: bool, global_std: bool) -> None:
    batch_size = 8
    steps = 10
    sample_level_rewards = torch.randn((batch_size, steps), dtype=torch.float32)
    uid = np.array([f"uid-{idx}" for idx in range(batch_size)], dtype=object)

    advantages, returns = diffusion_algos.compute_flow_grpo_outcome_advantage(
        sample_level_rewards=sample_level_rewards,
        index=uid,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        global_std=global_std,
    )

    assert advantages.shape == returns.shape == (batch_size, steps)


def test_dance_grpo_loss_registered_and_callable():
    """``dance_grpo`` loss function is registered and can be invoked."""
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig

    batch_size = 8
    rollout_log_probs = torch.randn((batch_size,), dtype=torch.float32)
    current_log_probs = torch.randn((batch_size,), dtype=torch.float32)
    advantages = torch.randn((batch_size,), dtype=torch.float32)

    with initialize_config_dir(
        config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor"), version_base=None
    ):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                "strategy=fsdp",
                "diffusion_loss.loss_mode=dance_grpo",
                "diffusion_loss.clip_ratio=0.0001",
                "diffusion_loss.adv_clip_max=5.0",
                "ppo_micro_batch_size_per_gpu=8",
            ],
        )
    actor_config: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)

    dance_grpo_loss = diffusion_algos.get_diffusion_loss_fn("dance_grpo")
    pg_loss, pg_metrics = dance_grpo_loss.compute_loss(
        old_log_prob=rollout_log_probs,
        log_prob=current_log_probs,
        advantages=advantages,
        config=actor_config,
    )

    assert pg_loss.shape == ()
    assert isinstance(pg_loss.item(), float)
    for key in ("actor/ppo_kl", "actor/pg_clipfrac", "actor/pg_clipfrac_higher", "actor/pg_clipfrac_lower"):
        assert key in pg_metrics


@pytest.mark.parametrize("norm_adv_by_std_in_grpo", [True, False])
@pytest.mark.parametrize("global_std", [True, False])
def test_flow_grpo_advantage_grouped_uids(norm_adv_by_std_in_grpo: bool, global_std: bool) -> None:
    """Exercises the len > 1 branch: multiple samples sharing the same prompt UID."""
    steps = 5
    # 4 samples: uid-0 × 2, uid-1 × 2  →  2 groups of size 2
    group_rewards = torch.tensor(
        [[1.0] * steps, [3.0] * steps, [0.0] * steps, [2.0] * steps],
        dtype=torch.float32,
    )
    uid = np.array(["uid-0", "uid-0", "uid-1", "uid-1"], dtype=object)

    advantages, returns = diffusion_algos.compute_flow_grpo_outcome_advantage(
        sample_level_rewards=group_rewards,
        index=uid,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        global_std=global_std,
    )

    assert advantages.shape == returns.shape == (4, steps)

    if not norm_adv_by_std_in_grpo:
        # Without std scaling: advantage = reward - group_mean
        # group uid-0 mean = (1+3)/2 = 2.0  →  advantages: -1, +1
        # group uid-1 mean = (0+2)/2 = 1.0  →  advantages: -1, +1
        torch.testing.assert_close(advantages[0], torch.full((steps,), -1.0))
        torch.testing.assert_close(advantages[1], torch.full((steps,), 1.0))
        torch.testing.assert_close(advantages[2], torch.full((steps,), -1.0))
        torch.testing.assert_close(advantages[3], torch.full((steps,), 1.0))
    else:
        # With std scaling: mean should be 0 for each group
        torch.testing.assert_close(advantages[0:2].mean(), torch.tensor(0.0), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(advantages[2:4].mean(), torch.tensor(0.0), atol=1e-6, rtol=1e-6)


def test_compute_policy_loss_flow_grpo() -> None:
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig

    batch_size = 8
    steps = 10
    rollout_log_probs = torch.randn((batch_size, steps), dtype=torch.float32)
    current_log_probs = torch.randn((batch_size, steps), dtype=torch.float32)
    advantages = torch.randn((batch_size, steps), dtype=torch.float32)

    with initialize_config_dir(
        config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor"), version_base=None
    ):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                "strategy=fsdp",
                "diffusion_loss.clip_ratio=0.0001",
                "diffusion_loss.adv_clip_max=5.0",
                "ppo_micro_batch_size_per_gpu=8",
            ],
        )
    actor_config: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)

    flow_grpo_loss = diffusion_algos.get_diffusion_loss_fn("flow_grpo")
    for step in range(steps):
        pg_loss, pg_metrics = flow_grpo_loss.compute_loss(
            old_log_prob=rollout_log_probs[:, step],
            log_prob=current_log_probs[:, step],
            advantages=advantages[:, step],
            config=actor_config,
        )

        assert pg_loss.shape == ()
        assert isinstance(pg_loss.item(), float)
        assert "actor/ppo_kl" in pg_metrics
        assert "actor/pg_clipfrac" in pg_metrics
        assert "actor/pg_clipfrac_higher" in pg_metrics
        assert "actor/pg_clipfrac_lower" in pg_metrics


def _reference_flow_dppo_loss(
    *,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    old_prev_sample_mean: torch.Tensor,
    prev_sample_mean: torch.Tensor,
    advantages: torch.Tensor,
    sigma_t: torch.Tensor,
    kl_mask_threshold: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the expected Flow-DPPO KL-advantage masking math."""
    log_diff = log_prob - old_log_prob
    ratio = torch.exp(log_diff)
    unclipped_loss = -advantages.detach() * ratio

    kl_per_elem = (prev_sample_mean - old_prev_sample_mean).pow(2) / (2 * sigma_t.pow(2))
    kl_per_sample = kl_per_elem.mean(dim=tuple(range(1, kl_per_elem.ndim)))
    kl_mask = kl_per_sample < kl_mask_threshold
    pos_rm_mask = (~kl_mask) & (ratio > 1.0) & (advantages > 0)
    neg_rm_mask = (~kl_mask) & (ratio < 1.0) & (advantages < 0)
    rm_mask = pos_rm_mask | neg_rm_mask
    keep_mask = (~rm_mask).detach()
    zero = torch.zeros((), dtype=unclipped_loss.dtype, device=unclipped_loss.device)
    loss = torch.where(keep_mask, unclipped_loss, zero).mean()
    return loss, {
        "kl_per_sample": kl_per_sample,
        "pos_rm_mask": pos_rm_mask,
        "neg_rm_mask": neg_rm_mask,
        "rm_mask": rm_mask,
    }


@pytest.mark.parametrize("add_kl_coefficient", [True, False])
def test_compute_policy_loss_flow_dppo_applies_asymmetric_kl_mask(add_kl_coefficient: bool) -> None:
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig

    old_log_prob = torch.zeros(4, dtype=torch.float32)
    log_prob = torch.log(torch.tensor([1.2, 0.8, 1.2, 0.8], dtype=torch.float32))
    advantages = torch.tensor([10.0, -10.0, -8.0, 8.0], dtype=torch.float32)
    old_prev_sample_mean = torch.zeros((4, 2, 2, 2), dtype=torch.float32)
    prev_sample_mean = torch.full_like(old_prev_sample_mean, 0.2)
    std_dev_t = torch.full((4, 1, 1, 1), 0.5, dtype=torch.float32)
    sqrt_dt = torch.full((4,), 0.5, dtype=torch.float32)
    kl_mask_threshold = 1e-5

    with initialize_config_dir(
        config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor"), version_base=None
    ):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                "strategy=fsdp",
                "diffusion_loss.loss_mode=flow_dppo",
                f"diffusion_loss.add_kl_coefficient={str(add_kl_coefficient)}",
                f"diffusion_loss.kl_mask_threshold={kl_mask_threshold}",
                "diffusion_loss.adv_clip_max=5.0",
                "ppo_micro_batch_size_per_gpu=4",
            ],
        )
    actor_config: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)

    flow_dppo_loss = diffusion_algos.get_diffusion_loss_fn("flow_dppo")
    loss, metrics = flow_dppo_loss.compute_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        config=actor_config,
        old_prev_sample_mean=old_prev_sample_mean,
        prev_sample_mean=prev_sample_mean,
        std_dev_t=std_dev_t,
        sqrt_dt=sqrt_dt,
    )

    if add_kl_coefficient:
        sigma_t = std_dev_t * sqrt_dt.reshape(4, 1, 1, 1)
    else:
        sigma_t = torch.ones_like(std_dev_t)
    ref_loss, ref_masks = _reference_flow_dppo_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        old_prev_sample_mean=old_prev_sample_mean,
        prev_sample_mean=prev_sample_mean,
        advantages=advantages,
        sigma_t=sigma_t,
        kl_mask_threshold=kl_mask_threshold,
    )

    torch.testing.assert_close(loss, ref_loss)
    clamped_advantages = torch.clamp(
        advantages,
        -actor_config.diffusion_loss.adv_clip_max,
        actor_config.diffusion_loss.adv_clip_max,
    )
    clamped_ref_loss, _ = _reference_flow_dppo_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        old_prev_sample_mean=old_prev_sample_mean,
        prev_sample_mean=prev_sample_mean,
        advantages=clamped_advantages,
        sigma_t=sigma_t,
        kl_mask_threshold=kl_mask_threshold,
    )
    assert not torch.isclose(loss, clamped_ref_loss)
    assert ref_masks["pos_rm_mask"].tolist() == [True, False, False, False]
    assert ref_masks["neg_rm_mask"].tolist() == [False, True, False, False]
    assert ref_masks["rm_mask"].tolist() == [True, True, False, False]
    assert metrics["actor/masked_fraction"] == pytest.approx(0.5)
    assert metrics["actor/unmasked_fraction"] == pytest.approx(0.5)
    assert "actor/kl_new_old_mean" in metrics


@pytest.mark.parametrize("norm_by_std", [True, False])
@pytest.mark.parametrize("global_std", [True, False])
def test_compute_diffusion_nft_group_advantages(norm_by_std: bool, global_std: bool) -> None:
    # 4 samples in 2 groups of 2
    rewards = torch.tensor([1.0, 3.0, 0.0, 2.0])
    uid = np.array(["uid-0", "uid-0", "uid-1", "uid-1"], dtype=object)

    advantages = diffusion_algos.DiffusionNFTLoss._compute_group_advantages(
        rewards=rewards, uid=uid, norm_by_std=norm_by_std, global_std=global_std
    )

    assert advantages.shape == (4,)
    if not norm_by_std:
        torch.testing.assert_close(advantages[0], torch.tensor(-1.0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(advantages[1], torch.tensor(1.0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(advantages[2], torch.tensor(-1.0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(advantages[3], torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    else:
        # group means are zero after std normalization
        torch.testing.assert_close(advantages[0:2].mean(), torch.tensor(0.0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(advantages[2:4].mean(), torch.tensor(0.0), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("adv_mode", ["continuous", "positive_only", "negative_only", "one_only", "binary"])
def test_diffusion_nft_advantage_to_reward_prob(adv_mode: str) -> None:
    adv_clip_max = 5.0
    advantages = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])

    reward_prob = diffusion_algos.DiffusionNFTLoss._advantage_to_reward_prob(
        advantages, adv_clip_max=adv_clip_max, adv_mode=adv_mode
    )

    assert reward_prob.shape == advantages.shape
    assert (reward_prob >= 0).all() and (reward_prob <= 1).all(), "reward_prob must be in [0, 1]"

    if adv_mode == "continuous":
        # clipped to [-5, 5] → mapped to [0, 1]
        torch.testing.assert_close(reward_prob[1], torch.tensor(0.0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reward_prob[2], torch.tensor(0.5), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reward_prob[3], torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    elif adv_mode == "positive_only":
        assert (reward_prob >= 0.5).all()
    elif adv_mode == "negative_only":
        assert (reward_prob <= 0.5).all()
    elif adv_mode == "one_only":
        # advantages binarized to {0, 1}, then mapped via (adv / adv_clip_max) / 2 + 0.5
        torch.testing.assert_close(reward_prob[0:3], torch.full((3,), 0.5), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reward_prob[3:5], torch.full((2,), 0.6), atol=1e-5, rtol=1e-5)
    elif adv_mode == "binary":
        # advantages signed to {-1, 0, 1}, then mapped via (adv / adv_clip_max) / 2 + 0.5
        torch.testing.assert_close(reward_prob[0:2], torch.full((2,), 0.4), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reward_prob[2], torch.tensor(0.5), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reward_prob[3:5], torch.full((2,), 0.6), atol=1e-5, rtol=1e-5)


def test_prepare_diffusion_nft_actor_batch() -> None:
    from types import SimpleNamespace

    from verl import DataProto

    B, T, C, H, W = 4, 6, 4, 8, 8
    rewards = torch.randn(B)
    uid = np.array([f"uid-{i // 2}" for i in range(B)], dtype=object)
    batch = DataProto.from_dict(
        tensors={
            "latents_clean": torch.randn(B, C, H, W),
            "train_timesteps": torch.randint(0, 1000, (B, T)),
            "prompts": torch.zeros(B, 16, dtype=torch.long),
        },
        non_tensors={"uid": uid},
    )
    algorithm_config = SimpleNamespace(
        norm_adv_by_std_in_grpo=True,
        global_std=True,
        adv_mode="continuous",
        timestep_fraction=0.5,
    )
    config = SimpleNamespace(
        algorithm=algorithm_config,
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(
                diffusion_loss=SimpleNamespace(adv_clip_max=5.0),
                data_loader_seed=42,
            )
        ),
    )

    result = diffusion_algos.DiffusionNFTLoss.prepare_actor_batch(batch, rewards, config)

    num_train = max(1, int(T * algorithm_config.timestep_fraction))
    assert result.batch["train_timesteps"].shape == (B, num_train)
    assert result.batch["advantages"].shape == (B, num_train)
    assert result.batch["reward_prob"].shape == (B, num_train)
    assert result.batch["returns"].shape == (B, num_train)
    assert result.batch["sample_level_rewards"].shape == (B, num_train)
    assert ((result.batch["reward_prob"] >= 0) & (result.batch["reward_prob"] <= 1)).all()


def test_prepare_online_dpo_actor_batch() -> None:
    from types import SimpleNamespace

    from verl import DataProto

    # Two prompts, two rollouts each; rewards pick high/low per uid.
    uid = np.array(["p0", "p0", "p1", "p1"], dtype=object)
    rewards = torch.tensor([1.0, 0.0, 0.5, 1.0])
    batch = DataProto.from_dict(
        tensors={"sample_level_scores": rewards.clone()},
        non_tensors={"uid": uid},
    )
    config = SimpleNamespace(algorithm=SimpleNamespace(sample_source="online"))

    result = diffusion_algos.DPOLoss.prepare_actor_batch(batch, rewards, config)

    assert len(result) == 4
    assert list(result.non_tensor_batch["uid"]) == ["p0", "p0", "p1", "p1"]
    chosen_rejected = result.batch["sample_level_scores"].squeeze(-1)
    assert chosen_rejected[0] >= chosen_rejected[1]
    assert chosen_rejected[2] >= chosen_rejected[3]


def test_compute_policy_loss_diffusion_nft() -> None:
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig

    B, C, H, W = 4, 4, 8, 8
    x0 = torch.randn(B, C, H, W)
    xt = torch.randn(B, C, H, W)
    t_expanded = torch.full((B, C, H, W), 0.5)
    forward_prediction = torch.randn(B, C, H, W)
    old_prediction = torch.randn(B, C, H, W)
    ref_forward_prediction = torch.randn(B, C, H, W)
    reward_prob = torch.rand(B)

    with initialize_config_dir(
        config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor"), version_base=None
    ):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                "strategy=fsdp",
                "diffusion_loss.loss_mode=diffusion_nft",
                "diffusion_loss.adv_clip_max=5.0",
                "ppo_micro_batch_size_per_gpu=4",
            ],
        )
    actor_config: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)

    nft_loss = diffusion_algos.get_diffusion_loss_fn("diffusion_nft")
    loss, metrics = nft_loss.compute_loss(
        forward_prediction=forward_prediction,
        old_prediction=old_prediction,
        ref_forward_prediction=ref_forward_prediction,
        x0=x0,
        xt=xt,
        t_expanded=t_expanded,
        reward_prob=reward_prob,
        config=actor_config,
    )

    assert loss.shape == ()
    assert isinstance(loss.item(), float)
    for key in (
        "actor/policy_loss",
        "actor/positive_loss",
        "actor/negative_loss",
        "actor/ref_kl_loss",
        "actor/old_deviate",
        "actor/reward_prob_mean",
        "actor/total_loss",
    ):
        assert key in metrics, key


def test_prepare_omni_nft_actor_batch_routes_component_rewards() -> None:
    from types import SimpleNamespace

    from verl import DataProto

    component_order = ["video_align", "hpsv3", "audiobox", "clap", "desync"]
    routing_weights = {
        "video_align": {"video": 1.0, "audio": 0.0},
        "hpsv3": {"video": 1.5, "audio": 0.0},
        "audiobox": {"video": 0.0, "audio": 0.5},
        "clap": {"video": 0.0, "audio": 1.0},
        "desync": {"video": 1.0, "audio": 1.0},
    }
    batch_size, steps = 4, 6
    scores = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [12.0, 22.0, 32.0, 42.0, 52.0],
        ]
    )
    batch = DataProto.from_dict(
        tensors={
            "video_latents_clean": torch.randn(batch_size, 12, 8),
            "audio_latents_clean": torch.randn(batch_size, 5, 8),
            "train_timesteps": torch.arange(steps).expand(batch_size, -1),
            "rm_scores": scores,
            "reward_valid_mask": torch.ones_like(scores, dtype=torch.bool),
        },
        non_tensors={"uid": np.array(["p0", "p0", "p1", "p1"], dtype=object)},
        meta_info={"reward_names": component_order},
    )
    config = SimpleNamespace(
        algorithm=SimpleNamespace(
            norm_adv_by_std_in_grpo=True,
            global_std=False,
            adv_mode="continuous",
            timestep_fraction=0.5,
        ),
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(
                diffusion_loss=SimpleNamespace(adv_clip_max=5.0),
                data_loader_seed=42,
            )
        ),
        reward=SimpleNamespace(
            component_order=component_order,
            reward_functions={
                name: {"routing_weights": weights} for name, weights in routing_weights.items()
            },
        ),
    )

    result = diffusion_algos.OmniNFTLoss.prepare_actor_batch(batch, scores, config)

    assert result.batch["train_timesteps"].shape == (batch_size, 3)
    assert result.batch["reward_advantages"].shape == (batch_size, 5)
    assert result.batch["modality_advantages"].shape == (batch_size, 2)
    assert result.batch["modality_reward_probs"].shape == (batch_size, 2)
    assert result.batch["timestep_modality_reward_probs"].shape == (batch_size, 3, 2)
    assert result.batch["video_reward_prob"].shape == (batch_size, 3)
    assert result.batch["audio_reward_prob"].shape == (batch_size, 3)

    unit_advantage = 1.0 / 1.0001
    signs = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    expected_reward_advantages = signs[:, None] * unit_advantage * torch.ones(1, 5)
    expected_modality_advantages = signs[:, None] * unit_advantage * torch.tensor([[3.5, 2.5]])
    expected_probabilities = 0.5 + 0.5 * expected_modality_advantages / 5.0
    torch.testing.assert_close(result.batch["reward_advantages"], expected_reward_advantages)
    torch.testing.assert_close(result.batch["modality_advantages"], expected_modality_advantages)
    torch.testing.assert_close(result.batch["modality_reward_probs"], expected_probabilities)
    torch.testing.assert_close(result.batch["video_reward_prob"], expected_probabilities[:, :1].expand(-1, 3))
    torch.testing.assert_close(result.batch["audio_reward_prob"], expected_probabilities[:, 1:].expand(-1, 3))


def test_modality_advantage_router_rejects_invalid_reward_contracts() -> None:
    from verl_omni.trainer.diffusion.modality_advantage import ModalityAdvantageRouter

    with pytest.raises(ValueError, match="finite"):
        ModalityAdvantageRouter.compute_reward_advantages(
            torch.tensor([[1.0, float("nan")]]),
            ["p0"],
            norm_by_std=True,
            global_std=False,
        )
    with pytest.raises(ValueError, match="exactly match"):
        ModalityAdvantageRouter.build_routing_matrix(
            reward_names=["clap", "desync"],
            component_order=["desync", "clap"],
            reward_functions={
                "clap": {"routing_weights": {"video": 0.0, "audio": 1.0}},
                "desync": {"routing_weights": {"video": 1.0, "audio": 1.0}},
            },
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_compute_omni_nft_joint_av_loss_backpropagates_both_modalities() -> None:
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig

    batch_size = 2
    video_x0 = torch.randn(batch_size, 7, 4)
    audio_x0 = torch.randn(batch_size, 3, 4)
    video_forward = torch.randn_like(video_x0, requires_grad=True)
    audio_forward = torch.randn_like(audio_x0, requires_grad=True)
    with initialize_config_dir(
        config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor"), version_base=None
    ):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                "strategy=fsdp2",
                "diffusion_loss.loss_mode=omni_nft",
                "diffusion_loss.video_weight=2.0",
                "diffusion_loss.audio_weight=3.0",
                "diffusion_loss.video_ref_kl_coef=0.1",
                "diffusion_loss.audio_ref_kl_coef=0.2",
                "ppo_micro_batch_size_per_gpu=2",
            ],
        )
    actor_config: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)

    loss, metrics = diffusion_algos.OmniNFTLoss.compute_loss(
        video_forward_prediction=video_forward,
        video_old_prediction=torch.randn_like(video_x0),
        video_ref_forward_prediction=torch.randn_like(video_x0),
        video_x0=video_x0,
        video_xt=torch.randn_like(video_x0),
        video_t_expanded=torch.full((batch_size, 1, 1), 0.5),
        video_reward_prob=torch.full((batch_size,), 0.5),
        audio_forward_prediction=audio_forward,
        audio_old_prediction=torch.randn_like(audio_x0),
        audio_ref_forward_prediction=torch.randn_like(audio_x0),
        audio_x0=audio_x0,
        audio_xt=torch.randn_like(audio_x0),
        audio_t_expanded=torch.full((batch_size, 1, 1), 0.5),
        audio_reward_prob=torch.full((batch_size,), 0.5),
        config=actor_config,
    )
    loss.backward()

    assert loss.shape == ()
    assert video_forward.grad is not None and torch.isfinite(video_forward.grad).all()
    assert audio_forward.grad is not None and torch.isfinite(audio_forward.grad).all()
    assert video_forward.grad.abs().sum() > 0
    assert audio_forward.grad.abs().sum() > 0
    for key in (
        "actor/video/policy_loss",
        "actor/audio/policy_loss",
        "actor/video/ref_kl_loss",
        "actor/audio/ref_kl_loss",
        "actor/total_loss",
    ):
        assert key in metrics

    expected_loss = (
        2.0 * (metrics["actor/video/policy_loss"] + 0.1 * metrics["actor/video/ref_kl_loss"])
        + 3.0 * (metrics["actor/audio/policy_loss"] + 0.2 * metrics["actor/audio/ref_kl_loss"])
    ) / 5.0
    assert loss.detach().item() == pytest.approx(expected_loss)


def test_compute_policy_loss_grpo_guard() -> None:
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig

    batch_size = 4
    rollout_log_probs = torch.randn((batch_size,), dtype=torch.float32)
    current_log_probs = torch.randn((batch_size,), dtype=torch.float32)
    advantages = torch.randn((batch_size,), dtype=torch.float32)
    old_prev_sample_mean = torch.randn((batch_size, 16, 8, 8), dtype=torch.float32)
    prev_sample_mean = old_prev_sample_mean + 0.01 * torch.randn_like(old_prev_sample_mean)
    std_dev_t = torch.full((batch_size, 1, 1, 1), 0.5, dtype=torch.float32)
    sqrt_dt = torch.full((batch_size,), 0.3, dtype=torch.float32)

    with initialize_config_dir(
        config_dir=os.path.abspath("verl_omni/trainer/config/diffusion/actor"), version_base=None
    ):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=[
                "strategy=fsdp",
                "diffusion_loss.loss_mode=grpo_guard",
                "diffusion_loss.clip_ratio=2e-6",
                "diffusion_loss.adv_clip_max=5.0",
                "ppo_micro_batch_size_per_gpu=8",
            ],
        )
    actor_config: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)

    grpo_guard_loss = diffusion_algos.get_diffusion_loss_fn("grpo_guard")
    pg_loss, pg_metrics = grpo_guard_loss.compute_loss(
        old_log_prob=rollout_log_probs,
        log_prob=current_log_probs,
        advantages=advantages,
        config=actor_config,
        old_prev_sample_mean=old_prev_sample_mean,
        prev_sample_mean=prev_sample_mean,
        std_dev_t=std_dev_t,
        sqrt_dt=sqrt_dt,
    )

    assert pg_loss.shape == ()
    assert isinstance(pg_loss.item(), float)
    for key in (
        "actor/ppo_kl",
        "actor/pg_clipfrac",
        "actor/pg_clipfrac_higher",
        "actor/pg_clipfrac_lower",
        "actor/ratio_mean",
        "actor/ratio_std",
    ):
        assert key in pg_metrics, key


@pytest.mark.parametrize("norm_adv_by_std_in_grpo", [True, False])
@pytest.mark.parametrize("global_std", [True, False])
def test_dance_grpo_advantage_return(norm_adv_by_std_in_grpo: bool, global_std: bool) -> None:
    """``dance_grpo`` reuses the ``flow_grpo`` advantage estimator."""
    batch_size = 8
    steps = 10
    sample_level_rewards = torch.randn((batch_size, steps), dtype=torch.float32)
    uid = np.array([f"uid-{idx}" for idx in range(batch_size)], dtype=object)

    advantages, returns = diffusion_algos.compute_flow_grpo_outcome_advantage(
        sample_level_rewards=sample_level_rewards,
        index=uid,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        global_std=global_std,
    )

    assert advantages.shape == returns.shape == (batch_size, steps)
