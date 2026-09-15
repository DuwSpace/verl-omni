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
"""Lightweight integration coverage for the OmniNFT reward-to-loss path."""

from types import SimpleNamespace

import numpy as np
import torch
from verl import DataProto

from verl_omni.pipelines.ltx2_omni_nft.config import OmniNFTLossConfig
from verl_omni.reward_loop.multimodal_reward_loop import assemble_batch_reward
from verl_omni.trainer.diffusion.diffusion_algos import OmniNFTLoss


def _worker_output(sample_uids, scores):
    reward_names = ["video_quality", "audio_quality"]
    return {
        "rm_scores": torch.tensor(scores, dtype=torch.float32),
        "reward_valid_mask": torch.ones((len(sample_uids), len(reward_names)), dtype=torch.bool),
        "reward_names": reward_names,
        "sample_uid": np.asarray(sample_uids, dtype=object),
        "reward_extra_info": {
            name: {
                "metrics": {"batch_size": len(sample_uids)},
                "model_revision": f"{name}-v1",
                "definition_version": "test-v1",
            }
            for name in reward_names
        },
    }


def _prepare_actor_batch():
    scoring_batch = DataProto.from_dict(
        tensors={"responses": torch.zeros((4, 1), dtype=torch.uint8)},
        non_tensors={"sample_uid": np.asarray(["s0", "s1", "s2", "s3"], dtype=object)},
    )
    chunks = scoring_batch.chunk(2)
    reward_batch = assemble_batch_reward(
        scoring_batch,
        chunks,
        [
            _worker_output(["s1", "s0"], [[3.0, 30.0], [1.0, 10.0]]),
            _worker_output(["s3", "s2"], [[2.0, 20.0], [4.0, 40.0]]),
        ],
    )
    rollout_batch = DataProto.from_dict(
        tensors={
            "video_latents_clean": torch.zeros((4, 2, 2)),
            "audio_latents_clean": torch.zeros((4, 2, 2)),
            "train_timesteps": torch.arange(3).expand(4, -1).clone(),
        },
        non_tensors={"uid": np.asarray(["p0", "p0", "p1", "p1"], dtype=object)},
    )
    actor_batch = rollout_batch.union(reward_batch)
    loss_config = OmniNFTLossConfig(adv_clip_max=2.0, mix_beta=1.0)
    config = SimpleNamespace(
        algorithm=SimpleNamespace(
            norm_adv_by_std_in_grpo=False,
            global_std=False,
            adv_mode="continuous",
            timestep_fraction=1.0,
        ),
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(diffusion_loss=loss_config, data_loader_seed=42)
        ),
        reward=SimpleNamespace(
            component_order=["video_quality", "audio_quality"],
            reward_functions={
                "video_quality": {"routing_weights": {"video": 1.0, "audio": 0.0}},
                "audio_quality": {"routing_weights": {"video": 0.0, "audio": 1.0}},
            },
        ),
    )
    return OmniNFTLoss.prepare_actor_batch(actor_batch, actor_batch.batch["rm_scores"], config), config


def test_worker_rewards_are_reordered_and_routed_to_modalities():
    actor_batch, _ = _prepare_actor_batch()

    torch.testing.assert_close(
        actor_batch.batch["rm_scores"],
        torch.tensor([[1.0, 10.0], [3.0, 30.0], [4.0, 40.0], [2.0, 20.0]]),
    )
    expected_video = torch.tensor([0.25, 0.75, 0.75, 0.25]).unsqueeze(1).expand(-1, 3)
    expected_audio = torch.tensor([0.0, 1.0, 1.0, 0.0]).unsqueeze(1).expand(-1, 3)
    torch.testing.assert_close(actor_batch.batch["video_reward_prob"], expected_video)
    torch.testing.assert_close(actor_batch.batch["audio_reward_prob"], expected_audio)


def test_routed_rewards_drive_both_omnift_loss_branches():
    actor_batch, config = _prepare_actor_batch()
    video_prediction = torch.full((4, 2, 2), 0.25, requires_grad=True)
    audio_prediction = torch.full((4, 2, 2), -0.25, requires_grad=True)

    def branch_inputs(forward_prediction):
        return {
            "forward_prediction": forward_prediction,
            "old_prediction": torch.zeros_like(forward_prediction),
            "ref_forward_prediction": torch.full_like(forward_prediction, 0.1),
            "x0": torch.zeros_like(forward_prediction),
            "xt": torch.ones_like(forward_prediction),
            "t_expanded": torch.full_like(forward_prediction, 0.5),
        }

    loss, metrics = OmniNFTLoss.compute_loss(
        **{f"video_{key}": value for key, value in branch_inputs(video_prediction).items()},
        video_reward_prob=actor_batch.batch["video_reward_prob"],
        **{f"audio_{key}": value for key, value in branch_inputs(audio_prediction).items()},
        audio_reward_prob=actor_batch.batch["audio_reward_prob"],
        config=config.actor_rollout_ref.actor,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert video_prediction.grad is not None and torch.isfinite(video_prediction.grad).all()
    assert audio_prediction.grad is not None and torch.isfinite(audio_prediction.grad).all()
    assert {"actor/video/policy_loss", "actor/audio/policy_loss", "actor/total_loss"} <= metrics.keys()
