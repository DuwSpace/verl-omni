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

"""Hybrid UniGRPO AR objective and OmniNFT's prediction-space image objective.

BAGEL produces text and images, not LTX audio/video. The text track retains
UniGRPO's discrete AR-GRPO; the image track uses the same NFT branch formula as
OmniNFT, reusing DiffusionNFTLoss instead of copying the modality algebra.
"""

import torch

from verl_omni.pipelines.bagel_unigrpo.joint_update import ar_grpo_loss
from verl_omni.trainer.diffusion.diffusion_algos import (
    DiffusionLossFn,
    DiffusionLossResult,
    DiffusionNFTLoss,
    register_diffusion_loss,
)


@register_diffusion_loss("unigrpo_nft")
class BagelUniGRPONFTLoss(DiffusionLossFn):
    """Separate branch forwards/backwards may accumulate into one engine-owned step."""

    required_model_output_keys = ("ar_log_probs", *DiffusionNFTLoss.required_model_output_keys)
    required_data_keys = ("ar_old_log_probs", "ar_advantages", "reward_prob")

    @staticmethod
    def branch_weights(config):
        """Normalize modality loss weights, independently of reward routing weights."""
        loss = config.diffusion_loss
        total = loss.text_loss_weight + loss.image_loss_weight
        return loss.text_loss_weight / total, loss.image_loss_weight / total

    @classmethod
    def compute_ar_loss(cls, *, new_log_probs, old_log_probs, advantages, config):
        """Clipped token objective on sampled tokens, excluding the fixed thinking BOS."""
        weight, _ = cls.branch_weights(config)
        loss, metrics = ar_grpo_loss(
            new_log_probs, old_log_probs.detach(), advantages.detach(), config.diffusion_loss.ar_clip_ratio
        )
        return loss * weight, {**metrics, "ar/policy_loss": float(loss.detach())}

    @classmethod
    def compute_image_loss(cls, *, model_output, reward_prob, config):
        """NFT positive/implicit-negative reconstruction with frozen old/reference velocities."""
        _, weight = cls.branch_weights(config)
        loss, metrics = DiffusionNFTLoss.compute_loss(
            **{key: model_output[key] for key in DiffusionNFTLoss.required_model_output_keys},
            reward_prob=reward_prob,
            config=config,
        )
        return loss * weight, {f"image/{key.split('/')[-1]}": value for key, value in metrics.items()}

    @classmethod
    def compute_loss(cls, *, model_output, data, config):
        """Pure joint loss, also used to verify the low-memory two-backward implementation."""
        ar_loss, ar_metrics = cls.compute_ar_loss(
            new_log_probs=model_output["ar_log_probs"],
            old_log_probs=data["ar_old_log_probs"],
            advantages=data["ar_advantages"],
            config=config,
        )
        image_loss, image_metrics = cls.compute_image_loss(
            model_output=model_output,
            reward_prob=data["reward_prob"],
            config=config,
        )
        loss = ar_loss + image_loss
        return loss, {**ar_metrics, **image_metrics, "joint/total_loss": float(loss.detach())}

    def __call__(self, *, config, model_output, data):
        """Validate and dispatch the registered hybrid loss."""
        self.validate_inputs(loss_name="unigrpo_nft", model_output=model_output, data=data)
        loss, metrics = self.compute_loss(model_output=model_output, data=data, config=config)
        return DiffusionLossResult(loss=loss, metrics=metrics)

    @staticmethod
    def prepare_actor_batch(batch, reward_tensor, config):
        """Route group-relative outcome credit to AR advantages and image NFT probabilities.

        Unlike the generic diffusion path, BAGEL times are continuous normalized
        sigma values in (0, 1); casting them to integer scheduler IDs is invalid.
        Noise and times are generated once and shared by current/old/reference.
        """
        loss = config.actor_rollout_ref.actor.diffusion_loss
        algorithm = config.algorithm
        if "uid" not in batch.non_tensor_batch or "bagel_joint_replay" not in batch.non_tensor_batch:
            raise ValueError("Joint actor preparation requires prompt UIDs and exact native BAGEL replay records")
        records = batch.non_tensor_batch["bagel_joint_replay"]
        for record in records:
            record.validate()
        rewards = reward_tensor.reshape(-1).detach().float().cpu()
        if rewards.numel() != len(records) or not bool(torch.isfinite(rewards).all()):
            raise ValueError("One finite outcome reward is required for each joint rollout")
        advantages = DiffusionNFTLoss._compute_group_advantages(
            rewards=rewards,
            uid=batch.non_tensor_batch["uid"],
            norm_by_std=algorithm.norm_adv_by_std_in_grpo,
            global_std=algorithm.global_std,
        )
        probabilities = DiffusionNFTLoss._advantage_to_reward_prob(
            advantages,
            adv_clip_max=loss.adv_clip_max,
            adv_mode=algorithm.adv_mode,
        )
        clean = torch.stack([record.latents_clean for record in records])
        times = torch.rand(len(records), loss.nft_timesteps_per_sample) * 0.98 + 0.01
        noise = torch.randn(len(records), loss.nft_timesteps_per_sample, *clean.shape[1:])
        batch.batch["latents_clean"] = clean
        batch.batch["train_timesteps"] = times
        batch.batch["nft_noise"] = noise
        batch.batch["advantages"] = advantages
        batch.batch["reward_prob"] = probabilities
        batch.batch["returns"] = advantages
        batch.batch["sample_level_rewards"] = rewards
        return batch
