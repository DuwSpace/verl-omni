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

"""Recaption credit derived from the unchanged two-probability OmniNFT pipeline."""

import torch
from tensordict import TensorDict

from verl_omni.trainer.diffusion.diffusion_algos import DiffusionLossResult, OmniNFTLoss, register_diffusion_loss
from verl_omni.trainer.diffusion.recaption_grpo import clipped_recaption_loss, probability_group_advantages


@register_diffusion_loss("ltx2_unigrpo_nft")
class LTXJointLoss(OmniNFTLoss):
    """Expose separate branch losses for two backwards before one optimizer step."""

    required_model_output_keys = (*OmniNFTLoss.required_model_output_keys, "ar_logprobs")
    required_data_keys = ("reward_prob", "ar_old_logprobs", "text_advantages", "ar_response_mask")

    def __call__(self, *, config, model_output, data):
        self.validate_inputs(loss_name="ltx2_unigrpo_nft", model_output=model_output, data=data)
        text, text_metrics = self.text_loss(
            current=model_output["ar_logprobs"],
            old=data["ar_old_logprobs"],
            advantages=data["text_advantages"],
            mask=data["ar_response_mask"],
            config=config,
        )
        av, av_metrics = self.av_loss(model_output=model_output, probabilities=data["reward_prob"], config=config)
        return DiffusionLossResult(loss=text + av, metrics={**text_metrics, **av_metrics})

    @staticmethod
    def text_loss(*, current, old, advantages, mask, config):
        loss, metrics = clipped_recaption_loss(
            current, old, advantages, mask, clip_ratio=config.diffusion_loss.ar_clip_ratio
        )
        return config.diffusion_loss.text_loss_weight * loss, metrics

    @staticmethod
    def av_loss(*, model_output, probabilities, config):
        result = OmniNFTLoss()(
            config=config,
            model_output=model_output,
            data=TensorDict({"reward_prob": probabilities}, batch_size=[probabilities.shape[0]]),
        )
        return config.diffusion_loss.av_loss_weight * result.loss, result.metrics

    @classmethod
    def prepare_actor_batch(cls, batch, reward_tensor, config):
        records = batch.non_tensor_batch.get("ltx_joint_replay")
        if records is None or len(records) != len(batch):
            raise ValueError("Joint actor preparation requires a replay record per sample")
        for record in records:
            record.validate()
        if len({record.policy_version for record in records}) != 1:
            raise ValueError("Joint actor batch must contain one rollout policy version")
        if "rm_scores" not in batch.batch or not bool(torch.isfinite(batch.batch["rm_scores"]).all()):
            raise ValueError("Joint actor preparation requires finite component reward scores")
        # Preserve the source normalization, routing, probability map and timestep selection.
        batch = OmniNFTLoss.prepare_actor_batch(batch, reward_tensor, config)
        loss_config = config.actor_rollout_ref.actor.diffusion_loss
        if loss_config.text_probability_reduction != "mean":
            raise ValueError("Unsupported scalar text probability reduction")
        probabilities = batch.batch["reward_prob"][:, 0].mean(dim=-1)
        advantages, metrics = probability_group_advantages(
            probabilities,
            batch.non_tensor_batch["uid"],
            epsilon=loss_config.text_advantage_epsilon,
        )
        batch.batch["text_advantages"] = advantages
        batch.batch["text_reward_prob"] = probabilities
        batch.meta_info["text_reward_metrics"] = metrics
        steps = batch.batch["train_timesteps"].shape[1]
        for modality in ("video", "audio"):
            clean = batch.batch[f"{modality}_latents_clean"]
            batch.batch[f"{modality}_forward_noise"] = torch.randn(
                (clean.shape[0], steps, *clean.shape[1:]),
                device=clean.device,
                dtype=torch.float32,
            )
        return batch
