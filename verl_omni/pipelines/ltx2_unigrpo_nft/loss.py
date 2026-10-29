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
from verl_omni.trainer.diffusion.recaption_grpo import (
    clipped_recaption_loss,
    probability_group_advantages,
    recaption_reference_kl,
    reward_group_advantages,
)


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
            reference=model_output.get("ar_ref_logprobs"),
        )
        av, av_metrics = self.av_loss(model_output=model_output, probabilities=data["reward_prob"], config=config)
        return DiffusionLossResult(loss=text + av, metrics={**text_metrics, **av_metrics})

    @staticmethod
    def text_loss(*, current, old, advantages, mask, config, reference=None):
        loss, metrics = clipped_recaption_loss(
            current,
            old,
            advantages,
            mask,
            clip_ratio=config.diffusion_loss.ar_clip_ratio,
            policy_loss=config.diffusion_loss.text_policy_loss,
        )
        coefficient = config.diffusion_loss.text_kl_coef
        kl = loss.new_zeros(())
        if coefficient:
            if reference is None:
                raise ValueError("Text KL requires frozen reference logprobs")
            kl = recaption_reference_kl(current, reference, mask)
        total = config.diffusion_loss.text_loss_weight * (loss + coefficient * kl)
        metrics.update(
            {
                "text/kl_loss": kl.detach().item(),
                "text/kl_coef": coefficient,
                "text/weighted_kl_loss": (config.diffusion_loss.text_loss_weight * coefficient * kl).detach().item(),
                "text/total_loss": total.detach().item(),
            }
        )
        return total, metrics

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
        cls._prepare_text_credit(batch, config.actor_rollout_ref.actor.diffusion_loss)
        cls._prepare_forward_noise(batch, records)
        return batch

    @staticmethod
    def _prepare_text_credit(batch, loss_config):
        """Derive one grouped text advantage per sample from the configured reward source."""
        if loss_config.text_probability_reduction != "mean":
            raise ValueError("Unsupported scalar text probability reduction")
        probabilities = batch.batch["reward_prob"][:, 0].mean(dim=-1)
        advantages, metrics = probability_group_advantages(
            probabilities,
            batch.non_tensor_batch["uid"],
            epsilon=loss_config.text_advantage_epsilon,
            global_std=loss_config.text_global_std,
        )
        source = getattr(loss_config, "text_advantage_source", "component_advantage_mean")
        scalar_reward = probabilities
        if source == "combined_reward":
            # Same raw sum as critic/reward/combined/mean in preserve-components runs.
            scalar_reward = batch.batch["rm_scores"].detach().float().sum(dim=-1)
            advantages, reward_metrics = reward_group_advantages(
                scalar_reward,
                batch.non_tensor_batch["uid"],
                epsilon=loss_config.text_advantage_epsilon,
                global_std=loss_config.text_global_std,
            )
            metrics.update(reward_metrics)
        elif source == "component_advantage_mean":
            rewards = batch.batch["rm_scores"].detach().float()
            if rewards.ndim != 2 or rewards.shape[1] == 0:
                raise ValueError("Component text advantages require rm_scores shaped [samples, rewards]")
            names = batch.meta_info.get("reward_names", [str(i) for i in range(rewards.shape[1])])
            if len(names) != rewards.shape[1] or len(set(names)) != len(names):
                raise ValueError("Component text advantages require one unique name per reward")
            component_advantages = []
            constant_fractions = []
            for i, name in enumerate(names):
                component_advantage, component_metrics = reward_group_advantages(
                    rewards[:, i],
                    batch.non_tensor_batch["uid"],
                    epsilon=loss_config.text_advantage_epsilon,
                    global_std=loss_config.text_global_std,
                )
                component_advantages.append(component_advantage)
                constant_fractions.append(component_metrics["text/constant_group_fraction"])
                metrics.update(
                    {
                        f"text/components/{name}/{key.removeprefix('text/')}": value
                        for key, value in component_metrics.items()
                    }
                )
            # Equal weight per scorer, including constant components. No second normalization.
            advantages = torch.stack(component_advantages, dim=-1).mean(dim=-1)
            # Diagnostic scalar only: advantages are computed before this raw reward mean.
            scalar_reward = rewards.mean(dim=-1)
            metrics["text/constant_group_fraction"] = sum(constant_fractions) / len(constant_fractions)
            metrics["text/advantage_abs_mean"] = advantages.abs().mean().item()
        elif source != "modality_probability":
            raise ValueError(
                "Supported text_advantage_source: modality_probability, combined_reward, component_advantage_mean"
            )
        metrics["text/reward_source_component_advantage_mean"] = float(source == "component_advantage_mean")
        metrics["text/reward_source_combined"] = float(source == "combined_reward")
        batch.batch["text_advantages"] = advantages
        batch.batch["text_reward_prob"] = probabilities
        batch.batch["text_reward"] = scalar_reward
        batch.meta_info["text_reward_source"] = source
        batch.meta_info["text_reward_metrics"] = metrics

    @staticmethod
    def _prepare_forward_noise(batch, records):
        """Seed video/audio noise from replay identity, independent of rank and batch order."""
        steps = batch.batch["train_timesteps"].shape[1]
        for modality, salt in (("video", 0x564944), ("audio", 0x415544)):
            clean = batch.batch[f"{modality}_latents_clean"]
            # Replay identity determines noise, independent of rank/order/global RNG.
            # Modality salts keep the video and audio random streams separate.
            noise = []
            for record in records:
                generator = torch.Generator(device=clean.device)
                generator.manual_seed((int(record.seed) + salt) % (2**63))
                noise.append(
                    torch.randn(
                        (steps, *clean.shape[1:]),
                        generator=generator,
                        device=clean.device,
                        dtype=torch.float32,
                    )
                )
            batch.batch[f"{modality}_forward_noise"] = torch.stack(noise)
