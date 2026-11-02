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

"""Text GRPO credit on top of the existing audiovisual OmniNFT objective."""

import torch

from verl_omni.trainer.diffusion.diffusion_algos import OmniNFTLoss, register_diffusion_loss

from ..replay import LTXJointReplay
from ..tree_rollout.advantages import av_reward_group_ids, compute_text_advantages
from ..tree_rollout.metrics import compute_tree_reward_metrics


@register_diffusion_loss("ltx2_unigrpo_nft")
class LTXJointLoss(OmniNFTLoss):
    @classmethod
    def prepare_actor_batch(cls, batch, reward_tensor, config):
        records = batch.non_tensor_batch.get("ltx_joint_replay")
        if records is None or len(records) != len(batch):
            raise ValueError("Missing sampled text replay for joint training")
        records = [record if isinstance(record, LTXJointReplay) else LTXJointReplay(**record) for record in records]
        for record in records:
            record.validate()
        if len({record.policy_version for record in records}) != 1:
            raise ValueError("Joint actor batch mixes rollout policy versions")
        loss_config = config.actor_rollout_ref.actor.diffusion_loss
        if loss_config.text_advantage_source != "component_advantage_mean":
            raise ValueError("Only component_advantage_mean is supported")
        if any(record.recaption_group_id is not None for record in records):
            prompt_uid = batch.non_tensor_batch["uid"]
            text_advantages = compute_text_advantages(
                batch.batch["rm_scores"],
                prompt_uid,
                records,
                cls._compute_component_advantages,
                global_std=loss_config.text_global_std,
                epsilon=loss_config.text_advantage_epsilon,
            )
            # Only tree AV rewards use recaption groups. Keep original prompt
            # identities for text grouping, replay and downstream consumers.
            batch.non_tensor_batch["uid"] = av_reward_group_ids(prompt_uid, records)
            try:
                batch = super().prepare_actor_batch(batch, reward_tensor, config)
            finally:
                batch.non_tensor_batch["uid"] = prompt_uid
            batch.meta_info.setdefault("actor_batch_metrics", {}).update(
                compute_tree_reward_metrics(
                    batch.batch["rm_scores"], prompt_uid, records, batch.meta_info["reward_names"]
                )
            )
        else:
            batch = super().prepare_actor_batch(batch, reward_tensor, config)
            components = cls._compute_component_advantages(
                scores=batch.batch["rm_scores"].detach().float(),
                uid=batch.non_tensor_batch["uid"],
                norm_by_std=True,
                global_std=loss_config.text_global_std,
                epsilon=loss_config.text_advantage_epsilon,
            )
            text_advantages = components.mean(dim=-1)
        batch.batch["text_advantages"] = text_advantages
        for modality, salt in (("video", 0x564944), ("audio", 0x415544)):
            clean = batch.batch[f"{modality}_latents_clean"]
            steps = batch.batch["train_timesteps"].shape[1]
            batch.batch[f"{modality}_forward_noise"] = torch.stack(
                [
                    torch.randn((steps, *clean.shape[1:]), generator=torch.Generator().manual_seed(record.seed ^ salt))
                    for record in records
                ]
            ).to(clean.device)
        return batch
