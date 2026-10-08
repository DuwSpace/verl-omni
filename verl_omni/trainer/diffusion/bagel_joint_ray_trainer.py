# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Joint BAGEL trainer using native vllm-omni MP inference, never actor sampling."""

import numpy as np
import ray

from verl_omni.pipelines.bagel_unigrpo_nft.loss import BagelUniGRPONFTLoss
from verl_omni.trainer.diffusion.native_ray_trainer import NativeRayDiffusionTrainer


class BagelJointRayTrainer(NativeRayDiffusionTrainer):
    """Reuse Ray transport/reward/checkpoints; replace generation and loss preparation."""

    def __init__(self, config, *args, **kwargs):
        model, actor, rollout = (
            config.actor_rollout_ref.model,
            config.actor_rollout_ref.actor,
            config.actor_rollout_ref.rollout,
        )
        if model.algorithm != "unigrpo_nft" or actor.diffusion_loss.loss_mode != "unigrpo_nft":
            raise ValueError("Joint BAGEL requires the unigrpo_nft model adapter and hybrid loss")
        if rollout.name != "vllm_omni" or rollout.tensor_model_parallel_size != 1:
            raise ValueError("Joint BAGEL requires independent TP=1 vllm-omni workers")
        if rollout.n < 2 or config.trainer.nnodes != 1 or config.algorithm.sample_source != "online":
            raise ValueError("Joint BAGEL requires online grouped samples on one node")
        if model.lora_rank or model.get("lora", {}).get("rank", 0) or model.lora_adapter_path:
            raise ValueError("Joint BAGEL currently supports full-weight training only")
        if rollout.seed is None:
            raise ValueError("Joint BAGEL requires an explicit seed")
        super().__init__(config, *args, **kwargs)

    def _generate_native(self, gen_batch_output):
        """Mix prompts across replicas, then restore exact group/seed output order."""
        count = len(gen_batch_output)
        group_size = self.config.actor_rollout_ref.rollout.n
        if count == 0 or count % group_size:
            raise ValueError("Joint rollout must contain complete repeated prompt groups")
        uids = gen_batch_output.non_tensor_batch.get("uid")
        if uids is None or not (uids.reshape(-1, group_size) == uids.reshape(-1, group_size)[:, :1]).all():
            raise ValueError("Joint rollout requires contiguous repeated prompt UID groups")
        version = self.global_steps - 1
        gen_batch_output.non_tensor_batch["_bagel_policy_version"] = np.full(count, version)
        gen_batch_output.non_tensor_batch["_bagel_sample_seed"] = (
            self.config.actor_rollout_ref.rollout.seed + version * count + np.arange(count)
        )
        # A contiguous prompt group otherwise monopolizes one replica; long
        # thinking on that prompt leaves replicas with short prompts idle.
        permutation = np.arange(count).reshape(-1, group_size).T.reshape(-1)
        output = super()._generate_native(gen_batch_output.select_idxs(permutation))
        return output.select_idxs(np.argsort(permutation))

    def _prepare_actor_batch(self, batch, reward_tensor):
        return BagelUniGRPONFTLoss.prepare_actor_batch(batch, reward_tensor, self.config)

    def _validate(self):
        return {"val/joint/skipped": 1.0}

    def fit(self):
        """Bound cleanup after failures without masking the original training error."""
        failed = False
        try:
            return super().fit()
        except BaseException:
            failed = True
            raise
        finally:
            try:
                pending = self.actor_rollout_wg.close_joint_rollout()
                ray.get(pending, timeout=30)
            except Exception:
                if not failed:
                    raise
