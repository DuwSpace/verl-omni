# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synchronous native LTX rollout and joint GRPO/OmniNFT actor updates."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.reward import extract_reward
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
from verl_omni.trainer.diffusion.bagel_joint_ray_trainer import BagelJointRayTrainer
from verl_omni.trainer.diffusion.diffusion_trainer_utils import _to_diffusion_worker_tensordict


class LTXJointRayTrainer(BagelJointRayTrainer):
    """Reuse grouped dispatch and cleanup; preserve actual-sample mini-batch sizes."""

    policy_version_key = "_ltx_policy_version"
    sample_seed_key = "_ltx_sample_seed"

    def _rollout_version(self):
        # Reserve an evaluation version after each completed update round.
        # Initial evaluation is 0, training step s is 2s-1, evaluation is 2s.
        # A checkpoint saved before evaluation can resume safely at 2(s+1)-1.
        return 2 * self.global_steps - (0 if getattr(self, "_evaluating", False) else 1)

    def _rollout_seeds(self, count, version):
        if getattr(self, "_evaluating", False):
            return self.config.actor_rollout_ref.rollout.seed + np.arange(count)
        return super()._rollout_seeds(count, version)

    def _validate(self):
        """Evaluate the current paired policy once, after all mini-batches finish."""
        from verl_omni.pipelines.ltx2_unigrpo_nft.artifacts import save_joint_samples

        started = perf_counter()
        batches = [DataProto.from_single_dict(item) for item in self.val_dataloader]
        if not batches:
            raise ValueError("Joint evaluation requires a nonempty fixed validation dataset")
        source = DataProto.concat(batches)
        source.non_tensor_batch["uid"] = np.array([f"eval-{i}" for i in range(len(source))], dtype=object)
        batch = self._get_gen_batch(source).repeat(self.config.actor_rollout_ref.rollout.n, interleave=True)
        if batch.batch is None:
            batch.batch = TensorDict({}, batch_size=[len(batch)])
        batch.meta_info["global_steps"] = self.global_steps
        self._evaluating = True
        try:
            batch = batch.union(self._generate_native(batch))
            batch = batch.union(self._compute_reward_colocate(batch))
            reward_tensor, _ = extract_reward(batch)
            batch.batch["sample_level_scores"] = reward_tensor
            # Preparation retains the source probability map. Its training
            # noise is unused here; preserve the CPU RNG for subsequent updates.
            with torch.random.fork_rng(devices=[]):
                batch = self._prepare_actor_batch(batch, reward_tensor)
            root = self.config.trainer.get("rollout_data_dir") or self.config.trainer.default_local_dir
            directory = Path(root) / "evaluation" / f"step-{self.global_steps:06d}"
            names = sorted(self.config.reward.reward_functions)
            index = save_joint_samples(batch, directory, names)
            metrics = {
                "val/joint/total_time": perf_counter() - started,
                "val/joint/samples": len(batch),
                "val/joint/policy_version": self._rollout_version(),
                "val/joint/eos_fraction": sum(item["ended_with_eos"] for item in index) / len(index),
                "val/joint/text_probability_mean": batch.batch["text_reward_prob"].mean().item(),
            }
            for i, name in enumerate(names):
                metrics[f"val/reward/{name}"] = batch.batch["rm_scores"][:, i].mean().item()
            for i, name in enumerate(("video", "audio")):
                metrics[f"val/joint/{name}_probability_mean"] = batch.batch["reward_prob"][:, 0, i].mean().item()
            print(f"Joint evaluation samples: {directory / 'samples.json'}")
            return metrics
        finally:
            self._evaluating = False

    @staticmethod
    def validate_joint_config(config):
        if config.data.get("validation_shuffle", False):
            raise ValueError("Joint fixed-prompt evaluation requires validation_shuffle=false")
        if config.algorithm.trainer_type != "direct_preference":
            raise ValueError("Joint LTX uses the online direct-preference trainer family")
        model, actor, rollout = (
            config.actor_rollout_ref.model,
            config.actor_rollout_ref.actor,
            config.actor_rollout_ref.rollout,
        )
        if model.algorithm != "ltx2_unigrpo_nft" or actor.diffusion_loss.loss_mode != "ltx2_unigrpo_nft":
            raise ValueError("Joint LTX requires its paired adapter and loss")
        if rollout.name != "vllm_omni" or rollout.tensor_model_parallel_size != 1:
            raise ValueError("Joint LTX requires independent TP=1 native vLLM-Omni workers")
        if rollout.n < 2 or config.trainer.nnodes != 1 or config.algorithm.sample_source != "online":
            raise ValueError("Joint LTX requires online grouped samples on one node")
        if model.lora_rank <= 0 or tuple(model.policy_state_adapters) != ("default",):
            raise ValueError("Joint LTX requires default Gemma/DiT LoRA and snapshot-owned old/ref")
        if rollout.seed is None or actor.ppo_micro_batch_size_per_gpu != 1:
            raise ValueError("Joint LTX requires an explicit seed and one-sample micro-batches")
        world = config.trainer.n_gpus_per_node
        samples = config.data.train_batch_size * rollout.n
        if world <= 0 or samples <= 0 or actor.ppo_mini_batch_size <= 0:
            raise ValueError("Joint actor world and batch sizes must be positive")
        if samples % world or actor.ppo_mini_batch_size % world or samples % actor.ppo_mini_batch_size:
            raise ValueError("Joint sample batches and actual-sample PPO mini-batches must divide across actor ranks")

    def _generate_native(self, gen_batch_output):
        if len(gen_batch_output) % self.config.trainer.n_gpus_per_node:
            raise ValueError("Joint rollout samples must divide across all actor ranks")
        output = super()._generate_native(gen_batch_output)
        # These original-prompt reward/media fields come from the native result;
        # avoid union conflicts with optional dataset-supplied placeholders.
        for key in ("reward_inputs", "fps", "audio_layout"):
            gen_batch_output.non_tensor_batch.pop(key, None)
        return output

    def _prepare_actor_batch(self, batch, reward_tensor):
        batch = LTXJointLoss.prepare_actor_batch(batch, reward_tensor, self.config)
        root = self.config.trainer.get("rollout_data_dir")
        frequency = self.config.trainer.get("rollout_data_save_freq", 1)
        if root and frequency > 0 and self.global_steps % frequency == 0 and not getattr(self, "_evaluating", False):
            # Retain costly native rollout/reward evidence even if the actor
            # subsequently fails. Checkpoints still follow successful updates.
            timings = {}
            self._log_rollout_data(batch, {}, timings, root)
            batch.meta_info["joint_artifacts_time"] = timings["dump_rollout_generations"]
        return batch

    def _compute_reward_colocate(self, batch):
        started = perf_counter()
        result = super()._compute_reward_colocate(batch)
        self._joint_reward_time = perf_counter() - started
        return result

    def _log_rollout_data(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
        from verl_omni.pipelines.ltx2_unigrpo_nft.artifacts import save_joint_samples

        if "joint_artifacts_time" in batch.meta_info:
            timing_raw["dump_rollout_generations"] = batch.meta_info["joint_artifacts_time"]
            return
        started = perf_counter()
        save_joint_samples(
            batch,
            Path(rollout_data_dir) / f"step-{self.global_steps:06d}",
            sorted(self.config.reward.reward_functions),
        )
        timing_raw["dump_rollout_generations"] = perf_counter() - started

    def _update_actor(self, batch):
        data = _to_diffusion_worker_tensordict(batch)
        actor = self.config.actor_rollout_ref.actor
        # ppo_mini_batch_size counts complete recaption+AV samples, not prompts.
        tu.assign_non_tensor(
            data,
            global_batch_size=actor.ppo_mini_batch_size,
            mini_batch_size=actor.ppo_mini_batch_size,
            epochs=actor.ppo_epochs,
            seed=actor.data_loader_seed,
            dataloader_kwargs={"shuffle": actor.shuffle},
        )
        started = perf_counter()
        result = self.actor_rollout_wg.update_actor(data)
        finished = perf_counter()
        metrics = {f"actor/{key}": value for key, value in tu.get(result, "metrics").items()}
        metrics.update(batch.meta_info.get("text_reward_metrics", {}))
        metrics["train/total_time"] = finished - started
        if hasattr(self, "_joint_reward_time"):
            metrics["reward/total_time"] = self._joint_reward_time
        timings = batch.non_tensor_batch.get("ltx_rollout_metrics")
        if timings is not None:
            # All actors are on one host: perf_counter shares its monotonic
            # origin. Measure synchronization through update completion, not
            # the sum of overlapping per-replica stage durations.
            wall = finished - min(item["sync_started_at"] for item in timings)
            metrics["step/total_time"] = wall
            metrics["step/samples_per_second"] = len(batch) / wall
            for key in ("rollout/sync_time", "rollout/model_load_time"):
                metrics[key] = max(item[key] for item in timings)
            for key in ("rollout/max_memory_allocated_gib", "rollout/max_memory_reserved_gib"):
                values = [item[key] for item in timings if key in item]
                if values:
                    metrics[key] = max(values)
            for key in ("rollout/recaption_time", "rollout/generation_time"):
                per_rank = {}
                for item in timings:
                    per_rank[item["rank"]] = per_rank.get(item["rank"], 0.0) + item[key]
                metrics[key] = max(per_rank.values())
            tokens = sum(record.response_token_ids.numel() for record in batch.non_tensor_batch["ltx_joint_replay"])
            metrics["rollout/recaption_tokens_per_second"] = tokens / metrics["rollout/recaption_time"]
        # The parent logs only after checkpointing and evaluation. Persist the
        # completed update independently so a later failure cannot erase its
        # timing and gradient evidence. This is not a resumable checkpoint.
        root = self.config.trainer.get("default_local_dir")
        if root:
            directory = Path(root) / "update_metrics"
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"step-{self.global_steps:06d}.pt"
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(
                {
                    "global_step": self.global_steps,
                    "rollout_policy_version": self._rollout_version(),
                    "phase": "actor_update_complete",
                    "metrics": metrics,
                },
                temporary,
            )
            temporary.replace(destination)
        return DataProto.from_single_dict(data={}, meta_info={"metrics": metrics})
