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

from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.reward import extract_reward
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
from verl_omni.trainer.diffusion.bagel_joint_ray_trainer import BagelJointRayTrainer
from verl_omni.trainer.diffusion.diffusion_metric_utils import compute_old_policy_metrics
from verl_omni.trainer.diffusion.diffusion_trainer_utils import _to_diffusion_worker_tensordict, old_policy_decay
from verl_omni.trainer.diffusion.native_ray_trainer import NativeRayDiffusionTrainer


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

    def _validation_n(self):
        value = self.config.actor_rollout_ref.rollout.get("val_kwargs", {}).get("n", 1)
        if type(value) is not int or value <= 0:
            raise ValueError("Joint validation val_kwargs.n must be a positive integer")
        return value

    def _validate(self):
        """Evaluate the EMA paired policy once, after all mini-batches finish."""
        from verl_omni.pipelines.ltx2_unigrpo_nft.artifacts import save_joint_samples

        started = perf_counter()
        batches = [DataProto.from_single_dict(item) for item in self.val_dataloader]
        if not batches:
            raise ValueError("Joint evaluation requires a nonempty fixed validation dataset")
        source = DataProto.concat(batches)
        source.non_tensor_batch["uid"] = np.array([f"eval-{i}" for i in range(len(source))], dtype=object)
        batch = self._get_gen_batch(source).repeat(self._validation_n(), interleave=True)
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
            index = save_joint_samples(
                batch,
                directory,
                names,
                save_tensors=self.config.trainer.get("joint_sample_save_tensors", False),
            )
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
        if rollout.seed is None:
            raise ValueError("Joint LTX requires an explicit seed")
        micro = actor.ppo_micro_batch_size_per_gpu
        if type(micro) is not int or micro <= 0:
            raise ValueError("Joint micro-batch size must be a positive integer")
        world = config.trainer.n_gpus_per_node
        samples = config.data.train_batch_size * rollout.n
        if world <= 0 or samples <= 0 or actor.ppo_mini_batch_size <= 0:
            raise ValueError("Joint actor world and batch sizes must be positive")
        if samples % world or actor.ppo_mini_batch_size % world or samples % actor.ppo_mini_batch_size:
            raise ValueError("Joint sample batches and actual-sample PPO mini-batches must divide across actor ranks")
        if (actor.ppo_mini_batch_size // world) % micro:
            raise ValueError("Joint per-rank mini-batch must divide by micro-batch size")

    def _generate_native(self, gen_batch_output):
        from omegaconf import OmegaConf

        from verl_omni.pipelines.ltx2_unigrpo_nft.config import sampling_params

        rollout = self.config.actor_rollout_ref.rollout
        if "pipeline" in rollout:
            pipeline = rollout.pipeline
            if getattr(self, "_evaluating", False):
                pipeline = OmegaConf.merge(pipeline, rollout.get("val_kwargs", {}).get("pipeline", {}))
            params = sampling_params(pipeline)
            gen_batch_output.non_tensor_batch["_ltx_sampling_params"] = np.array(
                [params for _ in range(len(gen_batch_output))], dtype=object
            )
        world = self.config.trainer.n_gpus_per_node
        if getattr(self, "_evaluating", False):
            count, group_size = len(gen_batch_output), self._validation_n()
            if count == 0 or count % group_size:
                raise ValueError("Joint validation requires complete prompt groups")
            version = self._rollout_version()
            gen_batch_output.non_tensor_batch[self.policy_version_key] = np.full(count, version)
            gen_batch_output.non_tensor_batch[self.sample_seed_key] = self._rollout_seeds(count, version)
            permutation = np.arange(count).reshape(-1, group_size).T.reshape(-1)
            # Every FSDP rank must join paired LoRA export. Pad only dispatch;
            # replicas of real requests keep their seeds and are never scored.
            padded, padding = pad_dataproto_to_divisor(gen_batch_output.select_idxs(permutation), world)
            output = NativeRayDiffusionTrainer._generate_native(self, padded)
            output = unpad_dataproto(output, padding).select_idxs(np.argsort(permutation))
        else:
            if len(gen_batch_output) % world:
                raise ValueError("Joint rollout samples must divide across all actor ranks")
            output = super()._generate_native(gen_batch_output)
        # Dataset modality targets must survive generation/reordering. Native
        # workers use the original AV prompt only as a fallback for plain tasks.
        targets = gen_batch_output.non_tensor_batch.get("reward_inputs")
        if targets is not None:
            for target in targets:
                text = target.get("text", {}) if isinstance(target, dict) else {}
                if any(not isinstance(text.get(key), str) or not text[key].strip() for key in ("video", "audio")):
                    raise ValueError("Joint reward_inputs requires nonempty text.video and text.audio")
            output.non_tensor_batch["reward_inputs"] = deepcopy(targets)
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
        batch.meta_info["reward_diagnostics_context"] = {
            "step": self.global_steps,
            "phase": "validation" if getattr(self, "_evaluating", False) else "train",
            "policy_version": self._rollout_version(),
        }
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
            save_tensors=self.config.trainer.get("joint_sample_save_tensors", False),
        )
        timing_raw["dump_rollout_generations"] = perf_counter() - started

    def _update_old_policy(self):
        """Refresh both LoRA components after all actor mini-batches, before saving."""
        algorithm = self.config.algorithm
        interval = algorithm.get("old_policy_update_interval", 1)
        if type(interval) is not int or interval <= 0:
            raise ValueError("Joint old_policy_update_interval must be a positive integer")
        if self.global_steps % interval:
            return False, 0.0, "none"
        decay = algorithm.get("old_policy_decay")
        if decay is None:
            decay = old_policy_decay(self.global_steps, algorithm.get("old_policy_decay_schedule", "copy"))
        self.actor_rollout_wg.ema_update_adapter(source="default", target="old", decay=decay)
        return True, float(decay), "copy" if decay == 0 else "ema"

    def _update_actor(self, batch):
        data = _to_diffusion_worker_tensordict(batch)
        actor = self.config.actor_rollout_ref.actor
        # ppo_mini_batch_size counts complete recaption+AV samples, not prompts.
        tu.assign_non_tensor(
            data,
            text_diagnostic_step=self.global_steps,
            global_batch_size=actor.ppo_mini_batch_size,
            mini_batch_size=actor.ppo_mini_batch_size,
            epochs=actor.ppo_epochs,
            seed=actor.data_loader_seed,
            dataloader_kwargs={"shuffle": actor.shuffle},
        )
        started = perf_counter()
        result = self.actor_rollout_wg.update_actor(data)
        old_policy_metrics = compute_old_policy_metrics(self._update_old_policy())
        finished = perf_counter()
        metrics = {f"actor/{key}": value for key, value in tu.get(result, "metrics").items()}
        metrics.update(old_policy_metrics)
        metrics.update(batch.meta_info.get("text_reward_metrics", {}))
        metrics["train/total_time"] = finished - started
        if hasattr(self, "_joint_reward_time"):
            metrics["reward/total_time"] = self._joint_reward_time
        metrics.update(self._rollout_timing_metrics(batch, finished))
        self._save_update_metrics(metrics)
        return DataProto.from_single_dict(data={}, meta_info={"metrics": metrics})

    def _rollout_timing_metrics(self, batch, finished):
        """Aggregate concurrent replica timings without summing overlapping wall time."""
        metrics = {}
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
            for key in (
                "rollout/max_memory_allocated_gib",
                "rollout/max_memory_reserved_gib",
            ):
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
        return metrics

    def _save_update_metrics(self, metrics):
        """Persist completed-update evidence before checkpointing and validation."""
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
