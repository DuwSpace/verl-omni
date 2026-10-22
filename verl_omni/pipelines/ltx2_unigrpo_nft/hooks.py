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

"""Joint GRPO/OmniNFT accumulation under the shared engine-owned optimizer."""

import os
from dataclasses import asdict
from time import perf_counter

import torch
from torch.distributed.tensor import DTensor
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionEngineHooks
from verl_omni.trainer.diffusion.recaption_grpo import old_policy_logprob_metrics

from .loss import LTXJointLoss
from .policy_state import LTXJointPolicyState


class LTXJointHooks(DiffusionEngineHooks):
    """One local PPO mini-batch per call, accumulated in configurable micro-batches.

    The engine clears gradients and steps after this call. All reference outputs
    precede each sample's differentiable forwards, and every current graph is
    consumed before switching state for another sample. Text updates Gemma LoRA;
    AV uses frozen Gemma conditioning and updates DiT LoRA. Both branches share
    one optimizer accumulation window.
    """

    requires_checkpoint_state = True

    def __init__(self, module, model_config, optimizer_config):
        self.module = module
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.policies = LTXJointPolicyState(module)
        self._rollout_backend = None
        from .text_diagnostics import TextDiagnostics

        settings = getattr(model_config, "ltx_joint", None)
        self.text_diagnostics = TextDiagnostics(
            getattr(settings, "text_diagnostics", None),
            side="actor",
            source_rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
        )
        device = self._device()
        if device.type == "npu" and "LTX_ACTOR_NPU_MEMORY_FRACTION" in os.environ:
            # CANN kernels also allocate outside PyTorch's caching allocator.
            # Keep headroom for those allocations and colocated worker contexts.
            fraction = float(os.environ["LTX_ACTOR_NPU_MEMORY_FRACTION"])
            if not 0 < fraction <= 1:
                raise ValueError("LTX_ACTOR_NPU_MEMORY_FRACTION must be in (0, 1]")
            torch.get_device_module(device).set_per_process_memory_fraction(fraction, device)

    def generate(self, data):
        from verl import DataProto

        from verl_omni.utils.determinism import configure_worker_determinism
        from verl_omni.workers.rollout.bagel_joint_backend import original_prompt_text
        from verl_omni.workers.rollout.ltx_joint_backend import LTXJointVllmBackend

        from .config import sampling_params
        from .rewards import reward_sample

        configure_worker_determinism()
        prompts = tu.get(data, "raw_prompt")
        versions = tu.get(data, "_ltx_policy_version")
        seeds = tu.get(data, "_ltx_sample_seed")
        if prompts is None or versions is None or seeds is None or not len(prompts):
            raise ValueError("Joint LTX generation needs original prompts, versions and seeds")
        if len(versions) != len(prompts) or len(seeds) != len(prompts) or len(set(int(v) for v in versions)) != 1:
            raise ValueError("Joint LTX generation requires one version and a seed per prompt")
        version = int(versions[0])
        texts = [original_prompt_text(prompt) for prompt in prompts]
        self.begin_rollout(version=version)
        device = self._device()
        if device.type != "cpu":
            from verl.utils.memory_utils import aggressive_empty_cache

            aggressive_empty_cache(force_sync=True)
        settings, geometry = self.model_config.ltx_joint, self.model_config.pipeline
        loading = 0.0
        if self._rollout_backend is None:
            started = perf_counter()
            self._rollout_backend = LTXJointVllmBackend(
                model_path=self.model_config.local_path,
                geometry=sampling_params(geometry),
                recaption_temperature=settings.temperature,
                recaption_max_new_tokens=settings.max_new_tokens,
                sync_timeout_seconds=settings.sync_timeout_seconds,
                enable_cpu_offload=settings.cpu_offload,
                vae_use_tiling=settings.vae_use_tiling,
                text_diagnostics=getattr(settings, "text_diagnostics", None),
            )
            loading = perf_counter() - started
        if device.type != "cpu":
            torch.get_device_module(device).reset_peak_memory_stats()
        started = perf_counter()
        # Training samples and old-policy replay must use the identical EMA.
        # Restore trainable current parameters immediately after adapter export.
        with self.policies.use("old"):
            self._rollout_backend.load_actor_policy(self.module, version=version)
        sync = perf_counter() - started
        request_configs = tu.get(data, "_ltx_sampling_params")
        overrides = {}
        if request_configs is not None:
            request_configs = list(request_configs)
            if len(request_configs) != len(texts) or any(item != request_configs[0] for item in request_configs):
                raise ValueError("Joint rollout requires uniform sampling settings within each worker batch")
            overrides["sampling_params"] = request_configs[0]
        if self.text_diagnostics.enabled:
            from .text_diagnostics import stop_collectively

            # A native diagnostic failure must not strand other actor ranks at
            # their next FSDP collective. All finish this phase before stopping.
            error = None
            try:
                results = self._rollout_backend.generate(texts, seeds, version=version, **overrides)
            except Exception as exc:
                error = f"rollout policy {version}: {type(exc).__name__}: {exc}"
            stop_collectively(error)
        else:
            results = self._rollout_backend.generate(texts, seeds, version=version, **overrides)
        media = DataProto.concat([reward_sample(sample) for sample in results])
        values = dict(media.batch.items())
        # Original data_source/reward_model remain on the trainer's source batch.
        values.update({key: list(media.non_tensor_batch[key]) for key in ("reward_inputs", "fps", "audio_layout")})
        values["ltx_joint_replay"] = [sample["replay"] for sample in results]
        for key in ("video_latents_clean", "audio_latents_clean", "train_timesteps"):
            values[key] = torch.cat([sample[key] for sample in results], dim=0)
        values["ltx_rollout_metrics"] = [
            {
                "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                "sync_started_at": started,
                "rollout/model_load_time": loading,
                "rollout/sync_time": sync,
                "rollout/recaption_time": sample["recaption_time"],
                "rollout/generation_time": sample["generation_time"],
                **sample.get("memory_metrics", {}),
            }
            for sample in results
        ]
        return tu.get_tensordict(values)

    def close_rollout(self):
        if self._rollout_backend is not None:
            self._rollout_backend.close()
            self._rollout_backend = None

    def begin_rollout(self, *, version):
        self.policies.begin_rollout(version=version)

    def ema_update_adapter(self, *, source, target, decay):
        if (source, target) != ("default", "old"):
            raise ValueError("Joint EMA only supports default -> old")
        self.policies.update_old(decay=decay)

    def state_dict(self):
        return self.policies.state_dict()

    def load_state_dict(self, state):
        self.policies.load_state_dict(state)

    def _device(self):
        # Use the accelerator even when FSDP keeps local parameter shards on CPU.
        from verl.utils.device import get_device_id, get_device_name

        name = get_device_name()
        return torch.device(name, get_device_id()) if name != "cpu" else torch.device("cpu")

    def _synchronize(self):
        device = self._device()
        if device.type != "cpu":
            torch.get_device_module(device).synchronize()

    def _av_inputs(self, record, video_clean, audio_clean, video_noise, audio_noise, timestep):
        device = self._device()
        t = timestep.to(device=device, dtype=torch.float32).reshape(1)
        sigma = t.reshape(1, 1, 1) / 1000.0
        clean = (video_clean.to(device).unsqueeze(0), audio_clean.to(device).unsqueeze(0))
        noise = (video_noise.to(device).unsqueeze(0), audio_noise.to(device).unsqueeze(0))
        noisy = tuple((1 - sigma) * x.float() + sigma * n.float() for x, n in zip(clean, noise, strict=True))
        params = record.generation_params
        dtype = next(self.module.transformer.parameters()).dtype
        inputs = dict(
            input_ids=record.condition_token_ids.to(device).unsqueeze(0),
            attention_mask=record.condition_attention_mask.to(device).unsqueeze(0),
            hidden_states=noisy[0].to(dtype),
            audio_hidden_states=noisy[1].to(dtype),
            timestep=t,
            sigma=t,
            audio_timestep=t,
            audio_sigma=t,
            num_frames=(params["num_frames"] - 1) // 8 + 1,
            height=params["height"] // 32,
            width=params["width"] // 32,
            fps=params["frame_rate"],
            audio_num_frames=clean[1].shape[1],
        )
        return inputs, clean, noisy, sigma

    def _ar_batch_inputs(self, records):
        from torch.nn.utils.rnn import pad_sequence

        inputs = [record.ar_inputs(self._device()) for record in records]
        temperatures = {item["temperature"] for item in inputs}
        if len(temperatures) != 1:
            raise ValueError("Joint text micro-batch requires a shared sampling temperature")
        # Right padding preserves each replay's original causal positions.
        return {
            **{
                key: pad_sequence([item[key][0] for item in inputs], batch_first=True, padding_value=0)
                for key in ("input_ids", "attention_mask", "response_mask")
            },
            "temperature": inputs[0]["temperature"],
        }

    def _ar_batch(self, records):
        return self.module(task="recaption", **self._ar_batch_inputs(records))

    def _validate_text_diagnostic_batches(self, records, micro_size):
        """Require equal forward/check counts before entering diagnostic collectives."""
        if self.text_diagnostics.enabled and torch.distributed.is_initialized():
            counts = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(counts, (len(records), micro_size))
            if any(count != counts[0] for count in counts):
                raise RuntimeError(f"Text diagnostics require equal actor batch/micro-batch counts: {counts}")

    def _check_text_diagnostic_batch(self, records, old_logprobs, inputs, *, capture_only=False):
        """Capture T at its original call site, before this micro-batch's backward.

        The caller still has old installed. Previous micro-batches may already
        have gradients, but the engine cannot step until this hook returns.
        """
        from .text_diagnostics import collective_errors, stop_collectively

        diagnostic = self.text_diagnostics
        if not diagnostic.enabled:
            return
        error, evidence_error = None, False
        try:
            self.policies._reshard()
            diagnostic.policy(self.module.gemma, self.policies.version)
            for row, record in enumerate(records):
                row_inputs = {
                    key: value[row : row + 1] if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                }
                mask = inputs["response_mask"][row, 1:].cpu()
                mismatch, artifact = diagnostic.actor(self.module.gemma, record, old_logprobs[row][mask], row_inputs)
                if mismatch and not capture_only:
                    error = f"text mismatch policy={record.policy_version}; evidence={artifact}"
        except Exception as exc:
            evidence_error = True
            error = f"text evidence failure: {type(exc).__name__}: {exc}"
        errors = collective_errors(error)
        if errors:
            # All actor ranks save their local old shards. Never condition a
            # full-model FSDP gather on only the rank that detects a mismatch.
            dump_error = None
            try:
                diagnostic.snapshot(self.module.gemma, "; ".join(errors))
                if self._rollout_backend is not None:
                    self._rollout_backend.engine.collective_rpc(
                        "snapshot_joint_text_failure",
                        args=("; ".join(errors),),
                        timeout=self._rollout_backend.timeout,
                    )
            except Exception as exc:
                dump_error = f"failure snapshot incomplete: {type(exc).__name__}: {exc}"
            # Pairing/I/O errors always stop; warn-only applies solely to a
            # successfully recorded numerical mismatch.
            fatal = diagnostic.config.stop_on_mismatch or evidence_error
            stop_collectively(dump_error or (error if fatal else None))

    @torch.no_grad()
    def _capture_text_step(self, records, micro_size, config, step):
        """Capture this PPO batch under old before any backward or optimizer step."""
        from .text_diagnostics import stop_collectively

        diagnostic = self.text_diagnostics
        self._validate_text_diagnostic_batches(records, micro_size)
        train, rollout = [], []
        with self.policies.use("old"):
            for index in range(0, len(records), micro_size):
                group = records[index : index + micro_size]
                inputs = self._ar_batch_inputs(group)
                values = self.module(task="recaption", **inputs).detach().cpu()
                # Numeric differences are evidence, not a reason to skip the
                # remaining samples. Identity and I/O errors still stop all ranks.
                self._check_text_diagnostic_batch(group, values, inputs, capture_only=True)
                train.append(values[inputs["response_mask"][:, 1:].cpu()])
                rollout.extend(record.rollout_logprobs for record in group)
            train = torch.cat(train).unsqueeze(0)
            rollout = torch.cat(rollout).unsqueeze(0)
            metrics = old_policy_logprob_metrics(
                train,
                rollout,
                torch.ones_like(train, dtype=torch.bool),
                clip_ratio=config.diffusion_loss.ar_clip_ratio,
            )
            print(f"LTX text capture step={step} rank={diagnostic.source_rank}: {metrics}", flush=True)
            error = None
            try:
                self.policies._reshard()
                reason = f"scheduled-text-capture-step-{step}-policy-{self.policies.version}"
                actor_path = diagnostic.snapshot(self.module.gemma, reason, force=True)
                if self._rollout_backend is None:
                    raise RuntimeError("Scheduled text capture requires the live rollout backend")
                rollout_paths = self._rollout_backend.engine.collective_rpc(
                    "snapshot_joint_text_failure",
                    args=(reason, True),
                    timeout=self._rollout_backend.timeout,
                )
                diagnostic.save(
                    f"capture-step-{step}",
                    {
                        "step": step,
                        "policy_version": self.policies.version,
                        "source_rank": diagnostic.source_rank,
                        "metrics": metrics,
                        "actor_snapshot": actor_path,
                        "rollout_snapshots": rollout_paths,
                    },
                    {"T": train, "R": rollout},
                )
            except Exception as exc:
                error = f"scheduled capture failed: {type(exc).__name__}: {exc}"
            stop_collectively(error)
        # Raising here prevents the engine-owned optimizer from stepping. Every
        # actor rank reaches this point only after all snapshots have completed.
        stop_collectively(f"Text capture complete at step {step}; stopped before backward/optimizer update")

    def _av_batch(self, records, video_clean, audio_clean, video_noise, audio_noise, timesteps):
        if any(record.generation_params != records[0].generation_params for record in records):
            raise ValueError("Joint AV micro-batch requires uniform generation geometry")
        # Rollout encodes conditions with padding=max_length. Do not introduce
        # additional padding here: that would change the replay's positions.
        if len({record.condition_token_ids.numel() for record in records}) != 1:
            raise ValueError("Joint AV micro-batch requires equal padded condition lengths")
        items = [
            self._av_inputs(record, video_clean[i], audio_clean[i], video_noise[i], audio_noise[i], timesteps[i])
            for i, record in enumerate(records)
        ]
        inputs = {
            key: torch.cat([item[0][key] for item in items], dim=0) if torch.is_tensor(value) else value
            for key, value in items[0][0].items()
        }
        clean = tuple(torch.cat([item[1][column] for item in items]) for column in (0, 1))
        noisy = tuple(torch.cat([item[2][column] for item in items]) for column in (0, 1))
        sigma = torch.cat([item[3] for item in items])
        predictions = self.module(task="av", **inputs)
        return tuple(value.float() for value in predictions), clean, noisy, sigma

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        """Validate once, then accumulate old/ref → text → AV for each micro-batch.

        The engine owns zero_grad/optimizer.step. Never move those operations
        into a branch: Gemma and DiT share this accumulation window.
        """
        from verl_omni.utils.determinism import configure_worker_determinism

        configure_worker_determinism()
        if forward_only:
            raise NotImplementedError("Use vLLM-Omni for joint rollout")
        self.policies.assert_frozen()
        config = getattr(loss_function, "keywords", {}).get("config")
        if config is None or config.diffusion_loss.loss_mode != "ltx2_unigrpo_nft":
            raise ValueError("LTX joint hooks require the joint loss configuration")
        micro_size = config.ppo_micro_batch_size_per_gpu
        if type(micro_size) is not int or micro_size <= 0:
            raise ValueError("Joint micro-batch size must be a positive integer")
        records = tu.get(data, "ltx_joint_replay")
        self._validate_replay_records(records)
        capture_step = self.text_diagnostics.config.capture_step
        if capture_step and tu.get(data, "text_diagnostic_step") == capture_step:
            self._capture_text_step(records, micro_size, config, capture_step)
        times = self._validate_actor_tensors(data, records)
        self._validate_text_diagnostic_batches(records, micro_size)
        total, metrics = 0.0, {}
        replay_train, replay_rollout = [], []
        started = perf_counter()

        def collect(values, scale=1.0):
            for key, value in values.items():
                metrics[key] = metrics.get(key, 0.0) + float(value) * scale

        for index in range(0, len(records), micro_size):
            group = records[index : index + micro_size]
            selection = slice(index, index + len(group))
            sample_scale = len(group) / len(records)

            def av(step, group=group, selection=selection):
                return self._av_batch(
                    group,
                    data["video_latents_clean"][selection],
                    data["audio_latents_clean"][selection],
                    data["video_forward_noise"][selection, step],
                    data["audio_forward_noise"][selection, step],
                    times[selection, step],
                )

            self._synchronize()
            reference_start = perf_counter()
            old_logprobs, old_predictions, ref_logprobs, ref_predictions = self._reference_predictions(
                group, av, times.shape[1], config
            )
            self._synchronize()
            collect({"train/reference_time": perf_counter() - reference_start})
            text_loss, train_logprobs, text_gradients = self._backward_text(
                data, group, selection, config, old_logprobs, ref_logprobs, sample_scale, collect
            )
            total += text_loss
            replay_train.append(train_logprobs)
            replay_rollout.extend(item.rollout_logprobs for item in group)
            total += self._backward_av(
                data, selection, config, av, old_predictions, ref_predictions, sample_scale, text_gradients, collect
            )
            del text_gradients
        replay_train = torch.cat(replay_train).unsqueeze(0)
        replay_rollout = torch.cat(replay_rollout).unsqueeze(0)
        metrics.update(
            old_policy_logprob_metrics(
                replay_train,
                replay_rollout,
                torch.ones_like(replay_train, dtype=torch.bool),
                clip_ratio=config.diffusion_loss.ar_clip_ratio,
            )
        )
        metrics["train/forward_backward_time"] = perf_counter() - started
        metrics["joint/total_loss"] = total
        self.policies.assert_frozen()
        metrics["joint/frozen_snapshots_verified"] = 1.0
        return {"loss": [total], "metrics": {key: [value] for key, value in metrics.items()}, "model_output": {}}

    def _validate_replay_records(self, records):
        """Validate replay identity on every rank before actor forwards."""
        validation_error = None
        try:
            if records is None or len(records) == 0:
                raise ValueError("Missing exact LTX joint rollout records")
            for record in records:
                record.validate()
                if not self.policies.rollout_started or record.policy_version != self.policies.version:
                    raise ValueError("Rollout policy version differs from frozen old")
        except Exception as exc:
            if not self.text_diagnostics.enabled:
                raise
            validation_error = f"invalid text replay: {type(exc).__name__}: {exc}"
            try:
                self.text_diagnostics.save(
                    "invalid-replay",
                    {"error": validation_error, "policy_version": self.policies.version},
                    {"records": [asdict(record) for record in records] if records is not None else []},
                )
            except Exception as dump_exc:
                validation_error += f"; replay capture failed: {dump_exc}"
        if self.text_diagnostics.enabled:
            from .text_diagnostics import stop_collectively

            stop_collectively(validation_error)

    def _validate_actor_tensors(self, data, records):
        """Check shapes, finite values and replay geometry without changing tensors."""
        times = data["train_timesteps"]
        if times.ndim != 2 or times.shape[0] != len(records) or times.shape[1] == 0:
            raise ValueError("Joint train timesteps must be [batch, nonempty steps]")
        if not bool(torch.isfinite(times).all() and ((times > 0) & (times <= 1000)).all()):
            raise ValueError("LTX train timesteps must be finite model-scale times in (0, 1000]")
        for modality in ("video", "audio"):
            clean, noise = data[f"{modality}_latents_clean"], data[f"{modality}_forward_noise"]
            if (
                clean.ndim != 3
                or clean.shape[0] != len(records)
                or noise.shape != (len(records), times.shape[1], *clean.shape[1:])
            ):
                raise ValueError(f"Invalid joint {modality} clean latent/noise shapes")
            if not bool(torch.isfinite(clean).all() and torch.isfinite(noise).all()):
                raise ValueError(f"Nonfinite joint {modality} latents/noise")
        if data["reward_prob"].shape != (len(records), times.shape[1], 2):
            raise ValueError("Joint reward probabilities must preserve [batch, steps, video/audio]")
        probabilities = data["reward_prob"]
        if not bool(torch.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()):
            raise ValueError("Joint reward probabilities must be finite and in [0, 1]")
        if data["text_advantages"].shape != (len(records),) or not bool(torch.isfinite(data["text_advantages"]).all()):
            raise ValueError("Joint text advantages must be finite with one value per sample")
        for record in records:
            params = record.generation_params
            rows = ((params["num_frames"] - 1) // 8 + 1) * (params["height"] // 32) * (params["width"] // 32)
            if rows != data["video_latents_clean"].shape[1]:
                raise ValueError("Clean video latents do not match the recorded generation geometry")
        return times

    def _reference_predictions(self, group, av, step_count, config):
        """Evaluate old/ref first; restore current before any differentiable forward."""
        with self.policies.use("old"):
            if self.text_diagnostics.enabled:
                text_inputs = self._ar_batch_inputs(group)
                old_logprobs = self.module(task="recaption", **text_inputs).detach().cpu()
                self._check_text_diagnostic_batch(group, old_logprobs, text_inputs)
            else:
                old_logprobs = self._ar_batch(group).detach().cpu()
            old_predictions = [tuple(p.cpu() for p in av(step)[0]) for step in range(step_count)]
        with self.policies.use("ref"):
            ref_logprobs = self._ar_batch(group).detach().cpu() if config.diffusion_loss.text_kl_coef else None
            ref_predictions = [tuple(p.cpu() for p in av(step)[0]) for step in range(step_count)]
        return old_logprobs, old_predictions, ref_logprobs, ref_predictions

    def _backward_text(self, data, group, selection, config, old_logprobs, ref_logprobs, sample_scale, collect):
        """Accumulate text GRPO/KL gradients and retain the baseline for AV gradient metrics."""
        text_start = perf_counter()
        before_text = gradient_snapshot(self.module)
        current = self._ar_batch(group)
        old = old_logprobs.to(current.device)
        mask = self._ar_batch_inputs(group)["response_mask"][:, 1:]
        loss, values = LTXJointLoss.text_loss(
            current=current,
            old=old,
            advantages=data["text_advantages"][selection].to(current.device),
            mask=mask,
            config=config,
            reference=ref_logprobs.to(current.device) if ref_logprobs is not None else None,
        )
        (loss * sample_scale).backward()
        after_text = gradient_snapshot(self.module)
        collect(gradient_delta_metrics(before_text, after_text, "text"), sample_scale)
        del before_text
        weighted_loss = loss.detach().item() * sample_scale
        collect(values, sample_scale)
        replay_logprobs = old[mask].detach().cpu()
        self._synchronize()
        collect({"train/text_time": perf_counter() - text_start})
        return weighted_loss, replay_logprobs, after_text

    def _backward_av(
        self, data, selection, config, av, old_predictions, ref_predictions, sample_scale, after_text, collect
    ):
        """Accumulate OmniNFT gradients, averaging over the sampled diffusion timesteps."""
        total = 0.0
        step_count = len(old_predictions)
        av_start = perf_counter()
        for step in range(step_count):
            predictions, clean, noisy, sigma = av(step)
            output = {}
            for column, modality in enumerate(("video", "audio")):
                prediction = predictions[column]
                output.update(
                    {
                        f"{modality}_forward_prediction": prediction,
                        f"{modality}_old_prediction": old_predictions[step][column].to(prediction.device),
                        f"{modality}_ref_forward_prediction": ref_predictions[step][column].to(prediction.device),
                        f"{modality}_x0": clean[column],
                        f"{modality}_xt": noisy[column],
                        f"{modality}_t_expanded": sigma,
                    }
                )
            loss, values = LTXJointLoss.av_loss(
                model_output=output,
                probabilities=data["reward_prob"][selection, step].to(predictions[0].device),
                config=config,
            )
            scale = sample_scale / step_count
            (loss * scale).backward()
            total += loss.detach().item() * scale
            collect(values, scale)
        self._synchronize()
        after_av = gradient_snapshot(self.module)
        collect(gradient_delta_metrics(after_text, after_av, "av"), sample_scale)
        del after_text, after_av
        collect({"train/av_time": perf_counter() - av_start})
        return total


def gradient_snapshot(module):
    """Copy accumulated gradients, so a subsequent backward can be attributed."""
    result = {}
    for name, parameter in module.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            value = parameter.grad
            value = value.to_local() if isinstance(value, DTensor) else value
            result[name] = value.detach().float().cpu().clone()
    return result


def gradient_delta_metrics(before, after, branch):
    """Report local shard norms; do not label a rank average as a global norm."""
    squared = {"gemma": 0.0, "dit": 0.0}
    nonfinite = 0
    for name in before.keys() | after.keys():
        if not name.startswith(("gemma.", "transformer.")):
            raise ValueError(f"Unexpected joint gradient parameter: {name}")
        value = after.get(name)
        previous = before.get(name)
        delta = -previous if value is None else value if previous is None else value - previous
        nonfinite += int((~torch.isfinite(delta)).sum())
        component = "gemma" if name.startswith("gemma.") else "dit"
        squared[component] += delta.double().square().sum().item()
    return {
        **{f"train/{branch}_{component}_grad_norm_local": value**0.5 for component, value in squared.items()},
        f"train/{branch}_nonfinite_gradient_elements_local": nonfinite,
    }
