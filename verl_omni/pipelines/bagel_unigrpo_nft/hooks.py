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

"""Low-memory two-backward hybrid update; the shared engine owns the optimizer."""

import torch
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available

from verl_omni.pipelines.bagel_flow_grpo.bagel_model import BagelForTraining
from verl_omni.pipelines.bagel_unigrpo.bagel_ar_thinking import replay_thinking_logprobs
from verl_omni.pipelines.model_base import DiffusionEngineHooks

from .loss import BagelUniGRPONFTLoss
from .policy_state import BagelPolicyState


class BagelUniGRPONFTHooks(DiffusionEngineHooks):
    """AR-GRPO plus image NFT over shared full-weight MoT; no native rollout fallback."""

    requires_checkpoint_state = True

    def __init__(self, module, model_config, optimizer_config):
        self.module = module
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.policies = BagelPolicyState(module)
        self._rollout_backend = None

    def generate(self, data):
        """Freeze/export this policy and delegate generation to a real MP worker."""
        from verl_omni.workers.rollout.bagel_joint_backend import BagelJointVllmBackend, original_prompt_text

        prompts = tu.get(data, "raw_prompt")
        versions = tu.get(data, "_bagel_policy_version")
        seeds = tu.get(data, "_bagel_sample_seed")
        if prompts is None or versions is None or seeds is None:
            raise ValueError("Joint generation requires original prompts, policy versions and per-sample seeds")
        if len(set(int(v) for v in versions)) != 1:
            raise ValueError("Joint generation batch must contain one policy version")
        version = int(versions[0])
        texts = [original_prompt_text(prompt) for prompt in prompts]
        self.begin_rollout(version=version)
        if self._rollout_backend is None:
            self._rollout_backend = BagelJointVllmBackend(self.model_config)
        self._rollout_backend.load_actor_policy(self.module, version=version)
        images, records = self._rollout_backend.generate(texts, seeds, version=version)
        return tu.get_tensordict({"responses": images, "bagel_joint_replay": records})

    def close_rollout(self):
        if self._rollout_backend is not None:
            self._rollout_backend.close()
            self._rollout_backend = None

    def begin_rollout(self, *, version: int):
        """Freeze old BEFORE exporting these same weights to the vllm-omni backend."""
        self.policies.begin_rollout(version=version)

    def state_dict(self):
        """Persist initial reference and old rollout policy, not just trainable weights."""
        return self.policies.state_dict()

    def load_state_dict(self, state):
        """Restore frozen policy state from the same-rank training checkpoint."""
        self.policies.load_state_dict(state)

    def _device(self):
        if is_cuda_available or is_npu_available:
            return torch.device(get_device_name(), get_device_id())
        return next(self.module.parameters()).device

    def _ar(self, record):
        return replay_thinking_logprobs(
            self.module,
            record.ar_prompt_token_ids.tolist(),
            record.response_token_ids.tolist(),
            temperature=record.temperature,
        )

    def _velocity(self, record, noise, sigma):
        device = self._device()
        clean = record.latents_clean.to(device).unsqueeze(0)
        time = sigma.to(device=device, dtype=torch.float32).reshape(1)
        if not bool(((time > 0) & (time < 1)).all()):
            raise ValueError("BAGEL NFT requires normalized continuous sigma in (0, 1)")
        noisy = (1 - time[:, None, None]) * clean + time[:, None, None] * noise.to(device).unsqueeze(0)
        condition = record.condition_token_ids.to(device).unsqueeze(0)
        velocity = BagelForTraining.forward(
            self.module,
            hidden_states=noisy.to(torch.bfloat16),
            timestep=time.to(torch.bfloat16),
            text_token_ids=condition,
            latent_pos_ids=record.latent_pos_ids.to(device).unsqueeze(0),
            text_attention_mask=torch.ones_like(condition, dtype=torch.bool),
        )[0].float()
        return velocity, clean, noisy, time[:, None, None]

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        """Accumulate weighted AR and NFT gradients; never clear, step, or schedule them."""
        if forward_only:
            raise NotImplementedError("Use the vllm-omni rollout backend for joint-policy inference")
        if loss_function is None or not hasattr(loss_function, "keywords"):
            raise ValueError("Joint update requires the configured hybrid loss")
        config = loss_function.keywords["config"]
        records = tu.get(data, "bagel_joint_replay")
        if records is None or len(records) == 0:
            raise ValueError("Joint update requires exact native rollout records")
        for record in records:
            record.validate()
            if record.policy_version != self.policies.version:
                raise ValueError("Rollout version differs from the frozen old policy")
            boundaries = [self.module.config.start_of_image_id, self.module.config.end_of_image_id]
            if record.image_boundary_token_ids.tolist() != boundaries:
                raise ValueError("Actor image-boundary IDs differ from the native rollout")
        times, noise = data["train_timesteps"], data["nft_noise"]
        if times.ndim != 2 or noise.shape[:2] != times.shape or times.shape[0] != len(records):
            raise ValueError("Joint NFT noise/times must match the actor batch")
        old_ar, old_velocity, ref_velocity = [], [], []
        with torch.no_grad(), self.policies.use("old"):
            for index, record in enumerate(records):
                old_ar.append(self._ar(record).detach().cpu())
                old_velocity.append(
                    [
                        self._velocity(record, noise[index, step], times[index, step])[0].detach().cpu()
                        for step in range(times.shape[1])
                    ]
                )
        with torch.no_grad(), self.policies.use("reference"):
            for index, record in enumerate(records):
                ref_velocity.append(
                    [
                        self._velocity(record, noise[index, step], times[index, step])[0].detach().cpu()
                        for step in range(times.shape[1])
                    ]
                )
        metrics, total = {}, 0.0

        def collect(values, scale):
            for key, value in values.items():
                metrics[key] = metrics.get(key, 0.0) + float(value) * scale

        for index, record in enumerate(records):
            current_logp = self._ar(record)
            old_logp = old_ar[index].to(current_logp.device)
            advantage = torch.full_like(current_logp, float(data["advantages"][index]))
            ar_loss, ar_metrics = BagelUniGRPONFTLoss.compute_ar_loss(
                new_log_probs=current_logp,
                old_log_probs=old_logp,
                advantages=advantage,
                config=config,
            )
            (ar_loss / len(records)).backward()
            total += float(ar_loss.detach()) / len(records)
            collect(ar_metrics, 1 / len(records))
            collect(
                {"ar/rollout_old_logp_mae": (old_logp.detach().cpu() - record.rollout_log_probs).abs().mean()},
                1 / len(records),
            )
            for step in range(times.shape[1]):
                velocity, clean, noisy, time = self._velocity(record, noise[index, step], times[index, step])
                image_output = {
                    "forward_prediction": velocity,
                    "old_prediction": old_velocity[index][step].to(velocity.device),
                    "ref_forward_prediction": ref_velocity[index][step].to(velocity.device),
                    "x0": clean,
                    "xt": noisy,
                    "t_expanded": time,
                }
                image_loss, image_metrics = BagelUniGRPONFTLoss.compute_image_loss(
                    model_output=image_output,
                    reward_prob=data["reward_prob"][index].reshape(1),
                    config=config,
                )
                scale = 1 / (len(records) * times.shape[1])
                (image_loss * scale).backward()
                total += float(image_loss.detach()) * scale
                collect(image_metrics, scale)
        metrics["joint/total_loss"] = total
        return {"loss": [total], "metrics": {key: [value] for key, value in metrics.items()}, "model_output": {}}
