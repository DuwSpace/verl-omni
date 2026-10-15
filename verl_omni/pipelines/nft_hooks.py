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
"""Algorithm hooks for the shared NFT forward-process engine."""

from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict
from torch.utils._pytree import tree_map
from torch.utils.checkpoint import checkpoint
from verl.utils import tensordict_utils as tu


def _flow_time(micro_batch, step, scheduler):
    timestep = micro_batch["train_timesteps"][:, step]
    scale = getattr(scheduler.config, "num_train_timesteps", None)
    if scale is None:
        raise ValueError(
            "Converting `train_timesteps` to flow time requires "
            f"scheduler.config.num_train_timesteps, but {type(scheduler).__name__} does not define it."
        )
    tu.assign_non_tensor(micro_batch, num_train_timesteps=int(scale))
    return timestep, timestep.float() / float(scale)


def _noise_latents(micro_batch, clean_key, noise_key, step, time):
    x0 = micro_batch[clean_key]
    t = time.view(-1, *([1] * (x0.ndim - 1)))
    noise = micro_batch.get(noise_key, None)
    if noise is None:
        noise = torch.randn_like(x0.float())
    elif noise.ndim == x0.ndim + 1:
        noise = noise[:, step]
    return {"x0": x0, "xt": (1.0 - t) * x0 + t * noise, "t_expanded": t}


class NFTTrainingHooks:
    """Default single-latent NFT preparation, also used by existing DGPO adapters."""

    @staticmethod
    def prepare_training_inputs(
        micro_batch: TensorDict, step: int, scheduler
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Prepare the default NFT noising context using the scheduler's timestep scale."""
        timestep, time = _flow_time(micro_batch, step, scheduler)
        context = _noise_latents(micro_batch, "latents_clean", "forward_noise", step, time)
        return context["xt"], timestep, context

    @staticmethod
    def prepare_model_outputs(predictions: dict, loss_context: dict) -> dict:
        """Combine policy predictions with algorithm-owned loss context."""
        return {**loss_context, **predictions}

    @staticmethod
    def prepare_loss_inputs(micro_batch: TensorDict, step: int) -> dict:
        """Slice per-timestep fields and retain per-sample group metadata."""
        inputs = {
            key: micro_batch[key][:, step]
            for key in ("reward_prob", "advantages")
            if micro_batch.get(key, None) is not None
        }
        inputs.update(
            {key: micro_batch[key] for key in ("group_index", "group_size") if micro_batch.get(key, None) is not None}
        )
        return inputs


class OmniNFTTrainingHooks(NFTTrainingHooks):
    """Compose video/audio noising and loss fields without duplicating engine execution."""

    @staticmethod
    def validate_engine_config(engine_config) -> None:
        """Reject unsupported sharding and sequence parallelism before initialization."""
        if engine_config.strategy != "fsdp2":
            raise NotImplementedError(
                f"OmniNFT currently supports only actor.strategy=fsdp2, got {engine_config.strategy!r}."
            )
        if engine_config.ulysses_sequence_parallel_size != 1:
            raise NotImplementedError(
                "OmniNFT FSDP2 does not implement Ulysses/context parallelism yet; "
                "set actor.fsdp_config.ulysses_sequence_parallel_size=1."
            )

    @staticmethod
    def configure_gradient_checkpointing(module, engine_config, *, preserve_fp32_modules: bool) -> None:
        """Configure the FSDP2 checkpoint wrapper only for OmniNFT training."""
        from verl.utils.torch_dtypes import PrecisionType

        precision = engine_config.mixed_precision
        param_dtype = PrecisionType.to_dtype(precision.get("param_dtype", "bf16") if precision is not None else "bf16")
        if getattr(module, "_keep_in_fp32_modules", None) and preserve_fp32_modules:
            param_dtype = None
        module.enable_gradient_checkpointing(gradient_checkpointing_func=_gradient_checkpointing_with_cast(param_dtype))

    @staticmethod
    def prepare_training_inputs(
        micro_batch: TensorDict, step: int, scheduler
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Noise modalities independently and retain their context separately."""
        timestep, time = _flow_time(micro_batch, step, scheduler)
        context = {
            modality: _noise_latents(micro_batch, f"{modality}_latents_clean", f"{modality}_forward_noise", step, time)
            for modality in ("video", "audio")
        }
        latents = torch.cat((context["video"]["xt"], context["audio"]["xt"]), dim=1)
        return latents, timestep, context

    @staticmethod
    def prepare_model_outputs(predictions: dict, loss_context: dict) -> dict:
        """Map joint model predictions and modality context to OmniNFT loss fields."""
        outputs = {
            f"{modality}_{name}": value for modality, context in loss_context.items() for name, value in context.items()
        }
        for name, values in predictions.items():
            for modality, value in zip(("video", "audio"), values, strict=True):
                outputs[f"{modality}_{name}"] = value
        return outputs

    @staticmethod
    def prepare_loss_inputs(micro_batch: TensorDict, step: int) -> dict:
        """Select reward probabilities and retain per-modality reward metrics."""
        from verl_omni.trainer.diffusion.diffusion_algos import OmniNFTLoss

        inputs = {"reward_prob": micro_batch["reward_prob"][:, step]}
        inputs.update(
            {
                key: micro_batch[key]
                for key in micro_batch.keys()
                if isinstance(key, str) and key.startswith(OmniNFTLoss._REWARD_METRIC_PREFIX)
            }
        )
        return inputs


def get_nft_training_hooks(algorithm: str) -> type[NFTTrainingHooks]:
    """Select built-in specialization; other adapters retain the existing NFT contract."""
    return OmniNFTTrainingHooks if algorithm == "omni_nft" else NFTTrainingHooks


def _gradient_checkpointing_with_cast(param_dtype: Optional[torch.dtype]) -> Callable:
    """Build a non-reentrant checkpoint wrapper using the FSDP parameter dtype."""

    def cast_fp_tensor(value):
        if (
            param_dtype is None
            or not isinstance(value, torch.Tensor)
            or not torch.is_floating_point(value)
            or value.dtype == param_dtype
        ):
            return value
        return value.to(param_dtype)

    def gradient_checkpointing_func(module, *args, **kwargs):
        def checkpointed_forward(*inner_args, **inner_kwargs):
            return module.__call__(*tree_map(cast_fp_tensor, inner_args), **tree_map(cast_fp_tensor, inner_kwargs))

        return checkpoint(checkpointed_forward, *args, use_reentrant=False, **kwargs)

    return gradient_checkpointing_func
