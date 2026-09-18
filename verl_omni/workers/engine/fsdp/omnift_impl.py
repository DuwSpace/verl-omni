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
"""FSDP engine for joint audio-video OmniNFT updates."""

from typing import Callable, Optional

import torch
from tensordict import TensorDict
from torch.utils._pytree import tree_map
from torch.utils.checkpoint import checkpoint
from verl.trainer.config import CheckpointConfig
from verl.utils.device import get_device_name
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig
from verl.workers.engine.base import EngineRegistry

from verl_omni.pipelines.ltx2_omni_nft.compat import apply_ltx_npu_rms_norm_workaround
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.utils import prepare_model_inputs
from verl_omni.workers.config import DiffusionModelConfig

from .diffusers_impl import NFTDiffusersFSDPEngine

device_name = get_device_name()


def _fsdp2_gradient_checkpointing_with_cast_func(param_dtype: Optional[torch.dtype]) -> Callable:
    """Build a non-reentrant checkpoint wrapper using effective FSDP param_dtype.

    On both initial forward and recomputation, cast floating Tensor leaves of
    positional/keyword PyTrees to param_dtype. None disables casting; integer,
    bool, non-Tensor, and already matching leaves pass through. Device and tree
    structure are preserved, and casts remain part of the autograd graph.
    """

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
            cast_args = tree_map(cast_fp_tensor, inner_args)
            cast_kwargs = tree_map(cast_fp_tensor, inner_kwargs)
            return module.__call__(*cast_args, **cast_kwargs)

        return checkpoint(checkpointed_forward, *args, use_reentrant=False, **kwargs)

    return gradient_checkpointing_func


def _validate_omni_nft_fsdp2_config(engine_config: FSDPEngineConfig) -> None:
    """Require FSDP2 with sequence-parallel size one for this engine."""
    if engine_config.strategy != "fsdp2":
        raise NotImplementedError(
            f"OmniNFT currently supports only actor.strategy=fsdp2, got {engine_config.strategy!r}."
        )
    if engine_config.ulysses_sequence_parallel_size != 1:
        raise NotImplementedError(
            "OmniNFT FSDP2 does not implement Ulysses/context parallelism yet; "
            "set actor.fsdp_config.ulysses_sequence_parallel_size=1."
        )


@EngineRegistry.register(model_type="omni_nft_model", backend=["fsdp2"], device=[device_name])
class OmniNFTDiffusersFSDPEngine(NFTDiffusersFSDPEngine):
    """FSDP2 actor engine for joint LTX video/audio DiffusionNFT updates."""

    def __init__(
        self,
        model_config: DiffusionModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        _validate_omni_nft_fsdp2_config(engine_config)
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _build_module(self):
        """Load LTX and apply its instance-local NPU compatibility fixes."""
        module = super()._build_module()
        if self.model_config.enable_gradient_checkpointing:
            self._enable_omni_gradient_checkpointing(module)
        return apply_ltx_npu_rms_norm_workaround(module)

    def _enable_omni_gradient_checkpointing(self, module: torch.nn.Module) -> None:
        """Install checkpoint input casting consistent with the effective FSDP policy.

        Recompute can bypass FSDP's input-casting boundary while LTX receives
        FP32 forward-process latents. Use mixed-precision param_dtype, not the
        loader dtype, for both checkpoint executions. When the adapter requests
        preservation and the module declares FP32 islands, use None just as
        _build_fsdp_module does, retaining the loader's mixed layout.
        """
        param_dtype, _, _ = self._get_mixed_precision_dtypes()
        keep_in_fp32 = getattr(module, "_keep_in_fp32_modules", None)
        if keep_in_fp32 and DiffusionModelBase.get_class(self.model_config).preserve_fp32_modules():
            param_dtype = None
        module.enable_gradient_checkpointing(
            gradient_checkpointing_func=_fsdp2_gradient_checkpointing_with_cast_func(param_dtype)
        )

    @staticmethod
    def _select_forward_noise(micro_batch: TensorDict, key: str, x0: torch.Tensor, step: int) -> torch.Tensor:
        """Select stored per-step/shared noise, or sample fresh FP32 noise like x0."""
        noise = micro_batch.get(key, None)
        if noise is None:
            return torch.randn_like(x0.float())
        return noise[:, step] if noise.ndim == x0.ndim + 1 else noise

    def prepare_model_inputs(self, micro_batch: TensorDict, step: int):
        """Build video/audio inputs at one shared model timestep without mutating the batch.

        Args:
            micro_batch: Clean video/audio latents ``[B, Sv/Sa, D]``, prompt
                fields, train_timesteps ``[B, T]``, video_seq_len, and optional
                video/audio_forward_noise. Noise may match its latent or add a
                timestep dimension after B; missing noise is sampled in FP32.
                Connector-specific audio prompts remain in the batch for the adapter.
            step: Selected timestep/noise-column index.

        Returns:
            ``(model_inputs, negative_model_inputs, (video_x0, audio_x0),
            (video_xt, audio_xt), (video_t, audio_t))``. Both modalities use
            ``xt = (1 - t) * x0 + t * noise``, with ``t = timestep / 1000``
            broadcast over latent axes. Latents are concatenated video-first
            along sequence for the adapter, which receives the unnormalized
            timestep. Nested positive/negative video prompts are unpadded locally.
        """
        video_x0 = micro_batch["video_latents_clean"]
        audio_x0 = micro_batch["audio_latents_clean"]
        timestep = micro_batch["train_timesteps"][:, step]
        t = timestep.float() / 1000.0
        video_t = t.view(-1, *([1] * (video_x0.ndim - 1)))
        audio_t = t.view(-1, *([1] * (audio_x0.ndim - 1)))

        video_noise = self._select_forward_noise(micro_batch, "video_forward_noise", video_x0, step)
        audio_noise = self._select_forward_noise(micro_batch, "audio_forward_noise", audio_x0, step)
        video_xt = (1.0 - video_t) * video_x0 + video_t * video_noise
        audio_xt = (1.0 - audio_t) * audio_x0 + audio_t * audio_noise

        prompt_embeds = micro_batch["prompt_embeds"]
        prompt_embeds_mask = micro_batch["prompt_embeds_mask"]
        negative_prompt_embeds = micro_batch.get("negative_prompt_embeds", None)
        negative_prompt_embeds_mask = micro_batch.get("negative_prompt_embeds_mask", None)
        if prompt_embeds.is_nested:
            prompt_embeds, prompt_embeds_mask = self._unpad_nested_embeds(prompt_embeds, prompt_embeds_mask)
        if isinstance(negative_prompt_embeds, torch.Tensor) and negative_prompt_embeds.is_nested:
            negative_prompt_embeds, negative_prompt_embeds_mask = self._unpad_nested_embeds(
                negative_prompt_embeds, negative_prompt_embeds_mask
            )

        packed_xt = torch.cat((video_xt, audio_xt), dim=1)
        model_inputs, negative_model_inputs = prepare_model_inputs(
            module=self.module,
            model_config=self.model_config,
            latents=packed_xt,
            timesteps=timestep,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            micro_batch=micro_batch,
            step=step,
        )
        return (
            model_inputs,
            negative_model_inputs,
            (video_x0, audio_x0),
            (video_xt, audio_xt),
            (video_t, audio_t),
        )

    @staticmethod
    def prepare_model_outputs(output, micro_batch: TensorDict) -> dict[str, torch.Tensor]:
        """Unpack paired NFT outputs into per-modality loss fields, reusing tensors.

        ``output`` is ``(old, current, reference, x0, xt, t_expanded)``; every
        entry must be a ``(video, audio)`` tuple or TypeError is raised. Map these
        to video_/audio_ old_prediction, forward_prediction,
        ref_forward_prediction, x0, xt, and t_expanded keys without detaching or
        casting. ``micro_batch`` is unused.
        """
        del micro_batch
        old_prediction, current_prediction, ref_prediction, x0, xt, t_expanded = output
        predictions = (old_prediction, current_prediction, ref_prediction)
        if not all(isinstance(value, tuple) and len(value) == 2 for value in predictions):
            raise TypeError("LTX-2.3 OmniNFT expects (video, audio) predictions from every policy forward.")
        contexts = (x0, xt, t_expanded)
        if not all(isinstance(value, tuple) and len(value) == 2 for value in contexts):
            raise TypeError("LTX-2.3 OmniNFT expects paired video/audio forward-process context.")
        video_old, audio_old = old_prediction
        video_current, audio_current = current_prediction
        video_ref, audio_ref = ref_prediction
        video_x0, audio_x0 = x0
        video_xt, audio_xt = xt
        video_t, audio_t = t_expanded
        return {
            "video_old_prediction": video_old,
            "audio_old_prediction": audio_old,
            "video_forward_prediction": video_current,
            "audio_forward_prediction": audio_current,
            "video_ref_forward_prediction": video_ref,
            "audio_ref_forward_prediction": audio_ref,
            "video_x0": video_x0,
            "audio_x0": audio_x0,
            "video_xt": video_xt,
            "audio_xt": audio_xt,
            "video_t_expanded": video_t,
            "audio_t_expanded": audio_t,
        }
