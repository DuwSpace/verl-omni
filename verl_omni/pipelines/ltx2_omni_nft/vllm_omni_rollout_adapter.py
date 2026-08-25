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

"""vLLM-Omni rollout adapter for LTX-2.3 OmniNFT."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.ltx2.ltx2_conditioning import LTXPromptContext
from vllm_omni.diffusion.models.ltx2.ltx2_denoise import (
    LTXDenoiseContext,
    LTXForwardContext,
    LTXPhaseResult,
)
from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState
from vllm_omni.diffusion.models.ltx2.ltx2_recipes import LTXPhaseRecipe
from vllm_omni.diffusion.models.ltx2.ltx2_request import LTXRequestInputs
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.diffusion_rollout_output import with_rollout_data
from verl_omni.pipelines.ltx2_omni_nft.prompt_mixin import LTXTokenIdPromptMixin, normalize_ltx_output_type
from verl_omni.pipelines.model_base import VllmOmniPipelineBase

__all__ = ["LTX23OmniNFTPipeline"]


def _resolve_omni_nft_output_type(sampling_params: Any) -> str:
    output_type = getattr(sampling_params, "output_type", None)
    if output_type is None:
        extra_args = getattr(sampling_params, "extra_args", None) or {}
        if isinstance(extra_args, dict):
            output_type = extra_args.get("output_type")
    output_type = normalize_ltx_output_type(output_type) or "pt"
    if output_type != "pt":
        raise ValueError(
            "LTX-2.3 OmniNFT requires decoded 'pt' tensor output, got "
            f"{output_type!r}; 'latent' or other formats break the training contract."
        )
    return output_type


@VllmOmniPipelineBase.register("LTX2Pipeline", algorithm="omni_nft")
class LTX23OmniNFTPipeline(LTXTokenIdPromptMixin, LTX2Pipeline):
    """Run the native LTX sampler and retain final OmniNFT training tensors."""

    supports_request_batch = False

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self.set_progress_bar_config(disable=True)
        self._omni_nft_prompt_context: LTXPromptContext | None = None
        self._omni_nft_clean_state: LTXAVState | None = None
        self._omni_nft_forward_context: LTXForwardContext | None = None

    def _prepare_prompt_context(self, **kwargs: Any) -> LTXPromptContext:
        prompt_context = super()._prepare_prompt_context(**kwargs)
        self._omni_nft_prompt_context = prompt_context
        return prompt_context

    def run_phase(
        self,
        req: DiffusionRequestBatch,
        request_inputs: LTXRequestInputs,
        *,
        noise_scale: float,
        sigmas: list[float] | None,
        timesteps: list[int] | None,
        attention_kwargs: dict[str, Any] | None,
        phase_recipe: LTXPhaseRecipe,
        image: Any | None = None,
        prompt_context: LTXPromptContext | None = None,
    ) -> LTXPhaseResult:
        """Delegate phase execution to LTX's native model and sampler."""
        return super().run_phase(
            req,
            request_inputs,
            noise_scale=noise_scale,
            sigmas=sigmas,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
            phase_recipe=phase_recipe,
            image=image,
            prompt_context=prompt_context,
        )

    def _denoise_step(
        self,
        index: int,
        timestep: torch.Tensor,
        state: LTXAVState,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> LTXAVState:
        clean_state = super()._denoise_step(
            index,
            timestep,
            state,
            forward_ctx,
            denoise_ctx,
        )
        self._omni_nft_clean_state = clean_state
        self._omni_nft_forward_context = forward_ctx
        return clean_state

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch, **kwargs: Any) -> DiffusionOutput | list[DiffusionOutput]:
        """Generate decoded audio-video and attach final clean latent metadata."""

        if req.num_reqs != 1:
            raise ValueError(f"LTX-2.3 OmniNFT expects one request, got {req.num_reqs}.")

        self._omni_nft_prompt_context = None
        self._omni_nft_clean_state = None
        self._omni_nft_forward_context = None
        request = req.requests[0]
        request.sampling_params.output_type = _resolve_omni_nft_output_type(request.sampling_params)
        self._inject_precomputed_prompt_embeds(request)
        output = super().forward(req, **kwargs)
        if isinstance(output, list):
            if len(output) != 1:
                raise RuntimeError(f"Single-request LTX rollout returned {len(output)} outputs.")
            output = output[0]

        clean_state = self._omni_nft_clean_state
        forward_context = self._omni_nft_forward_context
        prompt_context = self._omni_nft_prompt_context
        if clean_state is None or forward_context is None or prompt_context is None:
            raise RuntimeError("LTX-2.3 OmniNFT rollout did not capture final latent and prompt state.")
        if any(
            value is not None
            for value in (
                output.trajectory_latents,
                output.trajectory_timesteps,
                output.trajectory_log_probs,
            )
        ):
            raise RuntimeError("Native LTX rollout unexpectedly returned trajectory data for OmniNFT.")
        if not isinstance(output.output, (tuple, list)) or len(output.output) != 2:
            raise RuntimeError("Native LTX rollout did not return decoded (video, audio) output.")
        decoded_video, decoded_audio = output.output
        if isinstance(decoded_video, torch.Tensor) and decoded_video.ndim == 5:
            if decoded_video.shape[0] != 1:
                raise RuntimeError("Single-request LTX rollout returned a decoded video batch larger than one.")
            decoded_video = decoded_video[0]
        output = replace(output, output=(decoded_video, decoded_audio))

        batch_size = clean_state.video.shape[0]
        device = clean_state.video.device
        train_timesteps = forward_context.timesteps.to(device=device, dtype=torch.float32)
        if train_timesteps.ndim == 1:
            train_timesteps = train_timesteps.unsqueeze(0).expand(batch_size, -1)

        vocoder = getattr(self, "vocoder", None)
        audio_sample_rate = getattr(getattr(vocoder, "config", None), "output_sampling_rate", None)
        if audio_sample_rate is None:
            raise RuntimeError("LTX-2.3 OmniNFT rollout requires vocoder.config.output_sampling_rate.")
        audio_sample_rate = int(audio_sample_rate)
        if audio_sample_rate <= 0:
            raise RuntimeError(f"Invalid vocoder output_sampling_rate: {audio_sample_rate}")

        fps = getattr(getattr(forward_context, "request_inputs", None), "frame_rate", None)
        if fps is None:
            raise RuntimeError("LTX-2.3 OmniNFT rollout requires request_inputs.frame_rate.")
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"Invalid LTX frame_rate: {fps}")

        output = with_rollout_data(
            output,
            media_key="video",
            prompt_embeddings={
                "prompt_embeds": prompt_context.positive_connector_prompt_embeds,
                "audio_prompt_embeds": prompt_context.positive_connector_audio_prompt_embeds,
                "prompt_embeds_mask": prompt_context.positive_connector_attention_mask,
                "negative_prompt_embeds": prompt_context.negative_connector_prompt_embeds,
                "negative_audio_prompt_embeds": prompt_context.negative_connector_audio_prompt_embeds,
                "negative_prompt_embeds_mask": prompt_context.negative_connector_attention_mask,
            },
            rl={
                "audio": decoded_audio,
                "video_latents_clean": clean_state.video.detach().float(),
                "audio_latents_clean": clean_state.audio.detach().float(),
                "train_timesteps": train_timesteps,
                "video_latent_shape": torch.tensor(
                    [clean_state.video.shape[1:]], device=device, dtype=torch.long
                ).expand(batch_size, -1),
                "audio_latent_shape": torch.tensor(
                    [clean_state.audio.shape[1:]], device=device, dtype=torch.long
                ).expand(batch_size, -1),
                "video_seq_len": torch.full(
                    (batch_size,), clean_state.video.shape[1], device=device, dtype=torch.long
                ),
                "audio_seq_len": torch.full(
                    (batch_size,), clean_state.audio.shape[1], device=device, dtype=torch.long
                ),
                "fps": torch.full((batch_size,), fps, device=device, dtype=torch.float32),
                "audio_sample_rate": torch.full(
                    (batch_size,), audio_sample_rate, device=device, dtype=torch.long
                ),
            },
            to_cpu=True,
        )
        return output
