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
import os
from copy import copy
from dataclasses import replace
from typing import Any

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.ltx2.ltx2_denoise import LTXForwardContext
from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.diffusion_rollout_output import with_rollout_data
from verl_omni.pipelines.ltx2_flow_grpo.common import normalize_ltx_output_type
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec

from .prompt_utils import LTXTokenIdPromptMixin

__all__ = ["LTX23OmniNFTPipeline"]


def _resolve_omni_nft_output_type(sampling_params: Any) -> str:
    """Resolve the explicit output type before extra_args and require decoded tensors."""
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


def _slice_batch_row(value: torch.Tensor, index: int, expected_batch: int, name: str) -> torch.Tensor:
    """Validate the packed batch size and return a row view retaining its batch axis."""
    shape = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
    if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != expected_batch:
        raise RuntimeError(f"LTX-2.3 OmniNFT packed {name} must have leading size {expected_batch}, got {shape}.")
    return value[index : index + 1]


def _rollout_progress_bar_enabled() -> bool:
    return os.environ.get("OMNIFT_ROLLOUT_PROGRESS", "").strip().lower() in {"1", "true", "yes"}


def _require_one_stage_recipe(pipeline_recipe: Any) -> None:
    """Require a single denoising phase that supplies both decoded modalities."""
    phases = tuple(getattr(pipeline_recipe, "phases", ()) or ())
    video_output_phase = getattr(pipeline_recipe, "video_output_phase", None)
    audio_output_phase = getattr(pipeline_recipe, "audio_output_phase", None)
    # Both indices select the sole phase; -1 is the native final-phase convention.
    valid_output_phases = {-1, 0}
    if (
        len(phases) != 1
        or video_output_phase not in valid_output_phases
        or audio_output_phase not in valid_output_phases
    ):
        raise NotImplementedError(
            "LTX-2.3 OmniNFT requires one native denoising phase with video and audio emitted from that phase."
        )


def _require_audio_sample_rate(pipeline: Any) -> int:
    vocoder = getattr(pipeline, "vocoder", None)
    audio_sample_rate = getattr(getattr(vocoder, "config", None), "output_sampling_rate", None)
    if audio_sample_rate is None:
        raise RuntimeError("LTX-2.3 OmniNFT rollout requires vocoder.config.output_sampling_rate.")
    audio_sample_rate = int(audio_sample_rate)
    if audio_sample_rate <= 0:
        raise RuntimeError(f"Invalid vocoder output_sampling_rate: {audio_sample_rate}")
    return audio_sample_rate


def _require_frame_rate(forward_context: Any) -> float:
    fps = getattr(getattr(forward_context, "request_inputs", None), "frame_rate", None)
    if fps is None:
        raise RuntimeError("LTX-2.3 OmniNFT rollout requires request_inputs.frame_rate.")
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Invalid LTX frame_rate: {fps}")
    return fps


def _attach_omni_nft_rollout_data(
    output: DiffusionOutput,
    *,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    prompt_embeddings: dict[str, torch.Tensor],
    train_timesteps: torch.Tensor,
    fps: float,
    audio_sample_rate: int,
) -> DiffusionOutput:
    """Attach one request's clean AV latents and replay metadata to decoded output.

    Args:
        output: Native decoded ``(video, audio)`` output without trajectory fields.
        video_latents: Normalized clean tokens, ``[1, S_video, D_video]``.
        audio_latents: Clean ``[1, S_audio, D_audio]`` tokens after padding removal.
        prompt_embeddings: Positive/negative video and audio connector embeddings
            and masks, with a leading singleton batch axis where present.
        train_timesteps: Model-scale schedule, ``[1, T]``; no /1000 conversion.
        fps: Video frames per second.
        audio_sample_rate: Decoded audio samples per second.

    Returns:
        A replacement DiffusionOutput with decoded media, prompt fields, detached
        FP32 clean latents, sequence lengths/shapes, and timing metadata. Requests
        CPU conversion through the native ``to_cpu`` contract; no independent
        storage copy is guaranteed for tensors already on CPU.

    Raises:
        RuntimeError: Trajectory fields are present, the decoded pair is missing,
            or a five-dimensional video contains more than one sample.
    """
    if any(
        value is not None
        for value in (output.trajectory_latents, output.trajectory_timesteps, output.trajectory_log_probs)
    ):
        raise RuntimeError("Native LTX rollout unexpectedly returned trajectory data for OmniNFT.")
    if not isinstance(output.output, tuple | list) or len(output.output) != 2:
        raise RuntimeError("Native LTX rollout did not return decoded (video, audio) output.")
    decoded_video, decoded_audio = output.output
    if isinstance(decoded_video, torch.Tensor) and decoded_video.ndim == 5:
        if decoded_video.shape[0] != 1:
            raise RuntimeError("Per-request LTX rollout returned a decoded video batch larger than one.")
        decoded_video = decoded_video[0]
    output = replace(output, output=(decoded_video, decoded_audio))

    batch_size = video_latents.shape[0]
    device = video_latents.device
    return with_rollout_data(
        output,
        media_key="video",
        prompt_embeddings=prompt_embeddings,
        rl={
            "audio": decoded_audio,
            "video_latents_clean": video_latents.detach().float(),
            "audio_latents_clean": audio_latents.detach().float(),
            "train_timesteps": train_timesteps,
            "video_latent_shape": torch.tensor([video_latents.shape[1:]], device=device, dtype=torch.long).expand(
                batch_size, -1
            ),
            "audio_latent_shape": torch.tensor([audio_latents.shape[1:]], device=device, dtype=torch.long).expand(
                batch_size, -1
            ),
            "video_seq_len": torch.full((batch_size,), video_latents.shape[1], device=device, dtype=torch.long),
            "audio_seq_len": torch.full((batch_size,), audio_latents.shape[1], device=device, dtype=torch.long),
            "fps": torch.full((batch_size,), fps, device=device, dtype=torch.float32),
            "audio_sample_rate": torch.full((batch_size,), audio_sample_rate, device=device, dtype=torch.long),
        },
        to_cpu=True,
    )


@VllmOmniPipelineBase.register("LTX2Pipeline", algorithm="omni_nft")
class LTX23OmniNFTPipeline(LTXTokenIdPromptMixin, LTX2Pipeline):
    """Run one-phase text-to-AV sampling and retain final OmniNFT training tensors.

    I2V is unsupported because replay data does not include image conditioning
    or its masks. Captured sampler state is instance-local and scoped to forward.
    """

    supports_request_batch = True
    support_image_input = False
    diffusion_io_spec = DiffusionIOSpec(
        primary=MediaSpec("video"),
        auxiliary=(MediaSpec("audio", sample_rate=24000),),
    )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        _require_one_stage_recipe(self.pipeline_recipe)
        if _rollout_progress_bar_enabled():
            self.set_progress_bar_config(desc="LTX denoise")
        else:
            self.set_progress_bar_config(disable=True)
        self._omni_nft_clean_state: LTXAVState | None = None
        self._omni_nft_forward_context: LTXForwardContext | None = None

    def _unpack_and_denormalize_stage(
        self,
        forward_ctx: LTXForwardContext,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Capture the final normalized sampler state before native decoding."""
        self._omni_nft_clean_state = LTXAVState(video=latents, audio=audio_latents)
        self._omni_nft_forward_context = forward_ctx
        return super()._unpack_and_denormalize_stage(
            forward_ctx,
            latents,
            audio_latents,
        )

    def _clear_captured_state(self) -> None:
        """Drop request-local device tensors as soon as one call is complete."""
        self._omni_nft_clean_state = None
        self._omni_nft_forward_context = None

    @staticmethod
    def _copy_request_batch(
        request_batch: DiffusionRequestBatch,
        prompt_kwargs: dict[str, torch.Tensor],
    ) -> DiffusionRequestBatch:
        """Copy requests and sampling params, normalizing output and prompt precedence.

        Dict prompts are copied, and embedded prompt aliases superseded by
        ``prompt_kwargs`` are removed. Copies are shallow: tensors and untouched
        nested values remain shared with the caller. For reserved dummy request
        IDs, remove ``multi_modal_data.image`` without inspecting its pixels;
        copy that mapping before changing it and preserve other modalities.

        This accommodates native warmup deriving image support from the base
        pipeline signature despite this adapter's text-to-AV-only contract.
        Return a new DiffusionRequestBatch; caller-owned mappings are unchanged.
        Unsupported output formats raise ``ValueError``.
        """
        prompt_fields = {
            "prompt_embeds": ("prompt_embeds",),
            "prompt_attention_mask": ("prompt_attention_mask", "attention_mask"),
            "negative_prompt_embeds": ("negative_prompt_embeds",),
            "negative_prompt_attention_mask": (
                "negative_prompt_attention_mask",
                "negative_attention_mask",
            ),
        }
        requests = []
        for request in request_batch.requests:
            cloned = copy(request)
            cloned.sampling_params = copy(request.sampling_params)
            cloned.sampling_params.output_type = _resolve_omni_nft_output_type(request.sampling_params)
            if isinstance(request.prompt, dict):
                cloned.prompt = dict(request.prompt)
                if OmniDiffusionRequest.is_dummy_run_request_id(request.request_id):
                    multi_modal_data = cloned.prompt.get("multi_modal_data")
                    if isinstance(multi_modal_data, dict) and "image" in multi_modal_data:
                        multi_modal_data = dict(multi_modal_data)
                        multi_modal_data.pop("image")
                        if multi_modal_data:
                            cloned.prompt["multi_modal_data"] = multi_modal_data
                        else:
                            cloned.prompt.pop("multi_modal_data")
                for field, aliases in prompt_fields.items():
                    if field in prompt_kwargs:
                        for alias in aliases:
                            cloned.prompt.pop(alias, None)
            requests.append(cloned)
        return DiffusionRequestBatch(requests=requests)

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest | DiffusionRequestBatch,
        **kwargs: Any,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        """Generate decoded AV and replay fields without recording gradients.

        Accept one request or a non-empty request batch; return a DiffusionOutput
        or an ordered list respectively. Token-ID-derived prompt kwargs override
        caller kwargs and embedded prompt aliases. Clear captured latent/context
        references before generation and in ``finally`` on success or failure.
        Concurrent calls on the same pipeline must be serialized by the caller.
        """
        self._clear_captured_state()
        try:
            return self._forward_impl(req, **kwargs)
        finally:
            self._clear_captured_state()

    def _forward_impl(
        self,
        req: OmniDiffusionRequest | DiffusionRequestBatch,
        **kwargs: Any,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        """Run native sampling and assemble each request's forward-training payload.

        Encode token-ID prompts, apply their kwargs after caller kwargs, and use
        copied requests for native sampling. Split captured positive/negative
        connector embeddings, masks, clean latents, and model-scale timesteps
        into per-request rows. Crop normalized audio latents to the captured
        logical frame count, excluding sequence-parallel padding. Include latent
        shapes/lengths, decoded audio, fps, and vocoder sample rate in the payload.

        Return one output for a single request or a list for a batch, preserving
        request order. Empty batches raise ``ValueError``; inconsistent native
        output counts, capture state, lengths, or timing metadata raise
        ``RuntimeError``. Captured references are released by ``forward``.
        """
        request_batch = req if isinstance(req, DiffusionRequestBatch) else DiffusionRequestBatch(requests=[req])
        return_batch = isinstance(req, DiffusionRequestBatch)
        if request_batch.num_reqs < 1:
            raise ValueError("LTX-2.3 OmniNFT expects at least one request.")

        prompt_kwargs = self._prepare_batch_prompt_embeds(request_batch)
        native_request_batch = self._copy_request_batch(request_batch, prompt_kwargs)
        native_kwargs = dict(kwargs)
        native_kwargs.update(prompt_kwargs)

        raw_outputs = super().forward(native_request_batch, **native_kwargs)
        if return_batch and not isinstance(raw_outputs, list):
            raise RuntimeError("Pinned vLLM-Omni LTX batch rollout must return one DiffusionOutput per request.")
        outputs = raw_outputs if isinstance(raw_outputs, list) else [raw_outputs]
        if isinstance(raw_outputs, list) and len(outputs) != request_batch.num_reqs:
            raise RuntimeError(
                f"LTX-2.3 OmniNFT rollout returned {len(outputs)} outputs for {request_batch.num_reqs} requests."
            )

        clean_state = self._omni_nft_clean_state
        forward_context = self._omni_nft_forward_context
        if clean_state is None or forward_context is None:
            raise RuntimeError("LTX-2.3 OmniNFT rollout did not capture final latent and prompt state.")
        prompt_context = forward_context.prompt_context

        original_audio_num_frames = int(forward_context.original_audio_num_frames)
        if not 0 < original_audio_num_frames <= clean_state.audio.shape[1]:
            raise RuntimeError(
                "LTX-2.3 OmniNFT captured an invalid logical audio length: "
                f"{original_audio_num_frames} for {clean_state.audio.shape[1]} packed rows."
            )
        # Sequence-parallel padding is not part of the logical audio sequence.
        clean_audio = clean_state.audio[:, :original_audio_num_frames]

        packed_batch = clean_state.video.shape[0]
        if packed_batch != request_batch.num_reqs:
            raise RuntimeError(
                f"LTX-2.3 OmniNFT captured batch {packed_batch} does not match {request_batch.num_reqs} requests."
            )
        audio_sample_rate = _require_audio_sample_rate(self)
        fps = _require_frame_rate(forward_context)
        device = clean_state.video.device
        train_timesteps = forward_context.timesteps.to(device=device, dtype=torch.float32)
        if train_timesteps.ndim == 1:
            train_timesteps = train_timesteps.unsqueeze(0).expand(packed_batch, -1)
        elif train_timesteps.shape[0] != packed_batch:
            raise RuntimeError(
                f"LTX-2.3 OmniNFT train_timesteps batch {tuple(train_timesteps.shape)} does not match {packed_batch}."
            )

        prompt_fields = {
            "prompt_embeds": prompt_context.positive_connector_prompt_embeds,
            "audio_prompt_embeds": prompt_context.positive_connector_audio_prompt_embeds,
            "prompt_embeds_mask": prompt_context.positive_connector_attention_mask,
            "negative_prompt_embeds": prompt_context.negative_connector_prompt_embeds,
            "negative_audio_prompt_embeds": prompt_context.negative_connector_audio_prompt_embeds,
            "negative_prompt_embeds_mask": prompt_context.negative_connector_attention_mask,
        }

        finalized = [
            _attach_omni_nft_rollout_data(
                output,
                video_latents=_slice_batch_row(clean_state.video, index, packed_batch, "video_latents_clean"),
                audio_latents=_slice_batch_row(clean_audio, index, packed_batch, "audio_latents_clean"),
                prompt_embeddings={
                    key: None if value is None else _slice_batch_row(value, index, packed_batch, key)
                    for key, value in prompt_fields.items()
                },
                train_timesteps=_slice_batch_row(train_timesteps, index, packed_batch, "train_timesteps"),
                fps=fps,
                audio_sample_rate=audio_sample_rate,
            )
            for index, output in enumerate(outputs)
        ]
        return finalized if return_batch else finalized[0]
