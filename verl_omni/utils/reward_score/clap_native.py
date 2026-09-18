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

"""Batch-native CLAP reward adapted from zghhui/OmniNFT."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_CLAP_SAMPLE_RATE = 48_000
_DEFAULT_MODEL_REVISION = "laion/clap-htsat-unfused@8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
_DEFINITION_VERSION = "omninft-clap-cosine-v1"


@dataclass
class _ClapNativeState:
    model: Any
    processor: Any
    model_revision: str
    device: torch.device | None = None


def _load_components(model_path: str) -> tuple[Any, Any]:
    from transformers import AutoProcessor, ClapModel

    model = ClapModel.from_pretrained(model_path).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def _load_state(model_path: str, model_revision: str = _DEFAULT_MODEL_REVISION) -> _ClapNativeState:
    """Load CLAP assets from model_path and retain the configured revision label."""
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("CLAP Native Reward requires a non-empty model_path.")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("CLAP Native Reward requires a non-empty model_revision.")
    model, processor = _load_components(model_path)
    return _ClapNativeState(model=model, processor=processor, model_revision=model_revision)


def _resample_audio(waveform: torch.Tensor, source_rate: int) -> torch.Tensor:
    if source_rate == _CLAP_SAMPLE_RATE:
        return waveform
    import torchaudio.functional as audio_functional

    return audio_functional.resample(
        waveform.unsqueeze(0),
        orig_freq=source_rate,
        new_freq=_CLAP_SAMPLE_RATE,
    ).squeeze(0)


def _extract_inputs(batch) -> tuple[list[np.ndarray], list[str], list[int]]:
    """Validate audio/text and return mono 48 kHz FP32 arrays, prompts, and rates."""
    batch_size = len(batch)
    if batch_size <= 0:
        raise ValueError("CLAP Native Reward requires a non-empty local batch.")
    audio = batch.batch.get("audio")
    if not isinstance(audio, torch.Tensor) or audio.ndim != 3 or audio.shape[0] != batch_size:
        shape = None if not isinstance(audio, torch.Tensor) else tuple(audio.shape)
        raise ValueError(f"CLAP audio must have shape [B,C,S] with B={batch_size}, got {shape}.")
    if audio.shape[1] <= 0 or audio.shape[2] <= 0 or not audio.dtype.is_floating_point:
        raise ValueError("CLAP audio must be a non-empty floating-point tensor.")
    audio = audio.detach().float().cpu()
    if not torch.isfinite(audio).all():
        raise ValueError("CLAP audio must contain only finite values.")

    sample_rates = batch.batch.get("audio_sample_rate")
    if (
        not isinstance(sample_rates, torch.Tensor)
        or sample_rates.shape != (batch_size,)
        or sample_rates.dtype.is_floating_point
        or sample_rates.dtype == torch.bool
    ):
        raise ValueError(f"CLAP audio_sample_rate must be an integer tensor with shape ({batch_size},).")
    rates = [int(value) for value in sample_rates.detach().cpu().tolist()]
    if any(rate <= 0 for rate in rates):
        raise ValueError("CLAP audio_sample_rate values must be positive.")

    reward_inputs = batch.non_tensor_batch.get("reward_inputs")
    if reward_inputs is None or np.asarray(reward_inputs, dtype=object).shape != (batch_size,):
        raise ValueError(f"CLAP reward_inputs must have shape ({batch_size},).")

    prompts = []
    waveforms = []
    for index, (sample, rate) in enumerate(zip(audio, rates, strict=True)):
        reward_input = reward_inputs[index]
        try:
            prompt = reward_input["text"]["audio"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"CLAP reward_inputs[{index}] must contain text.audio.") from exc
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"CLAP reward_inputs[{index}].text.audio must be a non-empty string.")
        waveform = _resample_audio(sample.mean(dim=0), rate)
        if waveform.ndim != 1 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise ValueError(f"CLAP preprocessed audio at index {index} is invalid.")
        prompts.append(prompt)
        waveforms.append(waveform.numpy().astype(np.float32, copy=False))
    return waveforms, prompts, rates


async def compute_score_batch(batch, reward_model, *, micro_batch_size: int, **kwargs) -> dict[str, Any]:
    """Score aligned audio/text pairs with rescaled CLAP cosine similarity.

    Args:
        batch: Nonempty batch with finite floating ``audio[B, C, S]``, positive
            integer ``audio_sample_rate[B]``, and per-row
            ``reward_inputs.text.audio`` strings. Audio is detached to CPU,
            channel-averaged, and resampled to 48 kHz.
        reward_model: Active executor returning aligned audio/text embeddings.
        micro_batch_size: Positive number of original audio/text pairs per call.
        **kwargs: Ignored scorer options.

    Returns:
        CPU FP32 ``scores[B]`` equal to ``clamp((cosine + 1) / 2, 0, 1)``;
        higher means greater alignment. Also returns all-true CPU bool
        ``valid_mask[B]``. Row order is preserved.

    Raises:
        ValueError: Invalid inputs, micro-batch size, embedding shapes, or scores.
    """
    del kwargs
    if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int) or micro_batch_size <= 0:
        raise ValueError("CLAP micro_batch_size must be a positive integer.")

    waveforms, prompts, _ = _extract_inputs(batch)
    score_chunks = []
    for start in range(0, len(batch), micro_batch_size):
        stop = min(start + micro_batch_size, len(batch))
        output = await reward_model.infer(waveforms[start:stop], prompts[start:stop])
        audio_embeddings = F.normalize(output["audio_embeddings"].float(), p=2, dim=-1)
        text_embeddings = F.normalize(output["text_embeddings"].float(), p=2, dim=-1)
        if (
            audio_embeddings.ndim != 2
            or text_embeddings.shape != audio_embeddings.shape
            or audio_embeddings.shape[0] != stop - start
        ):
            raise ValueError("CLAP embeddings must preserve the aligned audio/text micro-batch dimension.")
        score_chunks.append(((audio_embeddings * text_embeddings).sum(dim=-1) + 1.0).div(2.0).clamp(0, 1).cpu())

    scores = torch.cat(score_chunks).to(dtype=torch.float32)
    if scores.shape != (len(batch),) or not torch.isfinite(scores).all():
        raise ValueError("CLAP scores must be finite and sample-aligned.")
    return {
        "scores": scores,
        "valid_mask": torch.ones(len(batch), dtype=torch.bool),
    }


class CLAPNativeModel:
    """Raw CLAP inference adapter owned by a native reward executor."""

    def __init__(self, model_path: str, device, **kwargs: Any) -> None:
        self._state = _load_state(model_path=model_path, **kwargs)
        self._state.device = torch.device(device)
        self._state.model.to(self._state.device).eval()

    def close(self) -> None:
        """Drop model/processor references; reuse requires constructing a new adapter."""
        self._state.model = None
        self._state.processor = None
        self._state.device = None

    @torch.inference_mode()
    def infer(self, waveforms: list[np.ndarray], prompts: list[str]) -> dict[str, torch.Tensor]:
        """Encode aligned mono 48 kHz arrays and text without gradients.

        The processor pads/truncates the batch; its tensor outputs move to the
        active device. Return detached CPU ``audio_embeddings`` and
        ``text_embeddings``, each ``[B, D]`` in model-output dtype. Cosine
        normalization and score scaling are performed by the scorer.
        """
        inputs = self._state.processor(
            text=prompts,
            audio=waveforms,
            sampling_rate=_CLAP_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {
            key: value.to(self._state.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        outputs = self._state.model(**inputs)
        return {
            "audio_embeddings": outputs.audio_embeds.detach().cpu(),
            "text_embeddings": outputs.text_embeds.detach().cpu(),
        }
