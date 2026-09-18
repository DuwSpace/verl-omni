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

"""Batch-native AudioBox Aesthetics reward adapted from zghhui/OmniNFT."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

_AUDIOBOX_SAMPLE_RATE = 16_000
_AUDIOBOX_WINDOW_SAMPLES = 10 * _AUDIOBOX_SAMPLE_RATE
_AUDIOBOX_HOP_SAMPLES = 10 * _AUDIOBOX_SAMPLE_RATE
_AXES = ("CE", "CU", "PC", "PQ")
_DEFAULT_MODEL_REVISION = "facebook/audiobox-aesthetics@9b1dd8e5df9af7216e836a98974fe3b82c56ded6"
_DEFINITION_VERSION = "omninft-audiobox-aesthetics-v1"


@dataclass
class _AudioBoxState:
    model: Any
    target_transform: dict[str, tuple[float, float]]
    model_revision: str
    device: torch.device | None = None


def _load_model(model_path: str) -> Any:
    from audiobox_aesthetics.model.aes import AesMultiOutput

    return AesMultiOutput.from_pretrained(model_path).eval()


def _load_state(model_path: str, model_revision: str = _DEFAULT_MODEL_REVISION) -> _AudioBoxState:
    """Load AudioBox, validate CE/CU/PC/PQ transforms, and record the revision label."""
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("AudioBox Native Reward requires a non-empty model_path.")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("AudioBox Native Reward requires a non-empty model_revision.")

    model = _load_model(model_path)
    target_transform = getattr(model, "target_transform", None)
    if not isinstance(target_transform, dict):
        raise ValueError("AudioBox model is missing target_transform metadata.")
    transforms = {}
    for axis in _AXES:
        values = target_transform.get(axis)
        if not isinstance(values, dict) or not isinstance(values.get("mean"), int | float):
            raise ValueError(f"AudioBox target_transform is missing {axis}.mean.")
        if not isinstance(values.get("std"), int | float) or values["std"] <= 0:
            raise ValueError(f"AudioBox target_transform is missing a positive {axis}.std.")
        transforms[axis] = (float(values["mean"]), float(values["std"]))
    return _AudioBoxState(model=model, target_transform=transforms, model_revision=model_revision)


def _resample_audio(waveform: torch.Tensor, source_rate: int) -> torch.Tensor:
    if source_rate == _AUDIOBOX_SAMPLE_RATE:
        return waveform
    import torchaudio.functional as audio_functional

    return audio_functional.resample(
        waveform.unsqueeze(0),
        orig_freq=source_rate,
        new_freq=_AUDIOBOX_SAMPLE_RATE,
    ).squeeze(0)


def _extract_inputs(batch) -> tuple[list[torch.Tensor], list[int]]:
    """Validate audio/rates and return channel-averaged CPU FP32 16 kHz waveforms."""
    batch_size = len(batch)
    if batch_size <= 0:
        raise ValueError("AudioBox Native Reward requires a non-empty local batch.")
    audio = batch.batch.get("audio")
    if not isinstance(audio, torch.Tensor) or audio.ndim != 3 or audio.shape[0] != batch_size:
        shape = None if not isinstance(audio, torch.Tensor) else tuple(audio.shape)
        raise ValueError(f"AudioBox audio must have shape [B,C,S] with B={batch_size}, got {shape}.")
    if audio.shape[1] <= 0 or audio.shape[2] <= 0 or not audio.dtype.is_floating_point:
        raise ValueError("AudioBox audio must be a non-empty floating-point tensor.")
    audio = audio.detach().float().cpu()
    if not torch.isfinite(audio).all():
        raise ValueError("AudioBox audio must contain only finite values.")

    sample_rates = batch.batch.get("audio_sample_rate")
    if (
        not isinstance(sample_rates, torch.Tensor)
        or sample_rates.shape != (batch_size,)
        or sample_rates.dtype.is_floating_point
        or sample_rates.dtype == torch.bool
    ):
        raise ValueError(f"AudioBox audio_sample_rate must be an integer tensor with shape ({batch_size},).")
    rates = [int(value) for value in sample_rates.detach().cpu().tolist()]
    if any(rate <= 0 for rate in rates):
        raise ValueError("AudioBox audio_sample_rate values must be positive.")

    waveforms = []
    for index, (sample, rate) in enumerate(zip(audio, rates, strict=True)):
        waveform = _resample_audio(sample.mean(dim=0), rate)
        if waveform.ndim != 1 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise ValueError(f"AudioBox preprocessed audio at index {index} is invalid.")
        waveforms.append(waveform)
    return waveforms, rates


def _make_windows(waveforms: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[int], list[float]]:
    """Expand waveforms into nonoverlapping 10 s windows, padding the last one.

    Return CPU ``[W, 1, 160000]`` audio and bool validity masks, local sample
    indices, and valid-duration fractions for per-sample weighted averaging.
    """
    windows = []
    masks = []
    sample_indices = []
    weights = []
    for sample_index, waveform in enumerate(waveforms):
        for start in range(0, waveform.numel(), _AUDIOBOX_HOP_SAMPLES):
            window = waveform[start : start + _AUDIOBOX_WINDOW_SAMPLES]
            valid_length = window.numel()
            if valid_length < _AUDIOBOX_WINDOW_SAMPLES:
                window = F.pad(window, (0, _AUDIOBOX_WINDOW_SAMPLES - valid_length))
            mask = torch.zeros(_AUDIOBOX_WINDOW_SAMPLES, dtype=torch.bool)
            mask[:valid_length] = True
            windows.append(window.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            sample_indices.append(sample_index)
            weights.append(valid_length / _AUDIOBOX_WINDOW_SAMPLES)
    return torch.stack(windows), torch.stack(masks), sample_indices, weights


def _validate_predictions(predictions: Any, window_count: int) -> dict[str, torch.Tensor]:
    if not isinstance(predictions, dict):
        raise ValueError("AudioBox model output must be a dict of axis tensors.")
    validated = {}
    for axis in _AXES:
        values = predictions.get(axis)
        if (
            not isinstance(values, torch.Tensor)
            or values.shape != (window_count,)
            or not values.dtype.is_floating_point
        ):
            raise ValueError(f"AudioBox {axis} output must be a floating-point tensor with shape ({window_count},).")
        if not torch.isfinite(values).all():
            raise ValueError(f"AudioBox {axis} output must contain only finite values.")
        validated[axis] = values
    return validated


def _score_windows(output: dict[str, Any], window_count: int) -> torch.Tensor:
    """Undo each axis's target transform and return CPU FP32 window rewards."""
    predictions = _validate_predictions(output["predictions"], window_count)
    target_transform = output["target_transform"]
    restored = {}
    for axis in _AXES:
        mean, std = target_transform[axis]
        values = predictions[axis].float().mul(std).add(mean).detach().cpu()
        restored[axis] = values
    return (restored["CE"] + restored["CU"] + restored["PQ"] - restored["PC"]) / 40.0


async def compute_score_batch(batch, reward_model, *, micro_batch_size: int, **kwargs) -> dict[str, Any]:
    """Score audio aesthetics, averaging window rewards by valid duration.

    Args:
        batch: Nonempty batch with finite floating ``audio[B, C, S]`` and
            positive integer ``audio_sample_rate[B]``. Audio is detached to CPU,
            channel-averaged, and resampled to 16 kHz; text is not used.
        reward_model: Active executor exposing AudioBox inference and metadata.
        micro_batch_size: Positive number of original samples per call. All
            their 10 s windows are expanded into one model batch, so this does
            not bound the number of windows.
        **kwargs: Ignored scorer options.

    Returns:
        ``scores`` as CPU FP32 ``[B]`` and all-true CPU bool ``valid_mask[B]``.
        Each window's score is ``(CE + CU + PQ - PC) / 40`` after restoring
        target mean/std; higher is preferred. Scores are not clamped.

    Raises:
        ValueError: Invalid inputs, micro-batch size, predictions, or final scores.
    """
    del kwargs
    if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int) or micro_batch_size <= 0:
        raise ValueError("AudioBox micro_batch_size must be a positive integer.")

    waveforms, _ = _extract_inputs(batch)
    score_chunks = []
    for start in range(0, len(waveforms), micro_batch_size):
        stop = min(start + micro_batch_size, len(waveforms))
        windows, masks, sample_indices, weights = _make_windows(waveforms[start:stop])
        output = await reward_model.infer(windows, masks)
        local_scores = _score_windows(output, windows.shape[0])
        sample_scores = []
        for sample_index in range(stop - start):
            selected = [index for index, owner in enumerate(sample_indices) if owner == sample_index]
            selected_weights = torch.tensor([weights[index] for index in selected], dtype=torch.float32)
            sample_scores.append((local_scores[selected] * selected_weights).sum() / selected_weights.sum())
        score_chunks.append(torch.stack(sample_scores))

    scores = torch.cat(score_chunks).to(dtype=torch.float32)
    if scores.shape != (len(batch),) or not torch.isfinite(scores).all():
        raise ValueError("AudioBox scores must be finite and sample-aligned.")
    return {
        "scores": scores,
        "valid_mask": torch.ones(len(batch), dtype=torch.bool),
    }


class AudioBoxNativeModel:
    """Raw AudioBox inference adapter owned by a native reward executor."""

    def __init__(self, model_path: str, device, **kwargs: Any) -> None:
        self._state = _load_state(model_path=model_path, **kwargs)
        self._state.device = torch.device(device)
        self._state.model.to(self._state.device).eval()

    def close(self) -> None:
        """Drop model/transform references; further inference requires a new adapter."""
        self._state.model = None
        self._state.target_transform = {}
        self._state.device = None

    @torch.inference_mode()
    def infer(self, windows: torch.Tensor, masks: torch.Tensor) -> dict[str, Any]:
        """Infer prepared ``[W, 1, 160000]`` audio and bool masks without gradients.

        Move inputs to the active device without changing dtype. Return detached
        CPU ``[W]`` predictions for CE/CU/PC/PQ in model-output dtype, plus the
        retained target-transform mapping; calibration belongs to the scorer.
        """
        inputs = {"wav": windows.to(self._state.device), "mask": masks.to(self._state.device)}
        predictions = self._state.model(inputs)
        return {
            "predictions": {name: predictions[name].detach().cpu() for name in _AXES},
            "target_transform": self._state.target_transform,
        }
