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
from verl.utils.device import get_device_name, get_torch_device

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


def initialize(model_path: str, model_revision: str = _DEFAULT_MODEL_REVISION) -> _ClapNativeState:
    """Load the fixed CLAP model and processor into CPU state."""
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("CLAP Native Reward requires a non-empty model_path.")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("CLAP Native Reward requires a non-empty model_revision.")
    model, processor = _load_components(model_path)
    return _ClapNativeState(model=model, processor=processor, model_revision=model_revision)


def _resolve_device(device: int | str | torch.device) -> torch.device:
    if isinstance(device, int):
        return torch.device(f"{get_device_name()}:{device}")
    return torch.device(device)


def activate(state: _ClapNativeState, device: int | str | torch.device) -> None:
    """Move CLAP to the runtime-selected accelerator."""
    if state.model is None:
        raise RuntimeError("CLAP Native Reward has already been finalized.")
    if state.device is not None:
        raise RuntimeError("CLAP Native Reward is already active.")
    state.device = _resolve_device(device)
    state.model.to(state.device).eval()


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


def score_batch(state: _ClapNativeState, batch, micro_batch_size: int, **kwargs) -> dict[str, Any]:
    """Score one complete local shard using batched CLAP forward calls."""
    del kwargs
    if state.device is None:
        raise RuntimeError("CLAP Native Reward must be active before scoring.")
    if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int) or micro_batch_size <= 0:
        raise ValueError("CLAP micro_batch_size must be a positive integer.")

    waveforms, prompts, source_rates = _extract_inputs(batch)
    score_chunks = []
    forward_calls = 0
    with torch.inference_mode():
        for start in range(0, len(batch), micro_batch_size):
            stop = min(start + micro_batch_size, len(batch))
            inputs = state.processor(
                text=prompts[start:stop],
                audio=waveforms[start:stop],
                sampling_rate=_CLAP_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {
                key: value.to(state.device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            outputs = state.model(**inputs)
            forward_calls += 1
            audio_embeddings = F.normalize(outputs.audio_embeds.float(), p=2, dim=-1)
            text_embeddings = F.normalize(outputs.text_embeds.float(), p=2, dim=-1)
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
        "metrics": {
            "batch_size": len(batch),
            "micro_batch_size": micro_batch_size,
            "forward_calls": forward_calls,
            "source_sample_rates": sorted(set(source_rates)),
            "target_sample_rate": _CLAP_SAMPLE_RATE,
        },
        "model_revision": state.model_revision,
        "definition_version": _DEFINITION_VERSION,
    }


def _release_accelerator_memory() -> None:
    accelerator = get_torch_device()
    accelerator.synchronize()
    accelerator.empty_cache()
    accelerator.synchronize()


def deactivate(state: _ClapNativeState) -> None:
    """Move CLAP back to CPU and release accelerator cache."""
    if state.device is None:
        return
    device = state.device
    state.model.to("cpu")
    state.device = None
    if device.type != "cpu":
        _release_accelerator_memory()


def finalize(state: _ClapNativeState) -> None:
    """Release all CLAP state owned by this Reward Manager."""
    deactivate(state)
    state.model = None
    state.processor = None
