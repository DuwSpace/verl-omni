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

"""Batch-native DeSync reward adapted from zghhui/OmniNFT."""

import importlib
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from verl.utils.device import get_device_name, get_torch_device

from .reward_utils import load_torch_state_dict

_DEFAULT_MODEL_REVISION = "zghhui/OmniNFT-Reward-Series@9e30061a1392d03bafdcf717e80a385ddf411b4d"
_DEFAULT_SOURCE_REVISION = "fb9237f6e74edf0d0f2a683f4d975b79fde588fe"
_DEFINITION_VERSION = "omninft-desync-synchformer-v3"
_TARGET_VIDEO_FPS = 25.0
_TARGET_AUDIO_RATE = 16_000
_MAX_SECONDS = 8
_VIDEO_FRAMES = 200
_AUDIO_SAMPLES = 128_000
_VIDEO_SEGMENT = 16
_VIDEO_STEP = 8
_AUDIO_SEGMENT = 10_240
_AUDIO_STEP = 5_120
_SEGMENTS = 24
_COMPARE_SEGMENTS = 14
_MEL_TIME = 66
_CLASS_GRID = torch.linspace(-2.0, 2.0, 21)


@dataclass
class _DeSyncNativeState:
    model: Any
    mel: Any
    model_revision: str
    source_revision: str
    device: torch.device | None = None
    mha_fastpath_enabled: bool | None = None


def _read_git_revision(source_root: Path) -> str:
    git_dir = source_root / ".git"
    if not git_dir.is_dir():
        raise ValueError("DeSync source_root must be a Git checkout with a .git directory.")
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose_ref = git_dir / ref
    if loose_ref.is_file():
        return loose_ref.read_text(encoding="utf-8").strip()
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                revision, name = line.split(" ", 1)
                if name == ref:
                    return revision
    raise ValueError(f"Cannot resolve DeSync source revision {ref!r}.")


@contextmanager
def _source_import_path(source_root: Path):
    sys.path.insert(0, str(source_root))
    try:
        yield
    finally:
        sys.path.remove(str(source_root))


def _ensure_transformers_legacy_api() -> None:
    """Restore APIs used by OmniNFT's pinned pre-v5 AST source."""
    from transformers import pytorch_utils
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                shifted_head = head - sum(pruned_head < head for pruned_head in already_pruned_heads)
                mask[shifted_head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(mask.shape[0])[mask].long()
            return heads, index

        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    if not hasattr(PreTrainedModel, "get_head_mask"):

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers
            if head_mask.ndim == 1:
                head_mask = head_mask[None, None, :, None, None].expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.ndim == 2:
                head_mask = head_mask[:, None, :, None, None]
            if head_mask.ndim != 5:
                raise ValueError("head_mask must have dimension 1, 2, or 5.")
            head_mask = head_mask.to(dtype=self.dtype)
            return head_mask.unsqueeze(-1) if is_attention_chunked else head_mask

        PreTrainedModel.get_head_mask = get_head_mask


def _load_components(model_path: str, source_root: str, source_revision: str) -> tuple[Any, Any]:
    root = Path(source_root).expanduser().resolve()
    module_path = root / "flow_grpo/audio_video_align/synchformer/synchformer.py"
    config_path = module_path.parent / "divided_224_16x4.yaml"
    if not module_path.is_file() or not config_path.is_file():
        raise ValueError("DeSync source_root is missing the OmniNFT Synchformer source or fixed config.")
    actual_revision = _read_git_revision(root)
    if actual_revision != source_revision:
        raise ValueError(f"DeSync source revision mismatch: expected {source_revision}, got {actual_revision}.")

    module_name = "flow_grpo.audio_video_align.synchformer.synchformer"
    existing = sys.modules.get(module_name)
    if existing is not None and Path(existing.__file__).resolve() != module_path:
        raise RuntimeError("A different Synchformer source_root is already imported in this process.")
    _ensure_transformers_legacy_api()
    with _source_import_path(root):
        module = existing or importlib.import_module(module_name)
    if Path(module.__file__).resolve() != module_path:
        raise RuntimeError("Imported Synchformer does not belong to the configured source_root.")

    model = module.Synchformer()
    state_dict = load_torch_state_dict(model_path)
    if not isinstance(state_dict, dict) or not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError("DeSync checkpoint must be a tensor state dict.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=_TARGET_AUDIO_RATE, win_length=400, hop_length=160, n_fft=1024, n_mels=128
    )
    return model, mel


def initialize(
    model_path: str,
    source_root: str,
    model_revision: str = _DEFAULT_MODEL_REVISION,
    source_revision: str = _DEFAULT_SOURCE_REVISION,
) -> _DeSyncNativeState:
    """Load the pinned Synchformer source and checkpoint into CPU state."""
    for name, value in (
        ("model_path", model_path),
        ("source_root", source_root),
        ("model_revision", model_revision),
        ("source_revision", source_revision),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"DeSync Native Reward requires a non-empty {name}.")
    model, mel = _load_components(model_path, source_root, source_revision)
    return _DeSyncNativeState(model, mel, model_revision, source_revision)


def _resolve_device(device: int | str | torch.device) -> torch.device:
    if isinstance(device, int):
        return torch.device(f"{get_device_name()}:{device}")
    return torch.device(device)


def activate(state: _DeSyncNativeState, device: int | str | torch.device) -> None:
    """Move Synchformer and its mel transform to the runtime accelerator."""
    if state.model is None:
        raise RuntimeError("DeSync Native Reward has already been finalized.")
    if state.device is not None:
        raise RuntimeError("DeSync Native Reward is already active.")
    state.device = _resolve_device(device)
    state.mha_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    state.model.to(state.device).eval()
    state.mel.to(state.device)


def _temporal_resample_video(video: torch.Tensor, source_fps: float) -> torch.Tensor:
    duration = video.shape[0] / source_fps
    frame_count = int(math.floor(duration * _TARGET_VIDEO_FPS + 1e-9))
    if frame_count <= 0:
        raise ValueError("DeSync video is too short for 25 fps resampling.")
    frame_count = min(frame_count, _VIDEO_FRAMES)
    # ffmpeg `fps=25` (torio): sample each output slot at its center.
    output_index = torch.arange(frame_count, dtype=torch.float64)
    indices = torch.ceil((output_index + 0.5) * source_fps / _TARGET_VIDEO_FPS).long() - 1
    return video[indices.clamp(0, video.shape[0] - 1)]


def _resize_crop_video(video: torch.Tensor) -> torch.Tensor:
    height, width = video.shape[-2:]
    if min(height, width) <= 0:
        raise ValueError("DeSync video spatial dimensions must be positive.")
    if height <= width:
        size = (224, int(width * 224 / height))
    else:
        size = (int(height * 224 / width), 224)
    video = F.interpolate(video.float().div(255), size=size, mode="bicubic", align_corners=False, antialias=True)
    top = (size[0] - 224) // 2
    left = (size[1] - 224) // 2
    return video[:, :, top : top + 224, left : left + 224].sub(0.5).div(0.5)


def _prepare_video(video: torch.Tensor, source_fps: float) -> torch.Tensor:
    video = _resize_crop_video(_temporal_resample_video(video, source_fps))
    if video.shape[0] < _VIDEO_FRAMES:
        video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, _VIDEO_FRAMES - video.shape[0]), value=-1.0)
    return video


def _resample_audio(waveform: torch.Tensor, source_rate: int) -> torch.Tensor:
    if source_rate == _TARGET_AUDIO_RATE:
        return waveform
    import torchaudio.functional as audio_functional

    return audio_functional.resample(waveform.unsqueeze(0), source_rate, _TARGET_AUDIO_RATE).squeeze(0)


def _prepare_audio(audio: torch.Tensor, source_rate: int) -> torch.Tensor:
    waveform = _resample_audio(audio.float().mean(dim=0), source_rate)[:_AUDIO_SAMPLES]
    return F.pad(waveform, (0, _AUDIO_SAMPLES - waveform.shape[0]))


def _extract_inputs(batch) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float], list[int]]:
    batch_size = len(batch)
    videos = batch.batch.get("responses")
    audio = batch.batch.get("audio")
    fps = batch.batch.get("fps")
    sample_rates = batch.batch.get("audio_sample_rate")
    if batch_size <= 0:
        raise ValueError("DeSync Native Reward requires a non-empty local batch.")
    if not isinstance(videos, torch.Tensor) or videos.ndim != 5 or videos.shape[0] != batch_size:
        raise ValueError(f"DeSync responses must have shape [B,T,3,H,W] with B={batch_size}.")
    if videos.shape[1] <= 0 or videos.shape[2] != 3 or videos.dtype != torch.uint8:
        raise ValueError("DeSync responses must be non-empty uint8 RGB video.")
    if not isinstance(audio, torch.Tensor) or audio.ndim != 3 or audio.shape[0] != batch_size:
        raise ValueError(f"DeSync audio must have shape [B,C,S] with B={batch_size}.")
    if audio.shape[1] <= 0 or audio.shape[2] <= 0 or not audio.dtype.is_floating_point:
        raise ValueError("DeSync audio must be a non-empty floating-point tensor.")
    if not torch.isfinite(audio).all():
        raise ValueError("DeSync audio must contain only finite values.")
    if not isinstance(fps, torch.Tensor) or fps.shape != (batch_size,) or not fps.dtype.is_floating_point:
        raise ValueError(f"DeSync fps must be a floating-point tensor with shape ({batch_size},).")
    if not torch.isfinite(fps).all():
        raise ValueError("DeSync fps must contain only finite values.")
    if (
        not isinstance(sample_rates, torch.Tensor)
        or sample_rates.shape != (batch_size,)
        or sample_rates.dtype.is_floating_point
        or sample_rates.dtype == torch.bool
    ):
        raise ValueError(f"DeSync audio_sample_rate must be an integer tensor with shape ({batch_size},).")
    rates = [int(value) for value in sample_rates.cpu().tolist()]
    source_fps = [float(value) for value in fps.cpu().tolist()]
    if any(value <= 0 for value in source_fps) or any(value <= 0 for value in rates):
        raise ValueError("DeSync source rates must be positive.")
    prepared_video = [_prepare_video(sample.cpu(), rate) for sample, rate in zip(videos, source_fps, strict=True)]
    prepared_audio = [_prepare_audio(sample.cpu(), rate) for sample, rate in zip(audio, rates, strict=True)]
    return prepared_video, prepared_audio, source_fps, rates


def _pad_mel_time(mel: torch.Tensor) -> torch.Tensor:
    if mel.shape[-1] < _MEL_TIME:
        return F.pad(mel, (0, _MEL_TIME - mel.shape[-1]))
    return mel[..., :_MEL_TIME]


def _score_micro_batch(state: _DeSyncNativeState, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
    batch_size = video.shape[0]
    video_segments = video.unfold(1, _VIDEO_SEGMENT, _VIDEO_STEP).movedim(-1, 2)
    audio_segments = audio.unfold(1, _AUDIO_SEGMENT, _AUDIO_STEP)
    if video_segments.shape[1] != _SEGMENTS or audio_segments.shape[1] != _SEGMENTS:
        raise ValueError("DeSync preprocessing must produce exactly 24 AV segments.")

    visual = video_segments.reshape(-1, _VIDEO_SEGMENT, *video.shape[2:]).unsqueeze(1).to(state.device)
    visual = state.model.extract_vfeats(visual)
    visual = visual.reshape(batch_size, _SEGMENTS, *visual.shape[2:])

    audio_segments = audio_segments.to(state.device)
    mel = _pad_mel_time(torch.log(state.mel(audio_segments) + 1e-6))
    mel = (mel - (-4.2677393)) / (2 * 4.5689974)
    auditory = state.model.extract_afeats(mel.unsqueeze(2))

    distances = []
    for start in (0, _SEGMENTS - _COMPARE_SEGMENTS):
        logits = state.model.compare_v_a(
            visual[:, start : start + _COMPARE_SEGMENTS], auditory[:, start : start + _COMPARE_SEGMENTS]
        )
        if not isinstance(logits, torch.Tensor) or logits.shape != (batch_size, 21):
            raise ValueError(f"DeSync logits must have shape ({batch_size}, 21).")
        if not torch.isfinite(logits).all():
            raise ValueError("DeSync logits must contain only finite values.")
        offsets = _CLASS_GRID.to(logits.device)[logits.argmax(dim=-1)].abs()
        distances.append(offsets)
    distance = torch.stack(distances).mean(dim=0)
    return (1.0 / (1.0 + distance)).float().cpu()


def score_batch(state: _DeSyncNativeState, batch, micro_batch_size: int, **kwargs) -> dict[str, Any]:
    """Score one complete local shard using batched Synchformer inference."""
    del kwargs
    if state.device is None:
        raise RuntimeError("DeSync Native Reward must be active before scoring.")
    if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int) or micro_batch_size <= 0:
        raise ValueError("DeSync micro_batch_size must be a positive integer.")
    videos, audio, source_fps, source_rates = _extract_inputs(batch)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(batch), micro_batch_size):
            stop = min(start + micro_batch_size, len(batch))
            chunks.append(_score_micro_batch(state, torch.stack(videos[start:stop]), torch.stack(audio[start:stop])))
    scores = torch.cat(chunks).to(torch.float32)
    if scores.shape != (len(batch),) or not torch.isfinite(scores).all():
        raise ValueError("DeSync scores must be finite and sample-aligned.")
    forward_calls = math.ceil(len(batch) / micro_batch_size)
    return {
        "scores": scores,
        "valid_mask": torch.ones(len(batch), dtype=torch.bool),
        "metrics": {
            "batch_size": len(batch),
            "micro_batch_size": micro_batch_size,
            "video_forward_calls": forward_calls,
            "audio_forward_calls": forward_calls,
            "compare_forward_calls": 2 * forward_calls,
            "segments_per_sample": _SEGMENTS,
            "source_fps": sorted(set(source_fps)),
            "source_sample_rates": sorted(set(source_rates)),
            "target_fps": _TARGET_VIDEO_FPS,
            "target_sample_rate": _TARGET_AUDIO_RATE,
            "source_revision": state.source_revision,
        },
        "model_revision": state.model_revision,
        "definition_version": _DEFINITION_VERSION,
    }


def _release_accelerator_memory() -> None:
    accelerator = get_torch_device()
    accelerator.synchronize()
    accelerator.empty_cache()
    accelerator.synchronize()


def deactivate(state: _DeSyncNativeState) -> None:
    """Move Synchformer and mel state back to CPU."""
    if state.device is None:
        return
    device = state.device
    state.model.to("cpu")
    state.mel.to("cpu")
    state.device = None
    if state.mha_fastpath_enabled is not None:
        torch.backends.mha.set_fastpath_enabled(state.mha_fastpath_enabled)
        state.mha_fastpath_enabled = None
    if device.type != "cpu":
        _release_accelerator_memory()


def finalize(state: _DeSyncNativeState) -> None:
    """Release all DeSync state owned by this Reward Manager."""
    deactivate(state)
    state.model = None
    state.mel = None
