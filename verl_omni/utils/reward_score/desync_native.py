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

"""Native DeSync reward adapted from zghhui/OmniNFT."""

import importlib
import math
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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
_MHA_FASTPATH_LOCK = threading.Lock()
_SOURCE_IMPORT_LOCK = threading.Lock()


@dataclass
class _DeSyncNativeState:
    model: Any
    mel: Any
    model_revision: str
    source_revision: str
    device: torch.device | None = None


@contextmanager
def _source_import_path(source_root: Path):
    """Temporarily prepend a process-wide import path; caller owns serialization."""
    sys.path.insert(0, str(source_root))
    try:
        yield
    finally:
        sys.path.remove(str(source_root))


def _legacy_find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        shifted_head = head - sum(pruned_head < head for pruned_head in already_pruned_heads)
        mask[shifted_head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(mask.shape[0])[mask].long()
    return heads, index


def _legacy_get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
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


@contextmanager
def _temporary_transformers_ast_import_compat():
    """Expose a missing legacy Transformers symbol for the duration of import.

    This changes the process-wide module, not one model instance. Remove only
    the symbol installed here on exit; the caller holds ``_SOURCE_IMPORT_LOCK``.
    """
    from transformers import pytorch_utils

    installed = not hasattr(pytorch_utils, "find_pruneable_heads_and_indices")
    if installed:
        pytorch_utils.find_pruneable_heads_and_indices = _legacy_find_pruneable_heads_and_indices
    try:
        yield
    finally:
        if installed:
            del pytorch_utils.find_pruneable_heads_and_indices


def _import_synchformer(root: Path, module_path: Path):
    """Import one Synchformer source tree under a process-local lock.

    Temporary path/symbol changes are restored after import. Imported modules
    remain cached, and a missing AST ``get_head_mask`` is installed permanently
    on that source class. Reject an already imported different source path;
    neither paths nor configured revision labels verify source contents.
    """
    module_name = "flow_grpo.audio_video_align.synchformer.synchformer"
    ast_module_name = "flow_grpo.audio_video_align.synchformer.hf_src.modeling_ast"
    with _SOURCE_IMPORT_LOCK:
        existing = sys.modules.get(module_name)
        if existing is not None and Path(existing.__file__).resolve() != module_path:
            raise RuntimeError("A different Synchformer source_root is already imported in this process.")
        with _source_import_path(root), _temporary_transformers_ast_import_compat():
            module = existing or importlib.import_module(module_name)
            ast_module = importlib.import_module(ast_module_name)
        ast_base = ast_module.ASTPreTrainedModel
        if not hasattr(ast_base, "get_head_mask"):
            ast_base.get_head_mask = _legacy_get_head_mask
    return module


def _load_components(model_path: str, source_root: str) -> tuple[Any, Any]:
    """Load the configured Synchformer source/state strictly and create a 16 kHz mel transform.

    Check the imported module's path and tensor-only checkpoint, freeze model
    parameters, and return model/mel components without moving them to an accelerator.
    """
    root = Path(source_root).expanduser().resolve()
    module_path = root / "flow_grpo/audio_video_align/synchformer/synchformer.py"
    config_path = module_path.parent / "divided_224_16x4.yaml"
    if not module_path.is_file() or not config_path.is_file():
        raise ValueError("DeSync source_root is missing the OmniNFT Synchformer source or fixed config.")
    module = _import_synchformer(root, module_path)
    if Path(module.__file__).resolve() != module_path:
        raise RuntimeError("Imported Synchformer does not belong to the configured source_root.")

    model = module.Synchformer()
    state_dict = _load_torch_state_dict(model_path)
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


def _load_state(
    model_path: str,
    source_root: str,
    model_revision: str = _DEFAULT_MODEL_REVISION,
    source_revision: str = _DEFAULT_SOURCE_REVISION,
) -> _DeSyncNativeState:
    """Load local Synchformer assets and retain configured revision metadata."""
    for name, value in (
        ("model_path", model_path),
        ("source_root", source_root),
        ("model_revision", model_revision),
        ("source_revision", source_revision),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"DeSync Native Reward requires a non-empty {name}.")
    model, mel = _load_components(model_path, source_root)
    return _DeSyncNativeState(model, mel, model_revision, source_revision)


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
    """Validate paired media/rates and prepare CPU video/audio plus source rates.

    Video is resampled to 25 fps, resized and center-cropped to 224 square,
    scaled around [-1, 1], and padded/truncated to 200 frames. Audio is averaged
    across channels, resampled to 16 kHz, and padded/truncated to 128000 samples.
    """
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


def _infer_micro_batch(state: _DeSyncNativeState, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
    """Expand each prepared AV sample into 24 overlapping aligned segments.

    Extract video and normalized log-mel features on the active device, then
    compare the first and last 14 segments separately. Return detached CPU
    logits ``[2, B, 21]`` in model-output dtype; the caller sets inference mode
    and the MHA compatibility switch.
    """
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

    logits_batches = []
    for start in (0, _SEGMENTS - _COMPARE_SEGMENTS):
        logits = state.model.compare_v_a(
            visual[:, start : start + _COMPARE_SEGMENTS], auditory[:, start : start + _COMPARE_SEGMENTS]
        )
        if not isinstance(logits, torch.Tensor) or logits.shape != (batch_size, 21):
            raise ValueError(f"DeSync logits must have shape ({batch_size}, 21).")
        if not torch.isfinite(logits).all():
            raise ValueError("DeSync logits must contain only finite values.")
        logits_batches.append(logits.detach().cpu())
    return torch.stack(logits_batches)


async def compute_score(batch, reward_model, **kwargs) -> dict[str, float]:
    """Score audiovisual synchrony from two Synchformer offset predictions.

    Args:
        batch: Single-sample batch of uint8 RGB ``responses[B, T, 3, H, W]``, finite
            floating ``audio[B, C, S]``, positive floating ``fps[B]``, and
            positive integer ``audio_sample_rate[B]``. No text is used. Media
            are prepared on CPU as 8 s of 25 fps video and 16 kHz mono audio.
        reward_model: Active executor returning offset logits ``[2, M, 21]``.
        **kwargs: Ignored scorer options.

    Returns:
        A scalar ``score`` in a dict. Each comparison selects its argmax on
        the 21-class [-2, 2] second grid; reward is ``1 / (1 + mean(abs(offset)))``.
        Higher is better, reaching 1 when both predicted offsets are zero.

    Raises:
        ValueError: Invalid inputs, logits shape, or scores.
    """
    del kwargs
    if len(batch) != 1:
        raise ValueError("compute_score requires exactly one sample.")
    if "audio_sample_rate" not in batch.batch and "audio_sample_rate" in batch.non_tensor_batch:
        batch.batch["audio_sample_rate"] = torch.as_tensor(batch.non_tensor_batch["audio_sample_rate"].tolist())
    videos, audio, _, _ = _extract_inputs(batch)
    logits = await reward_model.infer(torch.stack(videos), torch.stack(audio))
    if not isinstance(logits, torch.Tensor) or logits.shape != (2, 1, 21):
        raise ValueError("DeSync logits must have shape (2, 1, 21).")
    if not torch.isfinite(logits).all():
        raise ValueError("DeSync logits must contain only finite values.")
    offsets = _CLASS_GRID[logits.argmax(dim=-1)].abs()
    distance = offsets.mean(dim=0)
    scores = (1.0 / (1.0 + distance)).float()
    if scores.shape != (len(batch),) or not torch.isfinite(scores).all():
        raise ValueError("DeSync scores must be finite and sample-aligned.")
    return {"score": float(scores[0])}


class DeSyncNativeModel:
    """Raw Synchformer inference adapter owned by a native reward executor."""

    def __init__(self, model_path: str, device, **kwargs: Any) -> None:
        self._state = _load_state(model_path=model_path, **kwargs)
        self._state.device = torch.device(device)
        self._state.model.to(self._state.device).eval()
        self._state.mel.to(self._state.device)

    def close(self) -> None:
        """Drop model/mel references; reuse requires constructing a new adapter."""
        self._state.model = None
        self._state.mel = None
        self._state.device = None

    @torch.inference_mode()
    def infer(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Infer prepared video ``[B, 200, 3, 224, 224]`` and audio ``[B, 128000]``.

        Return detached CPU offset logits ``[2, B, 21]`` without gradients.
        Disable the process-wide MHA fastpath for the forward because the
        Synchformer attention path is unsupported on the target NPU. A module
        lock serializes these calls and ``finally`` restores the previous flag;
        unrelated callers that do not use this lock can observe the temporary flag.
        """
        with _MHA_FASTPATH_LOCK:
            fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            try:
                return _infer_micro_batch(self._state, video, audio)
            finally:
                torch.backends.mha.set_fastpath_enabled(fastpath_enabled)


def _load_torch_state_dict(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as exc:
        if "mmap can only be used with files saved with" not in str(exc):
            raise
        return torch.load(path, map_location="cpu", weights_only=True)
