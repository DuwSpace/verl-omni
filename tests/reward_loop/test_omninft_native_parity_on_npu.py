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

"""Ascend NPU parity test: five native rewards vs the OmniNFT reference implementation.

Scores the same decoded production sample with the batch-native scorers and the
OmniNFT reference code (fb9237f6e74edf0d0f2a683f4d975b79fde588fe) using the same
checkpoint files on the same device. Media-decoding gaps in the frozen container
(missing torio/cv2/decord/torchcodec) are bridged with in-test shims.
DeSync uses PyAV's ffmpeg ``fps`` filter as the torio ``StreamingMediaDecoder`` stand-in.
"""

import importlib
import json
import math
import os
import sys
import types

import numpy as np
import pytest
import soundfile as sf
import torch
import torchvision
from PIL import Image
from verl import DataProto

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Ascend NPU is required",
)

_DEFAULT_MEDIA = "/repo/outputs/omnift_rollout_prod/0.mp4"
_DEFAULT_REPLAY = "/repo/outputs/omnift_rollout_prod/replay_g8_81f.pkl"
_DEFAULT_REFERENCE = "/tmp/OmniNFT-reference"
_DEFAULT_CLAP = "/hub/omnift-rewards/checkpoints/clap-htsat-unfused"
_DEFAULT_AUDIOBOX = "/hub/omnift-rewards/audiobox-aesthetics"
_DEFAULT_HPSV3 = "/hub/omnift-rewards/HPSv3/HPSv3.safetensors"
_DEFAULT_HPSV3_BASE = "/hub/omnift-rewards/Qwen2-VL-7B-Instruct"
_DEFAULT_VIDEOALIGN = "/hub/omnift-rewards/VideoReward/checkpoint-11352/model.pth"
_DEFAULT_VIDEOALIGN_BASE = "/hub/omnift-rewards/Qwen2-VL-2B-Instruct"
_DEFAULT_DESYNC = "/hub/omnift-rewards/synchformer/synchformer_state_dict.pth"

_SOURCE_REVISION = "fb9237f6e74edf0d0f2a683f4d975b79fde588fe"
_MODEL_REVISION = "zghhui/OmniNFT-Reward-Series@9e30061a1392d03bafdcf717e80a385ddf411b4d"


def _env(name, default):
    return os.environ.get(name, default)


def _require(path, label):
    if not os.path.exists(path):
        pytest.skip(f"{label} not found: {path}")
    return path


def _clear_npu_cache():
    torch.npu.synchronize()
    torch.npu.empty_cache()
    torch.npu.synchronize()


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    media_path = _require(_env("OMNIFT_PARITY_MEDIA", _DEFAULT_MEDIA), "parity media")
    replay_path = _require(_env("OMNIFT_PARITY_REPLAY", _DEFAULT_REPLAY), "parity replay")
    reference_root = _require(_env("OMNIFT_PARITY_REFERENCE", _DEFAULT_REFERENCE), "OmniNFT reference")

    torch.npu.set_device(0)
    frames, audio, info = torchvision.io.read_video(media_path, pts_unit="sec")
    fps = float(info["video_fps"])
    sample_rate = int(info["audio_fps"])

    replay = DataProto.load_from_disk(replay_path)
    reward_input = replay.non_tensor_batch["reward_inputs"][0]
    prompt_video = reward_input["text"]["video"]
    prompt_audio = reward_input["text"]["audio"]
    assert isinstance(prompt_video, str) and prompt_video.strip()
    assert isinstance(prompt_audio, str) and prompt_audio.strip()

    wav_path = os.path.splitext(media_path)[0] + ".wav"
    sf.write(wav_path, audio.numpy().T, sample_rate)

    _install_reference_runtime(reference_root)

    yield {
        "media_path": media_path,
        "wav_path": wav_path,
        "frames": frames,
        "audio": audio,
        "fps": fps,
        "sample_rate": sample_rate,
        "prompt_video": prompt_video,
        "prompt_audio": prompt_audio,
        "reference_root": reference_root,
        "work_dir": str(tmp_path_factory.mktemp("omnift_parity")),
    }

    if os.path.exists(wav_path):
        os.remove(wav_path)
    _clear_npu_cache()


@pytest.fixture(scope="module")
def native_batch(media):
    reward_inputs = np.empty(1, dtype=object)
    reward_inputs[:] = [{"text": {"video": media["prompt_video"], "audio": media["prompt_audio"]}}]
    responses = media["frames"].permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    return DataProto.from_dict(
        tensors={
            "responses": responses,
            "audio": media["audio"].unsqueeze(0).to(torch.float32),
            "fps": torch.tensor([media["fps"]], dtype=torch.float32),
            "audio_sample_rate": torch.tensor([media["sample_rate"]], dtype=torch.int64),
        },
        non_tensors={"reward_inputs": reward_inputs},
    )


def _run_native_scorer(module_name, init_kwargs, batch):
    module = importlib.import_module(module_name)
    state = module.initialize(**init_kwargs)
    try:
        module.activate(state, "npu:0")
        output = module.score_batch(state, batch, 1)
    finally:
        module.deactivate(state)
        module.finalize(state)
    _clear_npu_cache()
    score = float(output["scores"][0].item())
    assert math.isfinite(score)
    return score, output.get("metrics", {})


def _install_reference_runtime(reference_root):
    os.environ.setdefault("CLAP_CKPT", _env("CLAP_MODEL_PATH", _DEFAULT_CLAP))
    os.environ.setdefault(
        "AUDIOBOX_CKPT", os.path.join(_env("AUDIOBOX_MODEL_PATH", _DEFAULT_AUDIOBOX), "checkpoint.pt")
    )
    os.environ.setdefault("SYNCHFORMER_CKPT", _env("DESYNC_MODEL_PATH", _DEFAULT_DESYNC))
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")

    _install_torchaudio_load_shim()
    _install_torio_shim()
    _stub_missing_top_level_modules()
    _install_processor_kwarg_shim()
    _install_transformers_trainer_aliases()
    _install_qwen2vl_pretrained_key_remap()

    for entry in (
        reference_root,
        os.path.join(reference_root, "flow_grpo", "HPSv3"),
        os.path.join(reference_root, "flow_grpo", "videoalign"),
    ):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    from verl_omni.utils.reward_score.desync_native import _ensure_transformers_legacy_api

    _ensure_transformers_legacy_api()


def _install_torchaudio_load_shim():
    import torchaudio

    if getattr(torchaudio.load, "_omnift_parity_shim", False):
        return

    def _soundfile_load(filepath, *args, **kwargs):
        data, sample_rate = sf.read(filepath, dtype="float32", always_2d=True)
        return torch.from_numpy(np.ascontiguousarray(data.T)), sample_rate

    _soundfile_load._omnift_parity_shim = True
    torchaudio.load = _soundfile_load


def _install_torio_shim():
    try:
        importlib.import_module("torio.io")
        return
    except ImportError:
        pass

    av = pytest.importorskip("av")

    class _StreamingMediaDecoder:
        """FFmpeg fps-filter stand-in for torio.io.StreamingMediaDecoder."""

        def __init__(self, src):
            self._src = src
            self._frames_per_chunk = None
            self._frame_rate = None
            self._chunks = None

        def add_basic_video_stream(self, frames_per_chunk, frame_rate, format=None):
            self._frames_per_chunk = frames_per_chunk
            self._frame_rate = frame_rate

        def fill_buffer(self):
            container = av.open(self._src)
            try:
                stream = container.streams.video[0]
                graph = av.filter.Graph()
                src = graph.add_buffer(template=stream)
                fps = graph.add("fps", f"fps={self._frame_rate}")
                sink = graph.add("buffersink")
                src.link_to(fps)
                fps.link_to(sink)
                graph.configure()
                frames = []
                limit = int(self._frames_per_chunk)
                for frame in container.decode(video=0):
                    graph.push(frame)
                    while len(frames) < limit:
                        try:
                            frames.append(graph.pull())
                        except av.error.BlockingIOError:
                            break
                    if len(frames) >= limit:
                        break
                graph.push(None)
                while len(frames) < limit:
                    try:
                        frames.append(graph.pull())
                    except (av.error.BlockingIOError, av.error.EOFError):
                        break
            finally:
                container.close()
            if not frames:
                raise RuntimeError(f"PyAV produced no video frames from {self._src}")
            chunk = torch.stack([torch.from_numpy(frame.to_ndarray(format="rgb24").copy()) for frame in frames])
            self._chunks = [chunk.permute(0, 3, 1, 2).contiguous()]

        def pop_chunks(self):
            chunks, self._chunks = self._chunks, None
            return chunks

    torio = types.ModuleType("torio")
    torio_io = types.ModuleType("torio.io")
    torio_io.StreamingMediaDecoder = _StreamingMediaDecoder
    torio.io = torio_io
    sys.modules.setdefault("torio", torio)
    sys.modules.setdefault("torio.io", torio_io)


def _stub_missing_top_level_modules():
    trl = types.ModuleType("trl")
    trl.get_kbit_device_map = lambda *args, **kwargs: {}
    trl.get_quantization_config = lambda *args, **kwargs: None
    trl.RewardTrainer = object
    sys.modules.setdefault("trl", trl)
    sys.modules.setdefault("fire", types.ModuleType("fire"))

    def _stub_if_missing(name, attributes=()):
        try:
            importlib.import_module(name)
            return
        except ImportError:
            pass
        module = types.ModuleType(name)
        for attribute in attributes:
            setattr(module, attribute, None)
        sys.modules.setdefault(name, module)

    _stub_if_missing("librosa")
    _stub_if_missing("cv2")
    _stub_if_missing("decord", ("VideoReader", "cpu"))
    try:
        importlib.import_module("matplotlib.pyplot")
        importlib.import_module("matplotlib.patches")
    except ImportError:
        matplotlib = sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
        for child in ("pyplot", "patches"):
            if f"matplotlib.{child}" not in sys.modules:
                child_module = types.ModuleType(f"matplotlib.{child}")
                sys.modules[f"matplotlib.{child}"] = child_module
                setattr(matplotlib, child, child_module)
    try:
        importlib.import_module("sklearn.metrics.pairwise")
    except ImportError:
        for name in ("sklearn", "sklearn.metrics", "sklearn.metrics.pairwise"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["sklearn"].metrics = sys.modules["sklearn.metrics"]
        sys.modules["sklearn.metrics"].pairwise = sys.modules["sklearn.metrics.pairwise"]
        sys.modules["sklearn.metrics.pairwise"].polynomial_kernel = lambda *args, **kwargs: None


def _install_processor_kwarg_shim():
    """Rename the pre-transformers-5 ``audios`` kwarg to ``audio`` for reference calls."""
    from transformers import ClapProcessor

    if getattr(ClapProcessor.__call__, "_omnift_parity_shim", False):
        return

    original_call = ClapProcessor.__call__

    def _call(self, *args, **kwargs):
        if "audios" in kwargs and "audio" not in kwargs:
            kwargs["audio"] = kwargs.pop("audios")
        return original_call(self, *args, **kwargs)

    _call._omnift_parity_shim = True
    ClapProcessor.__call__ = _call


def _install_transformers_trainer_aliases():
    """Restore transformers symbols that v5 moved or removed for reference imports."""
    import transformers.trainer as trainer_module
    import transformers.trainer_pt_utils as pt_utils

    if not hasattr(trainer_module, "nested_concat"):
        trainer_module.nested_concat = pt_utils.nested_concat
    for name in ("DistributedTensorGatherer", "SequentialDistributedSampler"):
        if not hasattr(trainer_module, name):
            setattr(trainer_module, name, type(name, (), {}))

    import transformers.image_utils as image_utils
    import transformers.video_utils as video_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = video_utils.VideoInput

    from transformers import Qwen2VLConfig

    if "__getattr__" not in Qwen2VLConfig.__dict__:

        def _config_getattr(self, name):
            text_config = self.__dict__.get("text_config")
            if text_config is not None and hasattr(text_config, name):
                return getattr(text_config, name)
            raise AttributeError(f"{type(self).__name__}!r object has no attribute {name!r}")

        Qwen2VLConfig.__getattr__ = _config_getattr


def _install_qwen2vl_pretrained_key_remap():
    """Map Transformers-4 Qwen2-VL keys onto the v5 `language_model` layout."""
    from transformers.modeling_utils import PreTrainedModel
    from transformers.modeling_utils import load_state_dict as hf_load_state_dict

    from verl_omni.utils.reward_score.hpsv3_reward import _remap_state_dict

    original = PreTrainedModel._load_pretrained_model
    if getattr(original, "_omnift_parity_shim", False):
        return

    @staticmethod
    def _load_pretrained_model(model, state_dict, checkpoint_files, load_config, expected_keys=None):
        expected = list(model.state_dict().keys()) if expected_keys is None else expected_keys
        if state_dict is None and checkpoint_files:
            merged = {}
            for checkpoint_file in checkpoint_files:
                merged.update(
                    hf_load_state_dict(
                        checkpoint_file,
                        map_location="cpu",
                        weights_only=getattr(load_config, "weights_only", True),
                        disable_mmap=getattr(load_config, "disable_mmap", False),
                    )
                )
            state_dict = merged
            checkpoint_files = None
        if state_dict is not None:
            state_dict = _remap_state_dict(state_dict, expected)
        return original(model, state_dict, checkpoint_files, load_config, expected_keys=expected)

    _load_pretrained_model._omnift_parity_shim = True
    PreTrainedModel._load_pretrained_model = _load_pretrained_model


def _local_hpsv3_config(media, base_model_path):
    reference_root = media["reference_root"]
    source = os.path.join(reference_root, "flow_grpo", "HPSv3", "hpsv3", "config", "HPSv3_7B.yaml")
    with open(source, encoding="utf-8") as handle:
        lines = handle.readlines()
    target = os.path.join(media["work_dir"], "HPSv3_7B_local.yaml")
    with open(target, "w", encoding="utf-8") as handle:
        for line in lines:
            if line.startswith("model_name_or_path:"):
                handle.write(f'model_name_or_path: "{base_model_path}"\n')
            else:
                handle.write(line)
    return target


def _local_videoalign_checkpoint(media, videoalign_path, base_model_path):
    source_config = os.path.join(videoalign_path, "model_config.json")
    with open(source_config, encoding="utf-8") as handle:
        config = json.load(handle)
    config["model_config"]["model_name_or_path"] = base_model_path

    local_root = os.path.join(media["work_dir"], "videoalign_local")
    os.makedirs(local_root, exist_ok=True)
    for entry in os.listdir(videoalign_path):
        if entry.startswith("checkpoint-"):
            os.symlink(os.path.join(videoalign_path, entry), os.path.join(local_root, entry))
    with open(os.path.join(local_root, "model_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle)
    return local_root


def test_clap_matches_omninft_reference(media, native_batch):
    score, _ = _run_native_scorer(
        "verl_omni.utils.reward_score.clap_native",
        {"model_path": _require(_env("CLAP_MODEL_PATH", _DEFAULT_CLAP), "CLAP checkpoint")},
        native_batch,
    )

    from flow_grpo.rewards import clap_score

    reference_fn = clap_score("npu:0")
    reference_scores, _ = reference_fn([media["media_path"]], None, [{"prompt_a": media["prompt_audio"]}])
    _clear_npu_cache()
    reference = float(reference_scores[0])

    print(f"[clap] native={score:.6f} reference={reference:.6f} diff={abs(score - reference):.3e}")
    assert abs(score - reference) <= 1e-3


def test_audiobox_matches_omninft_reference(media, native_batch):
    pytest.importorskip("audiobox_aesthetics")
    audiobox_path = _require(_env("AUDIOBOX_MODEL_PATH", _DEFAULT_AUDIOBOX), "AudioBox checkpoint")

    score, _ = _run_native_scorer(
        "verl_omni.utils.reward_score.audiobox_native", {"model_path": audiobox_path}, native_batch
    )

    from flow_grpo.rewards import audiobox_aesthetics_score

    reference_fn = audiobox_aesthetics_score("npu:0")
    reference_scores, _ = reference_fn([media["media_path"]], None, [{"prompt_a": media["prompt_audio"]}])
    _clear_npu_cache()
    reference = float(reference_scores[0])

    print(f"[audiobox] native={score:.6f} reference={reference:.6f} diff={abs(score - reference):.3e}")
    assert abs(score - reference) <= 1e-3


def test_hpsv3_matches_omninft_reference(media, native_batch):
    checkpoint = _require(_env("HPSV3_MODEL_PATH", _DEFAULT_HPSV3), "HPSv3 checkpoint")
    base_model = _require(_env("HPSV3_BASE_MODEL_PATH", _DEFAULT_HPSV3_BASE), "HPSv3 base model")

    score, _ = _run_native_scorer(
        "verl_omni.utils.reward_score.hpsv3_native",
        {"model_path": checkpoint, "base_model_path": base_model},
        native_batch,
    )

    from hpsv3.inference import HPSv3RewardInferencer
    from hpsv3.model.qwen2vl_trainer import Qwen2VLRewardModelBT as ReferenceHPSv3Model

    original_init = ReferenceHPSv3Model.__init__

    def _init(self, config, *args, **kwargs):
        kwargs.pop("use_cache", None)
        original_init(self, config, *args, **kwargs)

    if not getattr(ReferenceHPSv3Model.__init__, "_omnift_parity_shim", False):
        _init._omnift_parity_shim = True
        ReferenceHPSv3Model.__init__ = _init

    from verl_omni.utils.reward_score.hpsv3_reward import _remap_state_dict

    original_load_state_dict = ReferenceHPSv3Model.load_state_dict

    def _load_state_dict(self, state_dict, strict=True):
        remapped = _remap_state_dict(state_dict, list(self.state_dict().keys()))
        return original_load_state_dict(self, remapped, strict=strict)

    if not getattr(ReferenceHPSv3Model.load_state_dict, "_omnift_parity_shim", False):
        _load_state_dict._omnift_parity_shim = True
        ReferenceHPSv3Model.load_state_dict = _load_state_dict

    from verl_omni.utils.reward_score.hpsv3_native import ensure_omninft_qwen2vl_layout

    original_forward = ReferenceHPSv3Model.forward

    def _forward(self, *args, **kwargs):
        kwargs.pop("mm_token_type_ids", None)
        return original_forward(self, *args, **kwargs)

    if not getattr(ReferenceHPSv3Model.forward, "_omnift_parity_shim", False):
        _forward._omnift_parity_shim = True
        ReferenceHPSv3Model.forward = _forward
    inferencer = HPSv3RewardInferencer(
        config_path=_local_hpsv3_config(media, base_model),
        checkpoint_path=checkpoint,
        device="npu:0",
    )
    ensure_omninft_qwen2vl_layout(inferencer.model)
    frame_ids = [round(i * (media["frames"].shape[0] - 1) / 4) for i in range(5)]
    images = [Image.fromarray(media["frames"][frame_id].numpy()) for frame_id in frame_ids]
    with torch.inference_mode():
        rewards = inferencer.reward([media["prompt_video"]] * len(images), images)
    del inferencer
    _clear_npu_cache()
    frame_scores = [min(float(reward[0].item()), 15.0) for reward in rewards]
    sorted_scores = sorted(frame_scores, reverse=True)
    top_k = max(1, math.ceil(len(sorted_scores) * 0.3))
    reference = sum(sorted_scores[:top_k]) / top_k

    print(f"[hpsv3] native={score:.6f} reference={reference:.6f} diff={abs(score - reference):.3e}")
    assert abs(score - reference) <= 1e-3


def test_videoalign_matches_omninft_reference(media, native_batch):
    checkpoint = _require(_env("VIDEOALIGN_MODEL_PATH", _DEFAULT_VIDEOALIGN), "VideoReward checkpoint")
    base_model = _require(_env("VIDEOALIGN_BASE_MODEL_PATH", _DEFAULT_VIDEOALIGN_BASE), "VideoReward base model")

    score, _ = _run_native_scorer(
        "verl_omni.utils.reward_score.videoalign_native",
        {"model_path": checkpoint, "base_model_path": base_model},
        native_batch,
    )

    from inference import VideoVLMRewardInference
    from torch import nn
    from trainer import Qwen2VLRewardModelBT as VideoAlignRewardModel

    from verl_omni.utils.reward_score.hpsv3_native import ensure_omninft_qwen2vl_layout
    from verl_omni.utils.reward_score.hpsv3_reward import _remap_state_dict
    from verl_omni.utils.reward_score.videoalign_native import _remap_checkpoint_state_dict

    original_forward = VideoAlignRewardModel.forward

    def _forward(self, *args, **kwargs):
        kwargs.pop("mm_token_type_ids", None)
        return original_forward(self, *args, **kwargs)

    if not getattr(VideoAlignRewardModel.forward, "_omnift_parity_shim", False):
        _forward._omnift_parity_shim = True
        VideoAlignRewardModel.forward = _forward
    original_load_state_dict = nn.Module.load_state_dict

    def _load_state_dict(self, state_dict, *args, **kwargs):
        if state_dict and any(key.startswith("base_model.model.visual.") for key in state_dict):
            state_dict = _remap_checkpoint_state_dict(state_dict)
        elif state_dict:
            state_dict = _remap_state_dict(state_dict, list(self.state_dict().keys()))
        return original_load_state_dict(self, state_dict, *args, **kwargs)

    nn.Module.load_state_dict = _load_state_dict

    checkpoint_root = os.path.dirname(os.path.dirname(checkpoint))
    local_root = _local_videoalign_checkpoint(media, checkpoint_root, base_model)
    try:
        inferencer = VideoVLMRewardInference(local_root, device="npu:0", dtype=torch.bfloat16)
    finally:
        nn.Module.load_state_dict = original_load_state_dict
    ensure_omninft_qwen2vl_layout(inferencer.model)
    with torch.inference_mode():
        reward = inferencer.reward([media["media_path"]], [media["prompt_video"]], fps=24.0, use_norm=True)[0]
    del inferencer
    _clear_npu_cache()
    reference = (reward["VQ"] + reward["TA"]) / 2

    print(f"[videoalign] native={score:.6f} reference={reference:.6f} diff={abs(score - reference):.3e}")
    assert abs(score - reference) <= 2e-2


def test_desync_reference_recorded(media, native_batch):
    pytest.importorskip("timm")
    checkpoint = _require(_env("DESYNC_MODEL_PATH", _DEFAULT_DESYNC), "DeSync checkpoint")

    score, _ = _run_native_scorer(
        "verl_omni.utils.reward_score.desync_native",
        {
            "model_path": checkpoint,
            "source_root": media["reference_root"],
            "model_revision": _MODEL_REVISION,
            "source_revision": _SOURCE_REVISION,
        },
        native_batch,
    )

    from flow_grpo.audio_video_align.av_desync import av_desync_reward

    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    import torchaudio

    original_spectrogram_init = torchaudio.transforms.Spectrogram.__init__

    def _spectrogram_init_cpu_window(self, *args, **kwargs):
        wkwargs = dict(kwargs.get("wkwargs") or {})
        device = wkwargs.pop("device", None)
        if device is not None:
            original_window_fn = kwargs.get("window_fn", torch.hann_window)

            def _window_fn(window_length, *window_args, **window_kwargs):
                window = original_window_fn(window_length, *window_args, device="cpu", **window_kwargs)
                return window.to(device)

            kwargs["window_fn"] = _window_fn
            kwargs["wkwargs"] = wkwargs
        return original_spectrogram_init(self, *args, **kwargs)

    torch.backends.mha.set_fastpath_enabled(False)
    torchaudio.transforms.Spectrogram.__init__ = _spectrogram_init_cpu_window
    try:
        reference_fn = av_desync_reward(device="npu:0")
        reference_scores, _ = reference_fn([media["media_path"]], None, None)
    finally:
        torchaudio.transforms.Spectrogram.__init__ = original_spectrogram_init
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    _clear_npu_cache()
    reference = float(reference_scores[0])

    frame_count = int(media["frames"].shape[0])
    padded_segments = 24
    reference_windows = math.ceil(min(padded_segments, padded_segments) / 14)
    print(
        f"[desync] native={score:.6f} reference={reference:.6f} diff={abs(score - reference):.3e} "
        f"frames={frame_count} reference_windows={reference_windows}"
    )
    assert 1 / 3 <= score <= 1
    assert 1 / 3 <= reference <= 1
    assert abs(score - reference) <= 1e-3
