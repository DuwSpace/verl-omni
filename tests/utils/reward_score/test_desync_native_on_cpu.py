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

"""CPU contracts for the batch-native OmniNFT DeSync reward."""

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_omni.utils.reward_score import desync_native


class _FakeMel:
    def __init__(self):
        self.input_shapes = []
        self.to_devices = []

    def __call__(self, audio):
        self.input_shapes.append(tuple(audio.shape))
        return torch.ones(*audio.shape[:2], 128, 65, device=audio.device)

    def to(self, device):
        self.to_devices.append(torch.device(device))
        return self


class _FakeModel:
    def __init__(self):
        self.video_shapes = []
        self.audio_shapes = []
        self.compare_shapes = []
        self.to_devices = []
        self.compare_calls = 0
        self.return_nan = False
        self.bad_shape = False

    def eval(self):
        return self

    def to(self, device):
        self.to_devices.append(torch.device(device))
        return self

    def extract_vfeats(self, video):
        self.video_shapes.append(tuple(video.shape))
        return torch.zeros(video.shape[0], 1, 2, 4, device=video.device)

    def extract_afeats(self, audio):
        self.audio_shapes.append(tuple(audio.shape))
        return torch.zeros(audio.shape[0], audio.shape[1], 3, 4, device=audio.device)

    def compare_v_a(self, video, audio):
        self.compare_shapes.append((tuple(video.shape), tuple(audio.shape)))
        if self.bad_shape:
            return torch.zeros(video.shape[0], 20, device=video.device)
        logits = torch.full((video.shape[0], 21), -10.0, device=video.device)
        class_ids = (0, 5) if self.compare_calls % 2 == 0 else (10, 20)
        for index in range(video.shape[0]):
            logits[index, class_ids[index]] = 10.0
        self.compare_calls += 1
        return logits.fill_(torch.nan) if self.return_nan else logits


def _make_batch(batch_size=4, video_dtype=torch.uint8, audio_dtype=torch.float32, fps=None, rates=None):
    videos = torch.zeros(batch_size, 9, 3, 4, 6, dtype=video_dtype)
    audio = torch.zeros(batch_size, 2, 15_840, dtype=audio_dtype)
    fps = [24.0] * batch_size if fps is None else fps
    rates = [48_000] * batch_size if rates is None else rates
    sample_uids = np.asarray([f"sample-{index}" for index in range(batch_size)], dtype=object)
    return DataProto.from_dict(
        tensors={
            "responses": videos,
            "fps": torch.tensor(fps, dtype=torch.float32),
            "audio": audio,
            "audio_sample_rate": torch.tensor(rates, dtype=torch.int64),
        },
        non_tensors={"sample_uid": sample_uids},
    )


def _initialize(monkeypatch):
    model = _FakeModel()
    mel = _FakeMel()
    monkeypatch.setattr(desync_native, "_load_components", lambda model_path, source_root, revision: (model, mel))
    state = desync_native.initialize("/models/synchformer.pth", "/source/OmniNFT")
    return state, model, mel


def _stub_preprocessing(monkeypatch):
    monkeypatch.setattr(desync_native, "_prepare_video", lambda video, fps: torch.zeros(200, 3, 2, 2))
    monkeypatch.setattr(desync_native, "_prepare_audio", lambda audio, rate: torch.zeros(128_000))


def test_local_batch_uses_av_segments_micro_batches_and_offset_reward(monkeypatch):
    state, model, mel = _initialize(monkeypatch)
    _stub_preprocessing(monkeypatch)
    desync_native.activate(state, "cpu")

    result = desync_native.score_batch(
        state, _make_batch(fps=[24.0, 48.0, 12.0, 24.0], rates=[48_000, 16_000, 44_100, 48_000]), 2
    )

    torch.testing.assert_close(result["scores"], torch.tensor([0.5, 0.4, 0.5, 0.4]))
    assert result["valid_mask"].tolist() == [True] * 4
    assert model.video_shapes == [(48, 1, 16, 3, 2, 2)] * 2
    assert model.audio_shapes == [(2, 24, 1, 128, 66)] * 2
    assert mel.input_shapes == [(2, 24, 10_240)] * 2
    assert model.compare_shapes == [((2, 14, 2, 4), (2, 14, 3, 4))] * 4
    assert result["metrics"] == {
        "batch_size": 4,
        "micro_batch_size": 2,
        "video_forward_calls": 2,
        "audio_forward_calls": 2,
        "compare_forward_calls": 4,
        "segments_per_sample": 24,
        "source_fps": [12.0, 24.0, 48.0],
        "source_sample_rates": [16_000, 44_100, 48_000],
        "target_fps": 25.0,
        "target_sample_rate": 16_000,
        "source_revision": desync_native._DEFAULT_SOURCE_REVISION,
    }
    assert result["model_revision"] == desync_native._DEFAULT_MODEL_REVISION
    assert result["definition_version"] == "omninft-desync-synchformer-v3"


def test_video_resampling_resize_normalize_and_padding():
    frames = torch.arange(9, dtype=torch.uint8).view(9, 1, 1, 1).expand(-1, 3, 2, 2)
    sampled = desync_native._temporal_resample_video(frames, 48.0)
    assert sampled[:, 0, 0, 0].tolist() == [0, 2, 4, 6]

    src24 = torch.arange(79, dtype=torch.uint8).view(79, 1, 1, 1).expand(-1, 3, 2, 2)
    sampled24 = desync_native._temporal_resample_video(src24, 24.0)
    values24 = sampled24[:, 0, 0, 0].tolist()
    assert len(values24) == 82
    assert values24[12] == 11
    assert values24[37] == 35
    assert values24[62] == 59
    assert values24[-1] == 78

    constants = torch.stack((torch.zeros(3, 4, 8), torch.full((3, 4, 8), 255))).to(torch.uint8)
    resized = desync_native._resize_crop_video(constants)
    assert resized.shape == (2, 3, 224, 224)
    torch.testing.assert_close(resized[0], torch.full_like(resized[0], -1.0))
    torch.testing.assert_close(resized[1], torch.ones_like(resized[1]))

    prepared = desync_native._prepare_video(frames, 48.0)
    assert prepared.shape == (200, 3, 224, 224)
    assert torch.count_nonzero(prepared[4:].add(1)) == 0


def test_audio_downmix_resample_and_padding(monkeypatch):
    seen = {}

    def fake_resample(waveform, source_rate):
        seen["waveform"] = waveform
        seen["source_rate"] = source_rate
        return torch.tensor([2.0, 4.0, 6.0])

    monkeypatch.setattr(desync_native, "_resample_audio", fake_resample)
    prepared = desync_native._prepare_audio(torch.tensor([[1.0, 3.0], [3.0, 5.0]]), 48_000)
    torch.testing.assert_close(seen["waveform"], torch.tensor([2.0, 4.0]))
    assert seen["source_rate"] == 48_000
    assert prepared.shape == (128_000,)
    torch.testing.assert_close(prepared[:3], torch.tensor([2.0, 4.0, 6.0]))
    assert torch.count_nonzero(prepared[3:]) == 0


def test_lifecycle_is_isolated_and_finalized(monkeypatch):
    state, model, mel = _initialize(monkeypatch)
    original_fastpath = torch.backends.mha.get_fastpath_enabled()
    desync_native.activate(state, "cpu")
    assert torch.backends.mha.get_fastpath_enabled() is False
    with pytest.raises(RuntimeError, match="already active"):
        desync_native.activate(state, "cpu")
    desync_native.deactivate(state)
    assert torch.backends.mha.get_fastpath_enabled() is original_fastpath
    desync_native.deactivate(state)
    desync_native.finalize(state)
    assert model.to_devices == mel.to_devices == [torch.device("cpu"), torch.device("cpu")]
    assert state.model is None and state.mel is None
    with pytest.raises(RuntimeError, match="finalized"):
        desync_native.activate(state, "cpu")


@pytest.mark.parametrize(
    ("mutate", "match", "model_attr"),
    [
        (lambda batch: batch.batch.__setitem__("responses", torch.zeros(2, 9, 4, 6)), "shape", None),
        (lambda batch: batch.batch.__setitem__("responses", batch.batch["responses"].float()), "uint8", None),
        (lambda batch: batch.batch["fps"].fill_(0), "positive", None),
        (lambda batch: batch.batch["audio"].fill_(torch.nan), "finite", None),
        (lambda batch: batch.batch.__setitem__("audio", batch.batch["audio"].to(torch.int64)), "floating", None),
        (
            lambda batch: batch.batch.__setitem__("audio_sample_rate", batch.batch["audio_sample_rate"].float()),
            "integer",
            None,
        ),
        (lambda batch: None, "finite", "return_nan"),
        (lambda batch: None, "shape", "bad_shape"),
    ],
)
def test_invalid_inputs_fail_closed(monkeypatch, mutate, match, model_attr):
    state, model, _ = _initialize(monkeypatch)
    _stub_preprocessing(monkeypatch)
    batch = _make_batch(batch_size=2)
    mutate(batch)
    if model_attr is not None:
        setattr(model, model_attr, True)
    desync_native.activate(state, "cpu")
    with pytest.raises(ValueError, match=match):
        desync_native.score_batch(state, batch, 2)


def test_scoring_requires_active_state_and_positive_micro_batch(monkeypatch):
    state, _, _ = _initialize(monkeypatch)
    batch = _make_batch(batch_size=2)
    with pytest.raises(RuntimeError, match="active"):
        desync_native.score_batch(state, batch, 2)
    desync_native.activate(state, "cpu")
    with pytest.raises(ValueError, match="positive integer"):
        desync_native.score_batch(state, batch, 0)


def test_git_revision_supports_loose_and_detached_heads(tmp_path):
    git_dir = tmp_path / ".git"
    ref_dir = git_dir / "refs/heads"
    ref_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (ref_dir / "main").write_text("abc123\n", encoding="utf-8")
    assert desync_native._read_git_revision(tmp_path) == "abc123"
    (git_dir / "HEAD").write_text("def456\n", encoding="utf-8")
    assert desync_native._read_git_revision(tmp_path) == "def456"


def test_transformers_v5_compatibility_restores_legacy_pruning_helper(monkeypatch):
    from transformers import pytorch_utils
    from transformers.modeling_utils import PreTrainedModel

    monkeypatch.delattr(pytorch_utils, "find_pruneable_heads_and_indices", raising=False)
    monkeypatch.delattr(PreTrainedModel, "get_head_mask", raising=False)
    desync_native._ensure_transformers_legacy_api()
    heads, indices = pytorch_utils.find_pruneable_heads_and_indices({0}, 2, 2, set())
    assert heads == {0}
    assert indices.tolist() == [2, 3]
    assert PreTrainedModel.get_head_mask(None, None, 2) == [None, None]
