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

"""CPU contracts for the batch-native OmniNFT VideoAlign reward."""

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_omni.utils.reward_score import videoalign_native


class _FakeProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return [message[0]["content"][1]["text"] for message in messages]

    def __call__(self, *, text, videos, **kwargs):
        self.calls.append({"text": list(text), "videos": list(videos), **kwargs})
        values = torch.tensor([float(video[0, 0, 0, 0]) for video in videos])
        return {"video_values": values}


class _FakeModel:
    def __init__(self):
        self.forward_batch_sizes = []
        self.to_devices = []
        self.return_nan = False
        self.bad_shape = False

    def eval(self):
        return self

    def to(self, device):
        self.to_devices.append(torch.device(device))
        return self

    def __call__(self, *, video_values, return_dict):
        assert return_dict is True
        self.forward_batch_sizes.append(video_values.shape[0])
        if self.return_nan:
            return {"logits": torch.full((video_values.shape[0], 3), torch.nan)}
        if self.bad_shape:
            return {"logits": torch.zeros(video_values.shape[0], 2)}
        values = video_values.float()
        return {"logits": torch.stack((values, values + 100, values * 2), dim=1)}


def _make_batch(batch_size=4, frames=9, dtype=torch.uint8, fps=None, height=4, width=5):
    videos = torch.empty(batch_size, frames, 3, height, width, dtype=dtype)
    for sample_index in range(batch_size):
        for frame_index in range(frames):
            value = sample_index * 10 + frame_index
            if dtype.is_floating_point:
                value /= 255
            videos[sample_index, frame_index].fill_(value)
    if fps is None:
        fps = [24.0] * batch_size
    reward_inputs = np.empty(batch_size, dtype=object)
    reward_inputs[:] = [{"text": {"video": f"video prompt {index}"}} for index in range(batch_size)]
    return DataProto.from_dict(
        tensors={"responses": videos, "fps": torch.tensor(fps, dtype=torch.float32)},
        non_tensors={"reward_inputs": reward_inputs},
    )


def _initialize(monkeypatch):
    model = _FakeModel()
    processor = _FakeProcessor()
    monkeypatch.setattr(videoalign_native, "_load_components", lambda model_path, base_model_path: (model, processor))
    state = videoalign_native.initialize("/models/model.pth", "/models/Qwen2-VL-2B-Instruct")
    return state, model, processor


def test_local_batch_uses_fps_sampling_and_micro_batches(monkeypatch):
    state, model, processor = _initialize(monkeypatch)
    videoalign_native.activate(state, "cpu")

    result = videoalign_native.score_batch(state, _make_batch(fps=[24.0, 48.0, 12.0, 24.0]), micro_batch_size=2)

    assert model.forward_batch_sizes == [2, 2]
    assert [len(call["videos"]) for call in processor.calls] == [2, 2]
    assert all(
        video.dtype == torch.float32 and video.shape[1] == 3
        for call in processor.calls
        for video in call["videos"]
    )
    assert processor.calls[0]["videos"][0].shape[0] == 8
    assert processor.calls[0]["videos"][1].shape[0] == 4
    assert processor.calls[0]["videos"][0][:, 0, 0, 0].tolist() == [0, 1, 2, 3, 5, 6, 7, 8]
    assert processor.calls[0]["videos"][1][:, 0, 0, 0].tolist() == [10, 13, 15, 18]
    assert all(token in processor.calls[0]["text"][0] for token in videoalign_native._SPECIAL_TOKENS)
    expected = torch.tensor(
        [
            (
                (0 - videoalign_native._VQ_MEAN) / videoalign_native._VQ_STD
                + (0 - videoalign_native._TA_MEAN) / videoalign_native._TA_STD
            )
            / 2,
            (
                (10 - videoalign_native._VQ_MEAN) / videoalign_native._VQ_STD
                + (20 - videoalign_native._TA_MEAN) / videoalign_native._TA_STD
            )
            / 2,
            (
                (20 - videoalign_native._VQ_MEAN) / videoalign_native._VQ_STD
                + (40 - videoalign_native._TA_MEAN) / videoalign_native._TA_STD
            )
            / 2,
            (
                (30 - videoalign_native._VQ_MEAN) / videoalign_native._VQ_STD
                + (60 - videoalign_native._TA_MEAN) / videoalign_native._TA_STD
            )
            / 2,
        ]
    )
    torch.testing.assert_close(result["scores"], expected)
    assert result["metrics"]["forward_calls"] == 2
    assert result["metrics"]["frames_per_sample"] == [8, 4, 8, 8]


def test_float_video_reaches_processor_as_upstream_float_pixels(monkeypatch):
    state, _, processor = _initialize(monkeypatch)
    videoalign_native.activate(state, "cpu")
    videoalign_native.score_batch(state, _make_batch(batch_size=1, dtype=torch.float32), micro_batch_size=1)
    video = processor.calls[0]["videos"][0]
    assert video.dtype == torch.float32
    assert video[:, 0, 0, 0].tolist() == [0, 1, 2, 3, 5, 6, 7, 8]
    assert processor.calls[0]["videos_kwargs"] == {"do_rescale": True}


def test_upstream_spatial_resize_reaches_processor(monkeypatch):
    state, _, processor = _initialize(monkeypatch)
    videoalign_native.activate(state, "cpu")

    result = videoalign_native.score_batch(
        state,
        _make_batch(batch_size=1, frames=121, height=256, width=384),
        micro_batch_size=1,
    )

    video = processor.calls[0]["videos"][0]
    assert video.shape == (120, 3, 280, 392)
    assert (video.shape[0] // 2, video.shape[2] // 14, video.shape[3] // 14) == (60, 20, 28)
    assert result["definition_version"] == "omninft-videoalign-vq-ta-v3"


def test_upstream_frame_cap_is_applied_before_processor():
    video = torch.zeros(1000, 3, 4, 5, dtype=torch.uint8)

    sampled, indices = videoalign_native._sample_video(video, source_fps=24.0)

    assert sampled.shape[0] == 768
    assert len(indices) == 768
    assert indices[0] == 0 and indices[-1] == 999


def test_upstream_minimum_frame_count_is_applied_when_video_is_long_enough():
    video = torch.zeros(10, 3, 4, 5, dtype=torch.uint8)

    sampled, indices = videoalign_native._sample_video(video, source_fps=1000.0)

    assert sampled.shape[0] == 4
    assert len(indices) == 4
    assert indices[0] == 0 and indices[-1] == 9


def test_lifecycle_is_isolated_and_finalized(monkeypatch):
    state, model, _ = _initialize(monkeypatch)
    videoalign_native.activate(state, "cpu")
    with pytest.raises(RuntimeError, match="already active"):
        videoalign_native.activate(state, "cpu")
    videoalign_native.deactivate(state)
    videoalign_native.deactivate(state)
    videoalign_native.finalize(state)
    assert model.to_devices == [torch.device("cpu"), torch.device("cpu")]
    assert state.model is None and state.processor is None
    with pytest.raises(RuntimeError, match="finalized"):
        videoalign_native.activate(state, "cpu")


@pytest.mark.parametrize(
    ("mutate", "match", "model_attr"),
    [
        (lambda batch: batch.batch.__setitem__("responses", torch.zeros(2, 3, 4, 5)), "shape", None),
        (lambda batch: batch.batch["fps"].fill_(0), "positive", None),
        (lambda batch: batch.non_tensor_batch["reward_inputs"][0].clear(), "text.video", None),
        (
            lambda batch: batch.batch.__setitem__(
                "responses", torch.full(batch.batch["responses"].shape, 2.0, dtype=torch.float32)
            ),
            "values",
            None,
        ),
        (lambda batch: None, "finite", "return_nan"),
        (lambda batch: None, "shape", "bad_shape"),
    ],
)
def test_invalid_inputs_fail_closed(monkeypatch, mutate, match, model_attr):
    state, model, _ = _initialize(monkeypatch)
    batch = _make_batch(batch_size=2)
    mutate(batch)
    if model_attr is not None:
        setattr(model, model_attr, True)
    videoalign_native.activate(state, "cpu")
    with pytest.raises(ValueError, match=match):
        videoalign_native.score_batch(state, batch, micro_batch_size=2)


def test_scoring_requires_active_state_and_positive_micro_batch(monkeypatch):
    state, _, _ = _initialize(monkeypatch)
    batch = _make_batch(batch_size=2)
    with pytest.raises(RuntimeError, match="active"):
        videoalign_native.score_batch(state, batch, micro_batch_size=2)
    videoalign_native.activate(state, "cpu")
    with pytest.raises(ValueError, match="positive integer"):
        videoalign_native.score_batch(state, batch, micro_batch_size=0)


def test_checkpoint_remap_matches_video_reward_prefixes():
    state_dict = {
        "base_model.model.visual.x": torch.ones(1),
        "base_model.model.model.layers.0.x": torch.ones(1),
        "base_model.model.rm_head.weight": torch.ones(1),
    }
    remapped = videoalign_native._remap_checkpoint_state_dict(state_dict)
    assert set(remapped) == {
        "base_model.model.model.visual.x",
        "base_model.model.model.language_model.layers.0.x",
        "base_model.model.rm_head.weight",
    }
