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

"""CPU contracts for the batch-native OmniNFT HPSv3 reward."""

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_omni.utils.reward_score import hpsv3_native


class _FakeProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        return [message[0]["content"][1]["text"] for message in messages]

    def __call__(self, *, text, images, **kwargs):
        values = [float(np.asarray(image)[0, 0, 0]) for image in images]
        self.calls.append({"text": list(text), "images": list(images), **kwargs})
        return {"frame_values": torch.tensor(values)}


class _FakeModel:
    def __init__(self):
        self.forward_batch_sizes = []
        self.to_devices = []
        self.return_nan = False

    def eval(self):
        return self

    def to(self, device):
        self.to_devices.append(torch.device(device))
        return self

    def __call__(self, *, frame_values, return_dict):
        assert return_dict is True
        self.forward_batch_sizes.append(frame_values.shape[0])
        if self.return_nan:
            return {"logits": torch.full((frame_values.shape[0], 2), torch.nan)}
        return {"logits": torch.stack((frame_values, -frame_values), dim=1)}


def _make_batch(batch_size=4, frames=9, dtype=torch.uint8, height=4, width=5):
    videos = torch.empty(batch_size, frames, 3, height, width, dtype=dtype)
    for sample_index in range(batch_size):
        for frame_index in range(frames):
            value = sample_index * 10 + frame_index
            if dtype.is_floating_point:
                value /= 255
            videos[sample_index, frame_index].fill_(value)
    reward_inputs = np.empty(batch_size, dtype=object)
    reward_inputs[:] = [{"text": {"video": f"video prompt {index}"}} for index in range(batch_size)]
    return DataProto.from_dict(
        tensors={"responses": videos},
        non_tensors={
            "reward_inputs": reward_inputs,
            "sample_uid": np.asarray([f"sample-{index}" for index in range(batch_size)], dtype=object),
        },
    )


def _initialize(monkeypatch):
    model = _FakeModel()
    processor = _FakeProcessor()
    monkeypatch.setattr(hpsv3_native, "_load_components", lambda model_path, base_model_path: (model, processor))
    state = hpsv3_native.initialize("/models/HPSv3.safetensors", "/models/Qwen2-VL-7B-Instruct")
    return state, model, processor


def test_local_batch_uses_five_frames_and_sample_micro_batches(monkeypatch):
    state, model, processor = _initialize(monkeypatch)
    hpsv3_native.activate(state, "cpu")

    result = hpsv3_native.score_batch(state, _make_batch(), micro_batch_size=2)

    assert model.forward_batch_sizes == [10, 10]
    assert [len(call["images"]) for call in processor.calls] == [10, 10]
    assert "Textual prompt - video prompt 0" in processor.calls[0]["text"][0]
    assert [np.asarray(image)[0, 0, 0] for image in processor.calls[0]["images"][:5]] == [0, 2, 4, 6, 8]
    torch.testing.assert_close(result["scores"], torch.tensor([7.0, 15.0, 15.0, 15.0]))
    assert result["valid_mask"].all()
    assert result["definition_version"] == "omninft-hpsv3-top30-v4"
    assert result["metrics"] == {
        "batch_size": 4,
        "micro_batch_size": 2,
        "forward_calls": 2,
        "frames_per_sample": 5,
        "top_frame_count": 2,
        "reward_cap": 15.0,
    }


def test_float_video_is_converted_once_to_rgb_uint8(monkeypatch):
    state, _, processor = _initialize(monkeypatch)
    hpsv3_native.activate(state, "cpu")

    hpsv3_native.score_batch(state, _make_batch(batch_size=1, dtype=torch.float32), micro_batch_size=1)

    images = processor.calls[0]["images"]
    assert all(image.mode == "RGB" for image in images)
    assert [np.asarray(image)[0, 0, 0] for image in images] == [0, 2, 4, 6, 8]


def test_upstream_forced_image_size_reaches_processor(monkeypatch):
    state, _, processor = _initialize(monkeypatch)
    hpsv3_native.activate(state, "cpu")

    hpsv3_native.score_batch(
        state,
        _make_batch(batch_size=1, frames=5, height=256, width=384),
        micro_batch_size=1,
    )

    images = processor.calls[0]["images"]
    assert {image.size for image in images} == {(560, 392)}
    assert {(image.height // 14, image.width // 14) for image in images} == {(28, 40)}


def test_lifecycle_is_isolated_and_idempotently_deactivated(monkeypatch):
    state, model, _ = _initialize(monkeypatch)
    hpsv3_native.activate(state, "cpu")
    with pytest.raises(RuntimeError, match="already active"):
        hpsv3_native.activate(state, "cpu")
    hpsv3_native.deactivate(state)
    hpsv3_native.deactivate(state)
    hpsv3_native.finalize(state)

    assert model.to_devices == [torch.device("cpu"), torch.device("cpu")]
    assert state.model is None
    assert state.processor is None
    with pytest.raises(RuntimeError, match="finalized"):
        hpsv3_native.activate(state, "cpu")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda batch: batch.batch["responses"].fill_(torch.iinfo(torch.uint8).max), "scores"),
        (lambda batch: batch.batch.__setitem__("responses", torch.zeros(2, 3, 4, 5)), "shape"),
        (lambda batch: batch.non_tensor_batch["reward_inputs"][0].clear(), "text.video"),
    ],
)
def test_invalid_local_batch_fails_closed(monkeypatch, mutate, match):
    state, model, _ = _initialize(monkeypatch)
    batch = _make_batch(batch_size=2)
    mutate(batch)
    if match == "scores":
        model.return_nan = True
    hpsv3_native.activate(state, "cpu")

    if match == "scores":
        with pytest.raises(ValueError, match="finite"):
            hpsv3_native.score_batch(state, batch, micro_batch_size=2)
    else:
        with pytest.raises(ValueError, match=match):
            hpsv3_native.score_batch(state, batch, micro_batch_size=2)


def test_scoring_requires_active_state_and_positive_micro_batch(monkeypatch):
    state, _, _ = _initialize(monkeypatch)
    batch = _make_batch(batch_size=2)
    with pytest.raises(RuntimeError, match="active"):
        hpsv3_native.score_batch(state, batch, micro_batch_size=2)
    hpsv3_native.activate(state, "cpu")
    with pytest.raises(ValueError, match="positive integer"):
        hpsv3_native.score_batch(state, batch, micro_batch_size=0)
