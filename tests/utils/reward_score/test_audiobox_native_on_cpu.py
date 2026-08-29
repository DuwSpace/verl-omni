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

"""CPU contracts for the batch-native OmniNFT AudioBox reward."""

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_omni.utils.reward_score import audiobox_native


class _FakeModel:
    def __init__(self):
        self.forward_window_sizes = []
        self.to_devices = []
        self.target_transform = {
            "CE": {"mean": 10.0, "std": 2.0},
            "CU": {"mean": 20.0, "std": 3.0},
            "PC": {"mean": 4.0, "std": 1.0},
            "PQ": {"mean": 30.0, "std": 4.0},
        }

    def eval(self):
        return self

    def to(self, device):
        self.to_devices.append(torch.device(device))
        return self

    def __call__(self, inputs):
        windows = inputs["wav"]
        self.forward_window_sizes.append(windows.shape[0])
        return {
            "CE": torch.full((windows.shape[0],), 1.0, device=windows.device),
            "CU": torch.full((windows.shape[0],), 2.0, device=windows.device),
            "PC": torch.full((windows.shape[0],), 3.0, device=windows.device),
            "PQ": torch.full((windows.shape[0],), 4.0, device=windows.device),
        }


class _WeightedFakeModel(_FakeModel):
    def __call__(self, inputs):
        windows = inputs["wav"]
        self.forward_window_sizes.append(windows.shape[0])
        values = torch.arange(windows.shape[0], dtype=torch.float32, device=windows.device)
        return {axis: values for axis in ("CE", "CU", "PC", "PQ")}


def _make_batch(batch_size=8, sample_rate=16_000, sample_length=8):
    reward_inputs = np.empty(batch_size, dtype=object)
    reward_inputs[:] = [{"text": {"audio": f"unused-{index}"}} for index in range(batch_size)]
    channel_a = torch.ones((batch_size, 1, sample_length), dtype=torch.float32)
    channel_b = torch.full_like(channel_a, 3.0)
    return DataProto.from_dict(
        tensors={
            "audio": torch.cat((channel_a, channel_b), dim=1),
            "audio_sample_rate": torch.full((batch_size,), sample_rate, dtype=torch.long),
            "responses": torch.zeros((batch_size, 1), dtype=torch.uint8),
        },
        non_tensors={
            "reward_inputs": reward_inputs,
            "sample_uid": np.asarray([f"sample-{index}" for index in range(batch_size)], dtype=object),
        },
    )


def _initialize(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(audiobox_native, "_load_model", lambda model_path: model)
    state = audiobox_native.initialize("/models/audiobox")
    return state, model


def test_local_batch_uses_window_batch_for_each_sample_micro_batch(monkeypatch):
    state, model = _initialize(monkeypatch)
    audiobox_native.activate(state, "cpu")

    result = audiobox_native.score_batch(state, _make_batch(), micro_batch_size=4)

    assert model.forward_window_sizes == [4, 4]
    expected_score = (12.0 + 26.0 + 46.0 - 7.0) / 40.0
    torch.testing.assert_close(result["scores"], torch.full((8,), expected_score))
    assert result["valid_mask"].all()
    assert result["metrics"] == {
        "batch_size": 8,
        "micro_batch_size": 4,
        "forward_calls": 2,
        "window_count": 8,
        "source_sample_rates": [16_000],
        "target_sample_rate": 16_000,
        "window_seconds": 10,
        "hop_seconds": 10,
    }


def test_audio_is_downmixed_resampled_and_long_windows_are_weighted(monkeypatch):
    state, model = _initialize(monkeypatch)
    resample_calls = []

    def fake_resample(waveform, source_rate):
        resample_calls.append((waveform.clone(), source_rate))
        return waveform.repeat_interleave(2)

    monkeypatch.setattr(audiobox_native, "_resample_audio", fake_resample)
    audiobox_native.activate(state, "cpu")
    result = audiobox_native.score_batch(
        state,
        _make_batch(batch_size=2, sample_rate=8_000, sample_length=100_000),
        micro_batch_size=2,
    )

    assert model.forward_window_sizes == [4]
    assert [rate for _, rate in resample_calls] == [8_000, 8_000]
    torch.testing.assert_close(resample_calls[0][0], torch.full((100_000,), 2.0))
    assert result["metrics"]["window_count"] == 4
    assert result["metrics"]["forward_calls"] == 1


def test_partial_window_uses_valid_sample_weight(monkeypatch):
    model = _WeightedFakeModel()
    monkeypatch.setattr(audiobox_native, "_load_model", lambda model_path: model)
    state = audiobox_native.initialize("/models/audiobox")
    audiobox_native.activate(state, "cpu")
    state.target_transform = {axis: (0.0, 1.0) for axis in ("CE", "CU", "PC", "PQ")}

    result = audiobox_native.score_batch(
        state,
        _make_batch(batch_size=1, sample_rate=16_000, sample_length=160_001),
        micro_batch_size=1,
    )

    assert model.forward_window_sizes == [2]
    expected = (0.0 * 1.0 + 1.0 * (1 / 160_000)) / (1.0 + 1 / 160_000)
    torch.testing.assert_close(result["scores"], torch.tensor([expected]))


def test_lifecycle_is_isolated_and_idempotently_deactivated(monkeypatch):
    state, model = _initialize(monkeypatch)

    audiobox_native.activate(state, "cpu")
    with pytest.raises(RuntimeError, match="already active"):
        audiobox_native.activate(state, "cpu")
    audiobox_native.deactivate(state)
    audiobox_native.deactivate(state)
    audiobox_native.finalize(state)

    assert model.to_devices == [torch.device("cpu"), torch.device("cpu")]
    assert state.model is None
    assert state.target_transform == {}
    with pytest.raises(RuntimeError, match="finalized"):
        audiobox_native.activate(state, "cpu")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda batch: batch.batch["audio"].fill_(torch.nan), "finite"),
        (lambda batch: batch.batch.__setitem__("audio_sample_rate", torch.full((2,), 16_000.0)), "integer"),
    ],
)
def test_invalid_local_batch_fails_closed(monkeypatch, mutate, match):
    state, _ = _initialize(monkeypatch)
    batch = _make_batch(batch_size=2)
    mutate(batch)
    audiobox_native.activate(state, "cpu")

    with pytest.raises(ValueError, match=match):
        audiobox_native.score_batch(state, batch, micro_batch_size=2)


def test_scoring_requires_active_state_and_positive_micro_batch(monkeypatch):
    state, _ = _initialize(monkeypatch)
    batch = _make_batch(batch_size=2)

    with pytest.raises(RuntimeError, match="active"):
        audiobox_native.score_batch(state, batch, micro_batch_size=2)
    audiobox_native.activate(state, "cpu")
    with pytest.raises(ValueError, match="positive integer"):
        audiobox_native.score_batch(state, batch, micro_batch_size=0)
