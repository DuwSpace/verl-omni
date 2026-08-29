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

"""CPU contracts for the batch-native OmniNFT CLAP reward."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_omni.utils.reward_score import clap_native


class _FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, *, text, audio, sampling_rate, **kwargs):
        self.calls.append({"text": list(text), "audio": list(audio), "sampling_rate": sampling_rate, **kwargs})
        return {"fake_inputs": torch.ones(len(text), 1)}


class _FakeModel:
    def __init__(self):
        self.forward_batch_sizes = []
        self.to_devices = []

    def eval(self):
        return self

    def to(self, device):
        self.to_devices.append(torch.device(device))
        return self

    def __call__(self, fake_inputs):
        batch_size = fake_inputs.shape[0]
        self.forward_batch_sizes.append(batch_size)
        return SimpleNamespace(
            audio_embeds=torch.tensor([[3.0, 4.0]]).repeat(batch_size, 1),
            text_embeds=torch.tensor([[0.0, 5.0]]).repeat(batch_size, 1),
        )


def _make_batch(batch_size=8, sample_rate=48_000):
    reward_inputs = np.empty(batch_size, dtype=object)
    reward_inputs[:] = [{"text": {"audio": f"audio prompt {index}"}} for index in range(batch_size)]
    channel_a = torch.arange(batch_size, dtype=torch.float32).reshape(batch_size, 1, 1).expand(-1, 1, 8)
    channel_b = channel_a + 2
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
    processor = _FakeProcessor()
    monkeypatch.setattr(clap_native, "_load_components", lambda model_path: (model, processor))
    state = clap_native.initialize("/models/clap")
    return state, model, processor


def test_local_batch_is_forwarded_in_real_micro_batches(monkeypatch):
    state, model, processor = _initialize(monkeypatch)
    clap_native.activate(state, "cpu")

    result = clap_native.score_batch(state, _make_batch(), micro_batch_size=4)

    assert model.forward_batch_sizes == [4, 4]
    assert [len(call["audio"]) for call in processor.calls] == [4, 4]
    assert processor.calls[0]["text"] == [f"audio prompt {index}" for index in range(4)]
    assert processor.calls[0]["sampling_rate"] == 48_000
    torch.testing.assert_close(result["scores"], torch.full((8,), 0.9))
    assert result["valid_mask"].all()
    assert result["metrics"] == {
        "batch_size": 8,
        "micro_batch_size": 4,
        "forward_calls": 2,
        "source_sample_rates": [48_000],
        "target_sample_rate": 48_000,
    }


def test_audio_is_downmixed_resampled_and_paired_with_audio_prompt(monkeypatch):
    state, _, processor = _initialize(monkeypatch)
    resample_calls = []

    def fake_resample(waveform, source_rate):
        resample_calls.append((waveform.clone(), source_rate))
        return waveform.repeat_interleave(2)

    monkeypatch.setattr(clap_native, "_resample_audio", fake_resample)
    clap_native.activate(state, "cpu")
    result = clap_native.score_batch(state, _make_batch(batch_size=2, sample_rate=24_000), micro_batch_size=2)

    assert [rate for _, rate in resample_calls] == [24_000, 24_000]
    torch.testing.assert_close(resample_calls[0][0], torch.ones(8))
    assert [waveform.shape for waveform in processor.calls[0]["audio"]] == [(16,), (16,)]
    assert processor.calls[0]["text"] == ["audio prompt 0", "audio prompt 1"]
    assert result["metrics"]["source_sample_rates"] == [24_000]


def test_lifecycle_is_isolated_and_idempotently_deactivated(monkeypatch):
    state, model, _ = _initialize(monkeypatch)

    clap_native.activate(state, "cpu")
    with pytest.raises(RuntimeError, match="already active"):
        clap_native.activate(state, "cpu")
    clap_native.deactivate(state)
    clap_native.deactivate(state)
    clap_native.finalize(state)

    assert model.to_devices == [torch.device("cpu"), torch.device("cpu")]
    assert state.model is None
    assert state.processor is None
    with pytest.raises(RuntimeError, match="finalized"):
        clap_native.activate(state, "cpu")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda batch: batch.batch["audio"].fill_(torch.nan), "finite"),
        (lambda batch: batch.batch.__setitem__("audio_sample_rate", torch.full((2,), 48_000.0)), "integer"),
        (lambda batch: batch.non_tensor_batch["reward_inputs"][0].clear(), "text.audio"),
    ],
)
def test_invalid_local_batch_fails_closed(monkeypatch, mutate, match):
    state, _, _ = _initialize(monkeypatch)
    batch = _make_batch(batch_size=2)
    mutate(batch)
    clap_native.activate(state, "cpu")

    with pytest.raises(ValueError, match=match):
        clap_native.score_batch(state, batch, micro_batch_size=2)


def test_scoring_requires_active_state_and_positive_micro_batch(monkeypatch):
    state, _, _ = _initialize(monkeypatch)
    batch = _make_batch(batch_size=2)

    with pytest.raises(RuntimeError, match="active"):
        clap_native.score_batch(state, batch, micro_batch_size=2)
    clap_native.activate(state, "cpu")
    with pytest.raises(ValueError, match="positive integer"):
        clap_native.score_batch(state, batch, micro_batch_size=0)
