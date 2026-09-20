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
"""CPU contract tests for OmniNFT native audio and synchronization rewards."""

import asyncio

import numpy as np
import pytest
import torch
from verl.protocol import DataProto

from verl_omni.utils.reward_score import audiobox_native, clap_native, desync_native


def _audio_batch(sample_rate=16000, samples=16000):
    return DataProto.from_dict(
        tensors={
            "audio": torch.linspace(-1, 1, samples).reshape(1, 1, samples),
            "audio_sample_rate": torch.tensor([sample_rate], dtype=torch.int64),
        },
        non_tensors={
            "reward_inputs": np.array([{"text": {"audio": "clear birdsong"}}], dtype=object),
        },
    )


def test_audiobox_window_score_uses_axis_definition_and_valid_duration():
    class RewardModel:
        async def infer(self, windows, masks):
            assert windows.shape == (2, 1, 160000)
            assert masks.shape == windows.shape
            assert masks.sum(dim=(1, 2)).tolist() == [160000, 40000]
            predictions = {
                "CE": torch.tensor([40.0, 0.0]),
                "CU": torch.tensor([0.0, 0.0]),
                "PC": torch.tensor([0.0, 0.0]),
                "PQ": torch.tensor([0.0, 0.0]),
            }
            return {"predictions": predictions, "target_transform": {axis: (0.0, 1.0) for axis in predictions}}

    result = asyncio.run(audiobox_native.compute_score(_audio_batch(samples=200000), RewardModel()))

    assert result == {"score": pytest.approx(0.8)}


def test_clap_downmixes_audio_and_maps_cosine_to_unit_interval():
    batch = _audio_batch(sample_rate=48000, samples=4800)
    batch.batch["audio"] = torch.cat([batch.batch["audio"], -batch.batch["audio"]], dim=1)

    class RewardModel:
        async def infer(self, waveforms, prompts):
            assert len(waveforms) == 1
            np.testing.assert_allclose(waveforms[0], np.zeros(4800, dtype=np.float32), atol=1e-7)
            assert prompts == ["clear birdsong"]
            return {
                "audio_embeddings": torch.tensor([[1.0, 0.0]]),
                "text_embeddings": torch.tensor([[1.0, 0.0]]),
            }

    result = asyncio.run(clap_native.compute_score(batch, RewardModel()))

    assert result == {"score": pytest.approx(1.0)}


def test_desync_preprocessing_and_center_offset_score(monkeypatch):
    video = torch.randint(0, 256, (4, 3, 16, 24), dtype=torch.uint8)
    prepared_video = desync_native._prepare_video(video, source_fps=25.0)
    prepared_audio = desync_native._prepare_audio(torch.ones(1, 1600), source_rate=16000)
    assert prepared_video.shape == (200, 3, 224, 224)
    assert prepared_audio.shape == (128000,)

    batch = DataProto.from_dict(
        tensors={
            "responses": video.unsqueeze(0),
            "audio": torch.ones(1, 1, 1600),
            "fps": torch.tensor([25.0]),
            "audio_sample_rate": torch.tensor([16000]),
        }
    )
    monkeypatch.setattr(
        desync_native,
        "_extract_inputs",
        lambda _batch: ([prepared_video], [prepared_audio], [25.0], [16000]),
    )

    class RewardModel:
        async def infer(self, videos, audio):
            assert videos.shape == (1, 200, 3, 224, 224)
            assert audio.shape == (1, 128000)
            logits = torch.zeros(2, 1, 21)
            logits[:, :, 10] = 1
            return logits

    result = asyncio.run(desync_native.compute_score(batch, RewardModel()))

    assert result == {"score": pytest.approx(1.0)}


def test_desync_rejects_nonfinite_logits(monkeypatch):
    batch = _audio_batch()
    monkeypatch.setattr(
        desync_native,
        "_extract_inputs",
        lambda _batch: ([torch.zeros(1)], [torch.zeros(1)], [25.0], [16000]),
    )

    class RewardModel:
        async def infer(self, videos, audio):
            del videos, audio
            return torch.full((2, 1, 21), torch.nan)

    with pytest.raises(ValueError, match="finite"):
        asyncio.run(desync_native.compute_score(batch, RewardModel()))


@pytest.mark.parametrize("module", [audiobox_native, clap_native])
def test_native_audio_rewards_reject_nonfinite_audio(module):
    batch = _audio_batch()
    batch.batch["audio"][0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        asyncio.run(module.compute_score(batch, object()))
