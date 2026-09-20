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
"""CPU contract tests for OmniNFT native video rewards."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl.protocol import DataProto

from verl_omni.utils.reward_score import hpsv3_native, qwen2vl_reward_compat, videoalign_native


def _video_batch(frames=6, fps=24.0):
    frame_values = torch.arange(frames, dtype=torch.uint8).reshape(1, frames, 1, 1, 1)
    return DataProto.from_dict(
        tensors={
            "responses": frame_values.expand(1, frames, 3, 4, 4).clone(),
            "fps": torch.tensor([fps]),
        },
        non_tensors={
            "reward_inputs": np.array([{"text": {"video": "a bird takes flight"}}], dtype=object),
        },
    )


def test_hpsv3_samples_five_frames_and_returns_top_two_mean():
    class RewardModel:
        async def infer(self, images, prompts):
            assert len(images) == 5
            assert [np.asarray(image)[0, 0, 0] for image in images] == [0, 2, 4, 6, 8]
            assert prompts == ["a bird takes flight"] * 5
            return torch.tensor([[1.0, 0.0], [5.0, 0.0], [2.0, 0.0], [4.0, 0.0], [3.0, 0.0]])

    result = asyncio.run(hpsv3_native.compute_score(_video_batch(frames=9), RewardModel()))

    assert result == {"score": pytest.approx(4.5)}


def test_videoalign_preserves_sampled_video_and_score_definition():
    class RewardModel:
        async def infer(self, videos, prompts):
            assert len(videos) == 1
            assert videos[0].shape == (6, 3, 4, 4)
            assert videos[0].dtype == torch.uint8
            assert videos[0][:, 0, 0, 0].tolist() == [0, 2, 4, 7, 9, 11]
            assert prompts == ["a bird takes flight"]
            return torch.tensor([[videoalign_native._VQ_MEAN, 99.0, videoalign_native._TA_MEAN]])

    result = asyncio.run(videoalign_native.compute_score(_video_batch(frames=12, fps=48.0), RewardModel()))

    assert result == {"score": pytest.approx(0.0)}


@pytest.mark.parametrize("module", [hpsv3_native, videoalign_native])
def test_native_video_rewards_reject_nonfinite_logits(module):
    class RewardModel:
        async def infer(self, *_args):
            width = 2 if module is hpsv3_native else 3
            rows = 5 if module is hpsv3_native else 1
            return torch.full((rows, width), torch.nan)

    with pytest.raises(ValueError, match="finite"):
        asyncio.run(module.compute_score(_video_batch(), RewardModel()))


def test_qwen2vl_layout_adaptation_is_instance_local_and_idempotent():
    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(()))

        def forward(self, pixel_values, grid_thw=None, **kwargs):
            del grid_thw, kwargs
            return SimpleNamespace(last_hidden_state=pixel_values + 1)

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = Visual()
            self.language_model = SimpleNamespace(embed_tokens=torch.nn.Embedding(4, 2))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    adapted = Model()
    untouched = Model()
    original_visual_type = type(untouched.model.visual)

    assert qwen2vl_reward_compat.ensure_omninft_qwen2vl_layout(adapted) is adapted
    wrapper = adapted.model.visual
    assert qwen2vl_reward_compat.ensure_omninft_qwen2vl_layout(adapted).model.visual is wrapper
    torch.testing.assert_close(wrapper(torch.tensor([1.0])), torch.tensor([2.0]))
    assert adapted.model.embed_tokens is adapted.model.language_model.embed_tokens
    assert type(untouched.model.visual) is original_visual_type
