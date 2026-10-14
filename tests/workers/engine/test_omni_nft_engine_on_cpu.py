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
"""CPU tests for the model-independent OmniNFT actor engine."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict

from verl_omni.workers.engine.fsdp import diffusers_impl


@pytest.mark.parametrize(
    "strategy,sp_size,message",
    [("fsdp", 1, "only actor.strategy=fsdp2"), ("fsdp2", 2, "does not implement Ulysses")],
)
def test_omni_nft_rejects_unsupported_engine_config(strategy, sp_size, message):
    config = SimpleNamespace(strategy=strategy, ulysses_sequence_parallel_size=sp_size)
    with pytest.raises(NotImplementedError, match=message):
        diffusers_impl.OmniNFTDiffusersFSDPEngine(None, config, None, None)


@pytest.mark.parametrize("per_step_noise", [False, True])
def test_omni_nft_packs_video_audio_with_matching_forward_noise(monkeypatch, per_step_noise):
    engine = object.__new__(diffusers_impl.OmniNFTDiffusersFSDPEngine)
    engine.module = torch.nn.Identity()
    engine.model_config = SimpleNamespace()
    video = torch.full((2, 3, 4), 2.0)
    audio = torch.full((2, 2, 4), 3.0)
    video_noise = torch.full_like(video, -1.0)
    audio_noise = torch.full_like(audio, -2.0)
    batch = TensorDict(
        {
            "video_latents_clean": video,
            "audio_latents_clean": audio,
            "video_forward_noise": (
                torch.stack((torch.full_like(video, 99.0), video_noise), dim=1) if per_step_noise else video_noise
            ),
            "audio_forward_noise": (
                torch.stack((torch.full_like(audio, 99.0), audio_noise), dim=1) if per_step_noise else audio_noise
            ),
            "train_timesteps": torch.tensor([[500.0, 0.0], [500.0, 1000.0]]),
            "prompt_embeds": torch.zeros(2, 3, 4),
            "prompt_embeds_mask": torch.ones(2, 3, dtype=torch.bool),
        },
        batch_size=2,
    )
    prepare = Mock(return_value=({"positive": True}, None))
    monkeypatch.setattr(diffusers_impl, "prepare_model_inputs", prepare)

    positive, negative, clean, noised, times = engine.prepare_model_inputs(batch, step=1)

    expected_video = torch.stack((video[0], video_noise[1]))
    expected_audio = torch.stack((audio[0], audio_noise[1]))
    torch.testing.assert_close(prepare.call_args.kwargs["latents"], torch.cat((expected_video, expected_audio), dim=1))
    torch.testing.assert_close(prepare.call_args.kwargs["timesteps"], torch.tensor([0.0, 1000.0]))
    torch.testing.assert_close(noised[0], expected_video)
    torch.testing.assert_close(noised[1], expected_audio)
    torch.testing.assert_close(times[0], torch.tensor([0.0, 1.0]).view(2, 1, 1))
    torch.testing.assert_close(times[1], times[0])
    assert clean[0] is video and clean[1] is audio
    assert positive == {"positive": True} and negative is None


def test_omni_nft_checkpoint_wrapper_casts_inputs_and_preserves_gradients():
    module = torch.nn.Linear(4, 2, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        module.weight.fill_(1.0)
    inputs = torch.ones(2, 4, requires_grad=True)
    checkpoint = diffusers_impl.OmniNFTDiffusersFSDPEngine._fsdp2_gradient_checkpointing_with_cast_func(torch.bfloat16)

    result = checkpoint(module, inputs)
    result.float().sum().backward()

    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, torch.full((2, 2), 4.0, dtype=torch.bfloat16))
    torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 2.0))
    torch.testing.assert_close(module.weight.grad, torch.full_like(module.weight, 2.0))
