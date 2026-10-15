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
"""The LTX adapter applies its NPU workaround before preparing forward inputs."""

from types import SimpleNamespace

import pytest
import torch
from diffusers.models.normalization import RMSNorm
from tensordict import TensorDict

from verl_omni.pipelines.ltx2_omni_nft import compat
from verl_omni.pipelines.ltx2_omni_nft.diffusers_training_adapter import LTX23OmniNFT


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_ltx_adapter_scopes_workaround_to_model_instances(monkeypatch, device):
    monkeypatch.setattr(compat, "get_device_name", lambda: device)
    module = torch.nn.Sequential(RMSNorm(4, eps=1e-6, elementwise_affine=False), RMSNorm(4, eps=1e-6))
    untouched = RMSNorm(4, eps=1e-6, elementwise_affine=False)
    original_forward = RMSNorm.forward
    latents = torch.randn(1, 5, 4)
    embeds = torch.randn(1, 2, 4)
    config = SimpleNamespace(
        pipeline=SimpleNamespace(
            guidance_scale=1.0,
            num_frames=9,
            height=64,
            width=64,
            frame_rate=24,
        )
    )
    batch = TensorDict({"video_latents_clean": latents[:, :3], "audio_prompt_embeds": embeds}, batch_size=[1])

    inputs, negative = LTX23OmniNFT.prepare_model_inputs(
        module,
        config,
        latents,
        torch.tensor([500.0]),
        embeds,
        torch.ones(1, 2, dtype=torch.bool),
        None,
        None,
        batch,
        0,
    )

    expected = compat.ltx_affine_free_rms_norm_forward if device == "npu" else original_forward
    assert module[0].forward.__func__ is expected
    assert module[1].forward.__func__ is original_forward
    assert untouched.forward.__func__ is original_forward
    assert RMSNorm.forward is original_forward
    assert negative is None
    torch.testing.assert_close(inputs["hidden_states"], latents[:, :3])
    torch.testing.assert_close(inputs["audio_hidden_states"], latents[:, 3:])


def test_ltx_adapter_dispatches_joint_training_inputs():
    from verl_omni.pipelines.model_base import DiffusionModelBase

    config = SimpleNamespace(architecture="LTX2Pipeline", algorithm="omni_nft", external_lib=None)
    adapter = DiffusionModelBase.get_class(config)
    batch = TensorDict(
        {
            "video_latents_clean": torch.ones(1, 3, 4),
            "audio_latents_clean": torch.full((1, 2, 4), 2.0),
            "video_forward_noise": torch.zeros(1, 3, 4),
            "audio_forward_noise": torch.zeros(1, 2, 4),
            "train_timesteps": torch.tensor([[500.0]]),
        },
        batch_size=[1],
    )
    latents, timesteps, context = adapter.prepare_training_inputs(
        batch, step=0, scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    )
    assert adapter is LTX23OmniNFT
    torch.testing.assert_close(latents[:, :3], torch.full((1, 3, 4), 0.5))
    torch.testing.assert_close(latents[:, 3:], torch.ones(1, 2, 4))
    torch.testing.assert_close(timesteps, torch.tensor([500.0]))
    outputs = adapter.prepare_model_outputs({"forward_prediction": (latents[:, :3], latents[:, 3:])}, context)
    assert outputs["video_x0"] is batch["video_latents_clean"]
    assert outputs["audio_x0"] is batch["audio_latents_clean"]
