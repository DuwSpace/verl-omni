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

"""Focused CPU contracts for the final LTX-2.3 OmniNFT integration layer."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.ltx2_omni_nft.diffusers_training_adapter import LTX23OmniNFT
from verl_omni.pipelines.ltx2_omni_nft.vllm_omni_rollout_adapter import LTX23OmniNFTPipeline
from verl_omni.workers.engine.fsdp.diffusers_impl import OmniNFTDiffusersFSDPEngine


def _model_config():
    return SimpleNamespace(
        architecture="LTX2Pipeline",
        algorithm="omni_nft",
        external_lib=None,
        pipeline=SimpleNamespace(
            num_frames=121,
            height=256,
            width=384,
            frame_rate=24.0,
            guidance_scale=1.0,
            video_cfg_scale=1.0,
            audio_cfg_scale=1.0,
        ),
    )


def test_training_adapter_builds_one_joint_forward_with_float_timesteps():
    video = torch.zeros(2, 2, 3)
    audio = torch.zeros(2, 3, 3)
    prompt = torch.randn(2, 4, 8)
    audio_prompt = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    timesteps = torch.tensor([999.5, 250.25])
    micro_batch = TensorDict(
        {
            "video_seq_len": torch.tensor([2, 2]),
            "audio_prompt_embeds": audio_prompt,
        },
        batch_size=[2],
    )

    model_inputs, negative = LTX23OmniNFT.prepare_model_inputs(
        module=None,
        model_config=_model_config(),
        latents=torch.cat((video, audio), dim=1),
        timesteps=timesteps,
        prompt_embeds=prompt,
        prompt_embeds_mask=mask,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=micro_batch,
        step=0,
    )

    assert negative is None
    assert model_inputs["hidden_states"] is not video
    torch.testing.assert_close(model_inputs["hidden_states"], video)
    torch.testing.assert_close(model_inputs["audio_hidden_states"], audio)
    torch.testing.assert_close(model_inputs["timestep"], timesteps)
    torch.testing.assert_close(model_inputs["sigma"], timesteps)
    assert model_inputs["encoder_attention_mask"] is mask
    assert model_inputs["audio_encoder_attention_mask"] is mask
    assert model_inputs["audio_encoder_hidden_states"] is audio_prompt


def test_rollout_prompt_rows_are_left_padded_and_masks_remain_aligned():
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    pipeline.tokenizer_max_length = 4
    requests = [
        SimpleNamespace(
            prompt={
                "prompt_token_ids": [10, 11],
                "prompt_mask": [1, 1],
                "negative_prompt_ids": [30],
                "negative_prompt_mask": [1],
            },
            sampling_params=SimpleNamespace(max_sequence_length=4),
        ),
        SimpleNamespace(
            prompt={
                "prompt_ids": [20, 21, 22],
                "attention_mask": [1, 0, 1],
                "negative_prompt_token_ids": [31, 32],
                "negative_attention_mask": [1, 1],
            },
            sampling_params=SimpleNamespace(max_sequence_length=4),
        ),
    ]
    observed = {}

    def encode_rows(token_ids, attention_mask):
        observed["token_ids"] = token_ids.clone()
        observed["attention_mask"] = attention_mask.clone()
        return token_ids.unsqueeze(-1).float(), attention_mask, torch.arange(token_ids.shape[0])

    pipeline._encode_unique_prompt_rows = encode_rows
    result = pipeline._prepare_batch_prompt_embeds(SimpleNamespace(requests=requests))

    assert observed["token_ids"].tolist() == [
        [0, 0, 10, 11],
        [0, 20, 21, 22],
        [0, 0, 0, 30],
        [0, 0, 31, 32],
    ]
    assert observed["attention_mask"].tolist() == [
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
    ]
    assert result["prompt_embeds"].squeeze(-1).tolist() == [[0.0, 0.0, 10.0, 11.0], [0.0, 20.0, 21.0, 22.0]]
    assert result["negative_prompt_attention_mask"].tolist() == [
        [False, False, False, True],
        [False, False, True, True],
    ]


def test_omni_nft_engine_rejects_single_latent_timestep_staging():
    engine = object.__new__(OmniNFTDiffusersFSDPEngine)
    batch = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(batch, enable_timestep_staging=True)

    with pytest.raises(NotImplementedError, match="does not support actor.enable_timestep_staging"):
        engine.forward_backward_batch(batch, loss_function=object(), forward_only=False)


def test_fsdp_engine_replays_the_same_video_audio_forward_process():
    engine = object.__new__(OmniNFTDiffusersFSDPEngine)
    engine.module = None
    engine.model_config = _model_config()
    video_x0 = torch.full((2, 2, 3), 2.0)
    audio_x0 = torch.full((2, 3, 3), 4.0)
    video_noise = torch.stack((torch.zeros_like(video_x0), torch.full_like(video_x0, 10.0)), dim=1)
    audio_noise = torch.stack((torch.zeros_like(audio_x0), torch.full_like(audio_x0, 20.0)), dim=1)
    micro_batch = TensorDict(
        {
            "video_latents_clean": video_x0,
            "audio_latents_clean": audio_x0,
            "video_forward_noise": video_noise,
            "audio_forward_noise": audio_noise,
            "train_timesteps": torch.tensor([[999.5, 500.25], [750.0, 250.5]]),
            "prompt_embeds": torch.randn(2, 4, 8),
            "audio_prompt_embeds": torch.randn(2, 4, 8),
            "prompt_embeds_mask": torch.ones(2, 4, dtype=torch.bool),
            "video_seq_len": torch.tensor([2, 2]),
        },
        batch_size=[2],
    )

    model_inputs, negative, x0, xt, expanded_t = engine.prepare_model_inputs(micro_batch, step=1)

    expected_t = torch.tensor([0.50025, 0.2505]).view(2, 1, 1)
    expected_video = (1 - expected_t) * video_x0 + expected_t * video_noise[:, 1]
    expected_audio = (1 - expected_t) * audio_x0 + expected_t * audio_noise[:, 1]
    assert negative is None
    torch.testing.assert_close(x0[0], video_x0)
    torch.testing.assert_close(x0[1], audio_x0)
    torch.testing.assert_close(xt[0], expected_video)
    torch.testing.assert_close(xt[1], expected_audio)
    torch.testing.assert_close(model_inputs["hidden_states"], expected_video)
    torch.testing.assert_close(model_inputs["audio_hidden_states"], expected_audio)
    torch.testing.assert_close(model_inputs["timestep"], torch.tensor([500.25, 250.5]))
    torch.testing.assert_close(expanded_t[0], expected_t)
    torch.testing.assert_close(expanded_t[1], expected_t)


def test_fsdp_engine_rejects_unpaired_predictions():
    paired = (torch.zeros(1), torch.ones(1))
    output = (paired, torch.zeros(1), paired, paired, paired, paired)

    with pytest.raises(TypeError, match="expects.*predictions"):
        OmniNFTDiffusersFSDPEngine.prepare_model_outputs(output, TensorDict({}, batch_size=[]))


def test_rollout_crops_audio_padding_and_preserves_model_scale_timesteps(monkeypatch):
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    pipeline._omni_nft_clean_state = None
    pipeline._omni_nft_forward_context = None
    pipeline.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24_000))
    pipeline._prepare_batch_prompt_embeds = lambda request_batch: {}
    request = SimpleNamespace(
        request_id="r0",
        prompt={},
        sampling_params=SimpleNamespace(output_type="pt", extra_args={}),
    )
    request_batch = DiffusionRequestBatch(requests=[request])
    clean_state = LTXAVState(
        video=torch.arange(6, dtype=torch.bfloat16).reshape(1, 2, 3),
        audio=torch.arange(12, dtype=torch.bfloat16).reshape(1, 4, 3),
    )
    prompt_context = SimpleNamespace(
        positive_connector_prompt_embeds=torch.zeros(1, 2, 4),
        positive_connector_audio_prompt_embeds=torch.ones(1, 2, 4),
        positive_connector_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        negative_connector_prompt_embeds=torch.zeros(1, 2, 4),
        negative_connector_audio_prompt_embeds=torch.ones(1, 2, 4),
        negative_connector_attention_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    forward_context = SimpleNamespace(
        timesteps=torch.tensor([999.5, 500.25, 7.75]),
        request_inputs=SimpleNamespace(frame_rate=23.976),
        original_audio_num_frames=3,
        prompt_context=prompt_context,
    )

    def native_forward(self, req, **kwargs):
        self._omni_nft_clean_state = clean_state
        self._omni_nft_forward_context = forward_context
        return [DiffusionOutput(output=(torch.zeros(1, 3, 1, 2, 2), torch.zeros(1, 2, 5)))]

    monkeypatch.setattr(LTX2Pipeline, "forward", native_forward)
    output = pipeline.forward(request_batch)[0]
    rl = output.output["metadata"]["rl"]

    assert rl["audio_latents_clean"].shape == (1, 3, 3)
    torch.testing.assert_close(rl["audio_latents_clean"], clean_state.audio[:, :3].float())
    torch.testing.assert_close(rl["train_timesteps"], forward_context.timesteps.unsqueeze(0))
    assert rl["train_timesteps"].dtype == torch.float32
    assert rl["fps"].item() == pytest.approx(23.976)
    assert pipeline._omni_nft_clean_state is None
    assert pipeline._omni_nft_forward_context is None
