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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

from verl_omni.pipelines.ltx2_omni_nft.prompt_mixin import LTXTokenIdPromptMixin
from verl_omni.pipelines.ltx2_omni_nft.vllm_omni_rollout_adapter import LTX23OmniNFTPipeline
from verl_omni.pipelines.model_base import VllmOmniPipelineBase


def _prompt_context() -> SimpleNamespace:
    def values(offset: int) -> torch.Tensor:
        return torch.arange(offset, offset + 48, dtype=torch.float32).reshape(1, 6, 8)

    return SimpleNamespace(
        positive_connector_prompt_embeds=values(0),
        positive_connector_audio_prompt_embeds=values(100),
        positive_connector_attention_mask=torch.ones(1, 6, dtype=torch.long),
        negative_connector_prompt_embeds=values(200),
        negative_connector_audio_prompt_embeds=values(300),
        negative_connector_attention_mask=torch.zeros(1, 6, dtype=torch.long),
    )


def test_ltx2_omni_nft_registers_and_directly_reuses_ltx_pipeline() -> None:
    assert VllmOmniPipelineBase.get_class("LTX2Pipeline", "omni_nft") is LTX23OmniNFTPipeline
    assert LTX23OmniNFTPipeline.__bases__ == (LTXTokenIdPromptMixin, LTX2Pipeline)


def test_ltx2_omni_nft_phase_delegates_to_native_sampler() -> None:
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    phase_recipe = SimpleNamespace(sampler=object())
    expected = object()

    with (
        patch.object(LTX2Pipeline, "run_phase", return_value=expected) as native_phase,
    ):
        actual = pipeline.run_phase(
            MagicMock(),
            MagicMock(),
            noise_scale=1.0,
            sigmas=None,
            timesteps=None,
            attention_kwargs=None,
            phase_recipe=phase_recipe,
        )

    assert actual is expected
    assert native_phase.call_args.kwargs["phase_recipe"].sampler is phase_recipe.sampler


def test_ltx2_omni_nft_denoise_captures_native_clean_state() -> None:
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    clean_state = LTXAVState(video=torch.randn(1, 5, 8), audio=torch.randn(1, 7, 8))
    forward_context = SimpleNamespace()

    with patch.object(LTX2Pipeline, "_denoise_step", return_value=clean_state) as native_step:
        actual = pipeline._denoise_step(
            0,
            torch.tensor(1.0),
            MagicMock(),
            forward_context,
            MagicMock(),
        )

    assert actual is clean_state
    assert pipeline._omni_nft_clean_state is clean_state
    assert pipeline._omni_nft_forward_context is forward_context
    native_step.assert_called_once()


@pytest.mark.parametrize("output_type", ["image", "pt", None])
def test_ltx2_omni_nft_forward_runs_native_model_and_returns_contract(output_type: str | None) -> None:
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    pipeline._inject_precomputed_prompt_embeds = MagicMock()
    pipeline.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))
    request = SimpleNamespace(sampling_params=SimpleNamespace(output_type=output_type))
    request_batch = SimpleNamespace(num_reqs=1, requests=[request])
    video = torch.rand(1, 9, 3, 16, 16)
    audio = torch.rand(1, 24000)
    clean_state = LTXAVState(video=torch.randn(1, 5, 8), audio=torch.randn(1, 7, 8))
    forward_context = SimpleNamespace(
        timesteps=torch.tensor([1000.0, 500.0]),
        request_inputs=SimpleNamespace(frame_rate=24.0),
    )
    prompt_context = _prompt_context()

    def native_forward(*_args, **_kwargs):
        pipeline._omni_nft_clean_state = clean_state
        pipeline._omni_nft_forward_context = forward_context
        pipeline._omni_nft_prompt_context = prompt_context
        return DiffusionOutput(output=(video, audio))

    with (
        patch.object(LTX2Pipeline, "forward", side_effect=native_forward) as model_forward,
    ):
        output = pipeline.forward(request_batch)

    model_forward.assert_called_once()
    assert request.sampling_params.output_type == "pt"
    assert output.trajectory_latents is None
    assert output.trajectory_timesteps is None
    assert output.trajectory_log_probs is None

    metadata = output.output["metadata"]
    output_video, output_audio = output.output["payload"]["video"]
    torch.testing.assert_close(output_video, video[0])
    torch.testing.assert_close(output_audio, audio)
    assert set(metadata["rl"]) == {
        "audio",
        "video_latents_clean",
        "audio_latents_clean",
        "train_timesteps",
        "video_latent_shape",
        "audio_latent_shape",
        "video_seq_len",
        "audio_seq_len",
        "fps",
        "audio_sample_rate",
    }
    torch.testing.assert_close(metadata["rl"]["video_latents_clean"], clean_state.video.float())
    torch.testing.assert_close(metadata["rl"]["audio_latents_clean"], clean_state.audio.float())
    torch.testing.assert_close(metadata["rl"]["audio"], audio)
    assert metadata["rl"]["video_latent_shape"].tolist() == [[5, 8]]
    assert metadata["rl"]["audio_latent_shape"].tolist() == [[7, 8]]
    assert metadata["rl"]["train_timesteps"].tolist() == [[1000.0, 500.0]]
    assert metadata["rl"]["fps"].item() == 24.0
    assert metadata["rl"]["audio_sample_rate"] == 24000
    prompt_embeddings = metadata["prompt_embeddings"]
    expected_conditions = {
        "prompt_embeds": prompt_context.positive_connector_prompt_embeds,
        "audio_prompt_embeds": prompt_context.positive_connector_audio_prompt_embeds,
        "prompt_embeds_mask": prompt_context.positive_connector_attention_mask,
        "negative_prompt_embeds": prompt_context.negative_connector_prompt_embeds,
        "negative_audio_prompt_embeds": prompt_context.negative_connector_audio_prompt_embeds,
        "negative_prompt_embeds_mask": prompt_context.negative_connector_attention_mask,
    }
    assert set(prompt_embeddings) == set(expected_conditions)
    for key, expected in expected_conditions.items():
        torch.testing.assert_close(prompt_embeddings[key], expected)


@pytest.mark.parametrize("output_type", ["latent", "np"])
def test_ltx2_omni_nft_forward_rejects_non_pt_output_type(output_type: str) -> None:
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    pipeline._inject_precomputed_prompt_embeds = MagicMock()
    request = SimpleNamespace(sampling_params=SimpleNamespace(output_type=output_type))
    request_batch = SimpleNamespace(num_reqs=1, requests=[request])

    with (
        patch.object(LTX2Pipeline, "forward") as model_forward,
        pytest.raises(ValueError, match="decoded 'pt' tensor output"),
    ):
        pipeline.forward(request_batch)

    model_forward.assert_not_called()


def _forward_with(*, vocoder, request_inputs, output_type: str | None = "pt"):
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    pipeline._inject_precomputed_prompt_embeds = MagicMock()
    pipeline.vocoder = vocoder
    request = SimpleNamespace(sampling_params=SimpleNamespace(output_type=output_type))
    request_batch = SimpleNamespace(num_reqs=1, requests=[request])
    video = torch.rand(1, 9, 3, 16, 16)
    audio = torch.rand(1, 24000)
    clean_state = LTXAVState(video=torch.randn(1, 5, 8), audio=torch.randn(1, 7, 8))
    forward_context = SimpleNamespace(
        timesteps=torch.tensor([1000.0, 500.0]),
        request_inputs=request_inputs,
    )

    def native_forward(*_args, **_kwargs):
        pipeline._omni_nft_clean_state = clean_state
        pipeline._omni_nft_forward_context = forward_context
        pipeline._omni_nft_prompt_context = _prompt_context()
        return DiffusionOutput(output=(video, audio))

    with patch.object(LTX2Pipeline, "forward", side_effect=native_forward):
        return pipeline.forward(request_batch)


def test_ltx2_omni_nft_forward_records_vocoder_sample_rate() -> None:
    output = _forward_with(
        vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=48000)),
        request_inputs=SimpleNamespace(frame_rate=24.0),
    )
    assert output.output["metadata"]["rl"]["audio_sample_rate"] == 48000


@pytest.mark.parametrize(
    "request_inputs",
    [None, SimpleNamespace(), SimpleNamespace(frame_rate=None)],
)
def test_ltx2_omni_nft_forward_rejects_missing_frame_rate(request_inputs) -> None:
    with pytest.raises(RuntimeError, match="request_inputs.frame_rate"):
        _forward_with(
            vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000)),
            request_inputs=request_inputs,
        )


@pytest.mark.parametrize("frame_rate", [0.0, -1.0, float("nan"), float("inf")])
def test_ltx2_omni_nft_forward_rejects_invalid_frame_rate(frame_rate: float) -> None:
    with pytest.raises(RuntimeError, match="Invalid LTX frame_rate"):
        _forward_with(
            vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000)),
            request_inputs=SimpleNamespace(frame_rate=frame_rate),
        )


@pytest.mark.parametrize(
    "vocoder",
    [None, SimpleNamespace(), SimpleNamespace(config=SimpleNamespace())],
)
def test_ltx2_omni_nft_forward_rejects_missing_vocoder_sample_rate(vocoder) -> None:
    with pytest.raises(RuntimeError, match="vocoder.config.output_sampling_rate"):
        _forward_with(vocoder=vocoder, request_inputs=SimpleNamespace(frame_rate=24.0))


def test_ltx2_omni_nft_forward_rejects_invalid_vocoder_sample_rate() -> None:
    with pytest.raises(RuntimeError, match="Invalid vocoder output_sampling_rate"):
        _forward_with(
            vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=0)),
            request_inputs=SimpleNamespace(frame_rate=24.0),
        )
