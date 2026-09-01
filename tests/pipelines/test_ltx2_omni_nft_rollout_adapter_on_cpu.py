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
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.ltx2_omni_nft.prompt_mixin import LTXTokenIdPromptMixin
from verl_omni.pipelines.ltx2_omni_nft.vllm_omni_rollout_adapter import (
    LTX23OmniNFTPipeline,
    _rollout_progress_bar_enabled,
)
from verl_omni.pipelines.model_base import VllmOmniPipelineBase


def _prompt_context(batch_size: int = 1) -> SimpleNamespace:
    def values(offset: int) -> torch.Tensor:
        return torch.stack(
            [
                torch.arange(offset + row * 48, offset + row * 48 + 48, dtype=torch.float32).reshape(6, 8)
                for row in range(batch_size)
            ]
        )

    return SimpleNamespace(
        positive_connector_prompt_embeds=values(0),
        positive_connector_audio_prompt_embeds=values(100),
        positive_connector_attention_mask=torch.ones(batch_size, 6, dtype=torch.long),
        negative_connector_prompt_embeds=values(200),
        negative_connector_audio_prompt_embeds=values(300),
        negative_connector_attention_mask=torch.zeros(batch_size, 6, dtype=torch.long),
    )


def _one(outputs):
    assert isinstance(outputs, list) and len(outputs) == 1
    return outputs[0]


def test_ltx2_omni_nft_registers_and_directly_reuses_ltx_pipeline() -> None:
    assert VllmOmniPipelineBase.get_class("LTX2Pipeline", "omni_nft") is LTX23OmniNFTPipeline
    assert LTX23OmniNFTPipeline.__bases__ == (LTXTokenIdPromptMixin, LTX2Pipeline)
    assert LTX23OmniNFTPipeline.supports_request_batch is True


@pytest.mark.parametrize(
    ("value", "enabled"),
    [
        (None, False),
        ("0", False),
        ("1", True),
        ("true", True),
        ("YES", True),
    ],
)
def test_rollout_progress_bar_env_gate(monkeypatch, value, enabled) -> None:
    monkeypatch.delenv("OMNIFT_ROLLOUT_PROGRESS", raising=False)
    if value is not None:
        monkeypatch.setenv("OMNIFT_ROLLOUT_PROGRESS", value)
    assert _rollout_progress_bar_enabled() is enabled


def test_ltx2_omni_nft_prompt_encoding_calls_native_text_encoder() -> None:
    pipeline = object.__new__(LTXTokenIdPromptMixin)
    pipeline.device = torch.device("cpu")
    pipeline.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    pipeline.text_encoder = MagicMock(
        dtype=torch.float32,
        return_value=SimpleNamespace(hidden_states=(torch.zeros(1, 4, 3), torch.ones(1, 4, 3))),
    )

    prompt_embeds, prompt_mask = pipeline._encode_token_ids([10, 20], None, max_sequence_length=4)

    assert prompt_embeds.shape == (1, 4, 6)
    assert prompt_mask.tolist() == [[0, 0, 1, 1]]
    kwargs = pipeline.text_encoder.call_args.kwargs
    assert kwargs["output_hidden_states"] is True
    assert "use_cache" not in kwargs
    assert torch.equal(kwargs["attention_mask"], prompt_mask)
    assert kwargs["input_ids"].tolist() == [[0, 0, 10, 20]]


def test_ltx2_omni_nft_batch_injects_distinct_prompt_rows_in_one_encoder_call() -> None:
    pipeline = object.__new__(LTXTokenIdPromptMixin)
    pipeline.device = torch.device("cpu")
    pipeline.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    pipeline.tokenizer_max_length = 4

    def _fake_encoder(*, input_ids, attention_mask, output_hidden_states):
        del output_hidden_states
        batch, seq = input_ids.shape
        return SimpleNamespace(hidden_states=(torch.zeros(batch, seq, 3), torch.ones(batch, seq, 3)))

    pipeline.text_encoder = MagicMock(side_effect=_fake_encoder)
    pipeline.text_encoder.dtype = torch.float32

    def _request(prompt_ids, negative_ids):
        return SimpleNamespace(
            sampling_params=SimpleNamespace(max_sequence_length=4),
            prompt={
                "prompt_token_ids": prompt_ids,
                "negative_prompt_ids": negative_ids,
            },
        )

    req = SimpleNamespace(
        requests=[
            _request([10, 20], [1]),
            _request([30], [2, 3]),
        ]
    )
    pipeline._inject_batch_prompt_embeds(req)

    assert pipeline.text_encoder.call_count == 1
    encoded_ids = pipeline.text_encoder.call_args.kwargs["input_ids"]
    assert encoded_ids.tolist() == [
        [0, 0, 0, 1],
        [0, 0, 0, 30],
        [0, 0, 2, 3],
        [0, 0, 10, 20],
    ]
    assert encoded_ids.shape == (4, 4)
    assert req.requests[0].prompt["prompt_embeds"].shape == (4, 6)
    assert req.requests[1].prompt["negative_prompt_attention_mask"].tolist() == [0, 0, 1, 1]


def test_ltx2_omni_nft_batch_prompt_encoding_deduplicates_positive_and_negative_rows() -> None:
    pipeline = object.__new__(LTXTokenIdPromptMixin)
    pipeline.device = torch.device("cpu")
    pipeline.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    pipeline.tokenizer_max_length = 4

    def _fake_encoder(*, input_ids, attention_mask, output_hidden_states):
        del output_hidden_states
        batch, seq = input_ids.shape
        return SimpleNamespace(hidden_states=(torch.zeros(batch, seq, 3), torch.ones(batch, seq, 3)))

    pipeline.text_encoder = MagicMock(side_effect=_fake_encoder)
    pipeline.text_encoder.dtype = torch.float32
    requests = [
        SimpleNamespace(
            sampling_params=SimpleNamespace(max_sequence_length=4),
            prompt={"prompt_token_ids": [10, 20], "negative_prompt_ids": [1]},
        ),
        SimpleNamespace(
            sampling_params=SimpleNamespace(max_sequence_length=4),
            prompt={"prompt_token_ids": [10, 20], "negative_prompt_ids": [1]},
        ),
    ]

    pipeline._inject_batch_prompt_embeds(SimpleNamespace(requests=requests))

    assert pipeline.text_encoder.call_count == 1
    assert pipeline.text_encoder.call_args.kwargs["input_ids"].tolist() == [
        [0, 0, 0, 1],
        [0, 0, 10, 20],
    ]
    torch.testing.assert_close(
        requests[0].prompt["prompt_embeds"],
        requests[1].prompt["prompt_embeds"],
    )
    torch.testing.assert_close(
        requests[0].prompt["negative_prompt_embeds"],
        requests[1].prompt["negative_prompt_embeds"],
    )


def test_ltx2_omni_nft_gemma_receives_per_request_attention_mask() -> None:
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    pipeline.text_encoder = MagicMock(
        dtype=torch.float32,
        return_value=SimpleNamespace(hidden_states=(torch.zeros(1, 4, 3), torch.ones(1, 4, 3))),
    )
    token_ids = torch.tensor([[0, 0, 10, 20]])
    attention_mask = torch.tensor([[0, 0, 1, 1]])

    prompt_embeds, returned_mask = pipeline._encode_one_prepared_row(token_ids, attention_mask)

    assert prompt_embeds.shape == (1, 4, 6)
    assert pipeline.text_encoder.call_args.kwargs["attention_mask"].tolist() == [[0, 0, 1, 1]]
    torch.testing.assert_close(returned_mask, attention_mask)


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
    pipeline._inject_batch_prompt_embeds = MagicMock()
    pipeline.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))
    request = SimpleNamespace(sampling_params=SimpleNamespace(output_type=output_type))
    request_batch = DiffusionRequestBatch(requests=[request])
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
        output = _one(pipeline.forward(request_batch))

    model_forward.assert_called_once()
    assert pipeline._inject_batch_prompt_embeds.call_count == 1
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
    assert metadata["rl"]["audio_sample_rate"].dtype == torch.long
    assert metadata["rl"]["audio_sample_rate"].item() == 24000
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
    pipeline._inject_batch_prompt_embeds = MagicMock()
    request = SimpleNamespace(sampling_params=SimpleNamespace(output_type=output_type))
    request_batch = DiffusionRequestBatch(requests=[request])

    with (
        patch.object(LTX2Pipeline, "forward") as model_forward,
        pytest.raises(ValueError, match="decoded 'pt' tensor output"),
    ):
        pipeline.forward(request_batch)
    model_forward.assert_not_called()


def _forward_with(*, vocoder, request_inputs, output_type: str | None = "pt", as_batch: bool = True):
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    pipeline._inject_batch_prompt_embeds = MagicMock()
    pipeline.vocoder = vocoder
    request = SimpleNamespace(sampling_params=SimpleNamespace(output_type=output_type))
    request_batch = DiffusionRequestBatch(requests=[request])
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
        output = pipeline.forward(request_batch if as_batch else request)
    return _one(output) if as_batch else output


def test_ltx2_omni_nft_forward_single_request_returns_single_output() -> None:
    output = _forward_with(
        vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000)),
        request_inputs=SimpleNamespace(frame_rate=24.0),
        as_batch=False,
    )
    assert isinstance(output, DiffusionOutput)


def test_ltx2_omni_nft_forward_records_vocoder_sample_rate() -> None:
    output = _forward_with(
        vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=48000)),
        request_inputs=SimpleNamespace(frame_rate=24.0),
    )
    assert output.output["metadata"]["rl"]["audio_sample_rate"].dtype == torch.long
    assert output.output["metadata"]["rl"]["audio_sample_rate"].item() == 48000


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


def test_ltx2_omni_nft_forward_splits_packed_request_batch(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "request_batch.log"
    monkeypatch.setenv("OMNIFT_REQUEST_BATCH_LOG", str(log_path))
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    pipeline._inject_batch_prompt_embeds = MagicMock()
    pipeline.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))
    requests = [
        SimpleNamespace(sampling_params=SimpleNamespace(output_type="pt")),
        SimpleNamespace(sampling_params=SimpleNamespace(output_type="image")),
    ]
    request_batch = DiffusionRequestBatch(requests=requests)
    videos = [torch.rand(1, 9, 3, 16, 16), torch.rand(1, 9, 3, 16, 16)]
    audios = [torch.rand(1, 24000), torch.rand(1, 24000)]
    clean_state = LTXAVState(video=torch.randn(2, 5, 8), audio=torch.randn(2, 7, 8))
    forward_context = SimpleNamespace(
        timesteps=torch.tensor([1000.0, 500.0]),
        request_inputs=SimpleNamespace(frame_rate=24.0),
    )
    prompt_context = _prompt_context(batch_size=2)

    def native_forward(*_args, **_kwargs):
        pipeline._omni_nft_clean_state = clean_state
        pipeline._omni_nft_forward_context = forward_context
        pipeline._omni_nft_prompt_context = prompt_context
        return [
            DiffusionOutput(output=(videos[0], audios[0])),
            DiffusionOutput(output=(videos[1], audios[1])),
        ]

    with patch.object(LTX2Pipeline, "forward", side_effect=native_forward) as model_forward:
        outputs = pipeline.forward(request_batch)

    model_forward.assert_called_once()
    assert log_path.read_text(encoding="utf-8").strip() == "2 2"
    assert pipeline._inject_batch_prompt_embeds.call_count == 1
    assert [request.sampling_params.output_type for request in requests] == ["pt", "pt"]
    assert isinstance(outputs, list) and len(outputs) == 2

    for index, output in enumerate(outputs):
        assert output.trajectory_latents is None
        assert output.trajectory_timesteps is None
        assert output.trajectory_log_probs is None
        output_video, output_audio = output.output["payload"]["video"]
        torch.testing.assert_close(output_video, videos[index][0])
        torch.testing.assert_close(output_audio, audios[index])
        metadata = output.output["metadata"]
        torch.testing.assert_close(metadata["rl"]["video_latents_clean"], clean_state.video[index : index + 1].float())
        torch.testing.assert_close(metadata["rl"]["audio_latents_clean"], clean_state.audio[index : index + 1].float())
        torch.testing.assert_close(metadata["rl"]["audio"], audios[index])
        assert metadata["rl"]["video_latent_shape"].tolist() == [[5, 8]]
        assert metadata["rl"]["audio_latent_shape"].tolist() == [[7, 8]]
        assert metadata["rl"]["train_timesteps"].tolist() == [[1000.0, 500.0]]
        assert metadata["rl"]["fps"].item() == 24.0
        assert metadata["rl"]["audio_sample_rate"].dtype == torch.long
        assert metadata["rl"]["audio_sample_rate"].item() == 24000
        torch.testing.assert_close(
            metadata["prompt_embeddings"]["prompt_embeds"],
            prompt_context.positive_connector_prompt_embeds[index : index + 1],
        )
        torch.testing.assert_close(
            metadata["prompt_embeddings"]["negative_audio_prompt_embeds"],
            prompt_context.negative_connector_audio_prompt_embeds[index : index + 1],
        )


def test_ltx2_omni_nft_forward_rejects_captured_batch_mismatch() -> None:
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    pipeline._inject_batch_prompt_embeds = MagicMock()
    pipeline.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))
    requests = [
        SimpleNamespace(sampling_params=SimpleNamespace(output_type="pt")),
        SimpleNamespace(sampling_params=SimpleNamespace(output_type="pt")),
    ]
    request_batch = DiffusionRequestBatch(requests=requests)

    def native_forward(*_args, **_kwargs):
        pipeline._omni_nft_clean_state = LTXAVState(video=torch.randn(1, 5, 8), audio=torch.randn(1, 7, 8))
        pipeline._omni_nft_forward_context = SimpleNamespace(
            timesteps=torch.tensor([1000.0, 500.0]),
            request_inputs=SimpleNamespace(frame_rate=24.0),
        )
        pipeline._omni_nft_prompt_context = _prompt_context(batch_size=1)
        return [
            DiffusionOutput(output=(torch.rand(1, 9, 3, 16, 16), torch.rand(1, 24000))),
            DiffusionOutput(output=(torch.rand(1, 9, 3, 16, 16), torch.rand(1, 24000))),
        ]

    with (
        patch.object(LTX2Pipeline, "forward", side_effect=native_forward),
        pytest.raises(RuntimeError, match="captured batch 1 does not match 2 requests"),
    ):
        pipeline.forward(request_batch)
