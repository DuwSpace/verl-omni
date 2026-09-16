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

"""CPU contracts for the native LTX-2.3 OmniNFT rollout boundary."""

import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.ltx2_omni_nft.vllm_omni_rollout_adapter import (
    LTX23OmniNFTPipeline,
    _require_one_stage_recipe,
)
from verl_omni.pipelines.model_base import VllmOmniPipelineBase


def _request(request_id: str, prompt_ids: list[int], prompt_mask: list[int], seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        prompt={
            "prompt_token_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "negative_prompt_ids": [seed % 7 + 30],
            "negative_prompt_mask": [1],
        },
        sampling_params=SimpleNamespace(
            output_type="pt",
            extra_args={},
            max_sequence_length=4,
            seed=seed,
        ),
    )


def _bare_pipeline() -> LTX23OmniNFTPipeline:
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    pipeline._omni_nft_prompt_context = None
    pipeline._omni_nft_clean_state = None
    pipeline._omni_nft_forward_context = None
    pipeline.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))
    return pipeline


def _captured_batch(batch_size: int = 2) -> tuple[LTXAVState, SimpleNamespace, SimpleNamespace]:
    clean_state = LTXAVState(
        video=torch.arange(batch_size * 6, dtype=torch.bfloat16).reshape(batch_size, 2, 3),
        audio=(100 + torch.arange(batch_size * 9, dtype=torch.bfloat16)).reshape(batch_size, 3, 3),
    )
    timesteps = torch.tensor([999.5, 500.25, 7.75], dtype=torch.float32)
    forward_context = SimpleNamespace(
        timesteps=timesteps,
        request_inputs=SimpleNamespace(frame_rate=23.976),
    )

    def values(offset: int) -> torch.Tensor:
        return (offset + torch.arange(batch_size * 8)).reshape(batch_size, 2, 4)

    prompt_context = SimpleNamespace(
        positive_connector_prompt_embeds=values(0),
        positive_connector_audio_prompt_embeds=values(100),
        positive_connector_attention_mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        negative_connector_prompt_embeds=values(200),
        negative_connector_audio_prompt_embeds=values(300),
        negative_connector_attention_mask=torch.tensor([[1, 0], [1, 1]], dtype=torch.bool),
    )
    return clean_state, forward_context, prompt_context


def test_omni_nft_pipeline_registry_and_media_contract() -> None:
    assert VllmOmniPipelineBase.get_class("LTX2Pipeline", "omni_nft") is LTX23OmniNFTPipeline
    assert LTX23OmniNFTPipeline.supports_request_batch is True
    assert LTX23OmniNFTPipeline.diffusion_io_spec.primary.modality == "video"
    assert LTX23OmniNFTPipeline.diffusion_io_spec.auxiliary[0].modality == "audio"
    assert LTX23OmniNFTPipeline.diffusion_io_spec.auxiliary[0].sample_rate == 24000


def test_omni_nft_rejects_multi_phase_or_mixed_output_recipes() -> None:
    _require_one_stage_recipe(SimpleNamespace(phases=[object()], video_output_phase=0, audio_output_phase=0))

    with pytest.raises(NotImplementedError, match="requires one native denoising phase"):
        _require_one_stage_recipe(
            SimpleNamespace(phases=[object(), object()], video_output_phase=1, audio_output_phase=0)
        )
    with pytest.raises(NotImplementedError, match="requires one native denoising phase"):
        _require_one_stage_recipe(SimpleNamespace(phases=[object()], video_output_phase=0, audio_output_phase=1))


def test_native_clean_state_capture_matches_pinned_private_hook(monkeypatch) -> None:
    assert tuple(inspect.signature(LTX2Pipeline._denoise_step).parameters) == (
        "self",
        "index",
        "timestep",
        "state",
        "forward_ctx",
        "denoise_ctx",
    )
    pipeline = _bare_pipeline()
    first = LTXAVState(video=torch.full((1, 2, 3), 1.0), audio=torch.full((1, 3, 3), 2.0))
    final = LTXAVState(video=torch.full((1, 2, 3), 3.0), audio=torch.full((1, 3, 3), 4.0))
    states = iter((first, final))
    monkeypatch.setattr(LTX2Pipeline, "_denoise_step", lambda *args, **kwargs: next(states))
    forward_context = SimpleNamespace()
    denoise_context = SimpleNamespace()
    input_state = LTXAVState(video=torch.zeros(1, 2, 3), audio=torch.zeros(1, 3, 3))

    pipeline._denoise_step(0, torch.tensor(900.5), input_state, forward_context, denoise_context)
    returned = pipeline._denoise_step(1, torch.tensor(10.25), input_state, forward_context, denoise_context)

    assert returned is final
    assert pipeline._omni_nft_clean_state is final
    assert pipeline._omni_nft_forward_context is forward_context


def test_token_id_prompt_batch_preserves_per_request_masks_and_seeds() -> None:
    pipeline = _bare_pipeline()
    pipeline.device = torch.device("cpu")
    pipeline.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    pipeline.tokenizer_max_length = 4
    requests = [
        _request("r0", [10, 11], [1, 1], 101),
        _request("r1", [20, 21, 22], [1, 0, 1], 202),
    ]
    request_batch = DiffusionRequestBatch(requests=requests)
    observed = {}

    def encode_rows(token_ids: torch.Tensor, attention_mask: torch.Tensor):
        observed["token_ids"] = token_ids.clone()
        observed["attention_mask"] = attention_mask.clone()
        inverse = torch.arange(token_ids.shape[0])
        return token_ids.unsqueeze(-1).float(), attention_mask, inverse

    pipeline._encode_unique_prompt_rows = encode_rows
    pipeline._inject_batch_prompt_embeds(request_batch)

    assert [request.sampling_params.seed for request in requests] == [101, 202]
    assert observed["token_ids"][:2].tolist() == [[0, 0, 10, 11], [0, 20, 21, 22]]
    assert observed["attention_mask"][:2].tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
    assert requests[0].prompt["prompt_embeds"].squeeze(-1).tolist() == [0.0, 0.0, 10.0, 11.0]
    assert requests[1].prompt["prompt_attention_mask"].tolist() == [0, 1, 0, 1]
    assert requests[0].prompt["negative_prompt_embeds"].shape == (4, 1)
    assert requests[1].prompt["negative_prompt_attention_mask"].tolist() == [0, 0, 0, 1]


def test_packed_forward_splits_all_rollout_fields_and_releases_state(monkeypatch) -> None:
    pipeline = _bare_pipeline()
    pipeline._inject_batch_prompt_embeds = lambda request_batch: None
    requests = [
        _request("r0", [10, 11], [1, 1], 101),
        _request("r1", [20, 21], [1, 1], 202),
    ]
    request_batch = DiffusionRequestBatch(requests=requests)
    clean_state, forward_context, prompt_context = _captured_batch()

    def native_forward(self, req, **kwargs):
        self._omni_nft_clean_state = clean_state
        self._omni_nft_forward_context = forward_context
        self._omni_nft_prompt_context = prompt_context
        return [
            DiffusionOutput(
                output=(
                    torch.full((1, 3, 1, 2, 2), float(index + 1)),
                    torch.full((1, 2, 5), float(index + 11)),
                )
            )
            for index in range(req.num_reqs)
        ]

    monkeypatch.setattr(LTX2Pipeline, "forward", native_forward)
    outputs = pipeline.forward(request_batch)

    assert len(outputs) == 2
    for index, output in enumerate(outputs):
        assert output.to_cpu is True
        assert output.trajectory_latents is None
        assert output.trajectory_timesteps is None
        assert output.trajectory_log_probs is None
        video, audio = output.output["payload"]["video"]
        assert video.shape == (3, 1, 2, 2)
        assert audio.shape == (1, 2, 5)
        assert video.unique().item() == index + 1
        assert audio.unique().item() == index + 11

        metadata = output.output["metadata"]
        prompt_fields = metadata["prompt_embeddings"]
        rollout_fields = metadata["rl"]
        assert prompt_fields["prompt_embeds"][0, 0, 0].item() == index * 8
        assert prompt_fields["audio_prompt_embeds"][0, 0, 0].item() == 100 + index * 8
        assert prompt_fields["prompt_embeds_mask"].tolist() == [[[True, True], [True, False]][index]]
        assert rollout_fields["video_latents_clean"].shape == (1, 2, 3)
        assert rollout_fields["audio_latents_clean"].shape == (1, 3, 3)
        assert rollout_fields["video_latents_clean"].dtype == torch.float32
        assert rollout_fields["audio_latents_clean"].dtype == torch.float32
        assert rollout_fields["video_latent_shape"].tolist() == [[2, 3]]
        assert rollout_fields["audio_latent_shape"].tolist() == [[3, 3]]
        assert rollout_fields["video_seq_len"].tolist() == [2]
        assert rollout_fields["audio_seq_len"].tolist() == [3]
        assert rollout_fields["train_timesteps"].tolist() == [[999.5, 500.25, 7.75]]
        assert rollout_fields["fps"].item() == pytest.approx(23.976)
        assert rollout_fields["audio_sample_rate"].item() == 24000
        for value in (*prompt_fields.values(), *rollout_fields.values()):
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"

    assert pipeline._omni_nft_prompt_context is None
    assert pipeline._omni_nft_clean_state is None
    assert pipeline._omni_nft_forward_context is None


def test_forward_releases_captured_state_after_native_error(monkeypatch) -> None:
    pipeline = _bare_pipeline()
    pipeline._inject_batch_prompt_embeds = lambda request_batch: None
    request_batch = DiffusionRequestBatch(requests=[_request("r0", [10], [1], 101)])
    clean_state, forward_context, prompt_context = _captured_batch(batch_size=1)

    def failing_forward(self, req, **kwargs):
        self._omni_nft_clean_state = clean_state
        self._omni_nft_forward_context = forward_context
        self._omni_nft_prompt_context = prompt_context
        raise RuntimeError("native denoise failed")

    monkeypatch.setattr(LTX2Pipeline, "forward", failing_forward)

    with pytest.raises(RuntimeError, match="native denoise failed"):
        pipeline.forward(request_batch)
    assert pipeline._omni_nft_prompt_context is None
    assert pipeline._omni_nft_clean_state is None
    assert pipeline._omni_nft_forward_context is None
