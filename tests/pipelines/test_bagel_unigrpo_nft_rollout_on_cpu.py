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

"""Exact native BAGEL token/latent capture boundaries, without model weights."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.bagel_unigrpo_nft.capture import BagelReplayCapture
from verl_omni.pipelines.bagel_unigrpo_nft.replay import replay_from_metadata
from verl_omni.pipelines.bagel_unigrpo_nft.weight_sync import native_actor_weight_name
from verl_omni.pipelines.model_base import VllmOmniPipelineBase


class _Head(torch.nn.Module):
    def forward(self, hidden):
        return torch.arange(12).float().reshape(1, 12) + hidden.reshape(1, 1)


class _LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = _Head()

    def forward(self, *, packed_text_ids, **kwargs):
        return packed_text_ids.float()


class _Bagel(torch.nn.Module):
    def __init__(self, draws):
        super().__init__()
        self.language_model = _LanguageModel()
        self.draws = draws
        self.final_latent = torch.arange(24).reshape(4, 6).float()

    def prepare_prompts(self, **kwargs):
        # A second call represents CFG, and must not replace the primary prompt.
        ids = torch.tensor([1, 4, 5, 2]) if kwargs.get("primary", True) else torch.tensor([1, 8, 2])
        return {"packed_text_ids": ids, "packed_text_position_ids": torch.arange(ids.numel())}, [len(ids)], [len(ids)]

    def generate_text(self, *, packed_start_tokens, max_length, end_token_id, **kwargs):
        cached, current = [], packed_start_tokens
        for index in range(max_length):
            cached.append(current)
            hidden = self.language_model(packed_text_ids=current)
            self.language_model.lm_head(hidden)
            current = torch.tensor([self.draws[index]])
            if int(current[0]) == end_token_id:
                break
        return torch.stack(cached)

    def generate_image(self, **kwargs):
        return [self.final_latent], None, None, None


def _collect(draws=(7, 9, 2), *, max_length=5):
    bagel = _Bagel(draws)
    capture = BagelReplayCapture(bagel)
    with capture.installed():
        bagel.prepare_prompts()
        bagel.prepare_prompts(primary=False)
        ids = bagel.generate_text(
            packed_start_tokens=torch.tensor([1]),
            packed_query_position_ids=torch.tensor([4]),
            max_length=max_length,
            end_token_id=2,
            do_sample=True,
            temperature=0.7,
        )
        context_length = 4 + len(ids)
        bagel.generate_image(
            past_key_values=SimpleNamespace(key_cache=[torch.empty(context_length, 1, 2)]),
            packed_vae_position_ids=torch.tensor([0, 1, 8, 9]),
            packed_position_ids=torch.full((6,), context_length),
            packed_text_ids=torch.tensor([10, 11]),
        )
    return capture.result(policy_version=3), bagel


def test_capture_keeps_exact_native_context_and_terminal_eos():
    record, bagel = _collect()
    assert record.prompt_token_ids.tolist() == [1, 4, 5, 2]
    assert record.ar_prompt_token_ids.tolist() == [1, 4, 5, 2, 1]
    assert record.cached_thinking_token_ids.tolist() == [1, 7, 9]
    assert record.response_token_ids.tolist() == [7, 9, 2]
    assert record.condition_token_ids.tolist() == [1, 4, 5, 2, 1, 7, 9]
    expected = torch.log_softmax(torch.arange(12).float() / 0.7, -1)[torch.tensor([7, 9, 2])]
    torch.testing.assert_close(record.rollout_log_probs, expected)
    torch.testing.assert_close(record.latents_clean, bagel.final_latent)
    bagel.final_latent.add_(100)
    torch.testing.assert_close(record.latents_clean, torch.arange(24).reshape(4, 6).float())
    assert record.policy_version == 3
    assert not bagel._joint_capture_running
    assert "generate_text" not in bagel.__dict__


def test_max_length_scores_only_observed_cached_draws():
    record, _ = _collect((7, 9, 8), max_length=3)
    assert record.cached_thinking_token_ids.tolist() == [1, 7, 9]
    assert record.response_token_ids.tolist() == [7, 9]
    assert record.rollout_log_probs.shape == (2,)


def test_immediate_eos_is_scored_but_not_image_condition():
    record, _ = _collect((2,), max_length=3)
    assert record.cached_thinking_token_ids.tolist() == [1]
    assert record.response_token_ids.tolist() == [2]
    assert record.condition_token_ids.tolist() == [1, 4, 5, 2, 1]


def test_failed_decode_removes_hooks_and_restores_instance_methods():
    bagel = _Bagel(())
    capture = BagelReplayCapture(bagel)
    with pytest.raises(IndexError), capture.installed():
        bagel.prepare_prompts()
        bagel.generate_text(
            packed_start_tokens=torch.tensor([1]),
            max_length=3,
            end_token_id=2,
            do_sample=True,
            temperature=1,
        )
    assert not bagel._joint_capture_running
    assert "generate_text" not in bagel.__dict__
    assert not bagel.language_model._forward_pre_hooks
    assert not bagel.language_model.lm_head._forward_hooks


def test_reentrant_capture_is_rejected_and_outer_capture_survives():
    bagel = _Bagel((2,))
    with BagelReplayCapture(bagel).installed():
        with pytest.raises(RuntimeError, match="Concurrent/reentrant"):
            with BagelReplayCapture(bagel).installed():
                pass
        assert bagel._joint_capture_running
    assert not bagel._joint_capture_running


@pytest.mark.parametrize(
    "field,value",
    [
        ("latents_clean", torch.full((4, 6), float("nan"))),
        ("latent_pos_ids", torch.arange(3)),
        ("image_position_ids", torch.full((6,), 99)),
        ("rollout_log_probs", torch.zeros(2)),
        ("response_token_ids", torch.tensor([7, 9, 3])),
        ("cached_thinking_token_ids", torch.tensor([1, 8, 9])),
        ("temperature", 0),
        ("policy_version", -1),
    ],
)
def test_replay_rejects_mismatched_or_nonfinite_wire_data(field, value):
    record, _ = _collect()
    with pytest.raises(ValueError):
        replace(record, **{field: value}).validate()


def test_wire_contract_is_versioned_and_never_reconstructs_from_text():
    record, _ = _collect()
    metadata = {"bagel_joint_contract_version": 1, "bagel_joint_replay": record}
    assert replay_from_metadata(metadata) is record
    with pytest.raises(ValueError):
        replay_from_metadata({"think_text": "a different tokenizer would change me"})


def test_rollout_registered_under_distinct_hybrid_algorithm():
    pipeline = VllmOmniPipelineBase.get_class("OmniBagelForConditionalGeneration", "unigrpo_nft")
    assert pipeline is not None
    assert pipeline.supports_request_batch is False
    assert pipeline.__name__ == "BagelUniGRPONFTPipeline"


def test_capture_against_installed_native_generate_text(monkeypatch):
    from vllm_omni.diffusion.models.bagel.bagel_transformer import Bagel

    class NativeLanguageModel(_LanguageModel):
        def forward(self, *, packed_text_ids, past_key_values, **kwargs):
            return SimpleNamespace(past_key_values=past_key_values, packed_query_sequence=packed_text_ids.float())

    bagel = _Bagel(())
    bagel.language_model = NativeLanguageModel()
    bagel.generate_text = Bagel.generate_text.__get__(bagel)
    draws = iter([7, 9, 2])
    monkeypatch.setattr(torch, "multinomial", lambda probs, num_samples: torch.tensor([[next(draws)]]))
    capture = BagelReplayCapture(bagel)
    with capture.installed():
        bagel.prepare_prompts()
        tokens = bagel.generate_text(
            past_key_values=SimpleNamespace(),
            packed_start_tokens=torch.tensor([1]),
            packed_query_position_ids=torch.tensor([4]),
            max_length=5,
            end_token_id=2,
            do_sample=True,
            temperature=0.7,
        )
        bagel.generate_image(
            past_key_values=SimpleNamespace(key_cache=[torch.empty(4 + len(tokens), 1, 2)]),
            packed_vae_position_ids=torch.tensor([0, 1, 8, 9]),
            packed_position_ids=torch.full((6,), 4 + len(tokens)),
            packed_text_ids=torch.tensor([10, 11]),
        )
    record = capture.result(policy_version=0)
    assert record.cached_thinking_token_ids.tolist() == [1, 7, 9]
    assert record.response_token_ids.tolist() == [7, 9, 2]
    expected = torch.log_softmax(torch.arange(12).float() / 0.7, -1)[torch.tensor([7, 9, 2])]
    torch.testing.assert_close(record.rollout_log_probs, expected)


@pytest.mark.parametrize(
    "actor,native",
    [
        ("layers.0.self_attn.q_proj.weight", "language_model.model.layers.0.self_attn.q_proj.weight"),
        ("layers.0.self_attn.q_proj_moe_gen.weight", "language_model.model.layers.0.self_attn.q_proj_moe_gen.weight"),
        ("norm_moe_gen.weight", "language_model.model.norm_moe_gen.weight"),
        ("lm_head.weight", "language_model.lm_head.weight"),
        ("vae2llm.weight", "vae2llm.weight"),
        ("llm2vae.bias", "llm2vae.bias"),
        ("time_embedder.mlp.0.weight", "time_embedder.mlp.0.weight"),
        ("latent_pos_embed.pos_embed", "latent_pos_embed.pos_embed"),
    ],
)
def test_weight_sync_maps_every_trainable_component(actor, native):
    assert native_actor_weight_name("transformer." + actor) == native


def test_unknown_actor_weight_never_silently_drops_from_policy_sync():
    with pytest.raises(ValueError, match="partial policy sync"):
        native_actor_weight_name("transformer.unknown_projection.weight")


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_missing_npu_norm_uses_native_only_on_this_pipeline(monkeypatch, device):
    from vllm_omni.diffusion.layers.custom_op import CustomOp
    from vllm_omni.diffusion.layers.mot.mot_layernorm import MoTRMSNorm
    from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline

    class ImplementedNorm(MoTRMSNorm):
        def forward_npu(self, *args, **kwargs):
            return self.forward_native(*args, **kwargs)

    def initialize(self):
        torch.nn.Module.__init__(self)
        self.device = SimpleNamespace(type=device)
        self.missing = MoTRMSNorm(4)
        self.implemented = ImplementedNorm(4)
        self.missing._forward_method = self.missing.forward_npu
        self.implemented._forward_method = self.implemented.forward_npu

    monkeypatch.setattr(BagelPipeline, "__init__", initialize)
    pipeline = VllmOmniPipelineBase.get_class("OmniBagelForConditionalGeneration", "unigrpo_nft")()
    expected = pipeline.missing.forward_native if device == "npu" else pipeline.missing.forward_npu
    assert pipeline.missing._forward_method == expected
    assert pipeline.implemented._forward_method == pipeline.implemented.forward_npu
    assert MoTRMSNorm.forward_npu is CustomOp.forward_npu
    if device == "npu":
        x = torch.arange(8).reshape(2, 4).float()
        torch.testing.assert_close(pipeline.missing(x), pipeline.missing.forward_native(x))
