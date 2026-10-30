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

"""Real tiny Gemma teacher forcing and LoRA gradients through frozen connectors."""

import pytest
import torch
from peft import LoraConfig
from transformers import Gemma3Config, Gemma3ForConditionalGeneration, Gemma3TextConfig, SiglipVisionConfig

from verl_omni.pipelines.ltx2_unigrpo_nft.model import LTXJointModel


class Connector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(32, 16)

    def forward(self, hidden, attention_mask, padding_side):
        assert padding_side == "left"
        result = self.proj(hidden)
        return result, result * 0.5, attention_mask


class DiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 4)
        self.config = {"model_type": "tiny_ltx_test"}

    def forward(self, encoder_hidden_states, audio_encoder_hidden_states, **kwargs):
        return self.proj(encoder_hidden_states), self.proj(audio_encoder_hidden_states)


@pytest.fixture
def model():
    torch.manual_seed(7)
    text = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        sliding_window=16,
        layer_types=["full_attention"],
        pad_token_id=0,
    )
    vision = SiglipVisionConfig(
        hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, image_size=8, patch_size=4
    )
    gemma = Gemma3ForConditionalGeneration(Gemma3Config(text_config=text, vision_config=vision))
    joint = LTXJointModel(gemma, DiT(), Connector())
    joint.add_adapter(
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["gemma.model.language_model.layers.0.self_attn.q_proj", "transformer.proj"],
        )
    )
    joint.eval()
    return joint


def test_teacher_forcing_matches_real_gemma_logits_and_masks(model):
    ids = torch.tensor([[0, 2, 3, 4, 1], [2, 3, 5, 1, 0]])
    attention = ids != 0
    response = torch.tensor([[False, False, False, True, True], [False, False, True, True, False]])
    temperature = 0.8
    actual = model(
        task="recaption", input_ids=ids, attention_mask=attention, response_mask=response, temperature=temperature
    )
    logits = model.gemma(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1].float()
    expected = (logits / temperature).log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    torch.testing.assert_close(actual[response[:, 1:]], expected[response[:, 1:]])
    assert torch.count_nonzero(actual[~response[:, 1:]]) == 0
    (-actual.sum()).backward()
    assert (
        sum(p.grad.abs().sum() for n, p in model.named_parameters() if n.startswith("gemma.") and p.grad is not None)
        > 0
    )
    assert all(p.grad is None for p in model.transformer.parameters())


def test_av_trains_only_dit_with_frozen_base_condition(model):
    ids = torch.tensor([[2, 3, 4, 1]])
    video, audio = model(task="av", input_ids=ids, attention_mask=torch.ones_like(ids))
    (video.square().mean() + audio.square().mean()).backward()
    assert all(p.grad is None for p in model.gemma.parameters())
    for prefix in ("transformer.",):
        gradients = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    assert all(p.grad is None and not p.requires_grad for p in model.connectors.parameters())
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    before = model.encode_condition(ids, torch.ones_like(ids))[0].detach().clone()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    optimizer.step()
    after = model.encode_condition(ids, torch.ones_like(ids))[0].detach()
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_fp32_base_gemma_produces_frozen_bf16_condition(model):
    model.connectors.bfloat16()
    model.transformer.bfloat16()
    ids = torch.tensor([[2, 3, 4, 1]])
    video, audio = model(task="av", input_ids=ids, attention_mask=torch.ones_like(ids))
    assert video.dtype == audio.dtype == torch.bfloat16
    (video.float().square().mean() + audio.float().square().mean()).backward()
    gradients = [p.grad for p in model.gemma.parameters() if p.requires_grad]
    assert gradients and all(g is None for g in gradients)
    assert all(p.grad is None for p in model.connectors.parameters())


def test_missing_response_context_rejected(model):
    with pytest.raises(ValueError, match="preceding"):
        model(
            task="recaption",
            input_ids=torch.tensor([[2, 3]]),
            attention_mask=torch.ones(1, 2),
            response_mask=torch.ones(1, 2, dtype=torch.bool),
        )


def test_joint_policy_snapshots_preserve_pair_and_initial_reference_on_resume(model):
    import io

    from verl_omni.pipelines.ltx2_unigrpo_nft.policy_state import LTXJointPolicyState

    policy = LTXJointPolicyState(model)
    initial = policy.state_dict()["reference"]
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for p in parameters.values():
            p.add_(1)
    policy.begin_rollout(version=0)
    with torch.no_grad():
        for p in parameters.values():
            p.add_(2)
    for state, offset in (("old", 1), ("ref", 0)):
        with policy.use(state):
            assert not torch.is_grad_enabled()
            for name, p in parameters.items():
                torch.testing.assert_close(p, initial[name] + offset)
        for name, p in parameters.items():
            torch.testing.assert_close(p, initial[name] + 3)
    with pytest.raises(ValueError, match="strictly increasing"):
        policy.begin_rollout(version=0)
    buffer = io.BytesIO()
    torch.save(policy.state_dict(), buffer)
    buffer.seek(0)
    restored = LTXJointPolicyState(model)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert restored.rollout_started and restored.version == 0
    with restored.use("ref"):
        for name, p in parameters.items():
            torch.testing.assert_close(p, initial[name])
    # Exported/checkpoint tensors cannot modify the live frozen snapshots.
    exported = restored.state_dict()
    for tensor in exported["reference"].values():
        tensor.add_(100)
    for name in initial:
        torch.testing.assert_close(restored.reference[name], initial[name])


@pytest.mark.parametrize("snapshot", ["old", "reference"])
def test_joint_policy_detects_frozen_snapshot_mutation(model, snapshot):
    from verl_omni.pipelines.ltx2_unigrpo_nft.policy_state import LTXJointPolicyState

    policy = LTXJointPolicyState(model)
    policy.begin_rollout(version=1)
    policy.assert_frozen()
    next(iter(getattr(policy, snapshot).values())).add_(1)
    for action in (policy.assert_frozen, policy.state_dict, lambda: policy.begin_rollout(version=2)):
        with pytest.raises(RuntimeError, match=f"Frozen joint {snapshot}"):
            action()


def test_joint_policy_restores_both_components_after_failure(model):
    from verl_omni.pipelines.ltx2_unigrpo_nft.policy_state import LTXJointPolicyState

    policy = LTXJointPolicyState(model)
    policy.begin_rollout(version=1)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(1)
    current = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    with pytest.raises(RuntimeError, match="failed"), policy.use("old"):
        raise RuntimeError("failed")
    for name, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p, current[name])


def test_joint_policy_rejects_checkpoint_missing_one_component_without_mutation(model):
    from verl_omni.pipelines.ltx2_unigrpo_nft.policy_state import LTXJointPolicyState

    policy = LTXJointPolicyState(model)
    checkpoint = policy.state_dict()
    checkpoint["old"] = {n: p for n, p in checkpoint["old"].items() if n.startswith("gemma.")}
    before = policy.state_dict()
    with pytest.raises(ValueError, match="Missing/extra"):
        policy.load_state_dict(checkpoint)
    for name in before["reference"]:
        torch.testing.assert_close(policy.reference[name], before["reference"][name])
        torch.testing.assert_close(policy.old[name], before["old"][name])


@pytest.mark.parametrize(
    "key,value", [("policy_version", True), ("rollout_started", 1), ("joint_policy_kind", "other")]
)
def test_joint_policy_rejects_ambiguous_checkpoint_metadata(model, key, value):
    from verl_omni.pipelines.ltx2_unigrpo_nft.policy_state import LTXJointPolicyState

    policy = LTXJointPolicyState(model)
    state = policy.state_dict()
    state[key] = value
    with pytest.raises(ValueError):
        policy.load_state_dict(state)
    assert not policy.rollout_started


def _joint_batch(model):
    import numpy as np
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu

    from verl_omni.pipelines.ltx2_unigrpo_nft.replay import LTXJointReplay

    records = []
    for token in (4, 5):
        record = LTXJointReplay(
            original_prompt="A bell rings.",
            recaption=f"A bell rings with event {token}.",
            prompt_token_ids=torch.tensor([2, 3]),
            response_token_ids=torch.tensor([token, 1]),
            rollout_logprobs=torch.zeros(2),
            condition_token_ids=torch.tensor([2, token, 3, 1]),
            condition_attention_mask=torch.ones(4, dtype=torch.bool),
            policy_version=0,
            seed=token,
            temperature=1.0,
            eos_token_ids=(1,),
            generation_params={
                "height": 32,
                "width": 32,
                "num_frames": 25,
                "frame_rate": 24.0,
                "num_inference_steps": 2,
            },
        )
        with torch.no_grad():
            logprobs = model(task="recaption", **record.ar_inputs(torch.device("cpu")))
            record.rollout_logprobs = logprobs[0, -2:].clone()
        records.append(record)
    data = TensorDict(
        {
            "video_latents_clean": torch.randn(2, 4, 4),
            "audio_latents_clean": torch.randn(2, 4, 4),
            "video_forward_noise": torch.randn(2, 2, 4, 4),
            "audio_forward_noise": torch.randn(2, 2, 4, 4),
            "train_timesteps": torch.tensor([[200.0, 700.0], [300.0, 800.0]]),
            "reward_prob": torch.tensor([[[0.8, 0.7], [0.8, 0.7]], [[0.2, 0.3], [0.2, 0.3]]]),
            "text_advantages": torch.tensor([1.0, -1.0]),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, ltx_joint_replay=np.array(records, dtype=object))
    return data, records


@pytest.mark.parametrize("override_sampling", [False, True])
def test_joint_generate_exports_frozen_pair_and_preserves_av_reward_replay(model, monkeypatch, override_sampling):
    from types import SimpleNamespace

    from verl.utils import tensordict_utils as tu

    from verl_omni.pipelines.ltx2_unigrpo_nft.hooks import LTXJointHooks
    from verl_omni.workers.config.diffusion import LTXJointRolloutConfig
    from verl_omni.workers.rollout import ltx_joint_backend

    data, records = _joint_batch(model)
    events = []

    class Backend:
        def __init__(self, **kwargs):
            assert kwargs["geometry"]["extra_args"]["video_cfg_scale"] == 1.0
            assert kwargs["recaption_prompt"] == "ltx_t2v"

        def load_actor_policy(self, module, version):
            assert module is model and hooks.policies.rollout_started and hooks.policies.version == version == 1
            for name, parameter in hooks.policies._parameters().items():
                expected = initial[name] + 1.0
                torch.testing.assert_close(parameter, expected)
            events.append("sync")

        def generate(self, prompts, seeds, version, sampling_params=None):
            if override_sampling:
                assert sampling_params["num_inference_steps"] == 40
                assert sampling_params["extra_args"]["audio_cfg_scale"] == 7.0
            else:
                assert sampling_params is None
            assert events == ["sync"] and prompts == ["A bell rings."] * 2 and seeds == [4, 5]
            events.append("generate")
            return [
                {
                    "replay": record,
                    "video": torch.full((25, 3, 32, 32), 0.5),
                    "audio": torch.ones(2, 480),
                    "audio_sample_rate": 48000,
                    "fps": 24.0,
                    "video_latents_clean": data["video_latents_clean"][i : i + 1],
                    "audio_latents_clean": data["audio_latents_clean"][i : i + 1],
                    "train_timesteps": data["train_timesteps"][i : i + 1],
                    "recaption_time": 0.1,
                    "generation_time": 0.2,
                }
                for i, record in enumerate(records)
            ]

        def close(self):
            events.append("close")

    monkeypatch.setattr(ltx_joint_backend, "LTXJointVllmBackend", Backend)
    geometry = SimpleNamespace(**records[0].generation_params, max_sequence_length=32)
    hooks = LTXJointHooks(
        model,
        SimpleNamespace(
            ltx_joint=LTXJointRolloutConfig(recaption_prompt="ltx_t2v"), pipeline=geometry, local_path="fixture"
        ),
        None,
    )
    monkeypatch.setattr(hooks, "_device", lambda: torch.device("cpu"))
    hooks.begin_rollout(version=0)
    initial = hooks.policies._snapshot()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(2)
    current = hooks.policies._snapshot()
    hooks.ema_update_adapter(source="default", target="old", decay=0.5)
    request_data = {"raw_prompt": ["A bell rings."] * 2, "_ltx_policy_version": [1, 1], "_ltx_sample_seed": [4, 5]}
    if override_sampling:
        request_data["_ltx_sampling_params"] = [{"num_inference_steps": 40, "extra_args": {"audio_cfg_scale": 7.0}}] * 2
    result = hooks.generate(tu.get_tensordict(request_data))
    for name, parameter in hooks.policies._parameters().items():
        torch.testing.assert_close(parameter, current[name])
    assert result["responses"].shape == (2, 25, 3, 32, 32)
    assert result["responses"].dtype == torch.uint8
    assert result["audio_sample_rate"].tolist() == [48000, 48000]
    torch.testing.assert_close(result["video_latents_clean"], data["video_latents_clean"])
    assert tu.get(result, "reward_inputs")[0]["text"]["video"] == records[0].original_prompt
    assert tu.get(result, "ltx_joint_replay")[1].seed == 5
    hooks.close_rollout()
    assert events == ["sync", "generate", "close"]


def test_joint_hooks_accumulate_both_losses_before_engine_step_and_keep_old_ref(model):
    from functools import partial
    from types import SimpleNamespace

    from verl.workers.engine.base import BaseEngine

    from verl_omni.pipelines.ltx2_unigrpo_nft.hooks import LTXJointHooks
    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    cfg = SimpleNamespace(diffusion_loss=LTXJointLossConfig(ar_clip_ratio=0.2), ppo_micro_batch_size_per_gpu=1)
    hooks = LTXJointHooks(model, None, None)
    hooks._device = lambda: torch.device("cpu")
    hooks.begin_rollout(version=0)
    frozen = hooks.state_dict()
    data, _ = _joint_batch(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    events = []

    def zero():
        events.append("zero")
        optimizer.zero_grad()

    def backward(batch, loss, forward_only):
        events.append("backward")
        result = hooks.forward_backward_batch(batch, loss, forward_only)
        # The hooks must not change parameters themselves.
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, before[name])
        for prefix in ("gemma.", "transformer."):
            assert (
                sum(
                    p.grad.abs().sum()
                    for n, p in model.named_parameters()
                    if n.startswith(prefix) and p.grad is not None
                )
                > 0
            )
        return result

    def step():
        events.append("step")
        optimizer.step()
        return 1.0

    engine = SimpleNamespace(
        optimizer_zero_grad=zero,
        forward_backward_batch=backward,
        optimizer_step=step,
        is_mp_src_rank_with_outputs=lambda: False,
    )
    output = BaseEngine.train_batch(engine, data, partial(LTXJointLoss(), config=cfg))
    assert events == ["zero", "backward", "step"]
    assert output["metrics"]["consistency/old_logprob_abs_max"][0] < 1e-5
    for prefix in ("gemma.", "transformer."):
        assert any(
            not torch.equal(p, before[n])
            for n, p in model.named_parameters()
            if n.startswith(prefix) and p.requires_grad
        )
    for name, p in model.named_parameters():
        if not p.requires_grad:
            torch.testing.assert_close(p, before[name])
    after = hooks.state_dict()
    for policy in ("old", "reference"):
        for name in frozen[policy]:
            torch.testing.assert_close(after[policy][name], frozen[policy][name])
    # Another mini-batch uses exactly the same old/ref and version.
    optimizer.zero_grad()
    hooks.forward_backward_batch(data, partial(LTXJointLoss(), config=cfg))
    for policy in ("old", "reference"):
        for name in frozen[policy]:
            torch.testing.assert_close(hooks.state_dict()[policy][name], frozen[policy][name])


@pytest.mark.parametrize("source", ["modality_probability", "combined_reward"])
def test_joint_actor_preparation_selects_text_credit_without_changing_av(model, source):
    from types import SimpleNamespace

    import numpy as np
    from verl import DataProto

    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    data, records = _joint_batch(model)
    data["rm_scores"] = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    names = ["video_align", "hpsv3", "audiobox", "clap", "desync"]
    routes = [(1.0, 0.0), (1.5, 0.0), (0.0, 0.5), (0.0, 1.0), (1.0, 1.0)]
    config = SimpleNamespace(
        algorithm=SimpleNamespace(
            norm_adv_by_std_in_grpo=True, global_std=True, adv_mode="continuous", timestep_fraction=1.0
        ),
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(diffusion_loss=LTXJointLossConfig(text_advantage_source=source), data_loader_seed=42)
        ),
        reward=SimpleNamespace(
            reward_functions={
                name: {"routing_weights": {"video": v, "audio": a}} for name, (v, a) in zip(names, routes, strict=True)
            }
        ),
    )
    batch = DataProto(
        batch=data,
        non_tensor_batch={"uid": np.array(["a", "a"]), "ltx_joint_replay": np.array(records, dtype=object)},
        meta_info={"reward_names": names},
    )
    result = LTXJointLoss.prepare_actor_batch(batch, data["rm_scores"], config)
    probability = result.batch["reward_prob"][:, 0].mean(-1)
    scalar = probability if source == "modality_probability" else data["rm_scores"].sum(-1)
    expected = (scalar - scalar.mean()) / (scalar.std(correction=0) + 1e-6)
    torch.testing.assert_close(result.batch["text_reward_prob"], probability)
    torch.testing.assert_close(result.batch["text_advantages"], expected)
    # Video favors sample 1, audio favors sample 0 more strongly.
    torch.testing.assert_close(probability, torch.tensor([0.525, 0.475]), atol=1e-5, rtol=0)
    torch.testing.assert_close(result.batch["text_reward"], scalar)
    assert result.meta_info["text_reward_source"] == source
    if source == "modality_probability":
        assert result.batch["text_advantages"][0] > 0 > result.batch["text_advantages"][1]
    else:
        # Equal raw totals must give zero text credit despite different AV preferences.
        torch.testing.assert_close(result.batch["text_advantages"], torch.zeros(2))
    assert result.batch["reward_prob"].shape == (2, 2, 2)


def test_joint_training_and_loss_are_registered():
    from verl_omni.pipelines.ltx2_unigrpo_nft.diffusers_training_adapter import LTXJointTrainingAdapter
    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.pipelines.model_base import DiffusionModelBase
    from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn

    assert DiffusionModelBase.get_class_by_name("LTX2Pipeline", "ltx2_unigrpo_nft") is LTXJointTrainingAdapter
    assert isinstance(get_diffusion_loss_fn("ltx2_unigrpo_nft"), LTXJointLoss)


@pytest.mark.parametrize("text_weight,av_weight", [(1.0, 0.0), (0.0, 1.0)])
def test_each_joint_loss_contributes_to_the_correct_parameters(model, text_weight, av_weight):
    from functools import partial
    from types import SimpleNamespace

    from verl_omni.pipelines.ltx2_unigrpo_nft.hooks import LTXJointHooks
    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    cfg = SimpleNamespace(
        diffusion_loss=LTXJointLossConfig(text_loss_weight=text_weight, av_loss_weight=av_weight),
        ppo_micro_batch_size_per_gpu=1,
    )
    hooks = LTXJointHooks(model, None, None)
    hooks._device = lambda: torch.device("cpu")
    hooks.begin_rollout(version=0)
    data, _ = _joint_batch(model)
    result = hooks.forward_backward_batch(data, partial(LTXJointLoss(), config=cfg))
    metrics = result["metrics"]
    assert (metrics["train/text_gemma_grad_norm_local"][0] > 0) == bool(text_weight)
    assert metrics["train/text_dit_grad_norm_local"][0] == 0
    assert metrics["train/av_gemma_grad_norm_local"][0] == 0
    assert (metrics["train/av_dit_grad_norm_local"][0] > 0) == bool(av_weight)
    assert metrics["train/text_nonfinite_gradient_elements_local"][0] == 0
    assert metrics["train/av_nonfinite_gradient_elements_local"][0] == 0
    gemma = sum(
        p.grad.abs().sum() for n, p in model.named_parameters() if n.startswith("gemma.") and p.grad is not None
    )
    dit = sum(
        p.grad.abs().sum() for n, p in model.named_parameters() if n.startswith("transformer.") and p.grad is not None
    )
    assert (gemma > 0) == bool(text_weight)
    assert (dit > 0) if av_weight else (dit == 0)


@pytest.mark.parametrize("corruption", ["version", "probability", "geometry", "noise"])
def test_joint_hooks_reject_inconsistent_replay_before_backward(model, corruption):
    from functools import partial
    from types import SimpleNamespace

    from verl_omni.pipelines.ltx2_unigrpo_nft.hooks import LTXJointHooks
    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    cfg = SimpleNamespace(diffusion_loss=LTXJointLossConfig(), ppo_micro_batch_size_per_gpu=1)
    hooks = LTXJointHooks(model, None, None)
    hooks._device = lambda: torch.device("cpu")
    hooks.begin_rollout(version=0)
    data, records = _joint_batch(model)
    if corruption == "version":
        records[0].policy_version = 7
    elif corruption == "probability":
        data["reward_prob"][0, 0, 0] = float("nan")
    elif corruption == "geometry":
        records[0].generation_params["width"] = 64
    else:
        data["video_forward_noise"][0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError):
        hooks.forward_backward_batch(data, partial(LTXJointLoss(), config=cfg))
    assert all(p.grad is None for p in model.parameters())


def _rollout_gemma(model):
    gemma = Gemma3ForConditionalGeneration(model.gemma.config)
    state = {
        name.replace(".base_layer.", "."): tensor
        for name, tensor in model.gemma.state_dict().items()
        if ".lora_" not in name
    }
    gemma.load_state_dict(state, strict=True)
    gemma.eval()
    return gemma


@pytest.mark.parametrize("selected_layers", [(0, 1), (1,)])
def test_joint_bundle_loads_exact_gemma_adapter_and_commits_pair(model, tmp_path, monkeypatch, selected_layers):
    import json
    import re
    from types import SimpleNamespace

    import peft.tuners.tuners_utils as tuner_utils
    from peft import LoraConfig

    from verl_omni.pipelines.ltx2_unigrpo_nft.weight_sync import write_joint_lora_bundle
    from verl_omni.workers.rollout.ltx_joint_worker_extension import LTXJointWorkerExtension

    monkeypatch.setattr(tuner_utils, "MIN_TARGET_MODULES_FOR_OPTIMIZATION", 1)
    # Two language o_proj targets collapse to one suffix under real PEFT
    # injection (Siglip uses out_proj), reproducing the full model's rewrite.
    from copy import deepcopy

    config = deepcopy(model.gemma.config)
    config.text_config.num_hidden_layers = 2
    config.text_config.layer_types = ["full_attention", "full_attention"]
    model = LTXJointModel(Gemma3ForConditionalGeneration(config), DiT(), Connector())
    model.add_adapter(
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=[
                *(f"gemma.model.language_model.layers.{layer}.self_attn.o_proj" for layer in selected_layers),
                "transformer.proj",
            ],
        )
    )
    model.eval()

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.03)
    weights = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    bundle = tmp_path / "policy-0"
    manifest = write_joint_lora_bundle(bundle, weights, model.peft_config["default"], version=0, base_model="/model")
    pattern = json.loads((bundle / "dit" / "adapter_config.json").read_text())["target_modules"]
    assert isinstance(pattern, str)
    assert re.search(pattern, "transformer.proj")
    assert not re.search(pattern, "transformerXproj")
    assert not re.search(pattern, "other.transformer.proj")
    assert not re.search(pattern, "transformer.proj.extra")
    native = _rollout_gemma(model)
    activated = []
    pipeline = SimpleNamespace(
        text_encoder=native,
        transformer=model.transformer,
        device=torch.device("cpu"),
        od_config=SimpleNamespace(model="/model"),
    )
    worker = LTXJointWorkerExtension()
    worker.model_runner = SimpleNamespace(pipeline=pipeline)
    worker.lora_manager = SimpleNamespace(set_active_adapter=lambda request, scale: activated.append((request, scale)))
    pipeline.od_config.model = "/different-model"
    with pytest.raises(ValueError, match="base model differs"):
        worker.load_joint_lora_bundle(str(bundle))
    assert not activated and not hasattr(pipeline, "_joint_sync_in_progress")
    assert not hasattr(native, "peft_config")
    pipeline.od_config.model = "/model"
    result = worker.load_joint_lora_bundle(str(bundle))
    assert result["policy_version"] == pipeline._joint_policy_version == 0
    assert set(result["components"]) == {"gemma", "dit"}
    native_layers = {
        int(re.search(r"layers\.(\d+)\.", name)[1]) for name, _ in native.named_parameters() if ".lora_" in name
    }
    assert native_layers == set(selected_layers)
    assert not pipeline._joint_sync_in_progress
    assert activated[0][0].lora_int_id == 1 and activated[0][1] == 1.0
    ids = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        expected = model.gemma(input_ids=ids, use_cache=False).logits
        actual = native(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected)
    assert manifest["components"]["gemma"]["weights_sha256"]
    with pytest.raises(ValueError, match="increase"):
        worker.load_joint_lora_bundle(str(bundle))
    original_config = LoraConfig.from_pretrained(bundle / "gemma").to_dict()
    if len(selected_layers) > 1:
        # Multiple targets exercise PEFT compression; a single target need not compress.
        assert native.peft_config["joint"].to_dict() != original_config
    assert pipeline._joint_gemma_config == original_config
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("gemma.") and "lora_B" in name:
                parameter.add_(0.01)
    newer = tmp_path / "policy-1"
    weights = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    write_joint_lora_bundle(newer, weights, model.peft_config["default"], version=1, base_model="/model")
    worker.load_joint_lora_bundle(str(newer))
    assert pipeline._joint_policy_version == 1
    assert len(activated) == 2 and activated[-1][0].lora_int_id == 2
    with torch.no_grad():
        torch.testing.assert_close(
            native(input_ids=ids, use_cache=False).logits, model.gemma(input_ids=ids, use_cache=False).logits
        )


def test_joint_bundle_tampering_is_rejected_before_live_policy_mutation(model, tmp_path):
    from verl_omni.pipelines.ltx2_unigrpo_nft.weight_sync import validate_joint_lora_bundle, write_joint_lora_bundle

    weights = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    bundle = tmp_path / "policy-0"
    write_joint_lora_bundle(bundle, weights, model.peft_config["default"], version=0, base_model="/model")
    with open(bundle / "dit" / "adapter_model.safetensors", "ab") as stream:
        stream.write(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        validate_joint_lora_bundle(bundle)
    incomplete = {name: tensor for name, tensor in weights.items() if name.startswith("gemma.")}
    with pytest.raises(ValueError, match="both Gemma and DiT"):
        write_joint_lora_bundle(
            tmp_path / "incomplete", incomplete, model.peft_config["default"], version=1, base_model="/model"
        )
    assert not (tmp_path / "incomplete").exists()


@pytest.mark.parametrize("offloaded", [False, True])
@pytest.mark.parametrize("preset", ["planner", "ltx_t2v"])
def test_worker_recaption_sampling_logprobs_match_teacher_forcing(model, monkeypatch, offloaded, preset):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from verl_omni.pipelines.ltx2_unigrpo_nft.replay import recaption_messages
    from verl_omni.pipelines.ltx2_unigrpo_nft.vllm_omni_rollout_adapter import LTXJointPipeline

    # Use the real chat-template API: a stub hid its BatchEncoding default on
    # the installed Transformers version during the first NPU rollout.
    vocabulary = Tokenizer(WordLevel({f"token{i}": i for i in range(32)}, unk_token="token2"))
    vocabulary.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=vocabulary,
        pad_token="token0",
        eos_token="token1",
        unk_token="token2",
        chat_template="token2 token3",
    )
    model.gemma.generation_config.eos_token_id = 1
    active = []
    if offloaded:
        from vllm_omni.diffusion.offloader import sequential_backend

        @contextmanager
        def residency(module):
            assert module is model.gemma
            active.append(True)
            try:
                yield
            finally:
                active.pop()

        original_generate = model.gemma.generate

        def generate(*args, **kwargs):
            assert active, "Gemma must be resident before generate() prepares its inputs"
            return original_generate(*args, **kwargs)

        monkeypatch.setattr(sequential_backend, "sequential_offload_component", residency)
        monkeypatch.setattr(model.gemma, "generate", generate)
    worker = SimpleNamespace(
        tokenizer=tokenizer,
        text_encoder=model.gemma,
        device=torch.device("cpu"),
        od_config=SimpleNamespace(enable_cpu_offload=offloaded),
    )
    apply_template = tokenizer.apply_chat_template

    def checked_template(messages, **kwargs):
        assert messages == recaption_messages("A bell rings.", preset=preset)
        return apply_template(messages, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", checked_template)
    text, prompt, response, logprobs, eos = LTXJointPipeline._recaption(
        worker, "A bell rings.", seed=13, temperature=0.8, max_new_tokens=3, recaption_prompt=preset
    )
    assert prompt.tolist() == [2, 3]
    assert not active
    ids = torch.cat((prompt, response)).unsqueeze(0)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, prompt.numel() :] = True
    replay = model(
        task="recaption", input_ids=ids, attention_mask=torch.ones_like(ids), response_mask=mask, temperature=0.8
    )
    torch.testing.assert_close(replay[mask[:, 1:]], logprobs, atol=1e-5, rtol=1e-5)
    assert text and eos == (1,)


def test_joint_checkpoint_config_saves_both_model_components(model, tmp_path):
    from verl_omni.pipelines.ltx2_unigrpo_nft.model import LTXJointConfig

    model.config.base_model_path = "/original-joint-snapshot"
    model.config.save_pretrained(tmp_path)
    restored = LTXJointConfig.from_pretrained(tmp_path)
    assert restored.model_type == "ltx2_unigrpo_nft"
    assert restored.architectures == ["LTXJointModel"]
    assert Gemma3Config.from_dict(restored.gemma_config).to_dict() == model.gemma.config.to_dict()
    assert restored.transformer_config == model.transformer.config
    assert restored.connectors_config == {}
    assert restored.base_model_path == "/original-joint-snapshot"


def test_nonaffine_norm_replacement_preserves_reference_values_gradients_and_parameters():
    from diffusers.models.normalization import RMSNorm

    from verl_omni.pipelines.ltx2_unigrpo_nft.model import NonAffineRMSNorm, _replace_nonaffine_rmsnorm

    layers = torch.nn.Sequential(RMSNorm(8, 1e-6, elementwise_affine=False), RMSNorm(8, 1e-6))
    affine = layers[1]
    before = {name: value.clone() for name, value in layers.state_dict().items()}
    _replace_nonaffine_rmsnorm(layers)
    assert isinstance(layers[0], NonAffineRMSNorm) and layers[1] is affine
    assert layers.state_dict().keys() == before.keys()
    for name, value in before.items():
        torch.testing.assert_close(layers.state_dict()[name], value)
    x = torch.randn(2, 3, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    weights = torch.randn_like(x)
    actual = layers[0](x)
    expected = torch.nn.functional.rms_norm(y, (8,), eps=1e-6)
    torch.testing.assert_close(actual, expected)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(x.grad, y.grad)


def test_native_connector_attention_preserves_frozen_parameter_input_gradients():
    from diffusers.models.transformers.transformer_ltx2 import LTX2Attention

    from verl_omni.pipelines.ltx2_unigrpo_nft.attention import NativeConnectorAttnProcessor

    torch.manual_seed(71)
    attention = LTX2Attention(
        query_dim=16, heads=2, kv_heads=2, dim_head=8, rope_type="split", apply_gated_attention=True
    )
    attention.requires_grad_(False)
    attention.set_processor(NativeConnectorAttnProcessor(has_learned_registers=True))
    hidden = torch.randn(1, 7, 16, requires_grad=True)
    angles = torch.randn(1, 2, 7, 4)
    result = attention(hidden, query_rotary_emb=(angles.cos(), angles.sin()))
    result.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in attention.parameters())


def test_native_connector_split_rope_gradient():
    from verl_omni.pipelines.ltx2_unigrpo_nft.attention import native_connector_rope

    torch.manual_seed(72)
    x = torch.randn(1, 3, 8, dtype=torch.float64, requires_grad=True)
    angle = torch.randn(1, 2, 3, 2, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda value: native_connector_rope(value, (angle.cos(), angle.sin()), "split", 4), (x,)
    )


def test_native_dit_projection_updates_fp32_lora_masters_through_bf16_compute():
    from peft import inject_adapter_in_model

    from verl_omni.pipelines.ltx2_unigrpo_nft.attention import native_fused_qkv

    torch.manual_seed(83)
    layers = torch.nn.ModuleDict({key: torch.nn.Linear(8, 8).to(torch.bfloat16) for key in ("q", "k", "v")})
    layers.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=2, lora_alpha=2, target_modules=["q", "k", "v"]), layers)
    for name, parameter in layers.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.float()
            if "lora_B" in name:
                torch.nn.init.normal_(parameter, std=0.1)
    masters = [p for p in layers.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in masters]
    optimizer = torch.optim.SGD(masters, lr=0.1)
    value = torch.randn(1, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    output = native_fused_qkv(tuple(layers.values()), value)
    sum(x.float().square().mean() for x in output).backward()
    assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    assert all(
        p.dtype == torch.float32 and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in masters
    )
    assert all(p.grad is None for name, p in layers.named_parameters() if "base_layer" in name)
    optimizer.step()
    assert all(not torch.equal(p, old) for p, old in zip(masters, before, strict=True))


def test_native_frozen_component_audit_detects_mutation_and_unfrozen_parameters():
    from types import SimpleNamespace

    from verl_omni.workers.rollout.ltx_joint_worker_extension import LTXJointWorkerExtension

    pipeline = SimpleNamespace(
        **{
            name: torch.nn.Linear(3, 2).to(torch.bfloat16).requires_grad_(False)
            for name in ("connectors", "vae", "audio_vae", "vocoder")
        }
    )
    worker = LTXJointWorkerExtension()
    worker.model_runner = SimpleNamespace(pipeline=pipeline)
    before = worker.snapshot_joint_frozen_components()
    pipeline.vae(torch.ones(1, 3, dtype=torch.bfloat16))
    assert worker.snapshot_joint_frozen_components() == before
    with torch.no_grad():
        pipeline.vae.weight.add_(1)
    after = worker.snapshot_joint_frozen_components()
    assert after["vae"]["sha256"] != before["vae"]["sha256"]
    assert all(after[name] == before[name] for name in ("connectors", "audio_vae", "vocoder"))
    pipeline.audio_vae.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="gradients enabled: audio_vae.weight"):
        worker.snapshot_joint_frozen_components()


def test_native_dit_attention_backpropagates_to_all_lora_and_condition_inputs():
    from diffusers.models.transformers.transformer_ltx2 import LTX2Attention
    from peft import inject_adapter_in_model

    from verl_omni.pipelines.ltx2_unigrpo_nft.attention import NativeDiTAttnProcessor

    torch.manual_seed(84)
    attention = LTX2Attention(
        query_dim=16, heads=2, kv_heads=2, dim_head=8, rope_type="split", apply_gated_attention=True
    ).to(torch.bfloat16)
    attention.requires_grad_(False)
    inject_adapter_in_model(
        LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_k", "to_v", "to_out.0"]), attention
    )
    for name, parameter in attention.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.float()
            if "lora_B" in name:
                torch.nn.init.normal_(parameter, std=0.1)
    attention.set_processor(NativeDiTAttnProcessor(pack_qkv=True))
    value = torch.randn(1, 7, 16, dtype=torch.bfloat16, requires_grad=True)
    angles = torch.randn(1, 2, 7, 4)
    output = attention(value, query_rotary_emb=(angles.cos(), angles.sin()), perturbation_mask=None, all_perturbed=None)
    output.float().square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    for name, parameter in attention.named_parameters():
        if "lora_" in name:
            assert parameter.dtype == torch.float32 and parameter.grad is not None
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0, name
        else:
            assert parameter.grad is None, name


def test_encoding_ignores_nonzero_planning_lora_and_restores_state(model):
    from verl_omni.pipelines.ltx2_unigrpo_nft.model import base_encoder

    ids = torch.tensor([[2, 3, 4, 1]])
    mask = torch.ones_like(ids)
    flags = [p.requires_grad for p in model.parameters()]
    before = model.encode_condition(ids, mask)[0]
    logits_before = model.gemma(input_ids=ids).logits.detach()
    with torch.no_grad():
        for name, parameter in model.gemma.named_parameters():
            if "lora_" in name:
                parameter.normal_(std=1.0)
    after = model.encode_condition(ids, mask)[0]
    assert not after.requires_grad
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert not torch.allclose(logits_before, model.gemma(input_ids=ids).logits)
    with pytest.raises(RuntimeError, match="probe"):
        with base_encoder(model.gemma):
            with base_encoder(model.gemma):
                raise RuntimeError("probe")
    assert [p.requires_grad for p in model.parameters()] == flags
    assert not model.gemma.model.language_model.layers[0].self_attn.q_proj.disable_adapters


def test_rollout_positive_negative_encoding_disable_planning_adapter(model, monkeypatch):
    from types import SimpleNamespace

    from verl_omni.pipelines.ltx2_unigrpo_nft.vllm_omni_rollout_adapter import LTX2Pipeline, LTXJointPipeline

    pipeline = object.__new__(LTXJointPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.text_encoder = model.gemma
    pipeline.connectors = model.connectors
    pipeline.device = torch.device("cpu")
    ids = torch.tensor([[2, 3, 4, 1]])
    mask = torch.ones_like(ids)

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_ids=ids, attention_mask=mask)

    pipeline.tokenizer = Tokenizer()
    layer = model.gemma.model.language_model.layers[0].self_attn.q_proj
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_()
    from verl_omni.pipelines.ltx2_unigrpo_nft.model import base_encoder

    with torch.no_grad(), base_encoder(model.gemma):
        hidden = model.gemma(input_ids=ids, attention_mask=mask.bool(), output_hidden_states=True).hidden_states
    expected = torch.stack(hidden, dim=-1).flatten(2, 3)
    actual, _, _ = pipeline._condition("caption", 4)
    torch.testing.assert_close(actual, expected)
    assert not layer.disable_adapters

    def negative(*args, **kwargs):
        assert layer.disable_adapters
        assert kwargs["dtype"] == next(model.connectors.parameters()).dtype
        raise RuntimeError("negative probe")

    monkeypatch.setattr(LTX2Pipeline, "encode_prompt", negative)
    with pytest.raises(RuntimeError, match="negative probe"):
        pipeline.encode_prompt(prompt="", negative_prompt="noise")
    assert not layer.disable_adapters


def test_joint_ema_freezes_reference_survives_evaluation_and_resume(model):
    from verl_omni.pipelines.ltx2_unigrpo_nft.policy_state import LTXJointPolicyState

    policy = LTXJointPolicyState(model)
    initial = policy.state_dict()["reference"]
    policy.begin_rollout(version=1)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(2)
    current = policy._snapshot()
    policy.update_old(decay=0.25)
    policy.begin_rollout(version=3)
    for name in initial:
        torch.testing.assert_close(policy.old[name], initial[name] + 1.5)
        torch.testing.assert_close(policy.reference[name], initial[name])
    with policy.use("old"):
        for name, parameter in policy._parameters().items():
            torch.testing.assert_close(parameter, policy.old[name])
    for name, parameter in policy._parameters().items():
        torch.testing.assert_close(parameter, current[name])
    saved = policy.state_dict()
    policy.begin_rollout(version=4)
    for name in initial:
        torch.testing.assert_close(policy.old[name], saved["old"][name])
    restored = LTXJointPolicyState(model)
    restored.load_state_dict(policy.state_dict())
    # Starting another generation after resume must not advance EMA again.
    restored.begin_rollout(version=5)
    for name in initial:
        torch.testing.assert_close(restored.old[name], saved["old"][name])
    policy.update_old(decay=0.5)
    restored.update_old(decay=0.5)
    policy.begin_rollout(version=5)
    restored.begin_rollout(version=6)
    for name in initial:
        torch.testing.assert_close(restored.old[name], initial[name] + 1.75)
        torch.testing.assert_close(restored.old[name], policy.old[name])
        torch.testing.assert_close(restored.reference[name], initial[name])
    restored.assert_frozen()


@pytest.mark.parametrize("decay", [-0.1, 1.1, float("nan")])
def test_joint_ema_rejects_invalid_decay_without_changing_snapshots(model, decay):
    from verl_omni.pipelines.ltx2_unigrpo_nft.policy_state import LTXJointPolicyState

    policy = LTXJointPolicyState(model)
    with pytest.raises(ValueError, match="decay"):
        policy.update_old(decay=decay)
    policy.assert_frozen()
    assert not policy.rollout_started


def test_joint_hook_text_kl_uses_frozen_initial_policy(model):
    from functools import partial
    from types import SimpleNamespace

    from verl_omni.pipelines.ltx2_unigrpo_nft.hooks import LTXJointHooks
    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    hooks = LTXJointHooks(model, None, None)
    hooks._device = lambda: torch.device("cpu")
    hooks.begin_rollout(version=0)
    data, _ = _joint_batch(model)
    data["text_advantages"].zero_()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and name.startswith("gemma."):
                parameter.add_(0.05)
    config = SimpleNamespace(
        diffusion_loss=LTXJointLossConfig(text_kl_coef=0.001, av_loss_weight=0.0), ppo_micro_batch_size_per_gpu=1
    )
    result = hooks.forward_backward_batch(data, partial(LTXJointLoss(), config=config))
    metrics = result["metrics"]
    assert metrics["text/kl_loss"][0] > 0
    assert metrics["text/weighted_kl_loss"][0] == pytest.approx(0.001 * metrics["text/kl_loss"][0])
    assert metrics["train/text_gemma_grad_norm_local"][0] > 0
    assert metrics["train/av_dit_grad_norm_local"][0] == 0
    hooks.policies.assert_frozen()


def test_engine_ema_rpc_routes_to_joint_snapshot_state(model):
    from types import SimpleNamespace

    from verl_omni.pipelines.ltx2_unigrpo_nft.hooks import LTXJointHooks
    from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine

    hooks = LTXJointHooks(model, None, None)
    hooks.begin_rollout(version=0)
    initial = hooks.state_dict()["old"]
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(2)
    engine = SimpleNamespace(_engine_hooks=hooks)
    DiffusersFSDPEngine.ema_update_adapter(engine, source="default", target="old", decay=0.25)
    saved = hooks.state_dict()
    for name in initial:
        torch.testing.assert_close(saved["old"][name], initial[name] + 1.5)
        torch.testing.assert_close(saved["reference"][name], initial[name])
    with pytest.raises(ValueError, match="default -> old"):
        DiffusersFSDPEngine.ema_update_adapter(engine, source="default", target="ref", decay=0.25)


def test_joint_forward_noise_replays_by_sample_seed_independently_of_global_rng(model):
    from types import SimpleNamespace

    import numpy as np
    from verl import DataProto

    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    data, records = _joint_batch(model)
    data["rm_scores"] = torch.tensor([[0.2], [0.8]])
    data["video_latents_clean"] = torch.zeros(2, 2, 4)
    data["audio_latents_clean"] = torch.zeros(2, 2, 4)
    config = SimpleNamespace(
        algorithm=SimpleNamespace(
            norm_adv_by_std_in_grpo=True, global_std=True, adv_mode="continuous", timestep_fraction=1.0
        ),
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(diffusion_loss=LTXJointLossConfig(), data_loader_seed=42)
        ),
        reward=SimpleNamespace(reward_functions={"synthetic": {"routing_weights": {"video": 1.0, "audio": 1.0}}}),
    )
    batch = DataProto(
        batch=data,
        non_tensor_batch={"uid": np.array(["a", "a"]), "ltx_joint_replay": np.array(records, dtype=object)},
        meta_info={"reward_names": ["synthetic"]},
    )
    torch.manual_seed(11)
    first = LTXJointLoss.prepare_actor_batch(batch, data["rm_scores"], config)
    saved = {m: first.batch[f"{m}_forward_noise"].clone() for m in ("video", "audio")}
    torch.manual_seed(999)
    second = LTXJointLoss.prepare_actor_batch(batch, data["rm_scores"], config)
    for m in ("video", "audio"):
        torch.testing.assert_close(second.batch[f"{m}_forward_noise"], saved[m], atol=0, rtol=0)
    assert not torch.equal(saved["video"], saved["audio"])
    reversed_batch = batch.select_idxs([1, 0])
    third = LTXJointLoss.prepare_actor_batch(reversed_batch, reversed_batch.batch["rm_scores"], config)
    for m in ("video", "audio"):
        torch.testing.assert_close(third.batch[f"{m}_forward_noise"], saved[m].flip(0), atol=0, rtol=0)


@pytest.mark.parametrize("policy_loss", ["grpo", "gspo"])
def test_microbatch_four_matches_single_sample_gradients_with_ragged_text(model, policy_loss):
    from copy import deepcopy
    from functools import partial
    from types import SimpleNamespace

    import numpy as np
    from verl.utils import tensordict_utils as tu

    from verl_omni.pipelines.ltx2_unigrpo_nft.hooks import LTXJointHooks
    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    hooks = LTXJointHooks(model, None, None)
    hooks._device = lambda: torch.device("cpu")
    hooks.begin_rollout(version=0)
    data, source_records = _joint_batch(model)
    indices = [0, 1, 0, 1, 0, 1, 0]
    data = data[torch.tensor(indices)].clone()
    records = [deepcopy(source_records[i]) for i in indices]
    for i, record in enumerate(records):
        record.prompt_token_ids = torch.cat([torch.full((i % 3,), 2), record.prompt_token_ids])
        with torch.no_grad():
            values = model(task="recaption", **record.ar_inputs("cpu"))
            record.rollout_logprobs = values[record.ar_inputs("cpu")["response_mask"][:, 1:]].clone()
    tu.assign_non_tensor(data, ltx_joint_replay=np.array(records, dtype=object))
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.01)
    observed = []
    handle = model.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append((kwargs["task"], kwargs["input_ids"].shape[0])),
        with_kwargs=True,
    )
    runs = []
    try:
        for micro_size in (1, 4):
            model.zero_grad(set_to_none=True)
            observed.clear()
            config = SimpleNamespace(
                diffusion_loss=LTXJointLossConfig(text_kl_coef=0.001, text_policy_loss=policy_loss),
                ppo_micro_batch_size_per_gpu=micro_size,
            )
            output = hooks.forward_backward_batch(data, partial(LTXJointLoss(), config=config))
            gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
            runs.append((output, gradients))
            if micro_size == 4:
                assert ("recaption", 4) in observed and ("av", 4) in observed
                assert ("recaption", 3) in observed and ("av", 3) in observed
    finally:
        handle.remove()
    assert runs[0][0]["loss"][0] == pytest.approx(runs[1][0]["loss"][0], rel=2e-5, abs=2e-6)
    assert runs[0][1].keys() == runs[1][1].keys()
    for name in runs[0][1]:
        torch.testing.assert_close(runs[0][1][name], runs[1][1][name], rtol=2e-4, atol=2e-6, msg=name)


@pytest.mark.parametrize("text_weight,av_weight", [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
@pytest.mark.parametrize("micro_size", [1, 2])
def test_ablation_freezes_parameters_optimizer_state_and_rollout_ema(model, text_weight, av_weight, micro_size):
    """A disabled branch stays bitwise fixed even with existing AdamW momentum."""
    from copy import deepcopy
    from functools import partial
    from types import SimpleNamespace

    from verl_omni.pipelines.ltx2_unigrpo_nft.hooks import LTXJointHooks
    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(params.values(), lr=0.01, weight_decay=0.1)
    # Populate optimizer moments as in a resumed checkpoint; leave zero tensors
    # behind instead of relying on optimizer.zero_grad's default set_to_none.
    for parameter in params.values():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=False)
    hooks = LTXJointHooks(model, None, None)
    hooks._device = lambda: torch.device("cpu")
    hooks.begin_rollout(version=0)
    data, _ = _joint_batch(model)
    # Make current != old, so a frozen branch's EMA cannot silently drift.
    with torch.no_grad():
        for parameter in params.values():
            parameter.add_(0.001)
    config = SimpleNamespace(
        diffusion_loss=LTXJointLossConfig(text_loss_weight=text_weight, av_loss_weight=av_weight, text_kl_coef=0.001),
        ppo_micro_batch_size_per_gpu=micro_size,
    )
    calls = []
    handle = model.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append((kwargs["task"], torch.is_grad_enabled())), with_kwargs=True
    )
    try:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=False)
            before = {n: p.detach().clone() for n, p in params.items()}
            states = {n: deepcopy(optimizer.state[p]) for n, p in params.items()}
            old = hooks.state_dict()["old"]
            result = hooks.forward_backward_batch(data, partial(LTXJointLoss(), config=config))
            assert result["metrics"]["joint/update_text"] == [text_weight]
            assert result["metrics"]["joint/update_av"] == [av_weight]
            assert "consistency/old_logprob_abs_max" in result["metrics"]
            for prefix, enabled in (("gemma.", text_weight), ("transformer.", av_weight)):
                selected = {n: p for n, p in params.items() if n.startswith(prefix)}
                if not enabled:
                    assert all(p.grad is None for p in selected.values())
            optimizer.step()
            hooks.ema_update_adapter(source="default", target="old", decay=0.5)
            for prefix, enabled in (("gemma.", text_weight), ("transformer.", av_weight)):
                selected = {n: p for n, p in params.items() if n.startswith(prefix)}
                if enabled:
                    assert any(not torch.equal(p, before[n]) for n, p in selected.items())
                    assert any(not torch.equal(hooks.policies.old[n], old[n]) for n in selected)
                else:
                    for n, p in selected.items():
                        torch.testing.assert_close(p, before[n], rtol=0, atol=0)
                        torch.testing.assert_close(hooks.policies.old[n], old[n], rtol=0, atol=0)
                        for key, value in optimizer.state[p].items():
                            torch.testing.assert_close(value, states[n][key], rtol=0, atol=0)
    finally:
        handle.remove()
    assert (("recaption", True) in calls) == bool(text_weight)
    assert (("av", True) in calls) == bool(av_weight)
    if not av_weight:
        assert not any(task == "av" for task, _ in calls)


@pytest.mark.parametrize("source", ["modality_probability", "combined_reward"])
@pytest.mark.parametrize("global_std", [True, False])
def test_text_credit_config_uses_full_batch_std(source, global_std):
    import numpy as np
    from tensordict import TensorDict
    from verl import DataProto

    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    rewards = torch.tensor([0.0, 0.5, 0.1, 0.8])
    batch = DataProto(
        batch=TensorDict(
            {"reward_prob": rewards[:, None, None], "rm_scores": (20 * rewards)[:, None]},
            batch_size=[4],
        ),
        non_tensor_batch={"uid": np.array(["a", "b", "a", "b"])},
    )
    config = LTXJointLossConfig(text_advantage_source=source, text_global_std=global_std)
    LTXJointLoss._prepare_text_credit(batch, config)
    scale = 20 if source == "combined_reward" else 1
    centered = scale * torch.tensor([-0.05, -0.15, 0.05, 0.15])
    std = scale * (rewards.std(correction=0) if global_std else torch.tensor([0.05, 0.15, 0.05, 0.15]))
    torch.testing.assert_close(batch.batch["text_advantages"], centered / (std + 1e-6))
    torch.testing.assert_close(batch.batch["reward_prob"], rewards[:, None, None])
    assert LTXJointLossConfig().text_global_std is True


@pytest.mark.parametrize("global_std", [True, False])
def test_text_component_advantages_are_normalized_before_equal_weight_mean(global_std):
    import numpy as np
    from tensordict import TensorDict
    from verl import DataProto

    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    # Two interleaved groups, five scorers with different scales, opposite
    # preferences and a constant scorer (which must still count in the mean).
    rewards = torch.tensor(
        [
            [0.0, 0.0, 2.0, 7.0, 0.0],
            [10.0, 100.0, 16.0, 7.0, 6.0],
            [2.0, 20.0, 0.0, 7.0, 4.0],
            [16.0, 160.0, 10.0, 7.0, 2.0],
        ],
        requires_grad=True,
    )
    probabilities = torch.full((4, 1, 2), 0.5)
    names = ["video_align", "hpsv3", "audiobox", "clap", "desync"]
    batch = DataProto(
        batch=TensorDict({"reward_prob": probabilities.clone(), "rm_scores": rewards}, batch_size=[4]),
        non_tensor_batch={"uid": np.array(["a", "b", "a", "b"])},
        meta_info={"reward_names": names},
    )
    config = LTXJointLossConfig(text_global_std=global_std)
    LTXJointLoss._prepare_text_credit(batch, config)

    centered = torch.tensor(
        [
            [-1.0, -10.0, 1.0, 0.0, -2.0],
            [-3.0, -30.0, 3.0, 0.0, 2.0],
            [1.0, 10.0, -1.0, 0.0, 2.0],
            [3.0, 30.0, -3.0, 0.0, -2.0],
        ]
    )
    if global_std:
        std = torch.tensor([41.0**0.5, 10 * 41.0**0.5, 41.0**0.5, 0.0, 5.0**0.5])
    else:
        std = torch.tensor([[1.0, 10.0, 1.0, 0.0, 2.0], [3.0, 30.0, 3.0, 0.0, 2.0]]).repeat(2, 1)
    expected = (centered / (std + 1e-6)).mean(-1)
    actual = batch.batch["text_advantages"]
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    torch.testing.assert_close(batch.batch["reward_prob"], probabilities)
    torch.testing.assert_close(batch.batch["rm_scores"], rewards)
    metrics = batch.meta_info["text_reward_metrics"]
    assert not any(key.endswith("/advantage_std") for key in metrics)
    for i, name in enumerate(names):
        assert metrics[f"text/components/{name}/reward_std"] == pytest.approx(
            rewards[:, i].detach().std(correction=0).item()
        )
    assert metrics["text/advantage_abs_mean"] == pytest.approx(expected.abs().mean().item())
    assert metrics["text/constant_group_fraction"] == pytest.approx(0.2)
    assert metrics["text/components/clap/advantage_abs_mean"] == 0
    assert metrics["text/reward_source_component_advantage_mean"] == 1
    assert batch.meta_info["text_reward_source"] == "component_advantage_mean"


def test_nested_text_credit_averages_rewards_before_normalization():
    from types import SimpleNamespace

    import numpy as np
    from tensordict import TensorDict
    from verl import DataProto

    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    # Rewrite means are 1 and 5. Per-video normalization would give different credit.
    scores = torch.tensor([[0.0], [2.0], [4.0], [6.0]])
    replay = [
        SimpleNamespace(response_token_ids=torch.tensor([j // 2, 9]), rollout_logprobs=torch.tensor([-1.0, -2.0]))
        for j in range(4)
    ]
    batch = DataProto(
        batch=TensorDict({"rm_scores": scores.clone(), "reward_prob": torch.full((4, 1, 2), 0.5)}, batch_size=[4]),
        non_tensor_batch={
            "uid": np.array(["a"] * 4, dtype=object),
            "_ltx_rewrite_id": np.array([0, 0, 1, 1]),
            "ltx_joint_replay": np.array(replay, dtype=object),
        },
    )
    LTXJointLoss._prepare_nested_text_credit(batch, LTXJointLossConfig())
    torch.testing.assert_close(
        batch.batch["text_advantages"], torch.tensor([-1.0, -1.0, 1.0, 1.0]), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(batch.batch["rm_scores"], scores)
    torch.testing.assert_close(batch.batch["text_reward"], torch.tensor([1.0, 1.0, 5.0, 5.0]))
    replay[1].response_token_ids = torch.tensor([7, 9])
    with pytest.raises(ValueError, match="tokens"):
        LTXJointLoss._prepare_nested_text_credit(batch, LTXJointLossConfig())


@pytest.mark.parametrize("global_std", [True, False])
def test_av_rewrite_group_centers_per_rewrite_and_preserves_text_groups(global_std):
    from types import SimpleNamespace

    import numpy as np
    from tensordict import TensorDict
    from verl import DataProto

    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss
    from verl_omni.workers.config.diffusion.actor import LTXJointLossConfig

    scores = torch.tensor([[0.0], [2.0], [10.0], [14.0]])
    config = SimpleNamespace(
        algorithm=SimpleNamespace(
            norm_adv_by_std_in_grpo=True, global_std=global_std, adv_mode="continuous", timestep_fraction=1.0
        ),
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(diffusion_loss=LTXJointLossConfig(av_advantage_group="rewrite"), data_loader_seed=42),
            rollout=SimpleNamespace(ltx_joint=SimpleNamespace(tree_rollout=True)),
        ),
        reward=SimpleNamespace(reward_functions={"score": {"routing_weights": {"video": 1.0, "audio": 1.0}}}),
    )
    uids = np.array(["a"] * 4, dtype=object)
    batch = DataProto(
        batch=TensorDict(
            {
                "rm_scores": scores.clone(),
                "train_timesteps": torch.tensor([[900.0, 500.0]]).repeat(4, 1),
                "video_latents_clean": torch.zeros(4, 2, 4),
                "audio_latents_clean": torch.zeros(4, 2, 4),
            },
            batch_size=[4],
        ),
        non_tensor_batch={"uid": uids, "_ltx_rewrite_id": np.array([0, 0, 1, 1])},
        meta_info={"reward_names": ["score"]},
    )
    LTXJointLoss._prepare_av_credit(batch, scores, config)
    denominator = scores.std(correction=0) if global_std else torch.tensor([1.0, 1.0, 2.0, 2.0])
    expected = torch.tensor([-1.0, 1.0, -2.0, 2.0]) / (denominator + 1e-4)
    torch.testing.assert_close(batch.batch["advantages"][:, 0], expected)
    torch.testing.assert_close(batch.batch["reward_prob"][:, 0, 0], expected / 10 + 0.5)
    assert batch.non_tensor_batch["uid"] is uids
    torch.testing.assert_close(batch.batch["rm_scores"], scores)
    # Evaluation uses original prompt groups even when training requests rewrite groups.
    batch.meta_info["ltx_is_evaluation"] = True
    LTXJointLoss._prepare_av_credit(batch, scores, config)
    expected_eval = (scores[:, 0] - scores.mean()) / (scores.std(correction=0) + 1e-4)
    torch.testing.assert_close(batch.batch["advantages"][:, 0], expected_eval)
    batch.meta_info["ltx_is_evaluation"] = False
    config.actor_rollout_ref.rollout.ltx_joint.tree_rollout = False
    LTXJointLoss._prepare_av_credit(batch, scores, config)
    torch.testing.assert_close(batch.batch["advantages"][:, 0], expected_eval)
    config.actor_rollout_ref.rollout.ltx_joint.tree_rollout = True
    batch.non_tensor_batch["_ltx_rewrite_id"] = np.arange(4)
    with pytest.raises(ValueError, match="at least two"):
        LTXJointLoss._prepare_av_credit(batch, scores, config)
    assert batch.non_tensor_batch["uid"] is uids


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("update_text", [True, False])
def test_text_ema_switch_keeps_av_ema_and_reference(model, enabled, update_text):
    from types import SimpleNamespace

    from verl_omni.pipelines.ltx2_unigrpo_nft.hooks import LTXJointHooks
    from verl_omni.workers.config.diffusion import LTXJointRolloutConfig

    hooks = LTXJointHooks(model, SimpleNamespace(ltx_joint=LTXJointRolloutConfig(text_ema_enabled=enabled)), None)
    hooks._update_text = update_text
    hooks.begin_rollout(version=0)
    initial = hooks.state_dict()["old"]
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(2)
    current = hooks.policies._snapshot()
    hooks.ema_update_adapter(source="default", target="old", decay=0.75)
    for name in initial:
        text = name.startswith("gemma.")
        delta = (0 if not update_text else (0.5 if enabled else 2)) if text else 0.5
        torch.testing.assert_close(hooks.policies.old[name], initial[name] + delta)
        torch.testing.assert_close(hooks.policies.reference[name], initial[name])
        torch.testing.assert_close(hooks.policies._snapshot()[name], current[name])
    saved = hooks.state_dict()
    hooks.load_state_dict(saved)
    hooks.begin_rollout(version=1)
    for name in initial:
        torch.testing.assert_close(hooks.policies.old[name], saved["old"][name])
