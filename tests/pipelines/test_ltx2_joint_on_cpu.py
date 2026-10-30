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

from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig
from transformers import Gemma3Config, Gemma3ForConditionalGeneration, Gemma3TextConfig, SiglipVisionConfig

from verl_omni.pipelines.ltx2_unigrpo_nft.config import LTXJointLossConfig, LTXJointPipelineConfig
from verl_omni.pipelines.ltx2_unigrpo_nft.training.adapter import LTXJointTrainingAdapter
from verl_omni.pipelines.ltx2_unigrpo_nft.training.model import LTXJointModel
from verl_omni.pipelines.ltx2_unigrpo_nft.training.update import JointUpdate
from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin


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
            target_modules=[
                *[f"gemma.model.language_model.layers.0.self_attn.{name}_proj" for name in ("q", "k", "v", "o")],
                *[f"gemma.model.language_model.layers.0.mlp.{name}_proj" for name in ("gate", "up", "down")],
                "transformer.proj",
            ],
        )
    )
    joint.add_adapter(joint.peft_config["default"], adapter_name="old")
    joint.set_adapter("default")
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
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = getattr(model.gemma.model.language_model.layers[0].mlp, name)
        grad = layer.lora_B["default"].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


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


def test_bf16_base_gemma_produces_frozen_bf16_condition(model):
    model.bfloat16()
    ids = torch.tensor([[2, 3, 4, 1]])
    video, audio = model(task="av", input_ids=ids, attention_mask=torch.ones_like(ids))
    assert video.dtype == audio.dtype == torch.bfloat16
    (video.float().square().mean() + audio.float().square().mean()).backward()
    gradients = [p.grad for p in model.gemma.parameters() if p.requires_grad]
    assert gradients and all(g is None for g in gradients)
    assert all(p.grad is None for p in model.connectors.parameters())


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
            response_text=f"A bell rings with event {token}.",
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


class Engine(LoRAAdapterMixin):
    def __init__(self, model, mode="joint", ema=False):
        self.module = model
        self.model_config = SimpleNamespace(pipeline=LTXJointPipelineConfig(update_mode=mode, text_ema_enabled=ema))

    @contextmanager
    def _adapter_state_context(self):
        try:
            yield
        finally:
            self.module.set_adapter("default")

    def adapter_ema_decay(self, name, decay):
        return LTXJointTrainingAdapter.adapter_ema_decay(self.model_config, name, decay)


@pytest.mark.parametrize("mode", ["ar_only", "av_only", "joint"])
@pytest.mark.parametrize("micro", [1, 2])
def test_ablation_updates_only_selected_branch_even_with_optimizer_momentum(model, monkeypatch, mode, micro):
    engine = Engine(model, mode)
    engine.copy_adapter()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01, weight_decay=0.1)
    # Seed optimizer momentum in both branches before the ablation.
    for p in optimizer.param_groups[0]["params"]:
        p.grad = torch.ones_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    engine.copy_adapter()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    data, _ = _joint_batch(model)
    config = SimpleNamespace(ppo_micro_batch_size_per_gpu=micro, diffusion_loss=LTXJointLossConfig(mix_beta=1.0))
    update = JointUpdate(model, engine.model_config, engine.use_adapter)
    monkeypatch.setattr(update, "_device", lambda: torch.device("cpu"))
    result = update.run(data, partial(lambda **kwargs: None, config=config), False)
    optimizer.step()
    for key in ("logprob_diff_mean", "logprob_diff_abs_mean", "logprob_diff_abs_max"):
        name = "text/" + key
        assert (name in result["metrics"]) == (mode != "av_only")
        if mode != "av_only":
            assert result["metrics"][name][0] == pytest.approx(0.0, abs=1e-5)
    assert torch.isfinite(torch.tensor(result["loss"])).all()
    for prefix, active in (("gemma.", mode != "av_only"), ("transformer.", mode != "ar_only")):
        parameters = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix) and ".default." in n]
        changed = any(not torch.equal(p, before[n]) for n, p in parameters)
        assert changed == active
        if not active:
            assert all(p.grad is None for _, p in parameters)
    for n, p in model.named_parameters():
        if "lora_" not in n or ".old." in n:
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_text_ema_default_copy_keeps_av_ema(model, enabled):
    engine = Engine(model, ema=enabled)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".default." in n:
                p.fill_(4)
            elif ".old." in n:
                p.fill_(2)
    engine.ema_update_adapter(decay=0.75)
    for n, p in model.named_parameters():
        if ".old." in n:
            expected = 4 if n.startswith("gemma.") and not enabled else 2.5
            torch.testing.assert_close(p, torch.full_like(p, expected))


def test_recipe_modes_and_prompt_default(monkeypatch, model):
    import re
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    monkeypatch.setenv("LTX_MODEL_PATH", "/models/ltx")
    monkeypatch.setenv("LTX_TRAIN_FILE", "/data/train.parquet")
    monkeypatch.setenv("LTX_VAL_FILE", "/data/val.parquet")
    monkeypatch.setenv("LTX_OUTPUT_DIR", "/tmp/ltx")
    monkeypatch.setenv("LTX_REWARD_ROOT", "/models/rewards")
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    for mode in ("joint", "ar_only", "av_only"):
        with initialize_config_dir(config_dir=str(root / "examples/unigrpo_nft_trainer/ltx2"), version_base=None):
            cfg = compose(config_name="joint_train", overrides=[f"ablation={mode}"])
        actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.model.pipeline)
        rollout = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout.pipeline)
        validation = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout.val_kwargs.pipeline)
        loss = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor.diffusion_loss)
        assert actor.update_mode == mode
        assert not actor.text_ema_enabled
        assert actor.recaption_prompt == rollout.recaption_prompt == validation.recaption_prompt == "ltx_t2av"
        assert loss.text_advantage_source == "component_advantage_mean"
        optim = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor.optim)
        assert cfg.actor_rollout_ref.model.lora_dtype is None
        assert cfg.actor_rollout_ref.actor.fsdp_config.model_dtype == "bfloat16"
        assert optim.lr == 3e-5
        assert optim.param_group_lrs == {"gemma": 1e-6}
        assert optim.weight_decay == 0.01
        assert optim.override_optimizer_config == {"foreach": False}
        assert cfg.data.train_batch_size == 32
        assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n == 256
        assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 8
        assert cfg.algorithm.timestep_fraction == 0.4
        assert loss.mix_beta == 1.0
        assert loss.text_kl_coef == 0.001
        assert loss.video_ref_kl_coef == loss.audio_ref_kl_coef == 1e-4
        assert cfg.trainer.total_training_steps == 100
        assert cfg.trainer.save_freq == cfg.trainer.test_freq == 2
        assert cfg.actor_rollout_ref.rollout.pipeline.num_inference_steps == 20
        assert cfg.actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps == 40
        assert cfg.actor_rollout_ref.model.policy_state_adapters == ["default", "old"]
        selected = {
            name
            for name, _ in model.gemma.named_modules()
            if re.fullmatch(cfg.actor_rollout_ref.model.target_modules, "gemma." + name)
        }
        assert selected == {
            *[f"model.language_model.layers.0.self_attn.{name}_proj" for name in ("q", "k", "v", "o")],
            *[f"model.language_model.layers.0.mlp.{name}_proj" for name in ("gate", "up", "down")],
        }


def test_batch_four_recipe_delivers_one_microbatch_per_rank(monkeypatch):
    from pathlib import Path
    from unittest.mock import MagicMock

    from hydra import compose, initialize_config_dir
    from verl import DataProto
    from verl.utils import tensordict_utils as tu

    from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1

    monkeypatch.setenv("LTX_MODEL_PATH", "/models/ltx")
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "examples/unigrpo_nft_trainer/ltx2"), version_base=None):
        cfg = compose(
            config_name="joint_train",
            overrides=[
                "data.train_batch_size=4",
                "actor_rollout_ref.rollout.n=8",
                "actor_rollout_ref.actor.ppo_mini_batch_size=4",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
            ],
        )
    actor = MagicMock()
    actor.update_actor.return_value = tu.get_tensordict({}, non_tensor_dict={"metrics": {}})
    trainer = SimpleNamespace(config=cfg, _is_direct_preference=True, actor_rollout_wg=actor)
    batch = DataProto.from_dict(tensors={"video_latents_clean": torch.zeros(32, 2, 2)})
    PolicyGradientDiffusionTrainerV1._update_actor(trainer, batch)
    sent = actor.update_actor.call_args.args[0]
    global_size = tu.get_non_tensor_data(sent, "mini_batch_size", None)
    assert global_size == len(batch) == 32
    assert global_size // cfg.trainer.n_gpus_per_node == cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
    rank_batch = sent[:4]
    assert len(list(tu.make_iterator(rank_batch, mini_batch_size=4, epochs=1))) == 1


def test_component_advantage_mean_is_not_renormalized(model):
    import numpy as np
    from omegaconf import OmegaConf
    from verl import DataProto

    from verl_omni.pipelines.ltx2_unigrpo_nft.training.loss import LTXJointLoss

    data, records = _joint_batch(model)
    data.pop("ltx_joint_replay")
    data["rm_scores"] = torch.tensor([[1.0, 3.0, 5.0], [3.0, 1.0, 5.0]])
    batch = DataProto(
        batch=data,
        non_tensor_batch={"uid": np.array(["a", "a"]), "ltx_joint_replay": np.array(records, dtype=object)},
        meta_info={"reward_names": ["x", "y", "z"]},
    )
    cfg = OmegaConf.create(
        {
            "algorithm": {
                "norm_adv_by_std_in_grpo": True,
                "global_std": True,
                "adv_mode": "continuous",
                "timestep_fraction": 1.0,
            },
            "actor_rollout_ref": {"actor": {"data_loader_seed": 0, "diffusion_loss": dict(LTXJointLossConfig())}},
            "reward": {"reward_functions": {k: {"routing_weights": {"video": 1.0, "audio": 1.0}} for k in "xyz"}},
        }
    )
    result = LTXJointLoss.prepare_actor_batch(batch, data["rm_scores"], cfg)
    torch.testing.assert_close(result.batch["text_advantages"], torch.zeros(2), rtol=0, atol=0)
    # Constant reward retains its equal share, reducing the mean to 2/3.
    data["rm_scores"] = torch.tensor([[1.0, 1.0, 5.0], [3.0, 3.0, 5.0]])
    result = LTXJointLoss.prepare_actor_batch(batch, data["rm_scores"], cfg)
    expected = torch.tensor([-2 / 3, 2 / 3]) / (1 + 1e-6)
    torch.testing.assert_close(result.batch["text_advantages"], expected)


def test_sampler_preserves_independent_request_rng_and_true_logprob():
    from verl_omni.pipelines.ltx2_unigrpo_nft.text.sampling import SeededSampler

    scores = torch.tensor([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]])
    sampler = SeededSampler([17, 19], 0.7, "cpu")
    singles = [SeededSampler([seed], 0.7, "cpu") for seed in (17, 19)]
    before = torch.random.get_rng_state()
    for _ in range(10):
        selected = sampler(None, scores).argmax(-1)
        expected = torch.cat([single(None, row[None]).argmax(-1) for single, row in zip(singles, scores, strict=True)])
        torch.testing.assert_close(selected, expected)
        expected_logprob = (scores / 0.7).log_softmax(-1).gather(-1, selected[:, None]).squeeze(-1)
        torch.testing.assert_close(sampler.logprobs[-1], expected_logprob)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_paired_lora_mapper_updates_gemma_and_routes_dit(model):
    from peft import get_peft_model_state_dict

    from verl_omni.pipelines.ltx2_unigrpo_nft.rollout.adapter import LTXJointPipeline

    pipeline = object.__new__(LTXJointPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.text_encoder = Gemma3ForConditionalGeneration(model.gemma.config)
    pipeline._joint_offload = None
    pipeline._gemma_config = None
    pipeline._prepared_text = {}
    weights = get_peft_model_state_dict(model, adapter_name="default")
    weights = {key: torch.full_like(value, 0.25) for key, value in weights.items()}
    dit, config = pipeline.map_lora_update_to_engine(weights, model.peft_config["default"].to_dict())
    assert pipeline._pending_dit and pipeline._joint_sync_in_progress
    assert all(key.startswith("base_model.model.transformer.") for key in dit)
    assert "gemma" not in config["target_modules"]
    actual = get_peft_model_state_dict(pipeline.text_encoder, adapter_name="joint")
    assert set(actual) == {key.removeprefix("gemma.") for key in weights if key.startswith("gemma.")}
    for value in actual.values():
        torch.testing.assert_close(value, torch.full_like(value, 0.25))


def test_joint_lora_routes_real_diffusion_engine_export(model):
    from peft import get_peft_model_state_dict

    from verl_omni.pipelines.ltx2_unigrpo_nft.rollout.adapter import split_lora
    from verl_omni.workers.engine.lora_export import export_lora_for_sync

    expected = get_peft_model_state_dict(model, adapter_name="old")
    stream, config = export_lora_for_sync(
        model,
        model_config=SimpleNamespace(lora_rank=2, fsdp_layer_prefixes=["gemma.model.language_model.layers."]),
        base_sync_done=True,
        adapter_name="old",
        layered_summon=True,
        is_diffusers=True,
        key_prefix="transformer.",
    )
    gemma, _, dit, _ = split_lora(dict(stream), config)
    actual = {"gemma." + key: value for key, value in gemma.items()}
    actual.update({key.removeprefix("base_model.model."): value for key, value in dit.items()})
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


@pytest.mark.parametrize("count", [2, 4])
def test_stage_prepares_text_before_av_and_keeps_replay(model, monkeypatch, count):
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

    from verl_omni.pipelines.ltx2_unigrpo_nft.rollout.adapter import LTX23OmniNFTPipeline, LTXJointPipeline

    pipeline = object.__new__(LTXJointPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline._joint_offload = None
    pipeline._joint_policy_version = 0
    pipeline._joint_lora_id = 1
    pipeline._joint_sync_in_progress = False
    pipeline._pending_dit = False
    pipeline._prepared_text = {}
    _, records = _joint_batch(model)
    records = (records * count)[:count]
    releases = []
    pipeline._joint_offload = SimpleNamespace(release=lambda: releases.append(True))
    requests = [
        SimpleNamespace(
            request_id=f"sample-{i}",
            sampling_params=SimpleNamespace(
                lora_request=SimpleNamespace(lora_int_id=1),
                seed=42,
                extra_args={"recaption_batch_size": 2, "prepared_request_id": f"sample-{i}"},
            ),
        )
        for i in range(count)
    ]
    events = []

    def prepare(incoming):
        events.append("text_batch")
        return [
            (request, record, torch.zeros(1, 4, 32), torch.ones(1, 4, dtype=torch.bool), None)
            for request, record in zip(incoming, records[: len(incoming)], strict=True)
        ]

    def native(self, batch):
        events.append(batch.requests[0].request_id)
        assert "prompt_embeds" in batch.requests[0].prompt
        return DiffusionOutput(
            output={"metadata": {"rl": {"video_latents_clean": torch.zeros(1, 4, 4)}}, "payload": {}}
        )

    monkeypatch.setattr(pipeline, "_prepare_text", prepare)
    monkeypatch.setattr(LTX23OmniNFTPipeline, "forward", native)
    pipeline.prepare_rollout_requests(requests)
    assert events == ["text_batch"] * (count // 2)
    assert len(releases) == 1
    outputs = [pipeline.forward(DiffusionRequestBatch(requests=[request])) for request in requests]
    assert not pipeline._prepared_text
    with pytest.raises(RuntimeError, match="requires prepared text"):
        pipeline.forward(DiffusionRequestBatch(requests=requests))
    assert events == ["text_batch"] * (count // 2) + [f"sample-{i}" for i in range(count)]
    for output, record in zip(outputs, records, strict=True):
        replay = output.output["metadata"]["rl"]["ltx_joint_replay"]
        assert replay["recaption"] == record.recaption
        assert "video_latents_clean" in output.output["metadata"]["rl"]


def test_offload_reuses_cpu_parameter_and_buffer_storage():
    from verl_omni.pipelines.ltx2_unigrpo_nft.rollout.cpu_weights import FrozenCpuWeights

    module = torch.nn.Linear(4, 4).requires_grad_(False)
    module.register_buffer("statistics", torch.arange(4.0))
    master = FrozenCpuWeights(module)
    old_weight = module.weight.detach()
    old_buffer = module.statistics
    module.weight.data = module.weight.detach().clone() + 1
    module.statistics = module.statistics.clone() + 1
    master.restore_host()
    assert module.weight.data_ptr() == old_weight.data_ptr()
    assert module.statistics.data_ptr() == old_buffer.data_ptr()


def test_official_prompt_keeps_the_original_task():
    from verl_omni.pipelines.ltx2_unigrpo_nft.text.prompt import recaption_messages

    messages = recaption_messages("A bell rings.")
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[-1]["content"] == "user prompt: A bell rings."
    assert "integrated audio" in messages[0]["content"]


def test_logprob_diff_masks_context_and_padding_and_weights_ragged_responses():
    from verl_omni.pipelines.ltx2_unigrpo_nft.text.metrics import recaption_logprob_diff

    old = torch.tensor([[float("nan"), -2.0, 999.0, 999.0], [999.0, -4.0, -1.0, -6.0]], requires_grad=True)
    mask = torch.tensor([[False, True, False, False], [False, True, True, True]])
    sampled = [torch.tensor([-3.0]), torch.tensor([-2.0, -2.0, -3.0])]
    # Valid differences: [1] and [-2, 1, -3]; responses have equal weight.
    metrics = recaption_logprob_diff(old, sampled, mask)
    assert metrics["text/logprob_diff_mean"] == pytest.approx((1 - 4 / 3) / 2)
    assert metrics["text/logprob_diff_abs_mean"] == pytest.approx((1 + 6 / 3) / 2)
    assert metrics["text/logprob_diff_abs_max"] == 3
    assert old.grad is None


@pytest.mark.parametrize("micro", [1, 2])
def test_logprob_diff_matches_replay_offsets_across_microbatches(model, monkeypatch, micro):
    from verl.utils.metric import reduce_metrics

    engine = Engine(model, "ar_only")
    engine.copy_adapter()
    data, records = _joint_batch(model)
    records[0].rollout_logprobs -= torch.tensor([1.0, -1.0])
    records[1].rollout_logprobs -= torch.tensor([-2.0, -4.0])
    update = JointUpdate(model, engine.model_config, engine.use_adapter)
    monkeypatch.setattr(update, "_device", lambda: torch.device("cpu"))
    config = SimpleNamespace(ppo_micro_batch_size_per_gpu=micro, diffusion_loss=LTXJointLossConfig())
    result = update.run(data, partial(lambda **kwargs: None, config=config), False)
    metrics = result["metrics"]
    assert metrics["text/logprob_diff_mean"][0] == pytest.approx(-1.5, abs=1e-5)
    assert metrics["text/logprob_diff_abs_mean"][0] == pytest.approx(2.0, abs=1e-5)
    assert metrics["text/logprob_diff_abs_max"][0] == pytest.approx(4.0, abs=1e-5)
    # Driver reduction keeps the maximum across ranks/minibatches, not their mean.
    combined = reduce_metrics({"actor/text/logprob_diff_abs_max": [2.0, metrics["text/logprob_diff_abs_max"][0]]})
    assert combined["actor/text/logprob_diff_abs_max"] == pytest.approx(4.0, abs=1e-5)


def test_bf16_text_backward_retains_fp32_logprobs(model):
    model.bfloat16()
    ids = torch.tensor([[2, 3, 4, 1]])
    mask = torch.tensor([[False, False, True, True]])
    logprobs = model(
        task="recaption",
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        response_mask=mask,
        temperature=1.0,
    )
    assert logprobs.dtype == torch.float32
    assert torch.isfinite(logprobs).all()
    (-logprobs.sum()).backward()
    gradients = [p.grad for p in model.gemma.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())


@pytest.mark.parametrize("count", [1, 5, 9])
def test_chunked_head_matches_dense_values_gradients_and_one_module_call(count):
    from verl_omni.pipelines.ltx2_unigrpo_nft.training.logprob_head import ChunkedLogprobHead

    torch.manual_seed(42)
    original = torch.nn.Linear(8, 31, bias=False).requires_grad_(False)
    head = ChunkedLogprobHead(original, chunk_size=2)
    assert head.weight is original.weight
    assert head.state_dict().keys() == original.state_dict().keys()
    hidden = torch.randn(count, 8, requires_grad=True)
    reference = hidden.detach().clone().requires_grad_(True)
    targets = torch.arange(count) % 31
    calls = []
    hook = head.register_forward_pre_hook(lambda *args: calls.append(True))
    actual = head(hidden, targets=targets, temperature=0.7)
    expected = (original(reference).float() / 0.7).log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden.grad, reference.grad)
    assert calls == [True]
    hook.remove()
