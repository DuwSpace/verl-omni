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
"""CPU checks for merged-LoRA weight export in the diffusers FSDP engine."""

import copy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist

import verl_omni.workers.engine.fsdp.diffusers_impl as diffusers_impl
from verl_omni.workers.config.diffusion import DiffusionModelConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)
        # Carrying a ``peft_config`` is all ``get_per_tensor_param`` needs to
        # take the LoRA branch.
        self.peft_config = {"default": SimpleNamespace(to_dict=lambda: {"r": 8})}


def _make_engine(module, lora_config: dict) -> PPODiffusersFSDPEngine:
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.module = module
    engine._is_offload_param = False
    engine._uses_fsdp2_cpu_offload_policy = False
    # DiffusionModelConfig.__post_init__ does I/O; set the fields under test directly.
    model_config = object.__new__(DiffusionModelConfig)
    object.__setattr__(model_config, "lora", lora_config)
    object.__setattr__(model_config, "fsdp_layer_prefixes", ["layers."])
    engine.model_config = model_config
    return engine


def _patch_sync_helpers(monkeypatch, merged_context=None):
    monkeypatch.setattr(diffusers_impl, "log_gpu_memory_usage", MagicMock())
    monkeypatch.setattr(diffusers_impl, "load_fsdp_model_to_gpu", MagicMock())
    monkeypatch.setattr(diffusers_impl, "offload_fsdp_model_to_cpu", MagicMock())
    monkeypatch.setattr(diffusers_impl, "get_device_id", lambda: torch.device("cpu"))
    if merged_context is not None:
        monkeypatch.setattr(diffusers_impl, "merged_lora_context", merged_context)


def _passthrough_names(monkeypatch):
    monkeypatch.setattr(diffusers_impl, "normalize_peft_param_name", lambda state: state)
    monkeypatch.setattr(diffusers_impl, "convert_weight_keys", lambda state, model: state)


def test_merge_branch_streams_full_weights_without_peft_config(monkeypatch):
    module = _ToyModel()

    @contextmanager
    def merged_context(actor, backup_adapters):
        assert actor is module
        assert backup_adapters
        yield

    _patch_sync_helpers(monkeypatch, merged_context)
    _passthrough_names(monkeypatch)
    collect = MagicMock(return_value={})
    monkeypatch.setattr(diffusers_impl, "collect_lora_params", collect)

    engine = _make_engine(module, lora_config={"merge": True})
    params, peft_config = engine.get_per_tensor_param(layered_summon=False, base_sync_done=True)

    weights = dict(params)
    assert peft_config is None  # routes the rollout to the full-weight branch
    assert set(weights) == {"transformer.proj.weight"}
    torch.testing.assert_close(weights["transformer.proj.weight"], module.proj.weight)
    collect.assert_not_called()  # merge mode bypasses the adapter path entirely


def test_merged_weights_materialized_and_actor_restored(monkeypatch):
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    class _Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4, bias=False)

    torch.manual_seed(0)
    peft_model = get_peft_model(_Base(), LoraConfig(r=8, lora_alpha=16, target_modules=["proj"]))
    wrapped = peft_model.base_model.model.proj
    base_weight = wrapped.base_layer.weight.detach().clone()
    with torch.no_grad():
        # Default zero-init lora_B would merge to a no-op; make the delta visible.
        wrapped.lora_B["default"].weight.fill_(1.0)

    @contextmanager
    def peft_merge_context(actor, backup_adapters):
        # Stand-in for verl's FSDP-dependent merged_lora_context: per-layer
        # merge/unmerge, the same mechanism verl's fsdp_merge_unmerge uses.
        from peft.tuners.lora import LoraLayer

        assert backup_adapters
        with torch.no_grad():
            for layer in actor.modules():
                if isinstance(layer, LoraLayer):
                    layer.merge()
        try:
            yield
        finally:
            with torch.no_grad():
                for layer in actor.modules():
                    if isinstance(layer, LoraLayer):
                        layer.unmerge()

    _patch_sync_helpers(monkeypatch, peft_merge_context)
    # Real normalize_peft_param_name / convert_weight_keys: exercises the actual
    # name cleaning (strips peft wrappers, drops lora_* keys) end to end.

    engine = _make_engine(peft_model, lora_config={"merge": True})
    weights = dict(engine._merged_lora_per_tensor_param())

    assert set(weights) == {"transformer.proj.weight"}
    expected = base_weight + (16 / 8) * (wrapped.lora_B["default"].weight @ wrapped.lora_A["default"].weight)
    torch.testing.assert_close(weights["transformer.proj.weight"], expected)
    # The actor must come back unmerged once the stream is consumed.
    torch.testing.assert_close(wrapped.base_layer.weight, base_weight)


def test_merge_branch_rejects_named_adapter(monkeypatch):
    module = _ToyModel()

    _patch_sync_helpers(monkeypatch)
    _passthrough_names(monkeypatch)
    monkeypatch.setattr(diffusers_impl, "merged_lora_context", MagicMock())

    engine = _make_engine(module, lora_config={"merge": True})
    with pytest.raises(ValueError, match="rollout_adapter='old'"):
        engine.get_per_tensor_param(base_sync_done=True, adapter_name="old")
    # "default" is what the weight-sync call sites pass and must stay accepted.
    _, peft_config = engine.get_per_tensor_param(base_sync_done=True, adapter_name="default")
    assert peft_config is None


def test_merged_stream_offloads_on_finally(monkeypatch):
    _patch_sync_helpers(
        monkeypatch,
        merged_context=MagicMock(),
    )
    _passthrough_names(monkeypatch)

    engine = _make_engine(_ToyModel(), lora_config={"merge": True})
    engine._is_offload_param = True
    list(engine._merged_lora_per_tensor_param())
    diffusers_impl.offload_fsdp_model_to_cpu.assert_called_once_with(engine.module)


def test_merged_stream_skips_offload_when_disabled(monkeypatch):
    _patch_sync_helpers(
        monkeypatch,
        merged_context=MagicMock(),
    )
    _passthrough_names(monkeypatch)

    engine = _make_engine(_ToyModel(), lora_config={"merge": True})
    list(engine._merged_lora_per_tensor_param())
    diffusers_impl.offload_fsdp_model_to_cpu.assert_not_called()


def test_adapter_branch_unchanged_when_merge_disabled(monkeypatch):
    module = _ToyModel()
    adapter_weight = torch.zeros(8, 4)

    _patch_sync_helpers(monkeypatch)
    monkeypatch.setattr(diffusers_impl, "convert_weight_keys", lambda state, model: state)
    collect = MagicMock(
        return_value={"layers.0.self_attn.q_proj_moe_gen.lora_A.weight": adapter_weight},
    )
    monkeypatch.setattr(diffusers_impl, "collect_lora_params", collect)

    engine = _make_engine(module, lora_config={})
    params, peft_config = engine.get_per_tensor_param(layered_summon=True, base_sync_done=True)

    weights = dict(params)
    assert peft_config == {"r": 8}  # still routed to the rollout's add_lora branch
    assert set(weights) == {"transformer.layers.0.self_attn.q_proj_moe_gen.lora_A.weight"}
    torch.testing.assert_close(weights["transformer.layers.0.self_attn.q_proj_moe_gen.lora_A.weight"], adapter_weight)
    collect.assert_called_once_with(
        module=module,
        layered_summon=True,
        base_sync_done=True,
        is_diffusers=True,
        adapter_name="default",
        layer_prefixes=["layers."],
    )


def test_real_fsdp1_checkpoint_restores_named_adapters_and_optimizer(tmp_path):
    """A shared FSDP checkpoint preserves distinct current and old policies."""
    from diffusers.loaders import PeftAdapterMixin
    from omegaconf import OmegaConf
    from peft import LoraConfig
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

    class _Config:
        name_or_path = ""

        def save_pretrained(self, path):
            del path

    class Model(torch.nn.Module, PeftAdapterMixin):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
            self.config = _Config()

        def forward(self, inputs):
            return self.transformer_blocks[0](inputs)

        def can_generate(self):
            return False

    def adapter_state(module):
        return {name: parameter.detach().clone() for name, parameter in module.named_parameters() if "lora_" in name}

    assert not dist.is_initialized()
    init_file = tmp_path / "fsdp_checkpoint_init"
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
    try:
        model = Model()
        lora_config = LoraConfig(r=1, lora_alpha=1, target_modules=["0"], bias="none")
        model.add_adapter(lora_config, adapter_name="default")
        with pytest.warns(UserWarning, match="Already found a `peft_config`"):
            model.add_adapter(lora_config, adapter_name="old")
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".default." in name:
                    parameter.fill_(4.0)
                elif ".old." in name:
                    parameter.fill_(3.5)
        model.set_adapter("default")

        model.transformer_blocks[0] = FSDP(
            model.transformer_blocks[0], device_id=torch.device("cpu"), use_orig_params=True
        )
        model = FSDP(model, device_id=torch.device("cpu"), use_orig_params=True)
        # torch_npu patches automatic foreach selection even for CPU optimizers;
        # an explicit scalar path keeps this repository CPU test device-free.
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.01,
            momentum=0.9,
            foreach=False,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        expected_adapters = adapter_state(model)
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_scheduler = copy.deepcopy(scheduler.state_dict())
        checkpoint_config = OmegaConf.create(
            {
                "save_contents": ["model", "optimizer", "extra"],
                "load_contents": ["model", "optimizer", "extra"],
            }
        )
        manager = FSDPCheckpointManager(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            processing_class=None,
            checkpoint_config=checkpoint_config,
        )
        checkpoint_dir = tmp_path / "checkpoint"
        torch.manual_seed(1234)
        manager.save_checkpoint(str(checkpoint_dir), global_step=7)
        expected_rng_draw = torch.rand(4)

        saved_model = torch.load(checkpoint_dir / "model_world_size_1_rank_0.pt", weights_only=False)
        assert {name for name in saved_model if "lora_" in name} == {
            "transformer_blocks.0.lora_A.default.weight",
            "transformer_blocks.0.lora_A.old.weight",
            "transformer_blocks.0.lora_B.default.weight",
            "transformer_blocks.0.lora_B.old.weight",
        }

        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    parameter.fill_(99.0)
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        value.fill_(99.0)
        optimizer.param_groups[0]["lr"] = 9.0
        scheduler.last_epoch = 99
        torch.rand(16)

        manager.load_checkpoint(str(checkpoint_dir))
        restored_adapters = adapter_state(model)
        assert restored_adapters.keys() == expected_adapters.keys()
        for name, expected in expected_adapters.items():
            torch.testing.assert_close(restored_adapters[name], expected)

        default_old_pairs = [
            (name, name.replace(".default.", ".old.")) for name in restored_adapters if ".default." in name
        ]
        assert all(old_name in restored_adapters for _, old_name in default_old_pairs)
        assert any(
            not torch.equal(restored_adapters[current_name], restored_adapters[old_name])
            for current_name, old_name in default_old_pairs
        )

        restored_optimizer = optimizer.state_dict()
        assert restored_optimizer["param_groups"] == expected_optimizer["param_groups"]
        assert restored_optimizer["state"].keys() == expected_optimizer["state"].keys()
        for parameter_id, expected_state in expected_optimizer["state"].items():
            for name, expected in expected_state.items():
                torch.testing.assert_close(restored_optimizer["state"][parameter_id][name], expected)
        assert scheduler.state_dict() == expected_scheduler
        torch.testing.assert_close(torch.rand(4), expected_rng_draw)

        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        continued_adapters = adapter_state(model)
        assert any(
            not torch.equal(continued_adapters[name], restored_adapters[name])
            for name in restored_adapters
            if ".default." in name
        )
        assert all(
            torch.equal(continued_adapters[name], restored_adapters[name])
            for name in restored_adapters
            if ".old." in name
        )
    finally:
        dist.destroy_process_group()


def test_bagel_pickscore_e2e_uses_merged_lora_like_the_recipe():
    """Bagel LoRA smoke must merge adapters: vLLM-Omni fused MoT cannot bind *_moe_gen."""
    repo = Path(__file__).resolve().parents[2]
    e2e = (repo / "tests/special_e2e/run_flowgrpo_bagel_pickscore.sh").read_text(encoding="utf-8")
    recipe = (repo / "examples/flowgrpo_trainer/bagel/run_bagel_pickscore_lora.sh").read_text(encoding="utf-8")
    assert "actor_rollout_ref.model.lora.merge=True" in recipe
    assert any(
        "actor_rollout_ref.model.lora.merge=True" in line and not line.lstrip().startswith("#")
        for line in e2e.splitlines()
    )
