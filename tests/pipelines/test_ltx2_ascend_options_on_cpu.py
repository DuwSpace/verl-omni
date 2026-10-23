# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend options reach model construction without requiring an accelerator."""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import torch


def test_actor_disables_internal_format_before_loading_weights(monkeypatch):
    import verl.utils.device as devices

    from verl_omni.pipelines.ltx2_unigrpo_nft.diffusers_training_adapter import LTXJointTrainingAdapter
    from verl_omni.pipelines.ltx2_unigrpo_nft.model import LTXJointModel

    config = SimpleNamespace(allow_internal_format=True)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(config=config), raising=False)
    monkeypatch.setattr(devices, "get_device_name", lambda: "npu")
    expected = object()

    def load(path, **kwargs):
        assert not config.allow_internal_format
        assert path == "checkpoint" and kwargs["torch_dtype"] == torch.bfloat16
        return expected

    monkeypatch.setattr(LTXJointModel, "from_pretrained", load)
    model_config = SimpleNamespace(lora_rank=8, policy_state_adapters=("default",), local_path="checkpoint")
    assert LTXJointTrainingAdapter.build_module(model_config, torch.bfloat16) is expected


def test_rollout_disables_internal_format_before_base_constructor(monkeypatch):
    import transformers
    import vllm_omni.platforms as platforms

    from verl_omni.pipelines.ltx2_unigrpo_nft.vllm_omni_rollout_adapter import LTX2Pipeline, LTXJointPipeline

    config = SimpleNamespace(allow_internal_format=True)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(config=config), raising=False)
    monkeypatch.setattr(platforms, "current_omni_platform", SimpleNamespace(device_type="npu"))
    monkeypatch.delenv("LTX_DISABLE_PROGRESS_BAR", raising=False)
    loaded = torch.nn.Linear(2, 2)

    def construct(self, **kwargs):
        assert not config.allow_internal_format
        torch.nn.Module.__init__(self)
        self.text_encoder = torch.nn.Linear(2, 2)

    def load(*args, **kwargs):
        assert not config.allow_internal_format
        assert kwargs["torch_dtype"] == torch.float32
        return loaded

    monkeypatch.setattr(LTX2Pipeline, "__init__", construct)
    monkeypatch.setattr(transformers.Gemma3ForConditionalGeneration, "from_pretrained", load)
    pipeline = LTXJointPipeline(od_config=SimpleNamespace(model="checkpoint"))
    assert pipeline.text_encoder is loaded
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_native_engine_receives_rl_mode_and_disables_nz(monkeypatch):
    from verl_omni.workers.rollout import ltx_joint_backend as backend

    fake_engine = ModuleType("vllm_omni.diffusion.diffusion_engine")
    fake_engine.DiffusionEngine = lambda config: SimpleNamespace(od_config=config)
    monkeypatch.setitem(sys.modules, fake_engine.__name__, fake_engine)
    monkeypatch.setattr(backend.multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "independent_rollout_port", nullcontext)
    worker = backend.LTXJointVllmBackend(
        model_path="checkpoint",
        geometry={},
        recaption_temperature=1.0,
        recaption_max_new_tokens=16,
        sync_timeout_seconds=10,
    )
    config = worker.engine.od_config
    assert config.additional_config["weight_nz_mode"] == 0
    assert config.additional_config["rl_config"] == {
        "enabled": True,
        "enable_batch_invariant": False,
        "enable_training_consistency": False,
        "sleep_mode_extra_cleanup": False,
    }
    assert config.diffusion_attention_config.default.backend == "TORCH_SDPA"
    assert config.enable_cpu_offload
