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
"""CPU regression tests for LTX-scoped OmniNFT module initialization."""

from types import SimpleNamespace

import pytest
import torch
from diffusers.models.normalization import RMSNorm

from verl_omni.pipelines.ltx2_omni_nft import compat
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.engine.fsdp import diffusers_impl


@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize(
    "architecture,algorithm,engine_cls,expect_patch",
    [
        ("LTX2Pipeline", "omni_nft", diffusers_impl.OmniNFTDiffusersFSDPEngine, True),
        ("QwenImagePipeline", "diffusion_nft", diffusers_impl.NFTDiffusersFSDPEngine, False),
        ("LTX2Pipeline", "flow_grpo", diffusers_impl.PPODiffusersFSDPEngine, False),
        ("LTX2Pipeline", "flow_grpo", diffusers_impl.OmniNFTDiffusersFSDPEngine, False),
        ("_OtherOmniNFTTest", "omni_nft", diffusers_impl.OmniNFTDiffusersFSDPEngine, False),
    ],
)
def test_ltx_workaround_is_scoped_before_model_wrapping(
    monkeypatch, use_lora, architecture, algorithm, engine_cls, expect_patch
):
    """Exercise real engine dispatch without loading weights or initializing FSDP."""

    class OtherOmniNFTAdapter(DiffusionModelBase):
        pass

    monkeypatch.setitem(DiffusionModelBase._registry, ("_OtherOmniNFTTest", "omni_nft"), OtherOmniNFTAdapter)
    monkeypatch.setattr(compat, "get_device_name", lambda: "npu")
    module = torch.nn.Sequential(RMSNorm(4, eps=1e-6, elementwise_affine=False), RMSNorm(4, eps=1e-6))
    original_forward = RMSNorm.forward
    stages = []
    engine = object.__new__(engine_cls)
    engine.model_config = SimpleNamespace(
        architecture=architecture, algorithm=algorithm, external_lib=None, enable_gradient_checkpointing=False
    )
    engine.engine_config = SimpleNamespace(forward_only=True)
    engine._is_lora = use_lora
    engine.use_ulysses_sp = False
    engine.rank = 1

    def loaded_module(self):
        stages.append("load")
        return module

    def check_module(actual, stage):
        assert actual is module
        expected = compat.ltx_affine_free_rms_norm_forward if expect_patch else original_forward
        assert module[0].forward.__func__ is expected
        assert module[1].forward.__func__ is original_forward
        assert RMSNorm.forward is original_forward
        stages.append(stage)
        return actual

    monkeypatch.setattr(diffusers_impl.DiffusersFSDPEngine, "_build_module", loaded_module)
    monkeypatch.setattr(engine, "_build_lora_module", lambda m: check_module(m, "lora"))
    monkeypatch.setattr(diffusers_impl, "_maybe_compile_repeated_blocks", lambda m, *args: check_module(m, "compile"))
    monkeypatch.setattr(engine, "_build_scheduler", lambda: None)
    monkeypatch.setattr(engine, "_build_fsdp_module", lambda m: check_module(m, "fsdp"))
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(diffusers_impl, "log_gpu_memory_usage", lambda *args, **kwargs: None)

    engine._build_model_optimizer()

    assert engine.module is module
    assert stages == (["load", "lora", "compile", "fsdp"] if use_lora else ["load", "compile", "fsdp"])
