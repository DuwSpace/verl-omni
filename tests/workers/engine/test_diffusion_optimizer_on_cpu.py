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

"""Per-module learning rates through the diffusion engine's optimizer builder."""

from types import SimpleNamespace

import pytest
import torch
from verl.workers.config.optimizer import FSDPOptimizerConfig

from verl_omni.workers.config.diffusion import FSDPDiffusionOptimizerConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine
from verl_omni.workers.engine.fsdp.optimizer import optimizer_parameters


def test_engine_applies_group_lrs_and_scheduler_preserves_ratio():
    module = torch.nn.Module()
    module.gemma = torch.nn.Linear(2, 2, bias=False)
    module.transformer = torch.nn.Linear(2, 2, bias=False)
    module.old = torch.nn.Linear(2, 2, bias=False).requires_grad_(False)
    config = FSDPDiffusionOptimizerConfig(
        lr=3e-5,
        param_group_lrs={"gemma": 1e-6},
        weight_decay=0.01,
        override_optimizer_config={"foreach": False},
        total_training_steps=10,
    )
    engine = SimpleNamespace(optimizer_config=config, rank=0)
    optimizer = DiffusersFSDPEngine._build_optimizer(engine, module)
    scheduler = DiffusersFSDPEngine._build_lr_scheduler(engine, optimizer)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == 2
    assert {id(p) for p in parameters} == {id(module.gemma.weight), id(module.transformer.weight)}
    frozen = module.old.weight.detach().clone()
    for p in parameters:
        p.data.zero_()
        p.grad = torch.ones_like(p)
    optimizer.step()
    scheduler.step()
    torch.testing.assert_close(module.gemma.weight, torch.full_like(module.gemma.weight, -1e-6))
    torch.testing.assert_close(module.transformer.weight, torch.full_like(module.transformer.weight, -3e-5))
    torch.testing.assert_close(module.old.weight, frozen)
    assert scheduler.get_last_lr() == [3e-5, 1e-6]


def test_ordinary_fsdp_optimizer_keeps_ungrouped_parameters():
    module = torch.nn.Linear(2, 2)
    module.bias.requires_grad_(False)
    engine = SimpleNamespace(optimizer_config=FSDPOptimizerConfig(lr=3e-5))
    optimizer = DiffusersFSDPEngine._build_optimizer(engine, module)
    assert len(optimizer.param_groups) == 1
    assert {id(p) for p in optimizer.param_groups[0]["params"]} == {id(p) for p in module.parameters()}
    assert optimizer.param_groups[0]["lr"] == 3e-5


@pytest.mark.parametrize("overrides", [{"": 1e-6}, {"gemma": -1}, {"gemma": float("nan")}, {"gemma": float("inf")}])
def test_invalid_parameter_group_lr_fails(overrides):
    with pytest.raises(ValueError, match="Invalid parameter LR group"):
        optimizer_parameters(torch.nn.Linear(2, 2), SimpleNamespace(lr=3e-5, param_group_lrs=overrides))
