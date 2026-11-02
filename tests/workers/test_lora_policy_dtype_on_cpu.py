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
"""Policy adapter masters retain precision independently of trainability."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin


def test_fp32_dtype_includes_frozen_old_adapter_and_optimizer_state():
    engine = LoRAAdapterMixin()
    engine.model_config = SimpleNamespace(
        lora_rank=2,
        lora_alpha=2,
        policy_state_adapters=["default", "old"],
        lora_init_weights=True,
        target_modules=["proj"],
        target_parameters=None,
        exclude_modules=None,
        lora_dtype="fp32",
    )
    module = MagicMock(spec=torch.nn.Module)
    module.add_adapter = MagicMock()
    module.set_adapter = MagicMock()
    base = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16), requires_grad=False)
    active = torch.nn.Parameter(torch.full((2, 2), 0.02, dtype=torch.bfloat16))
    old = torch.nn.Parameter(active.detach().clone(), requires_grad=False)
    module.named_parameters.return_value = [
        ("proj.base_layer.weight", base),
        ("proj.lora_A.default.weight", active),
        ("proj.lora_A.old.weight", old),
    ]
    module.modules.return_value = []
    engine._build_lora_module(module)
    assert base.dtype == torch.bfloat16
    assert active.dtype == old.dtype == torch.float32
    assert active.requires_grad and not old.requires_grad
    before = active.detach().clone()
    optimizer = torch.optim.AdamW([active], lr=3e-5, weight_decay=0)
    active.grad = torch.ones_like(active)
    optimizer.step()
    assert torch.all(active < before)
    assert optimizer.state[active]["exp_avg"].dtype == torch.float32
    assert optimizer.state[active]["exp_avg_sq"].dtype == torch.float32
    with torch.no_grad():
        old.lerp_(active, 0.5)
    torch.testing.assert_close(old, (before + active) / 2)
