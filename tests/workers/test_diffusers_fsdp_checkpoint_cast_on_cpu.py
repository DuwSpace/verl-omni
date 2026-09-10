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

import torch

from verl_omni.workers.engine.fsdp.diffusers_impl import _fsdp2_gradient_checkpointing_with_cast_func


class _DtypeProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_dtypes: list[tuple[torch.dtype, torch.dtype | None]] = []

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None
    ) -> torch.Tensor:
        encoder_dtype = None if encoder_hidden_states is None else encoder_hidden_states.dtype
        self.seen_dtypes.append((hidden_states.dtype, encoder_dtype))
        return hidden_states * 2


def test_fsdp2_checkpoint_cast_converts_fp32_inputs_to_param_dtype() -> None:
    param_dtype = torch.bfloat16
    checkpoint_fn = _fsdp2_gradient_checkpointing_with_cast_func(param_dtype)
    module = _DtypeProbe()
    hidden_states = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
    encoder_hidden_states = torch.randn(2, 4, dtype=torch.float32)

    output = checkpoint_fn(module, hidden_states, encoder_hidden_states=encoder_hidden_states)
    output.float().sum().backward()

    assert output.dtype == param_dtype
    assert hidden_states.grad is not None
    assert module.seen_dtypes == [(param_dtype, param_dtype)]


def test_fsdp2_checkpoint_cast_leaves_matching_dtype_unchanged() -> None:
    param_dtype = torch.bfloat16
    checkpoint_fn = _fsdp2_gradient_checkpointing_with_cast_func(param_dtype)
    module = _DtypeProbe()
    hidden_states = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)

    output = checkpoint_fn(module, hidden_states)

    assert output.dtype == param_dtype
    assert module.seen_dtypes == [(param_dtype, None)]
