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
"""CPU tests for the LTX OmniNFT RMSNorm compatibility workaround."""

import pytest
import torch
from diffusers.models.normalization import RMSNorm

from verl_omni.pipelines.ltx2_omni_nft import compat


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ltx_workaround_preserves_rms_norm_outside_npu(monkeypatch, device):
    monkeypatch.setattr(compat, "get_device_name", lambda: device)
    module = torch.nn.Sequential(RMSNorm(4, eps=1e-6, elementwise_affine=False))
    original = module[0].forward.__func__

    compat.apply_ltx_npu_rms_norm_workaround(module)

    assert module[0].forward.__func__ is original


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ltx_npu_workaround_preserves_output_and_gradients(monkeypatch, dtype):
    monkeypatch.setattr(compat, "get_device_name", lambda: "npu")
    norm = RMSNorm(4, eps=1e-6, elementwise_affine=False)
    untouched_norm = RMSNorm(4, eps=1e-6, elementwise_affine=False)
    original = RMSNorm.forward
    inputs = torch.tensor([[0.25, -0.5, 1.0, 2.0], [-1.5, 0.0, 0.75, 0.5]], dtype=dtype, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    before = inputs.detach().clone()
    state_keys = tuple(norm.state_dict())

    compat.apply_ltx_npu_rms_norm_workaround(norm)
    output = norm(inputs)
    reference = torch.nn.functional.rms_norm(reference_inputs.float(), (4,), eps=norm.eps).to(dtype)
    output.float().square().sum().backward()
    reference.float().square().sum().backward()

    assert output.dtype == dtype
    assert output.device == inputs.device
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(inputs, before)
    torch.testing.assert_close(
        inputs.grad, reference_inputs.grad, atol=2e-2 if dtype == torch.bfloat16 else 1e-6, rtol=1e-4
    )
    assert torch.isfinite(inputs.grad).all()
    assert tuple(norm.state_dict()) == state_keys
    assert untouched_norm.forward.__func__ is original
    assert RMSNorm.forward is original
