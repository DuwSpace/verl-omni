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

from types import SimpleNamespace
from unittest.mock import patch

import torch

from verl_omni.utils.diffusers_npu import _rms_norm_forward_native, apply_diffusers_npu_rms_norm_patch


def test_native_rms_norm_restores_input_dtype_when_weight_is_none() -> None:
    norm = SimpleNamespace(eps=1e-6, weight=None, bias=None)
    hidden_states = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16)

    actual = _rms_norm_forward_native(norm, hidden_states)
    hidden_fp32 = hidden_states.float()
    variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
    expected = (hidden_states * torch.rsqrt(variance + norm.eps)).to(hidden_states.dtype)

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == hidden_states.dtype


def test_apply_diffusers_npu_rms_norm_patch_is_noop_off_npu() -> None:
    with patch("verl.utils.device.get_device_name", return_value="cpu"):
        assert apply_diffusers_npu_rms_norm_patch() is None
