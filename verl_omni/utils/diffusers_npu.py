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

"""Diffusers compatibility workarounds for Ascend NPU."""

import torch


def _rms_norm_forward_native(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run Diffusers RMSNorm's native implementation.

    ``torch_npu.npu_rms_norm`` requires a tensor for ``gamma``.  Diffusers'
    affine-free RMSNorm intentionally stores ``weight`` as ``None``, so use
    the same native math as Diffusers for that case.
    """
    input_dtype = hidden_states.dtype
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

    if self.weight is not None:
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        hidden_states = hidden_states * self.weight
        if self.bias is not None:
            hidden_states = hidden_states + self.bias
    else:
        hidden_states = hidden_states.to(input_dtype)

    return hidden_states


def apply_diffusers_npu_rms_norm_patch() -> None:
    """Work around Diffusers passing ``gamma=None`` to the NPU RMSNorm op.

    The patch is deliberately limited to NPU and to affine-free RMSNorm
    instances.  Weighted RMSNorm instances retain Diffusers' fused NPU path.
    """
    from verl.utils.device import get_device_name

    if get_device_name() != "npu":
        return

    from diffusers.models.normalization import RMSNorm

    original_forward = RMSNorm.forward
    if getattr(original_forward, "_verl_omni_npu_patch", False):
        return

    def patched_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            return _rms_norm_forward_native(self, hidden_states)
        return original_forward(self, hidden_states)

    patched_forward._verl_omni_npu_patch = True
    RMSNorm.forward = patched_forward
