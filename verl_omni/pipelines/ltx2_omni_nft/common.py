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

"""Shared LTX-2.3 OmniNFT numerical helpers."""

import torch


def calculate_shift(
    sequence_length: int,
    base_sequence_length: int,
    max_sequence_length: int,
    base_shift: float,
    max_shift: float,
) -> float:
    """Calculate the native flow scheduler's dynamic timestep shift."""
    slope = (max_shift - base_shift) / (max_sequence_length - base_sequence_length)
    return sequence_length * slope + base_shift - slope * base_sequence_length


def apply_x0_cfg(
    sample: torch.Tensor,
    positive_velocity: torch.Tensor,
    negative_velocity: torch.Tensor,
    sigma: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Apply LTX-2.3 classifier-free guidance in clean-sample space."""
    positive_x0 = sample - sigma * positive_velocity
    negative_x0 = sample - sigma * negative_velocity
    guided_x0 = positive_x0 + (guidance_scale - 1.0) * (positive_x0 - negative_x0)
    return (sample - guided_x0) / sigma


__all__ = ["apply_x0_cfg", "calculate_shift"]
