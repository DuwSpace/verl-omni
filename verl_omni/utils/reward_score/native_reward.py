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

"""Shared device lifecycle for native reward scorers."""

from typing import Any, Protocol

import torch
from verl.utils.device import get_device_name, get_torch_device


class NativeRewardState(Protocol):
    model: Any
    device: torch.device | None


def resolve_reward_device(device: int | str | torch.device) -> torch.device:
    if isinstance(device, int):
        return torch.device(f"{get_device_name()}:{device}")
    return torch.device(device)


def activate_reward_model(
    state: NativeRewardState,
    device: int | str | torch.device,
    *,
    reward_name: str,
) -> None:
    """Move an inactive native reward model to its runtime device."""
    if state.model is None:
        raise RuntimeError(f"{reward_name} has already been finalized.")
    if state.device is not None:
        raise RuntimeError(f"{reward_name} is already active.")
    state.device = resolve_reward_device(device)
    state.model.to(state.device).eval()


def release_accelerator_memory() -> None:
    accelerator = get_torch_device()
    accelerator.synchronize()
    accelerator.empty_cache()
    accelerator.synchronize()


def deactivate_reward_model(state: NativeRewardState) -> None:
    """Move an active native reward model to CPU and release its device cache."""
    if state.device is None:
        return
    device = state.device
    state.model.to("cpu")
    state.device = None
    if device.type != "cpu":
        release_accelerator_memory()
