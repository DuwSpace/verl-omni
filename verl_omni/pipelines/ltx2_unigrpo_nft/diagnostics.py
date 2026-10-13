# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Branch gradient attribution on local, possibly CPU-offloaded FSDP shards."""

import torch
from torch.distributed.tensor import DTensor


def gradient_snapshot(module):
    """Copy accumulated gradients, so a subsequent backward can be attributed."""
    result = {}
    for name, parameter in module.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            value = parameter.grad
            value = value.to_local() if isinstance(value, DTensor) else value
            result[name] = value.detach().float().cpu().clone()
    return result


def gradient_delta_metrics(before, after, branch):
    """Report local shard norms; do not label a rank average as a global norm."""
    squared = {"gemma": 0.0, "dit": 0.0}
    nonfinite = 0
    for name in before.keys() | after.keys():
        if not name.startswith(("gemma.", "transformer.")):
            raise ValueError(f"Unexpected joint gradient parameter: {name}")
        value = after.get(name)
        previous = before.get(name)
        delta = -previous if value is None else value if previous is None else value - previous
        nonfinite += int((~torch.isfinite(delta)).sum())
        component = "gemma" if name.startswith("gemma.") else "dit"
        squared[component] += delta.double().square().sum().item()
    return {
        **{f"train/{branch}_{component}_grad_norm_local": value**0.5 for component, value in squared.items()},
        f"train/{branch}_nonfinite_gradient_elements_local": nonfinite,
    }
