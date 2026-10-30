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

"""Named learning-rate groups for diffusion FSDP optimizers."""

import math


def optimizer_parameters(module, config):
    """Build ordered name-substring LR groups, retaining the default ungrouped path.

    Only trainable parameters enter explicit groups. The first matching key wins;
    unmatched trainable parameters keep the base learning rate. Optimizer class,
    betas and other options remain the responsibility of verl's optimizer builder.
    """
    overrides = getattr(config, "param_group_lrs", None)
    if not overrides:
        return module.parameters()
    for key, lr in overrides.items():
        if not key or not math.isfinite(float(lr)) or float(lr) < 0:
            raise ValueError(f"Invalid parameter LR group {key!r}: {lr}")
    buckets = {key: [] for key in overrides}
    base = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        key = next((key for key in overrides if key in name), None)
        (base if key is None else buckets[key]).append(param)
    groups = [{"params": base, "lr": float(config.lr)}] if base else []
    groups.extend({"params": params, "lr": float(overrides[key])} for key, params in buckets.items() if params)
    return groups
