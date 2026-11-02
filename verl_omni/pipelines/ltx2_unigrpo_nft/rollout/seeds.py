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

"""Assign the legacy joint seeds before rollout requests are split across workers."""

import torch
from tensordict import NonTensorData


def assign_legacy_rollout_seeds(prompts, *, base_seed, n, validate):
    """Attach per-session seeds before partitioning prompts across workers."""
    if validate:
        indices = torch.tensor([int(index) for index in prompts["index"]], dtype=torch.int64)
        seeds = int(base_seed) + indices[:, None] * n + torch.arange(n)
    else:
        step = prompts["global_steps"]
        if isinstance(step, NonTensorData):
            step = step.data
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ValueError("Legacy joint training seeds require a positive integer global_steps")
        count = len(prompts) * n
        # The old trainer reserves even policy versions for validation.
        seeds = int(base_seed) + (2 * step - 1) * count + torch.arange(count).reshape(len(prompts), n)
    prompts["_rollout_seeds"] = seeds
