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

"""Text credit across mean recaption rewards; AV credit within each recaption."""

from collections import defaultdict

import numpy as np
import torch


def _recaption_groups(uid, records):
    groups = defaultdict(list)
    for index, (prompt_id, record) in enumerate(zip(uid, records, strict=True)):
        if record.recaption_group_id is None:
            raise ValueError("Actor batch mixes tree and flat rollout records")
        key = (prompt_id, record.recaption_group_id)
        groups[key].append(index)
    for indices in groups.values():
        if any(records[index].videos_per_recaption != len(indices) for index in indices) or {
            records[index].recaption_video_index for index in indices
        } != set(range(len(indices))):
            raise ValueError("Actor batch contains an incomplete recaption tree")
    return list(groups.values())


def compute_text_advantages(scores, uid, records, compute_components, *, global_std, epsilon):
    """Average raw sibling rewards before computing any text reward statistics."""
    groups = _recaption_groups(uid, records)
    scores = scores.detach().float()
    parent_scores = torch.stack([scores[indices].mean(dim=0) for indices in groups])
    parent_uid = [uid[indices[0]] for indices in groups]
    components = compute_components(
        scores=parent_scores, uid=parent_uid, norm_by_std=True, global_std=global_std, epsilon=epsilon
    )
    parent_advantages = components.mean(dim=-1)
    result = scores.new_empty(scores.shape[0])
    for index, indices in enumerate(groups):
        result[indices] = parent_advantages[index]
    return result


def av_reward_group_ids(uid, records):
    """Identify the k sibling videos used to center tree AV rewards."""
    groups = {}
    result = np.empty(len(records), dtype=np.int64)
    for index, (prompt_id, record) in enumerate(zip(uid, records, strict=True)):
        key = (prompt_id, record.recaption_group_id)
        result[index] = groups.setdefault(key, len(groups))
    return result
