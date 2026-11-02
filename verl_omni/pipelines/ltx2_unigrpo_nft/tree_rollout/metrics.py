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

"""Reward distributions across recaptions and within their video siblings."""

import torch

from verl_omni.trainer.diffusion.diffusion_metric_utils import compute_reward_extra_metrics_diffusion

from .advantages import _recaption_groups


def compute_tree_reward_metrics(scores, uid, records, reward_names):
    """Collect parent reward moments and the mean within-parent population std."""
    groups = _recaption_groups(uid, records)
    scores = scores.detach().cpu().double()
    if scores.ndim != 2 or scores.shape[1] != len(reward_names):
        raise ValueError("Tree reward metrics require one name per reward component")
    parent_scores = torch.stack([scores[indices].mean(dim=0) for indices in groups])
    sibling_stds = torch.stack([scores[indices].std(dim=0, correction=0) for indices in groups])
    # Each parent has the same configured k, so trainer video-count weighting
    # is also parent-count weighting across local updates. Merge moments rather
    # than averaging stds from separate local batches.
    metrics = compute_reward_extra_metrics_diffusion(
        {f"tree/recaption/{name}": parent_scores[:, i].numpy() for i, name in enumerate(reward_names)},
        include_second_moment=True,
    )
    for i, name in enumerate(reward_names):
        metrics[f"critic/tree/within_recaption/{name}/std_mean"] = sibling_stds[:, i].mean().item()
    return metrics
