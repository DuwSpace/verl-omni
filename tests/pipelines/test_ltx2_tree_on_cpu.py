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

"""Tree shape and malformed sibling groups fail before inference."""

from types import SimpleNamespace

import pytest

from verl_omni.pipelines.ltx2_unigrpo_nft.config import LTXJointPipelineConfig
from verl_omni.pipelines.ltx2_unigrpo_nft.tree_rollout.requests import group_recaption_requests, tree_width


def test_tree_shape_and_defaults():
    assert LTXJointPipelineConfig().rollout_mode == "flat"
    assert tree_width({"rollout_mode": "tree", "videos_per_recaption": 4}, 12) == 4
    with pytest.raises(ValueError, match="multiple"):
        tree_width({"rollout_mode": "tree", "videos_per_recaption": 4}, 10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rollout_mode": "invalid"},
        {"videos_per_recaption": 2},
        {"rollout_mode": "tree", "videos_per_recaption": 0},
        {"rollout_mode": "tree", "videos_per_recaption": True},
    ],
)
def test_invalid_tree_config(kwargs):
    with pytest.raises(ValueError):
        LTXJointPipelineConfig(**kwargs)


def request(index, seed):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(
            seed=seed,
            extra_args={
                "prepared_request_id": str(index),
                "recaption_group_id": "parent",
                "rollout_mode": "tree",
                "videos_per_recaption": 2,
                "recaption_video_index": index,
            },
        )
    )


def test_incomplete_duplicate_and_reused_seed_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        group_recaption_requests([request(0, 42)])
    with pytest.raises(ValueError, match="index"):
        group_recaption_requests([request(0, 42), request(0, 43)])
    with pytest.raises(ValueError, match="distinct seeds"):
        group_recaption_requests([request(0, 42), request(1, 42)])
    group = group_recaption_requests([request(1, 43), request(0, 42)])[0]
    assert [r.sampling_params.seed for r in group] == [42, 43]


@pytest.mark.parametrize("global_std", [True, False])
def test_text_normalization_uses_mean_rewards_per_recaption(global_std):
    import torch

    from verl_omni.pipelines.ltx2_unigrpo_nft.tree_rollout.advantages import (
        av_reward_group_ids,
        compute_text_advantages,
    )
    from verl_omni.trainer.diffusion.diffusion_algos import OmniNFTLoss

    # Deliberately interleaved siblings and reused group names across prompts.
    uid = ["a"] * 4 + ["b"] * 4
    records = [
        SimpleNamespace(recaption_group_id=str(i % 2), recaption_video_index=(i % 4) // 2, videos_per_recaption=2)
        for i in range(8)
    ]
    scores = torch.tensor([[v, 5.0] for v in [1, 2, 5, 12, 8, 14, 12, 22]], requires_grad=True)
    original = scores.detach().clone()
    result = compute_text_advantages(
        scores, uid, records, OmniNFTLoss._compute_component_advantages, global_std=global_std, epsilon=1e-6
    )
    # Parent rewards: a=[3,7], b=[10,18]. Global std=5.5; local stds=2,4.
    denominators = torch.full((8,), 5.5) if global_std else torch.tensor([2.0] * 4 + [4.0] * 4)
    expected = torch.tensor([-2.0, 2.0, -2.0, 2.0, -4.0, 4.0, -4.0, 4.0]) / (denominators + 1e-6) / 2
    torch.testing.assert_close(result, expected)
    assert not result.requires_grad
    torch.testing.assert_close(scores.detach(), original)
    assert av_reward_group_ids(uid, records).tolist() == [0, 1, 0, 1, 2, 3, 2, 3]


@pytest.mark.parametrize("global_std", [True, False])
def test_equal_parent_rewards_produce_zero_text_advantage(global_std):
    import torch

    from verl_omni.pipelines.ltx2_unigrpo_nft.tree_rollout.advantages import compute_text_advantages
    from verl_omni.trainer.diffusion.diffusion_algos import OmniNFTLoss

    records = [
        SimpleNamespace(recaption_group_id=str(i // 2), recaption_video_index=i % 2, videos_per_recaption=2)
        for i in range(4)
    ]
    scores = torch.tensor([[0.0], [10.0], [4.0], [6.0]])
    result = compute_text_advantages(
        scores, ["p"] * 4, records, OmniNFTLoss._compute_component_advantages, global_std=global_std, epsilon=1e-6
    )
    torch.testing.assert_close(result, torch.zeros(4), atol=0, rtol=0)


def test_tree_reward_metrics_merge_parent_moments_across_unequal_local_batches():
    import numpy as np
    import torch
    from verl.trainer.ppo.v1.utils import MetricsAggregator

    from verl_omni.pipelines.ltx2_unigrpo_nft.tree_rollout.metrics import compute_tree_reward_metrics
    from verl_omni.trainer.diffusion.diffusion_metric_utils import finalize_reward_extra_metrics_diffusion

    aggregator = MetricsAggregator()
    for values in ([1.0, 5.0, 2.0, 12.0], [8.0, 12.0]):
        records = [
            SimpleNamespace(recaption_group_id=str(i // 2), recaption_video_index=i % 2, videos_per_recaption=2)
            for i in range(len(values))
        ]
        metrics = compute_tree_reward_metrics(
            torch.tensor([[v, 5.0] for v in values]), ["p"] * len(values), records, ["quality", "constant"]
        )
        aggregator.add_step_metrics(metrics, sample_count=len(values))
    result = aggregator.get_aggregated_metrics()
    finalize_reward_extra_metrics_diffusion(result)
    assert result["critic/tree/recaption/quality/mean"] == pytest.approx(np.mean([3.0, 7.0, 10.0]))
    assert result["critic/tree/recaption/quality/std"] == pytest.approx(np.std([3.0, 7.0, 10.0]))
    assert result["critic/tree/within_recaption/quality/std_mean"] == pytest.approx(3.0)
    assert result["critic/tree/recaption/constant/std"] == pytest.approx(0.0)
    assert result["critic/tree/within_recaption/constant/std_mean"] == pytest.approx(0.0)
    assert not any("_second_moment" in key for key in result)


def test_single_video_recaption_has_zero_internal_std():
    import torch

    from verl_omni.pipelines.ltx2_unigrpo_nft.tree_rollout.metrics import compute_tree_reward_metrics

    record = SimpleNamespace(recaption_group_id="p:0", recaption_video_index=0, videos_per_recaption=1)
    result = compute_tree_reward_metrics(torch.tensor([[3.0]]), ["p"], [record], ["quality"])
    assert result["critic/tree/recaption/quality/mean"] == pytest.approx(3.0)
    assert result["critic/tree/recaption/quality/std"] == pytest.approx(0.0)
    assert result["critic/tree/within_recaption/quality/std_mean"] == pytest.approx(0.0)
