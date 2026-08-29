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
"""CPU contracts for the OmniNFT-specific batch Reward execution path."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from verl import DataProto
from verl.experimental.reward_loop import RewardLoopManager

from verl_omni.reward_loop.multimodal_reward_loop import (
    BatchRewardCoordinator,
    MultiModalRewardLoopManager,
    MultiModalRewardLoopWorker,
    assemble_batch_reward,
)
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager
from verl_omni.reward_loop.reward_manager import MultiModalRewardManager


class _BatchManager:
    def __init__(self):
        self.batch_sizes = []
        self.shutdown_count = 0

    async def run_batch(self, data):
        self.batch_sizes.append(len(data))
        return {"sample_uid": list(data.non_tensor_batch["sample_uid"])}

    def shutdown(self):
        self.shutdown_count += 1


def _make_data(uids):
    return DataProto.from_dict(
        tensors={"responses": torch.zeros((len(uids), 1), dtype=torch.uint8)},
        non_tensors={"sample_uid": np.asarray(uids, dtype=object)},
    )


def _local_output(uids, offset=0.0, names=None):
    names = names or ["video", "audio"]
    size = len(uids)
    return {
        "rm_scores": torch.arange(size * len(names), dtype=torch.float32).reshape(size, len(names)) + offset,
        "reward_valid_mask": torch.ones((size, len(names)), dtype=torch.bool),
        "reward_names": names,
        "sample_uid": np.asarray(uids, dtype=object),
        "reward_extra_info": {name: {"batch": size, "name": name} for name in names},
    }


def test_common_reward_loop_manager_keeps_original_execution_methods():
    assert "_init_reward_loop_workers" not in OmniRewardLoopManager.__dict__
    assert "compute_rm_score" not in OmniRewardLoopManager.__dict__
    assert "shutdown" not in OmniRewardLoopManager.__dict__


def test_batch_worker_dispatches_one_call_for_the_complete_chunk():
    worker = object.__new__(MultiModalRewardLoopWorker)
    manager = _BatchManager()
    worker.reward_manager = manager
    data = _make_data(["s0", "s1", "s2"])

    result = asyncio.run(worker.compute_score_batch(data))

    assert manager.batch_sizes == [3]
    assert result["sample_uid"] == ["s0", "s1", "s2"]


def test_batch_worker_rejects_scalar_manager():
    worker = object.__new__(MultiModalRewardLoopWorker)
    worker.reward_manager = object()

    with pytest.raises(TypeError, match="batch-capable"):
        asyncio.run(worker.compute_score_batch(_make_data(["s0"])))


def test_batch_worker_shutdown_delegates_to_manager():
    worker = object.__new__(MultiModalRewardLoopWorker)
    manager = _BatchManager()
    worker.reward_manager = manager

    worker.shutdown()

    assert manager.shutdown_count == 1


def test_gather_restores_input_order_and_preserves_component_matrix():
    data = _make_data(["s2", "s0", "s3", "s1"])
    chunks = [data[:2], data[2:]]
    outputs = [
        _local_output(["s0", "s2"], offset=10.0),
        _local_output(["s1", "s3"], offset=20.0),
    ]

    result = assemble_batch_reward(data, chunks, outputs)

    assert result.non_tensor_batch["sample_uid"].tolist() == ["s2", "s0", "s3", "s1"]
    assert result.meta_info["reward_names"] == ["video", "audio"]
    torch.testing.assert_close(
        result.batch["rm_scores"],
        torch.tensor([[12.0, 13.0], [10.0, 11.0], [22.0, 23.0], [20.0, 21.0]]),
    )
    assert result.batch["reward_valid_mask"].all()
    assert result.non_tensor_batch["reward_extra_info"][0]["video"]["name"] == "video"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda output: output.update(sample_uid=np.asarray(["s0", "s0"], dtype=object)), "unique"),
        (lambda output: output.update(rm_scores=torch.ones((2, 1))), "float32 with shape"),
        (lambda output: output["reward_valid_mask"].fill_(False), "invalid required"),
        (lambda output: output.update(reward_names=["audio", "video"]), "disagree"),
        (lambda output: output.update(sample_uid=np.asarray(["unknown", "s2"], dtype=object)), "does not match"),
    ],
)
def test_malformed_worker_output_fails_closed(mutate, match):
    data = _make_data(["s0", "s1", "s2", "s3"])
    chunks = [data[:2], data[2:]]
    outputs = [_local_output(["s0", "s1"]), _local_output(["s2", "s3"])]
    mutate(outputs[0])

    with pytest.raises((TypeError, ValueError), match=match):
        assemble_batch_reward(data, chunks, outputs)


def test_missing_sample_uid_fails_closed():
    data = _make_data(["s0", "s1"])
    output = _local_output(["s0", "s1"])
    output.pop("sample_uid")

    with pytest.raises(ValueError, match="missing sample_uid"):
        assemble_batch_reward(data, [data], [output])


def _fake_resource_pool():
    pool = SimpleNamespace(world_size=3, store=[2, 1], max_colocate_count=4)
    pool.get_placement_groups = MagicMock(return_value=["pg-0", "pg-1"])
    return pool


def _make_loop_manager(num_workers, resource_pool, reward_manager_cls=MultiModalRewardManager):
    manager = object.__new__(MultiModalRewardLoopManager)
    manager.config = SimpleNamespace(reward=SimpleNamespace(num_workers=num_workers))
    manager.reward_router_address = None
    manager.reward_manager_cls = reward_manager_cls
    manager._resource_pool = resource_pool
    return manager


def test_batch_workers_bind_to_unique_actor_pool_bundles(monkeypatch):
    resource_pool = _fake_resource_pool()
    remote_class = MagicMock()
    remote_class.options.return_value.remote.side_effect = ["worker-0", "worker-1", "worker-2"]
    platform = SimpleNamespace(ray_resource_options=MagicMock(return_value={"resources": {"NPU": 0.25}}))
    monkeypatch.setattr("verl_omni.reward_loop.multimodal_reward_loop.ray.remote", lambda cls: remote_class)
    monkeypatch.setattr(
        "verl_omni.reward_loop.multimodal_reward_loop.PlacementGroupSchedulingStrategy",
        lambda placement_group, placement_group_bundle_index: (placement_group, placement_group_bundle_index),
    )
    monkeypatch.setattr("verl_omni.reward_loop.multimodal_reward_loop.get_device_name", lambda: "npu")
    monkeypatch.setattr("verl_omni.reward_loop.multimodal_reward_loop.get_platform", lambda: platform)
    manager = _make_loop_manager(num_workers=3, resource_pool=resource_pool)

    manager._init_reward_loop_workers()

    options = [remote_call.kwargs for remote_call in remote_class.options.call_args_list]
    assert [item["scheduling_strategy"] for item in options] == [("pg-0", 0), ("pg-0", 1), ("pg-1", 0)]
    assert [item["name"] for item in options] == [
        "multimodal_reward_loop_worker_0",
        "multimodal_reward_loop_worker_1",
        "multimodal_reward_loop_worker_2",
    ]
    assert all(item["resources"] == {"NPU": 0.25} for item in options)
    platform.ray_resource_options.assert_called_once_with(0.25)
    resource_pool.get_placement_groups.assert_called_once_with(device_name="npu")
    assert manager.reward_loop_workers == ["worker-0", "worker-1", "worker-2"]


@pytest.mark.parametrize(
    ("num_workers", "resource_pool", "reward_manager_cls", "match"),
    [
        (1, None, MultiModalRewardManager, "require an actor/rollout resource pool"),
        (0, _fake_resource_pool(), MultiModalRewardManager, "greater than 0"),
        (4, _fake_resource_pool(), MultiModalRewardManager, "exceeds the actor/rollout resource pool"),
        (1, _fake_resource_pool(), object, "batch-capable"),
    ],
)
def test_invalid_multimodal_reward_worker_configuration_fails_closed(
    num_workers, resource_pool, reward_manager_cls, match
):
    manager = _make_loop_manager(num_workers, resource_pool, reward_manager_cls)

    with pytest.raises((TypeError, ValueError), match=match):
        manager._init_reward_loop_workers()


def test_actor_pool_is_available_during_upstream_initialization(monkeypatch):
    seen_pools = []

    def fake_base_init(self, config, rm_resource_pool):
        self.reward_manager_cls = MultiModalRewardManager
        self.reward_router_address = None
        self._init_reward_loop_workers()

    monkeypatch.setattr(RewardLoopManager, "__init__", fake_base_init)
    monkeypatch.setattr(
        MultiModalRewardLoopManager,
        "_init_reward_loop_workers",
        lambda self: seen_pools.append(self._resource_pool),
    )
    config = SimpleNamespace(reward=SimpleNamespace(reward_model=SimpleNamespace(enable=False)))
    resource_pool = object()

    MultiModalRewardLoopManager(config=config, resource_pool=resource_pool)

    assert seen_pools == [resource_pool]


def test_controller_shutdown_dispatches_to_all_workers_once(monkeypatch):
    workers = [MagicMock(), MagicMock()]
    workers[0].shutdown.remote.return_value = "ref-0"
    workers[1].shutdown.remote.return_value = "ref-1"
    ray_get = MagicMock()
    monkeypatch.setattr("verl_omni.reward_loop.multimodal_reward_loop.ray.get", ray_get)
    manager = object.__new__(MultiModalRewardLoopManager)
    manager.reward_loop_workers = workers
    manager._shutdown = False

    manager.shutdown()
    manager.shutdown()

    workers[0].shutdown.remote.assert_called_once_with()
    workers[1].shutdown.remote.assert_called_once_with()
    ray_get.assert_called_once_with(["ref-0", "ref-1"])


def test_batch_coordinator_dispatches_chunks_and_assembles(monkeypatch):
    data = _make_data(["s0", "s1", "s2", "s3"])
    workers = [MagicMock(), MagicMock()]
    workers[0].compute_score_batch.remote.return_value = "ref-0"
    workers[1].compute_score_batch.remote.return_value = "ref-1"
    outputs = [_local_output(["s0", "s1"]), _local_output(["s2", "s3"], offset=10.0)]
    ray_get = MagicMock(return_value=outputs)
    monkeypatch.setattr("verl_omni.reward_loop.multimodal_reward_loop.ray.get", ray_get)

    result = BatchRewardCoordinator(workers).compute(data)

    ray_get.assert_called_once_with(["ref-0", "ref-1"])
    assert result.batch["rm_scores"].shape == (4, 2)
    assert result.non_tensor_batch["sample_uid"].tolist() == ["s0", "s1", "s2", "s3"]


def test_batch_coordinator_requires_workers():
    with pytest.raises(ValueError, match="at least one Worker"):
        BatchRewardCoordinator([])
