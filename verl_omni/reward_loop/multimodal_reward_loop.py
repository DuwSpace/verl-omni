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
"""OmniNFT-specific batch reward workers, dispatch, and result assembly."""

from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from verl import DataProto
from verl.experimental.reward_loop import RewardLoopManager, RewardLoopWorker
from verl.plugin.platform import get_platform
from verl.utils.device import get_device_name

from verl_omni.reward_loop.reward_manager import SupportsBatchScoring


class MultiModalRewardLoopWorker(RewardLoopWorker):
    """Run one complete local chunk through a batch-capable reward manager."""

    async def compute_score_batch(self, data: DataProto) -> dict[str, Any]:
        if not isinstance(self.reward_manager, SupportsBatchScoring):
            raise TypeError("MultiModalRewardLoopWorker requires a batch-capable reward manager.")
        return await self.reward_manager.run_batch(data)

    def shutdown(self) -> None:
        """Finalize reward state owned by this Worker."""
        self.reward_manager.shutdown()


class MultiModalRewardLoopManager(RewardLoopManager):
    """Create accelerator-bound batch reward Workers for OmniNFT."""

    def __init__(self, config, resource_pool):
        if config.reward.reward_model.enable:
            raise ValueError("MultiModalRewardLoopManager does not use a RewardModel server.")
        self._resource_pool = resource_pool
        self._shutdown = False
        super().__init__(config=config, rm_resource_pool=None)

    def _init_reward_loop_workers(self) -> None:
        if not issubclass(self.reward_manager_cls, SupportsBatchScoring):
            raise TypeError("MultiModalRewardLoopManager requires a batch-capable reward manager.")
        if self._resource_pool is None:
            raise ValueError("MultiModal reward Workers require an actor/rollout resource pool.")

        num_workers = self.config.reward.num_workers
        if num_workers <= 0:
            raise ValueError("reward.num_workers must be greater than 0 for MultiModal reward Workers.")
        if num_workers > self._resource_pool.world_size:
            raise ValueError(
                f"reward.num_workers ({num_workers}) exceeds the actor/rollout resource pool world size "
                f"({self._resource_pool.world_size})."
            )

        placement_groups = self._resource_pool.get_placement_groups(device_name=get_device_name())
        placements = [
            (placement_group, bundle_index)
            for placement_group, local_world_size in zip(
                placement_groups, self._resource_pool.store, strict=True
            )
            for bundle_index in range(local_world_size)
        ]
        resource_options = get_platform().ray_resource_options(1 / self._resource_pool.max_colocate_count)
        worker_class = ray.remote(MultiModalRewardLoopWorker)
        self.reward_loop_workers = []
        for worker_index, (placement_group, bundle_index) in enumerate(placements[:num_workers]):
            self.reward_loop_workers.append(
                worker_class.options(
                    name=f"multimodal_reward_loop_worker_{worker_index}",
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_bundle_index=bundle_index,
                    ),
                    **resource_options,
                ).remote(self.config, self.reward_router_address)
            )

    def shutdown(self) -> None:
        """Finalize every batch reward Worker exactly once."""
        if self._shutdown:
            return
        self._shutdown = True
        ray.get([worker.shutdown.remote() for worker in self.reward_loop_workers])


def _sample_uids(data: DataProto) -> list[str]:
    values = data.non_tensor_batch.get("sample_uid")
    if values is None:
        raise ValueError("sample_uid is required for batch reward gathering.")
    values = np.asarray(values, dtype=object)
    if values.shape != (len(data),):
        raise ValueError(f"sample_uid must have shape ({len(data)},), got {values.shape}.")
    uids = [str(value) for value in values]
    if len(set(uids)) != len(uids):
        raise ValueError("sample_uid must be unique within a reward batch.")
    return uids


def assemble_batch_reward(
    data: DataProto, chunks: list[DataProto], outputs: list[dict[str, Any]]
) -> DataProto:
    """Validate local reward matrices and restore the global input order."""
    expected_uids = _sample_uids(data)
    if len(chunks) != len(outputs):
        raise ValueError(f"Expected {len(chunks)} Worker outputs, got {len(outputs)}.")

    names = None
    rows: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = {}
    for chunk, output in zip(chunks, outputs, strict=True):
        if not isinstance(output, dict):
            raise TypeError("Batch-capable Worker output must be a dict.")
        local_uids = output.get("sample_uid")
        if local_uids is None:
            raise ValueError("Batch reward output is missing sample_uid.")
        local_uids = [str(value) for value in np.asarray(local_uids, dtype=object).tolist()]
        if len(local_uids) != len(chunk) or len(set(local_uids)) != len(local_uids):
            raise ValueError("Batch reward output sample_uid must be unique and match the local chunk size.")
        if set(local_uids) != set(_sample_uids(chunk)):
            raise ValueError("Batch reward output sample_uid does not match its local chunk.")

        reward_names = output.get("reward_names")
        if not isinstance(reward_names, (list, tuple)) or not reward_names:
            raise ValueError("Batch reward output reward_names must be a non-empty list.")
        reward_names = list(reward_names)
        if any(not isinstance(name, str) for name in reward_names) or len(set(reward_names)) != len(reward_names):
            raise ValueError("Batch reward output reward_names must contain unique strings.")
        if names is None:
            names = reward_names
        elif reward_names != names:
            raise ValueError("Batch reward output reward_names disagree across Workers.")

        scores = output.get("rm_scores")
        mask = output.get("reward_valid_mask")
        if (
            not isinstance(scores, torch.Tensor)
            or scores.dtype != torch.float32
            or scores.shape != (len(chunk), len(names))
        ):
            raise ValueError("Batch reward output rm_scores must be float32 with shape (B_local, K).")
        if not torch.isfinite(scores).all():
            raise ValueError("Batch reward output rm_scores must contain only finite values.")
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != scores.shape:
            raise ValueError("Batch reward output reward_valid_mask must be bool with shape (B_local, K).")
        if not mask.all():
            raise ValueError("Batch reward output contains invalid required reward samples.")

        extra_info = output.get("reward_extra_info")
        if not isinstance(extra_info, dict) or set(extra_info) != set(names):
            raise ValueError("Batch reward output reward_extra_info must be keyed by reward_names.")
        if any(not isinstance(extra_info[name], dict) for name in names):
            raise ValueError("Batch reward output reward metadata values must be dicts.")

        for row, sample_uid in enumerate(local_uids):
            if sample_uid in rows:
                raise ValueError(f"Duplicate sample_uid across reward Workers: {sample_uid}.")
            rows[sample_uid] = (scores[row], mask[row], {name: extra_info[name] for name in names})

    if names is None or set(rows) != set(expected_uids):
        raise ValueError("Batch reward outputs do not cover the input sample_uid set exactly.")

    ordered = [rows[sample_uid] for sample_uid in expected_uids]
    return DataProto.from_dict(
        tensors={
            "rm_scores": torch.stack([row[0] for row in ordered]),
            "reward_valid_mask": torch.stack([row[1] for row in ordered]),
        },
        non_tensors={
            "sample_uid": np.asarray(expected_uids, dtype=object),
            "reward_extra_info": np.asarray([row[2] for row in ordered], dtype=object),
        },
        meta_info={"reward_names": names, "reward_extra_keys": ["reward_extra_info"]},
    )


class BatchRewardCoordinator:
    """Shard a global batch, dispatch local scoring, and assemble ``[B, K]``."""

    def __init__(self, worker_handles):
        self.worker_handles = list(worker_handles)
        if not self.worker_handles:
            raise ValueError("Batch reward requires at least one Worker.")

    def compute(self, data: DataProto) -> DataProto:
        chunks = data.chunk(len(self.worker_handles))
        outputs = ray.get(
            [
                worker.compute_score_batch.remote(chunk)
                for worker, chunk in zip(self.worker_handles, chunks, strict=True)
            ]
        )
        return assemble_batch_reward(data, chunks, outputs)
