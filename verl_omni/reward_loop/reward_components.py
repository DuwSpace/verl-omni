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
"""Dispatch and assemble independently deployed component rewards."""

from typing import Any
from uuid import uuid4

import numpy as np
import torch
from verl import DataProto

from .async_utils import gather_complete


def ensure_sample_uids(data: DataProto) -> list[str]:
    """Return stable row identities, creating them when the caller has none."""
    values = data.non_tensor_batch.get("sample_uid")
    if values is None:
        values = np.asarray([uuid4().hex for _ in range(len(data))], dtype=object)
    else:
        values = np.asarray(values, dtype=object)
        if values.shape != (len(data),):
            raise ValueError(f"sample_uid must have shape ({len(data)},), got {values.shape}.")
    sample_uids = [str(value) for value in values]
    if len(set(sample_uids)) != len(sample_uids):
        raise ValueError("sample_uid must be unique within a reward batch.")
    data.non_tensor_batch["sample_uid"] = np.asarray(sample_uids, dtype=object)
    return sample_uids


def split_component_batch(data: DataProto, available_workers: int) -> list[DataProto]:
    """Split one model's full batch into balanced non-padding shards."""
    if len(data) <= 0:
        raise ValueError("Cannot compute component rewards for an empty DataProto.")
    if available_workers <= 0:
        raise ValueError("Component reward scoring requires at least one worker.")
    worker_count = min(len(data), available_workers)
    base_size, remainder = divmod(len(data), worker_count)
    sizes = [base_size + int(index < remainder) for index in range(worker_count)]
    chunks = []
    start = 0
    for size in sizes:
        chunks.append(data[start : start + size])
        start += size
    return chunks


async def compute_component_rewards(
    worker_groups: dict[str, list[Any]],
    data: DataProto,
    expected_reward_names: set[str],
) -> DataProto:
    """Score every sample with every model group and restore global column order."""
    ensure_sample_uids(data)
    requests = []
    request_meta = []
    for group_name, workers in worker_groups.items():
        chunks = split_component_batch(data, len(workers))
        for worker, chunk in zip(workers[: len(chunks)], chunks, strict=True):
            requests.append(worker.compute_score_components.remote(chunk))
            request_meta.append((group_name, chunk))
    outputs = await gather_complete(requests)
    grouped = [(group_name, chunk, output) for (group_name, chunk), output in zip(request_meta, outputs, strict=True)]
    return assemble_component_rewards(data, grouped, expected_reward_names)


def assemble_component_rewards(
    data: DataProto,
    grouped_outputs: list[tuple[str, DataProto, dict[str, Any]]],
    expected_reward_names: set[str],
) -> DataProto:
    """Merge results by ``(sample_uid, reward_name)`` and reject partial coverage."""
    expected_uids = ensure_sample_uids(data)
    if not expected_reward_names:
        raise ValueError("Expected reward names must be non-empty.")

    cells: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for group_name, chunk, output in grouped_outputs:
        if not isinstance(output, dict):
            raise TypeError(f"Reward group {group_name!r} must return a dict.")
        local_uids = [str(value) for value in np.asarray(output.get("sample_uid"), dtype=object).tolist()]
        expected_local_uids = ensure_sample_uids(chunk)
        if len(local_uids) != len(chunk) or len(set(local_uids)) != len(local_uids):
            raise ValueError(f"Reward group {group_name!r} returned invalid sample_uid values.")
        if set(local_uids) != set(expected_local_uids):
            raise ValueError(f"Reward group {group_name!r} sample_uid values do not match its shard.")

        names = list(output.get("reward_names", []))
        if not names or len(set(names)) != len(names) or not set(names).issubset(expected_reward_names):
            raise ValueError(f"Reward group {group_name!r} returned invalid reward_names {names}.")
        scores = output.get("rm_scores")
        mask = output.get("reward_valid_mask")
        if not isinstance(scores, torch.Tensor) or scores.shape != (len(chunk), len(names)):
            shape = None if not isinstance(scores, torch.Tensor) else tuple(scores.shape)
            raise ValueError(f"Reward group {group_name!r} scores have invalid shape {shape}.")
        if not isinstance(mask, torch.Tensor) or mask.shape != scores.shape:
            shape = None if not isinstance(mask, torch.Tensor) else tuple(mask.shape)
            raise ValueError(f"Reward group {group_name!r} valid mask has invalid shape {shape}.")
        scores = scores.detach().to(device="cpu", dtype=torch.float32)
        mask = mask.detach().to(device="cpu", dtype=torch.bool)

        output_definitions = output.get("reward_definitions", {})
        if set(output_definitions) != set(names):
            raise ValueError(f"Reward group {group_name!r} definitions must match reward_names.")
        for name in names:
            definition = dict(output_definitions[name])
            existing = definitions.setdefault(name, definition)
            if existing != definition:
                raise ValueError(f"Reward definition for {name!r} differs across replicas.")
        for row, sample_uid in enumerate(local_uids):
            for column, name in enumerate(names):
                key = (sample_uid, name)
                if key in cells:
                    raise ValueError(f"Duplicate component result for sample_uid={sample_uid!r}, reward={name!r}.")
                cells[key] = (scores[row, column], mask[row, column])

    reward_names = sorted(expected_reward_names)
    expected_cells = {(sample_uid, name) for sample_uid in expected_uids for name in reward_names}
    if set(cells) != expected_cells:
        missing = sorted(expected_cells - set(cells))
        extra = sorted(set(cells) - expected_cells)
        raise ValueError(f"Component reward coverage mismatch; missing={missing[:5]}, extra={extra[:5]}.")

    scores = torch.stack(
        [torch.stack([cells[(sample_uid, name)][0] for name in reward_names]) for sample_uid in expected_uids]
    )
    valid_mask = torch.stack(
        [torch.stack([cells[(sample_uid, name)][1] for name in reward_names]) for sample_uid in expected_uids]
    )
    if not torch.isfinite(scores).all() or not valid_mask.all():
        raise ValueError("Required component rewards must be finite and fully valid before actor update.")

    non_tensors = {"sample_uid": np.asarray(expected_uids, dtype=object)}
    reward_extra_keys = []
    for column, name in enumerate(reward_names):
        key = f"reward/{name}"
        non_tensors[key] = scores[:, column].numpy()
        reward_extra_keys.append(key)
    return DataProto.from_dict(
        tensors={"rm_scores": scores, "reward_valid_mask": valid_mask},
        non_tensors=non_tensors,
        meta_info={
            "reward_names": reward_names,
            "reward_definitions": definitions,
            "reward_extra_keys": reward_extra_keys,
        },
    )


__all__ = [
    "assemble_component_rewards",
    "compute_component_rewards",
    "ensure_sample_uids",
    "split_component_batch",
]
