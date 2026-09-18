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
"""Dispatch and assemble independently deployed component rewards.

Rows follow ``sample_uid`` and columns follow ``reward_names``. Scorers may
calibrate scores according to their own definitions; assembly preserves those
components without cross-reward weighting or advantage normalization. Modality
routing belongs to the algorithm layer. Missing or invalid cells abort assembly.
"""

import asyncio
from typing import Any, TypedDict
from uuid import uuid4

import numpy as np
import torch
from verl import DataProto

class ComponentRewardOutput(TypedDict):
    """One worker shard of named component rewards.

    ``rm_scores`` is CPU float32 ``[B, K]`` and ``reward_valid_mask`` is CPU
    bool with the same shape. ``sample_uid[B]`` identifies rows;
    ``reward_names[K]`` identifies columns.
    """

    rm_scores: torch.Tensor
    reward_valid_mask: torch.Tensor
    reward_names: list[str]
    sample_uid: np.ndarray


def ensure_sample_uids(data: DataProto) -> list[str]:
    """Validate/stringify row IDs and write them back to data.non_tensor_batch.

    Missing IDs are generated with UUID4 and are not deterministic across runs.
    Return IDs in sample order; mismatched shape or duplicate stringified IDs
    raise ValueError. These identify rows, independently of prompt-group uid.
    """
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
    """Dispatch all model groups over the batch and assemble aligned components.

    ``data`` carries responses and scorer-specific media/text fields; sample_uid
    is ensured in place. Each group receives all samples through balanced,
    unpadded shards, then validate complete coverage of
    ``expected_reward_names`` for each row.

    Return a reward-only DataProto in input row order and sorted reward-name
    order, following ``ComponentRewardOutput`` tensor dtypes and shapes.
    Assembly preserves scorer values without weighted merging or normalization.
    """
    ensure_sample_uids(data)
    requests = []
    request_meta = []
    for group_name, workers in worker_groups.items():
        chunks = split_component_batch(data, len(workers))
        for worker, chunk in zip(workers[: len(chunks)], chunks, strict=True):
            requests.append(worker.compute_score_components.remote(chunk))
            request_meta.append((group_name, chunk))
    outputs = await asyncio.gather(*requests)
    grouped = [(group_name, chunk, output) for (group_name, chunk), output in zip(request_meta, outputs, strict=True)]
    return assemble_component_rewards(data, grouped, expected_reward_names)


def assemble_component_rewards(
    data: DataProto,
    grouped_outputs: list[tuple[str, DataProto, ComponentRewardOutput]],
    expected_reward_names: set[str],
) -> DataProto:
    """Merge worker shard outputs into a fully covered reward-only DataProto.

    Args:
        data: Source batch whose sample_uid order defines output rows. IDs are
            validated or generated and written back in place.
        grouped_outputs: ``(group_name, original_shard, ComponentRewardOutput)``
            triples. Output rows may be permuted but must match the shard IDs.
        expected_reward_names: Nonempty expected component set; sorted for output.

    Returns:
        CPU FP32 rm_scores and boolean reward_valid_mask, both ``[B, K]``,
        plus sample IDs and reward column names.
        Scores retain their scorer-defined scale; no cross-reward aggregation
        or advantage normalization is performed.

    Raises:
        TypeError: A worker result is not a dict.
        ValueError: IDs, names, or shapes disagree; coverage
            is missing/extra/duplicated; or any score is nonfinite or invalid.
    """
    expected_uids = ensure_sample_uids(data)
    if not expected_reward_names:
        raise ValueError("Expected reward names must be non-empty.")

    cells: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
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

    return DataProto.from_dict(
        tensors={"rm_scores": scores, "reward_valid_mask": valid_mask},
        non_tensors={"sample_uid": np.asarray(expected_uids, dtype=object)},
        meta_info={"reward_names": reward_names},
    )


__all__ = [
    "ComponentRewardOutput",
    "assemble_component_rewards",
    "compute_component_rewards",
    "ensure_sample_uids",
    "split_component_batch",
]
