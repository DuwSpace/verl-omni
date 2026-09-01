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
"""Batch-preserving reward manager for multimodal training."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import torch
from verl import DataProto
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.import_utils import load_module

from .multi import MultiVisualRewardManager
from .visual import VisualRewardManager, _validate_visual_response


@runtime_checkable
class SupportsBatchScoring(Protocol):
    """Runtime capability for managers that score a complete local batch."""

    async def run_batch(self, data: DataProto) -> dict[str, Any]:
        """Score a local batch while preserving reward components."""
        ...


@dataclass
class RewardRuntimeEntry:
    """Runtime state and scoring hook for one reward component."""

    name: str
    state: Any
    activate: Callable[..., Any]
    score_batch: Callable[..., Any]
    deactivate: Callable[..., Any]
    finalize: Callable[..., Any] | None
    micro_batch_size: int


class MultiModalRewardManager(MultiVisualRewardManager):
    """Score each configured reward once per local batch without aggregation."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        # MultiVisualRewardManager.__init__ is specific to path/name/weight aggregation.
        VisualRewardManager.__init__(
            self,
            config,
            tokenizer,
            compute_score,
            reward_router_address,
            reward_model_tokenizer,
        )
        self.component_order = self._validate_config()
        self._parallel_groups = self._validate_parallel_groups()
        self._reward_entries = [
            self._initialize_reward(name, config.reward.reward_functions[name]) for name in self.component_order
        ]
        entries_by_name = {entry.name: entry for entry in self._reward_entries}
        grouped_names = {name for names in self._parallel_groups.values() for name in names}
        self._schedule_units = [
            tuple(entries_by_name[name] for name in names)
            for names in self._build_schedule_names(grouped_names)
        ]
        self._shutdown = False

    def _validate_config(self) -> list[str]:
        if self.config.reward.aggregation != "preserve_components":
            raise ValueError("MultiModalRewardManager requires reward.aggregation='preserve_components'.")

        reward_functions = self.config.reward.reward_functions
        component_order = list(self.config.reward.component_order)
        if (
            not component_order
            or len(component_order) != len(reward_functions)
            or set(component_order) != set(reward_functions)
        ):
            raise ValueError("reward.component_order must be non-empty, unique, and match reward.reward_functions.")
        return component_order

    def _validate_parallel_groups(self) -> dict[str, tuple[str, ...]]:
        reward_config = self.config.reward
        native_config = (
            reward_config.get("native", {})
            if isinstance(reward_config, Mapping)
            else getattr(reward_config, "native", {})
        )
        if native_config is None:
            native_config = {}
        if not isinstance(native_config, Mapping):
            raise ValueError("reward.native must be a mapping.")

        parallel_groups = native_config.get("parallel_groups", {})
        if parallel_groups is None:
            parallel_groups = {}
        if not isinstance(parallel_groups, Mapping):
            raise ValueError("reward.native.parallel_groups must be a mapping.")

        member_to_group = {}
        validated = {}
        for group_name, group_config in parallel_groups.items():
            if not isinstance(group_name, str) or not group_name:
                raise ValueError("parallel group names must be non-empty strings.")
            if not isinstance(group_config, Mapping):
                raise ValueError(f"Parallel group '{group_name}' must be a mapping.")
            members = group_config.get("rewards")
            if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
                raise ValueError(f"Parallel group '{group_name}' rewards must be a list.")
            members = list(members)
            if len(members) < 2:
                raise ValueError(f"Parallel group '{group_name}' must contain at least two rewards.")
            if any(not isinstance(name, str) or not name for name in members):
                raise ValueError(f"Parallel group '{group_name}' rewards must be non-empty strings.")
            if len(set(members)) != len(members):
                raise ValueError(f"Parallel group '{group_name}' contains duplicate rewards.")

            unknown = [name for name in members if name not in self.component_order]
            if unknown:
                raise ValueError(f"Parallel group '{group_name}' contains unknown rewards: {unknown}.")
            positions = sorted(self.component_order.index(name) for name in members)
            expected = list(range(positions[0], positions[0] + len(positions)))
            if positions != expected:
                raise ValueError(
                    f"Parallel group '{group_name}' rewards must be a contiguous component_order subsequence."
                )
            canonical_members = tuple(self.component_order[index] for index in expected)
            for name in canonical_members:
                if name in member_to_group:
                    raise ValueError(f"Reward '{name}' belongs to multiple parallel groups.")
                member_to_group[name] = group_name
            validated[group_name] = canonical_members
        return validated

    def _build_schedule_names(self, grouped_names: set[str]) -> list[tuple[str, ...]]:
        group_by_reward = {
            reward_name: group_name
            for group_name, reward_names in self._parallel_groups.items()
            for reward_name in reward_names
        }
        units = []
        seen_groups = set()
        for reward_name in self.component_order:
            group_name = group_by_reward.get(reward_name)
            if group_name is None:
                units.append((reward_name,))
            elif group_name not in seen_groups:
                units.append(self._parallel_groups[group_name])
                seen_groups.add(group_name)
        if grouped_names != set(group_by_reward):
            raise RuntimeError("Parallel group schedule does not match its validated members.")
        return units

    def _initialize_reward(self, name: str, entry_config) -> RewardRuntimeEntry:
        config = dict(entry_config)
        path = config.pop("path")

        required = config.pop("required", None)
        if required is not True:
            raise ValueError(f"Reward '{name}' must set required=true for MultiModalRewardManager.")

        micro_batch_size = config.pop("micro_batch_size")
        if micro_batch_size <= 0:
            raise ValueError(f"Reward '{name}' micro_batch_size must be a positive integer.")

        # Reserved for the controller-side modality router. It remains in the
        # original OmegaConf entry but is not part of the scorer initializer.
        config.pop("routing_weights", None)

        module = load_module(path)
        initialize = getattr(module, "initialize")
        activate = getattr(module, "activate")
        score_batch = getattr(module, "score_batch")
        deactivate = getattr(module, "deactivate")
        finalize = getattr(module, "finalize", None)
        if finalize is not None and not callable(finalize):
            raise TypeError(f"Reward '{name}' finalize hook must be callable.")
        state = initialize(**config)
        return RewardRuntimeEntry(
            name=name,
            state=state,
            activate=activate,
            score_batch=score_batch,
            deactivate=deactivate,
            finalize=finalize,
            micro_batch_size=micro_batch_size,
        )

    @staticmethod
    def _validate_sample_uids(data: DataProto) -> list[str]:
        sample_uids = data.non_tensor_batch.get("sample_uid")
        if sample_uids is None:
            raise ValueError("sample_uid is required for batch reward scoring.")

        sample_uids = np.asarray(sample_uids, dtype=object)
        if sample_uids.shape != (len(data),):
            raise ValueError(f"sample_uid must have shape ({len(data)},), got {sample_uids.shape}.")
        return [str(uid) for uid in sample_uids]

    @staticmethod
    def _promote_audio_sample_rate(data: DataProto) -> None:
        """Move the rollout scalar metadata into the native batch contract."""
        if "audio_sample_rate" in data.batch:
            return
        value = data.non_tensor_batch.get("audio_sample_rate")
        if value is None:
            return
        rates = torch.tensor(np.asarray(value, dtype=object).tolist(), dtype=torch.long)
        if rates.ndim == 0:
            rates = rates.repeat(len(data))
        data.batch["audio_sample_rate"] = rates

    @staticmethod
    def _synchronize_device(device) -> None:
        synchronize = getattr(get_torch_device(), "synchronize", None)
        if synchronize is None:
            return
        try:
            synchronize(device)
        except TypeError:
            synchronize()

    @staticmethod
    def _run_sequential_entry(entry: RewardRuntimeEntry, data: DataProto, device):
        try:
            entry.activate(entry.state, device)
            return entry.score_batch(entry.state, data, micro_batch_size=entry.micro_batch_size)
        finally:
            entry.deactivate(entry.state)

    async def _run_parallel_unit(self, entries, data: DataProto, device):
        activated = []
        results = None
        failure = None
        try:
            for entry in entries:
                activated.append(entry)
                entry.activate(entry.state, device)

            gathered = await asyncio.gather(
                *(
                    asyncio.to_thread(entry.score_batch, entry.state, data, micro_batch_size=entry.micro_batch_size)
                    for entry in entries
                ),
                return_exceptions=True,
            )
            errors = [result for result in gathered if isinstance(result, BaseException)]
            if errors:
                raise errors[0]
            results = gathered
            self._synchronize_device(device)
        except BaseException as exc:
            failure = exc
        finally:
            for entry in activated:
                try:
                    entry.deactivate(entry.state)
                except BaseException as exc:
                    if failure is None:
                        failure = exc
        if failure is not None:
            raise failure
        return results

    async def run_batch(self, data: DataProto) -> dict[str, Any]:
        """Return sample-aligned reward components for one local batch."""
        if self._shutdown:
            raise RuntimeError("MultiModalRewardManager cannot score after shutdown.")
        sample_uids = self._validate_sample_uids(data)
        self._promote_audio_sample_rate(data)
        _validate_visual_response(
            data.batch["responses"], self.config, is_validate=bool(data.meta_info.get("validate", False))
        )

        score_columns = []
        mask_columns = []
        reward_extra_info = {}
        device = get_device_id()
        for entries in self._schedule_units:
            if len(entries) == 1:
                results = [self._run_sequential_entry(entries[0], data, device)]
            else:
                results = await self._run_parallel_unit(entries, data, device)
            for entry, result in zip(entries, results, strict=True):
                score_columns.append(result["scores"].to(dtype=torch.float32))
                mask_columns.append(result["valid_mask"])
                reward_extra_info[entry.name] = {
                    key: result[key] for key in ("metrics", "model_revision", "definition_version")
                }

        return {
            "rm_scores": torch.stack(score_columns, dim=1),
            "reward_valid_mask": torch.stack(mask_columns, dim=1),
            "reward_names": list(self.component_order),
            "sample_uid": sample_uids,
            "reward_extra_info": reward_extra_info,
        }

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        """Run the batch contract for one sample."""
        if len(data) != 1:
            raise ValueError(f"run_single requires batch size 1, got {len(data)}.")
        return await self.run_batch(data)

    def shutdown(self) -> None:
        """Finalize every reward entry exactly once."""
        if self._shutdown:
            return
        self._shutdown = True

        first_error = None
        for entry in self._reward_entries:
            try:
                if entry.finalize is not None:
                    entry.finalize(entry.state)
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if first_error is not None:
            raise first_error
