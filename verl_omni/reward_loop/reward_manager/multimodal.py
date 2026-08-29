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

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import torch
from verl import DataProto
from verl.utils.device import get_device_id
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
        self._reward_entries = [
            self._initialize_reward(name, config.reward.reward_functions[name]) for name in self.component_order
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
    def _validate_reward_result(
        entry: RewardRuntimeEntry, result: dict[str, Any], batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        scores = result["scores"]
        if (
            not isinstance(scores, torch.Tensor)
            or not scores.dtype.is_floating_point
            or scores.shape != (batch_size,)
        ):
            raise ValueError(
                f"Reward '{entry.name}' scores must be a floating-point tensor with shape ({batch_size},)."
            )
        if not torch.isfinite(scores).all():
            raise ValueError(f"Reward '{entry.name}' scores must contain only finite values.")

        valid_mask = result["valid_mask"]
        if (
            not isinstance(valid_mask, torch.Tensor)
            or valid_mask.dtype != torch.bool
            or valid_mask.shape != (batch_size,)
        ):
            raise ValueError(f"Reward '{entry.name}' valid_mask must be a boolean tensor with shape ({batch_size},).")
        if not valid_mask.all():
            raise ValueError(f"Required reward '{entry.name}' returned invalid samples.")

        extra_info = {
            "metrics": result["metrics"],
            "model_revision": result["model_revision"],
            "definition_version": result["definition_version"],
        }
        return scores, valid_mask, extra_info

    async def run_batch(self, data: DataProto) -> dict[str, Any]:
        """Return sample-aligned reward components for one local batch."""
        if self._shutdown:
            raise RuntimeError("MultiModalRewardManager cannot score after shutdown.")
        sample_uids = self._validate_sample_uids(data)
        _validate_visual_response(
            data.batch["responses"], self.config, is_validate=bool(data.meta_info.get("validate", False))
        )

        score_columns = []
        mask_columns = []
        reward_extra_info = {}
        device = get_device_id()
        for entry in self._reward_entries:
            try:
                entry.activate(entry.state, device)
                result = entry.score_batch(entry.state, data, micro_batch_size=entry.micro_batch_size)
                scores, valid_mask, extra_info = self._validate_reward_result(entry, result, len(data))
            finally:
                entry.deactivate(entry.state)
            score_columns.append(scores.to(dtype=torch.float32))
            mask_columns.append(valid_mask)
            reward_extra_info[entry.name] = extra_info

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
