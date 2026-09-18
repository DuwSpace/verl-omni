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
"""Executor-driven component reward manager for multimodal training."""

import inspect
from typing import Any

import numpy as np
import torch
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.import_utils import load_extern_object

from verl_omni.reward_loop.reward_components import ComponentRewardOutput
from verl_omni.workers.config.reward import get_reward_model_entries, resolve_reward_model_name

from .multi import _filter_kwargs
from .visual import _validate_visual_response


class MultiModalRewardManager(RewardManagerBase):
    """Score a worker group's terms through its bound reward-model executors.

    Model construction, placement, wake and sleep are owned by
    ``MultiRewardModelManager`` and ``NativeRewardExecutor``. This manager only
    validates the multimodal batch, invokes each configured score definition,
    and returns sample-aligned component columns.
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        del reward_router_address, reward_model_tokenizer
        self._engine_reward_executors = {}
        self._native_reward_executors = {}

        if config.reward.aggregation != "preserve_components":
            raise ValueError("MultiModalRewardManager requires reward.aggregation='preserve_components'.")
        reward_functions = config.reward.reward_functions
        self.component_names = list(reward_functions)
        if not self.component_names:
            raise ValueError("MultiModalRewardManager requires non-empty reward_functions.")

        models = get_reward_model_entries(config)
        reserved = {"path", "name", "weight", "required", "model", "routing_weights"}
        self._components = []
        for component_name in self.component_names:
            entry = reward_functions[component_name]
            path, function_name = entry.get("path"), entry.get("name")
            if path is None or function_name is None:
                raise ValueError(f"Component reward {component_name!r} requires path and name.")
            if entry.get("required") is not True:
                raise ValueError(f"Component reward {component_name!r} must set required=true.")
            weight = float(entry.get("weight", 1.0))
            if weight != 1.0:
                raise ValueError(
                    f"Component reward {component_name!r} must use weight=1.0; modality routing applies weights later."
                )
            model_name = resolve_reward_model_name(component_name, entry, models)
            if model_name is None:
                raise ValueError(f"Component reward {component_name!r} must reference a named reward model.")
            fn = load_extern_object(path, function_name)
            self._components.append(
                {
                    "name": component_name,
                    "model": model_name,
                    "fn": fn,
                    "signature": inspect.signature(fn),
                    "is_async": inspect.iscoroutinefunction(fn),
                    "score_options": {key: value for key, value in entry.items() if key not in reserved},
                }
            )

    def set_reward_executors(self, engine_reward_executors, native_reward_executors) -> None:
        """Bind worker-owned executor mappings without transferring lifecycle ownership."""
        self._engine_reward_executors = engine_reward_executors or {}
        self._native_reward_executors = native_reward_executors or {}

    @staticmethod
    def _sample_uids(data: DataProto) -> list[str]:
        values = np.asarray(data.non_tensor_batch.get("sample_uid"), dtype=object)
        if values.shape != (len(data),):
            raise ValueError(f"sample_uid must have shape ({len(data)},), got {values.shape}.")
        sample_uids = [str(value) for value in values]
        if len(set(sample_uids)) != len(sample_uids):
            raise ValueError("sample_uid must be unique within a component-scoring shard.")
        return sample_uids

    @staticmethod
    def _promote_audio_sample_rate(data: DataProto) -> None:
        """Copy non-tensor audio rates to a CPU long batch field in place when absent."""
        if "audio_sample_rate" in data.batch:
            return
        value = data.non_tensor_batch.get("audio_sample_rate")
        if value is None:
            return
        rates = torch.tensor(np.asarray(value, dtype=object).tolist(), dtype=torch.long)
        if rates.ndim == 0:
            rates = rates.repeat(len(data))
        data.batch["audio_sample_rate"] = rates

    def _executor(self, model_name: str):
        executor = self._engine_reward_executors.get(model_name)
        if executor is None:
            executor = self._native_reward_executors.get(model_name)
        if executor is None:
            raise RuntimeError(f"Reward model {model_name!r} is not available in this worker.")
        return executor

    async def run_batch(self, data: DataProto) -> ComponentRewardOutput:
        """Score a shard in input row order and configured component order.

        ``data`` must include unique sample_uid rows and responses matching the
        train/validation media contract; each scorer also consumes its own audio,
        text, and timing fields. If only non-tensor audio_sample_rate is present,
        promote it in place to a CPU long batch tensor before scoring.

        Invoke components sequentially through bound executors. Require one
        finite score and true validity flag per sample, then return the
        ``ComponentRewardOutput`` protocol with CPU FP32 scores and boolean
        masks. Preserve scorer-defined calibration without weighting or
        algorithm-side normalization. No row reordering occurs in this method.

        Scorer exceptions become RuntimeError with the component name; malformed
        shapes, duplicate IDs, and invalid scores raise ValueError.
        """
        sample_uids = self._sample_uids(data)
        self._promote_audio_sample_rate(data)
        _validate_visual_response(
            data.batch["responses"], self.config, is_validate=bool(data.meta_info.get("validate", False))
        )

        columns = []
        masks = []
        for component in self._components:
            executor = self._executor(component["model"])
            kwargs = {
                "batch": data,
                **component["score_options"],
                **executor.reward_kwargs(),
            }
            kwargs = _filter_kwargs(kwargs, component["signature"])
            try:
                if component["is_async"]:
                    result = await component["fn"](**kwargs)
                else:
                    result = await self.loop.run_in_executor(
                        None, lambda fn=component["fn"], options=kwargs: fn(**options)
                    )
            except Exception as exc:
                raise RuntimeError(f"Required component reward {component['name']!r} failed: {exc}") from exc

            scores = result.get("scores")
            valid_mask = result.get("valid_mask")
            if not isinstance(scores, torch.Tensor) or scores.shape != (len(data),):
                shape = None if not isinstance(scores, torch.Tensor) else tuple(scores.shape)
                raise ValueError(f"Component {component['name']!r} scores must have shape ({len(data)},), got {shape}.")
            if not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != scores.shape:
                shape = None if not isinstance(valid_mask, torch.Tensor) else tuple(valid_mask.shape)
                raise ValueError(
                    f"Component {component['name']!r} valid_mask must have shape {tuple(scores.shape)}, got {shape}."
                )
            scores = scores.detach().to(device="cpu", dtype=torch.float32)
            valid_mask = valid_mask.detach().to(device="cpu", dtype=torch.bool)
            if not torch.isfinite(scores).all() or not valid_mask.all():
                raise ValueError(f"Required component {component['name']!r} must return finite, fully valid scores.")
            columns.append(scores)
            masks.append(valid_mask)
        return {
            "rm_scores": torch.stack(columns, dim=1),
            "reward_valid_mask": torch.stack(masks, dim=1),
            "reward_names": list(self.component_names),
            "sample_uid": np.asarray(sample_uids, dtype=object),
        }

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        """Reject the scalar API; component scoring requires run_batch."""
        raise RuntimeError("Component-preserving reward scoring uses compute_score_components(), not run_single().")
