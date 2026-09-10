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

"""Reward normalization and modality routing for OmniNFT."""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import torch


class ModalityAdvantageRouter:
    """Route independently normalized reward components to video and audio."""

    modalities = ("video", "audio")

    @staticmethod
    def compute_reward_advantages(
        scores: torch.Tensor,
        uid: Sequence[Any],
        *,
        norm_by_std: bool,
        global_std: bool,
        epsilon: float = 1e-4,
    ) -> torch.Tensor:
        """Normalize every reward column independently within prompt groups."""
        if scores.ndim != 2:
            raise ValueError(f"OmniNFT reward scores must have shape [B, K], got {tuple(scores.shape)}.")
        if scores.shape[1] == 0:
            raise ValueError("OmniNFT reward scores must contain at least one component.")
        if not torch.isfinite(scores).all():
            raise ValueError("OmniNFT reward scores must contain only finite values.")
        if len(uid) != scores.shape[0]:
            raise ValueError(f"OmniNFT uid count {len(uid)} does not match reward batch size {scores.shape[0]}.")
        if epsilon <= 0:
            raise ValueError(f"OmniNFT advantage epsilon must be positive, got {epsilon}.")

        scores = scores.detach().float()
        groups: dict[Any, list[int]] = defaultdict(list)
        for index, group_id in enumerate(uid):
            groups[group_id].append(index)

        advantages = torch.empty_like(scores)
        batch_std = scores.std(dim=0, correction=0) if global_std and norm_by_std else None
        for indices in groups.values():
            index_tensor = torch.tensor(indices, device=scores.device)
            group_scores = scores.index_select(0, index_tensor)
            centered = group_scores - group_scores.mean(dim=0, keepdim=True)
            if norm_by_std:
                std = batch_std if batch_std is not None else group_scores.std(dim=0, correction=0)
                centered = centered / (std.unsqueeze(0) + epsilon)
            advantages.index_copy_(0, index_tensor, centered)
        return advantages

    @classmethod
    def build_routing_matrix(
        cls,
        *,
        reward_names: Sequence[str],
        component_order: Sequence[str],
        reward_functions: Mapping[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a `[K, 2]` matrix after validating reward identity and weights."""
        reward_names = list(reward_names)
        component_order = list(component_order)
        if not component_order or len(component_order) != len(set(component_order)):
            raise ValueError("reward.component_order must be non-empty and unique for OmniNFT routing.")
        if reward_names != component_order:
            raise ValueError(
                f"OmniNFT reward_names must exactly match component_order; got {reward_names} and {component_order}."
            )
        if set(reward_functions) != set(component_order):
            raise ValueError("reward.reward_functions keys must exactly match component_order for OmniNFT routing.")

        rows: list[list[float]] = []
        for name in component_order:
            entry = reward_functions[name]
            routing_weights = entry.get("routing_weights")
            if routing_weights is None:
                raise ValueError(f"Reward '{name}' must define routing_weights.video and routing_weights.audio.")
            row = []
            for modality in cls.modalities:
                value = routing_weights.get(modality)
                if not isinstance(value, Real) or not math.isfinite(float(value)):
                    raise ValueError(f"Reward '{name}' routing weight for {modality} must be finite.")
                if value < 0:
                    raise ValueError(f"Reward '{name}' routing weight for {modality} must be non-negative.")
                row.append(float(value))
            if not any(row):
                raise ValueError(f"Reward '{name}' must route to at least one modality.")
            rows.append(row)
        return torch.tensor(rows, device=device, dtype=dtype)

    @staticmethod
    def route(reward_advantages: torch.Tensor, routing_matrix: torch.Tensor) -> torch.Tensor:
        """Combine `[B, K]` reward advantages into `[B, 2]` modality advantages."""
        if reward_advantages.ndim != 2 or routing_matrix.ndim != 2:
            raise ValueError("OmniNFT routing expects rank-2 advantage and routing tensors.")
        if routing_matrix.shape != (reward_advantages.shape[1], 2):
            raise ValueError(
                f"OmniNFT routing matrix must have shape {(reward_advantages.shape[1], 2)}, "
                f"got {tuple(routing_matrix.shape)}."
            )
        return reward_advantages @ routing_matrix

    @staticmethod
    def to_probability(
        modality_advantages: torch.Tensor,
        *,
        adv_clip_max: float,
        adv_mode: str,
    ) -> torch.Tensor:
        """Map modality advantages to DiffusionNFT optimality probabilities."""
        if adv_clip_max <= 0:
            raise ValueError(f"OmniNFT adv_clip_max must be positive, got {adv_clip_max}.")
        advantages = torch.clamp(modality_advantages, -adv_clip_max, adv_clip_max)
        if adv_mode == "positive_only":
            advantages = torch.clamp(advantages, 0, adv_clip_max)
        elif adv_mode == "negative_only":
            advantages = torch.clamp(advantages, -adv_clip_max, 0)
        elif adv_mode == "one_only":
            advantages = torch.where(advantages > 0, torch.ones_like(advantages), torch.zeros_like(advantages))
        elif adv_mode == "binary":
            advantages = torch.sign(advantages)
        elif adv_mode != "continuous":
            raise ValueError(f"Unsupported OmniNFT adv_mode: {adv_mode!r}.")
        return torch.clamp(0.5 + 0.5 * advantages / adv_clip_max, 0, 1)


__all__ = ["ModalityAdvantageRouter"]
