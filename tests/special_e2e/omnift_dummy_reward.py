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
"""Tiny native reward adapter for the LTX-2.3 OmniNFT GPU smoke test."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from verl import DataProto

_DEFINITION_VERSION = "omnift-smoke-v1"


class TinyOmniRewardModel:
    """Exercise native placement with an anchor tensor and UID-derived fake scores.

    Scores depend only on channel and sample_uid, not media quality; fixed IDs
    reproduce them, while newly generated sample IDs need not do so across runs.
    """

    def __init__(self, device: torch.device | str, **_: Any) -> None:
        self._device = torch.device(device)
        self._anchor = torch.tensor(1.0, device=self._device)

    def activate(self, device: torch.device | str) -> None:
        self._device = torch.device(device)
        self._anchor = self._anchor.to(self._device)

    def offload_to_cpu(self) -> None:
        self._device = torch.device("cpu")
        self._anchor = self._anchor.cpu()

    def close(self) -> None:
        self._anchor = torch.empty(0)
        self._device = torch.device("cpu")

    def metadata(self) -> dict[str, str]:
        return {
            "model_revision": "tiny-random-local",
            "definition_version": _DEFINITION_VERSION,
        }

    def infer(self, sample_uids: list[str], channel: str) -> torch.Tensor:
        values = []
        for uid in sample_uids:
            digest = hashlib.sha256(f"{channel}:{uid}".encode()).digest()
            values.append(int.from_bytes(digest[:4], "big") / 2**32)
        return torch.tensor(values, device=self._device, dtype=torch.float32) * self._anchor


async def compute_score_batch(
    batch: DataProto,
    reward_model,
    *,
    channel: str,
) -> dict[str, Any]:
    """Return one deterministic, sample-aligned component column."""
    sample_uids = [str(value) for value in np.asarray(batch.non_tensor_batch["sample_uid"], dtype=object)]
    scores = (await reward_model.infer(sample_uids, channel)).detach().cpu()
    return {
        "scores": scores,
        "valid_mask": torch.ones_like(scores, dtype=torch.bool),
        **reward_model.metadata(),
    }
