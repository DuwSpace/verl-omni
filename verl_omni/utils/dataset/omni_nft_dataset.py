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
"""Dataset adapter for the text prompt groups used by OmniNFT."""

from __future__ import annotations

import json
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


_REQUIRED_FIELDS = ("idx", "category", "prompt_av", "prompt_v", "prompt_a")


def _load_records(path: Path, max_samples: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_samples >= 0 and len(records) >= max_samples:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            missing = [key for key in _REQUIRED_FIELDS if key not in record]
            if missing:
                raise ValueError(f"Missing fields at {path}:{line_number}: {', '.join(missing)}")
            if isinstance(record["idx"], bool) or not isinstance(record["idx"], int):
                raise ValueError(f"Field 'idx' must be an integer at {path}:{line_number}")
            for key in ("category", "prompt_av", "prompt_v", "prompt_a"):
                if not isinstance(record[key], str) or not record[key].strip():
                    raise ValueError(f"Field '{key}' must be a non-empty string at {path}:{line_number}")

            uid = str(record["idx"])
            if uid in seen_uids:
                raise ValueError(f"Duplicate prompt-group idx={uid} at {path}:{line_number}")
            seen_uids.add(uid)
            records.append(
                {
                    "uid": uid,
                    "prompt": record["prompt_av"],
                    "reward_inputs": {"text": {"video": record["prompt_v"], "audio": record["prompt_a"]}},
                    "source": {"index": record["idx"], "category": record["category"]},
                    "metadata": dict(record),
                }
            )
    return records


class OmniNFTPromptDataset(Dataset):
    """Load OmniNFT JSONL metadata as prompt-group records."""

    def __init__(
        self,
        data_files: str | PathLike[str] | Sequence[str | PathLike[str]],
        tokenizer: Any = None,
        processor: Any = None,
        config: Any = None,
        max_samples: int = -1,
    ) -> None:
        if isinstance(data_files, str | PathLike):
            paths = [Path(data_files)]
        else:
            paths = [Path(path) for path in data_files]
        if len(paths) != 1:
            raise ValueError(f"OmniNFT requires exactly one JSONL data file, got {len(paths)}")
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < -1:
            raise ValueError(f"max_samples must be -1 or a non-negative integer, got {max_samples!r}")

        self.data_file = paths[0]
        if not self.data_file.is_file():
            raise FileNotFoundError(f"OmniNFT dataset file not found: {self.data_file}")

        self._records = _load_records(self.data_file, max_samples)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._records[index]


def collate_omni_nft_prompt_groups(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate prompt-group records and add LTX chat-message `raw_prompt`.

    Object columns stay in `DataProto.non_tensor_batch`. `idx` is an int64 tensor so
    `DataProto.batch` has a batch size and can `union` rollout tensors.
    """

    if not batch:
        raise ValueError("Cannot collate an empty OmniNFT prompt batch")
    columns: dict[str, Any] = {}
    for key in batch[0]:
        columns[key] = np.empty(len(batch), dtype=object)
        columns[key][:] = [record[key] for record in batch]
    columns["raw_prompt"] = np.empty(len(batch), dtype=object)
    columns["raw_prompt"][:] = [[{"role": "user", "content": record["prompt"]}] for record in batch]
    columns["idx"] = torch.tensor([int(record["uid"]) for record in batch], dtype=torch.long)
    return columns


__all__ = ["OmniNFTPromptDataset", "collate_omni_nft_prompt_groups"]
