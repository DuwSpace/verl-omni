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
"""CPU tests for the OmniNFT prompt-group dataset adapter."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from verl.protocol import DataProto

from verl_omni.utils.dataset.rl_dataset import create_rl_dataset, get_collate_fn
from verl_omni.utils.dataset.omni_nft_dataset import OmniNFTPromptDataset, collate_omni_nft_prompt_groups


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _record(idx: int = 7) -> dict:
    return {
        "prompt_v": "a video prompt",
        "prompt_a": "an audio prompt",
        "prompt_av": "a joint prompt",
        "idx": idx,
        "category": "other",
    }


def test_dataset_maps_native_schema_to_prompt_group_record(tmp_path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record()])

    dataset = OmniNFTPromptDataset(path)
    item = dataset[0]

    assert len(dataset) == 1
    assert item["uid"] == "7"
    assert item["prompt"] == "a joint prompt"
    assert item["reward_inputs"]["text"] == {"video": "a video prompt", "audio": "an audio prompt"}
    assert item["source"] == {"index": 7, "category": "other"}
    assert item["metadata"] == _record()


def test_dataset_skips_blank_lines_and_collates_without_candidate_expansion(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(_record(1)) + "\n\n" + json.dumps(_record(2)) + "\n", encoding="utf-8")

    dataset = OmniNFTPromptDataset(path)
    batch = collate_omni_nft_prompt_groups([dataset[0], dataset[1]])

    assert batch["uid"].tolist() == ["1", "2"]
    assert batch["idx"].tolist() == [1, 2]
    assert [source["index"] for source in batch["source"]] == [1, 2]
    assert batch["prompt"].tolist() == ["a joint prompt", "a joint prompt"]
    assert batch["raw_prompt"].tolist() == [
        [{"role": "user", "content": "a joint prompt"}],
        [{"role": "user", "content": "a joint prompt"}],
    ]
    assert "raw_prompt" not in dataset[0]


@pytest.mark.parametrize(
    "record, error",
    [
        (_record(1) | {"prompt_a": ""}, "prompt_a"),
        (_record(1) | {"idx": "1"}, "idx"),
        ({}, "idx"),
    ],
)
def test_dataset_rejects_invalid_records(tmp_path, record, error):
    path = tmp_path / "invalid.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises(ValueError, match=error):
        OmniNFTPromptDataset(path)


def test_dataset_rejects_duplicate_prompt_group_ids(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    _write_jsonl(path, [_record(3), _record(3)])

    with pytest.raises(ValueError, match="Duplicate prompt-group"):
        OmniNFTPromptDataset(path)


def test_dataset_supports_standard_custom_dataset_factory_and_max_samples(tmp_path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record(1), _record(2), _record(3)])
    config = OmegaConf.create(
        {
            "custom_cls": {
                "path": "pkg://verl_omni.utils.dataset.omni_nft_dataset",
                "name": "OmniNFTPromptDataset",
                "collate_fn": "collate_omni_nft_prompt_groups",
            }
        }
    )

    dataset = create_rl_dataset([str(path)], config, tokenizer=None, processor=None, max_samples=2)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=get_collate_fn(config))))
    data_proto = DataProto.from_single_dict(batch)

    assert len(dataset) == 2
    assert batch["uid"].tolist() == ["1", "2"]
    assert batch["raw_prompt"].tolist() == [
        [{"role": "user", "content": "a joint prompt"}],
        [{"role": "user", "content": "a joint prompt"}],
    ]
    assert len(data_proto) == 2
    assert data_proto.batch is not None
    assert data_proto.batch["idx"].tolist() == [1, 2]
    assert data_proto.non_tensor_batch["uid"].tolist() == ["1", "2"]
    assert data_proto.non_tensor_batch["raw_prompt"].tolist() == batch["raw_prompt"].tolist()
    assert len(OmniNFTPromptDataset(path, max_samples=0)) == 0


@pytest.mark.parametrize("data_files", [[], ["one.jsonl", "two.jsonl"]])
def test_dataset_rejects_invalid_file_count(data_files):
    with pytest.raises(ValueError, match="exactly one"):
        OmniNFTPromptDataset(data_files)


@pytest.mark.parametrize("max_samples", [-2, True, 1.5])
def test_dataset_rejects_invalid_max_samples(tmp_path, max_samples):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_record()])

    with pytest.raises(ValueError, match="max_samples"):
        OmniNFTPromptDataset(path, max_samples=max_samples)


def test_bundled_native_training_metadata_has_expected_contract():
    path = Path(__file__).parents[3] / "data/omninft/vggsound/train_metadata_20k.jsonl"
    dataset = OmniNFTPromptDataset(path)

    assert len(dataset) == 19_487
    assert len({item["uid"] for item in dataset}) == len(dataset)
    assert set(dataset[0]["metadata"]) == {"prompt_v", "prompt_a", "prompt_av", "idx", "category"}
