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

import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest


def _load_module():
    path = Path(__file__).parents[3] / "examples/omnift_trainer/ltx2/prepare_data.py"
    spec = importlib.util.spec_from_file_location("omninft_prepare_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare_data = _load_module()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _record(index: int = 7) -> dict:
    return {
        "prompt_av": "joint prompt",
        "prompt_v": "video prompt",
        "prompt_a": "audio prompt",
        "idx": index,
        "category": "other",
    }


def test_convert_split_writes_standard_rlhf_parquet(tmp_path):
    source = tmp_path / "train.jsonl"
    _write_jsonl(source, [_record()])

    dataframe = prepare_data._convert_split(source, "train", max_samples=-1)
    output = tmp_path / "train.parquet"
    dataframe.to_parquet(output, index=False)
    row = pq.read_table(output).to_pylist()[0]

    assert row["data_source"] == "omninft_vggsound"
    assert row["prompt"] == [{"content": "joint prompt", "role": "user"}]
    assert row["negative_prompt"] == [{"content": "", "role": "user"}]
    assert row["ability"] == "text_to_audio_video"
    assert row["uid"] == "7"
    assert row["reward_inputs"] == {"text": {"audio": "audio prompt", "video": "video prompt"}}
    assert row["extra_info"]["index"] == 7
    assert row["extra_info"]["category"] == "other"
    assert row["extra_info"]["metadata"]["prompt_av"] == "joint prompt"


def test_convert_validation_synthesizes_provenance_and_honors_limit(tmp_path):
    source = tmp_path / "test.jsonl"
    prompt_only = {key: value for key, value in _record().items() if key not in {"idx", "category"}}
    _write_jsonl(source, [prompt_only, prompt_only | {"prompt_av": "second"}])

    dataframe = prepare_data._convert_split(source, "test", max_samples=1)
    row = dataframe.iloc[0].to_dict()

    assert len(dataframe) == 1
    assert row["uid"] == "0"
    assert row["extra_info"]["index"] == 0
    assert row["extra_info"]["category"] == "validation"


@pytest.mark.parametrize(
    "record, error",
    [
        (_record() | {"prompt_a": ""}, "prompt_a"),
        (_record() | {"idx": "7"}, "idx"),
        ({key: value for key, value in _record().items() if key != "category"}, "category"),
    ],
)
def test_convert_rejects_invalid_native_records(tmp_path, record, error):
    source = tmp_path / "invalid.jsonl"
    _write_jsonl(source, [record])

    with pytest.raises(ValueError, match=error):
        prepare_data._convert_split(source, "train", max_samples=-1)
