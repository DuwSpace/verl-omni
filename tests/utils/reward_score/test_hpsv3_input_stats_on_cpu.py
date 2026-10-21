# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import pytest
import torch

from verl_omni.utils.reward_score.hpsv3_input_stats import HPSInputStats, tensor_statistics


def _request(recorder, value=1):
    video = torch.full((2, 3, 2, 2), value, dtype=torch.uint8)
    return recorder.begin_request(video, "prompt", uid=f"uid-{value}", data_source="test", settings={})


def _records(root, name):
    return [json.loads(line) for path in root.glob(f"{name}-*.jsonl") for line in path.read_text().splitlines()]


def test_statistics_cover_nonfinite_values_and_preserve_input():
    value = torch.tensor([float("nan"), float("inf"), -2.0, 3.0])
    stats = tensor_statistics(value)
    assert stats["finite_count"] == 2 and stats["numel"] == 4
    assert stats["min"] == -2 and stats["max"] == 3
    assert len(stats["sha256"]) == 64 and torch.isnan(value[0])


def test_cache_is_bounded_by_samples_bytes_and_batches(tmp_path):
    recorder = HPSInputStats({"max_samples": 2, "max_bytes": 48, "max_batches": 2}, tmp_path)
    recorder.begin_batch({"step": 1, "phase": "train"})
    for value in (1, 2, 3):
        record = _request(recorder, value)
        recorder.finish_request(record, result={"score": float(value)})
    path = Path(recorder.dump(reason="test"))
    evidence = torch.load(path, weights_only=True)
    assert evidence["tensor_bytes"] == 48
    assert [sample["record"]["uid"] for sample in evidence["requests"]] == ["uid-2", "uid-3"]
    recorder.finish_batch()
    recorder.begin_batch({"step": 2, "phase": "validation"})
    recorder.finish_batch()
    recorder.begin_batch({"step": 3, "phase": "train"})
    evidence = torch.load(recorder.dump(reason="test"), weights_only=True)
    assert not evidence["requests"] and evidence["tensor_bytes"] == 0
    assert recorder.dump(reason="quota exhausted") is None
    assert len(list(tmp_path.glob("input-anomaly-*"))) == 2


def test_oversize_raw_input_and_growing_forward_capture_are_evicted(tmp_path):
    recorder = HPSInputStats({"max_bytes": 32}, tmp_path)
    recorder.begin_batch({})
    record = _request(recorder)
    recorder.record_call(record, torch.ones(1, 3, 2, 2, dtype=torch.uint8), {}, frame_offset=0)
    recorder.finish_request(record, result={"score": 1.0})
    assert not _records(tmp_path, "requests")[0]["raw_input_cached"]
    oversized = recorder.begin_request(torch.ones(100), "p", uid="large", data_source="test", settings={})
    recorder.finish_request(oversized, result={"score": 1.0})
    evidence = torch.load(recorder.dump(reason="test"), weights_only=True)
    assert evidence["requests"] == [] and evidence["tensor_bytes"] == 0
    assert len(_records(tmp_path, "requests")) == 2


def test_drop_alert_compares_same_phase_only_and_does_not_fail_scoring(tmp_path):
    recorder = HPSInputStats({"min_batch_samples": 2, "score_drop_threshold": 3}, tmp_path)
    for step, phase, score in [(1, "train", 10), (1, "validation", 2), (2, "train", 9), (2, "validation", -2)]:
        recorder.begin_batch({"step": step, "phase": phase})
        for _ in range(2):
            recorder.finish_request(_request(recorder), result={"score": score})
        recorder.finish_batch()
    batches = _records(tmp_path, "batches")
    assert [batch["drop_alert"] for batch in batches] == [False, False, False, True]
    assert batches[-1]["mean_drop"] == 4
    assert batches[-1]["previous_same_phase"]["context"] == {"step": 1, "phase": "validation"}
    evidence = torch.load(batches[-1]["evidence"], weights_only=True)
    assert len(evidence["requests"]) == 4


def test_error_capture_has_input_and_strict_json_even_with_nan_logits(tmp_path):
    recorder = HPSInputStats({}, tmp_path)
    recorder.begin_batch({"step": 14, "phase": "validation"})
    record = _request(recorder)
    recorder.record_call(record, torch.zeros(1, 3, 2, 2), {"logits": torch.tensor([[float("nan"), 1]])}, frame_offset=0)
    recorder.finish_request(record, error=ValueError("nonfinite logits"))
    report = _records(tmp_path, "requests")[0]
    assert report["calls"][0]["logits"] == [["nan", 1.0]]
    evidence = torch.load(next(tmp_path.glob("input-anomaly-*/recent_requests.pt")), weights_only=True)
    assert torch.isnan(evidence["requests"][0]["calls"][0]["logits"][0, 0])
    # The health-failure directory gets the cache even if anomaly dump quota is used up.
    recorder.dump(reason="second")
    assert recorder.dump(reason="third") is None
    destination = tmp_path / "health-failure"
    recorder.dump(reason="fixed probe failed", destination=destination)
    assert (destination / "recent_requests.pt").is_file()


@pytest.mark.parametrize(
    "config",
    [{"max_bytes": -1}, {"max_samples": True}, {"max_batches": 0}, {"score_drop_threshold": float("nan")}, {"typo": 1}],
)
def test_invalid_configuration_rejected(tmp_path, config):
    with pytest.raises(ValueError):
        HPSInputStats(config, tmp_path)


def test_late_completion_keeps_original_context_without_polluting_next_batch(tmp_path):
    recorder = HPSInputStats({}, tmp_path)
    recorder.begin_batch({"step": 14, "phase": "train"})
    record = _request(recorder)
    recorder.finish_batch()
    recorder.begin_batch({"step": 14, "phase": "validation"})
    recorder.finish_request(record, result={"score": 9.0})
    recorder.finish_batch()
    request = _records(tmp_path, "requests")[0]
    assert request["late_completion"] and request["context"]["phase"] == "train"
    assert all(batch["count"] == 0 for batch in _records(tmp_path, "batches"))
