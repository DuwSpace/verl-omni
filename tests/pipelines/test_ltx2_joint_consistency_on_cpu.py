# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.ltx2_unigrpo_nft.consistency_probe import compare_boundaries, probe_identity, probe_selected


@pytest.fixture
def paired(tmp_path):
    record = SimpleNamespace(original_prompt="bell", condition_token_ids=torch.tensor([1, 2]), seed=7, policy_version=3)
    identity = probe_identity(record, 500.0)
    kwargs = {
        key: torch.ones(1, 2, 3)
        for key in ("hidden_states", "audio_hidden_states", "encoder_hidden_states", "audio_encoder_hidden_states")
    }
    kwargs.update({key: torch.tensor([500.0]) for key in ("timestep", "audio_timestep", "sigma", "audio_sigma")})
    kwargs.update(
        {key: torch.ones(1, 2, dtype=torch.bool) for key in ("encoder_attention_mask", "audio_encoder_attention_mask")}
    )
    kwargs.update(num_frames=1, height=1, width=2, fps=24.0, audio_num_frames=2)
    data = {
        "connectors": {"args": (torch.ones(1, 2, 6), torch.ones(1, 2, dtype=torch.bool))},
        "transformer": {"kwargs": kwargs, "output": (torch.ones(1, 2, 3), torch.ones(1, 2, 3))},
    }
    paths = []
    for side in ("actor", "generate"):
        directory = tmp_path / side
        directory.mkdir()
        (directory / "correlation.json").write_text(json.dumps({**identity, "side": side, "complete": True}))
        torch.save(data, directory / "boundaries.pt")
        paths.append(directory)
    return paths


def test_matching_inputs_report_prediction_difference_without_accepting_it(paired):
    path = paired[1] / "boundaries.pt"
    data = torch.load(path, weights_only=True)
    data["transformer"]["output"][0].add_(0.5)
    torch.save(data, path)
    result = compare_boundaries(*paired)
    assert result["inputs_exact"] is True
    assert result["boundaries"]["gemma_stacked_embeddings"]["exact"] is True
    assert result["boundaries"]["video_prediction"]["abs_max"] == pytest.approx(0.5)
    assert result["acceptance"].startswith("Requires")


@pytest.mark.parametrize(
    "field,value", [("policy_version", 5), ("seed", 8), ("complete", False), ("branch", "current_cond")]
)
def test_mismatched_identity_is_not_numerical_evidence(paired, field, value):
    path = paired[1] / "correlation.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        compare_boundaries(*paired)


def test_same_metadata_with_different_noisy_latents_is_rejected(paired):
    path = paired[1] / "boundaries.pt"
    data = torch.load(path, weights_only=True)
    data["transformer"]["kwargs"]["hidden_states"].add_(1)
    torch.save(data, path)
    with pytest.raises(ValueError, match="Different forward inputs: hidden_states"):
        compare_boundaries(*paired)


def test_probe_window_excludes_evaluation_and_disabled_mode(monkeypatch):
    monkeypatch.setenv("DUMP_ON", "0")
    assert probe_selected(None) is False
    monkeypatch.setenv("DUMP_ON", "1")
    monkeypatch.setenv("MSPROBE_DUMP_GLOBAL_STEP", "2")
    assert probe_selected(3) is True
    assert probe_selected(4) is False
    with pytest.raises(ValueError):
        probe_selected([3])
