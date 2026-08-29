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
"""CPU contract tests for MultiModalRewardManager."""

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop.reward_manager import MultiModalRewardManager, SupportsBatchScoring

FAKE_HOOKS_PATH = "tests/reward_loop/test_multi_modal_reward_manager_on_cpu.py"


def initialize(offset=0.0, invalid_case=None, model_revision="fake-model-v1", definition_version="fake-def-v1"):
    """Create isolated fake reward state."""
    return {
        "offset": float(offset),
        "invalid_case": invalid_case,
        "model_revision": model_revision,
        "definition_version": definition_version,
        "initialize_calls": 1,
        "score_calls": 0,
    }


def activate(state, device):
    """Declare the later lifecycle hook without exercising it in S3-1."""
    del state, device


def score_batch(state, batch, micro_batch_size, **kwargs):
    """Return deterministic sample-aligned fake scores."""
    del kwargs
    state["score_calls"] += 1
    batch_size = len(batch)
    result = {
        "scores": torch.arange(batch_size, dtype=torch.float64) + state["offset"],
        "valid_mask": torch.ones(batch_size, dtype=torch.bool),
        "metrics": {"batch_size": batch_size, "micro_batch_size": micro_batch_size},
        "model_revision": state["model_revision"],
        "definition_version": state["definition_version"],
    }
    invalid_case = state["invalid_case"]
    if invalid_case == "bad_score_shape":
        result["scores"] = result["scores"].unsqueeze(-1)
    elif invalid_case == "nonfinite_score":
        result["scores"][0] = torch.nan
    elif invalid_case == "bad_mask_dtype":
        result["valid_mask"] = result["valid_mask"].float()
    elif invalid_case == "invalid_required":
        result["valid_mask"][0] = False
    return result


def deactivate(state):
    """Declare the later lifecycle hook without exercising it in S3-1."""
    del state


def finalize(state):
    """Declare the later lifecycle hook without exercising it in S3-1."""
    del state


def _reward_config(**overrides):
    entry = {
        "path": FAKE_HOOKS_PATH,
        "required": True,
        "micro_batch_size": 2,
        "offset": 0.0,
        "model_revision": "fake-model-v1",
        "definition_version": "fake-def-v1",
    }
    entry.update(overrides)
    return entry


def _make_config(reward_functions=None, component_order=None, aggregation="preserve_components"):
    if reward_functions is None:
        reward_functions = {"visual": _reward_config()}
    if component_order is None:
        component_order = list(reward_functions)

    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer")
    config.reward.aggregation = aggregation
    config.reward.component_order = component_order
    config.reward.reward_functions = OmegaConf.create(reward_functions)
    config.reward.reward_model.enable = False
    return config


def _make_batch(batch_size=3):
    return DataProto.from_dict(
        tensors={"responses": torch.randint(256, (batch_size, 3, 8, 8), dtype=torch.uint8)},
        non_tensors={"sample_uid": np.array([f"sample-{index}" for index in range(batch_size)], dtype=object)},
    )


def _build_manager(reward_functions=None, component_order=None, aggregation="preserve_components"):
    return MultiModalRewardManager(
        _make_config(reward_functions, component_order, aggregation),
        MagicMock(),
        compute_score=None,
    )


def _run(manager, data):
    return manager.loop.run_until_complete(manager.run_batch(data))


def test_run_batch_preserves_component_order_and_metadata():
    reward_functions = {
        "audio": _reward_config(offset=20.0, model_revision="audio-r1", definition_version="audio-d1"),
        "visual": _reward_config(offset=10.0, model_revision="visual-r1", definition_version="visual-d1"),
    }
    manager = _build_manager(reward_functions, component_order=["visual", "audio"])

    result = _run(manager, _make_batch())

    assert isinstance(manager, SupportsBatchScoring)
    assert result["reward_names"] == ["visual", "audio"]
    assert result["sample_uid"] == ["sample-0", "sample-1", "sample-2"]
    assert result["rm_scores"].dtype == torch.float32
    assert result["reward_valid_mask"].dtype == torch.bool
    torch.testing.assert_close(
        result["rm_scores"],
        torch.tensor([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]),
    )
    assert result["reward_valid_mask"].all()
    assert result["reward_extra_info"] == {
        "visual": {
            "metrics": {"batch_size": 3, "micro_batch_size": 2},
            "model_revision": "visual-r1",
            "definition_version": "visual-d1",
        },
        "audio": {
            "metrics": {"batch_size": 3, "micro_batch_size": 2},
            "model_revision": "audio-r1",
            "definition_version": "audio-d1",
        },
    }
    assert [entry.state["initialize_calls"] for entry in manager._reward_entries] == [1, 1]
    assert [entry.state["score_calls"] for entry in manager._reward_entries] == [1, 1]


def test_run_single_uses_same_batch_contract():
    manager = _build_manager()
    data = _make_batch(batch_size=1)

    result = manager.loop.run_until_complete(manager.run_single(data))

    assert result["rm_scores"].shape == (1, 1)
    assert result["reward_valid_mask"].shape == (1, 1)
    with pytest.raises(ValueError, match="batch size 1"):
        manager.loop.run_until_complete(manager.run_single(_make_batch(batch_size=2)))


@pytest.mark.parametrize(
    ("component_order", "aggregation", "match"),
    [
        ([], "preserve_components", "non-empty"),
        (["visual", "visual"], "preserve_components", "unique"),
        (["unknown"], "preserve_components", "match reward.reward_functions"),
        (["visual"], "weighted_sum", "preserve_components"),
    ],
)
def test_invalid_manager_config_fails_closed(component_order, aggregation, match):
    with pytest.raises((TypeError, ValueError), match=match):
        _build_manager(component_order=component_order, aggregation=aggregation)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"required": False}, "required=true"),
        ({"micro_batch_size": 0}, "positive integer"),
    ],
)
def test_invalid_reward_config_fails_closed(overrides, match):
    with pytest.raises((TypeError, ValueError), match=match):
        _build_manager({"visual": _reward_config(**overrides)})


@pytest.mark.parametrize(
    ("invalid_case", "match"),
    [
        ("bad_score_shape", "floating-point tensor with shape"),
        ("nonfinite_score", "finite values"),
        ("bad_mask_dtype", "boolean tensor with shape"),
        ("invalid_required", "Required reward"),
    ],
)
def test_invalid_hook_result_fails_closed(invalid_case, match):
    manager = _build_manager({"visual": _reward_config(invalid_case=invalid_case)})

    with pytest.raises((TypeError, ValueError), match=match):
        _run(manager, _make_batch())


def test_invalid_sample_uids_fail_closed():
    manager = _build_manager()
    data = _make_batch()
    del data.non_tensor_batch["sample_uid"]
    with pytest.raises(ValueError, match="sample_uid is required"):
        _run(manager, data)

    data = _make_batch()
    data.non_tensor_batch["sample_uid"] = np.array(["short"], dtype=object)
    with pytest.raises(ValueError, match="must have shape"):
        _run(manager, data)


def test_float_pixel_responses_are_rejected():
    manager = _build_manager()
    data = _make_batch()
    data.batch["responses"] = data.batch["responses"].float()

    with pytest.raises(ValueError, match="Expected uint8 pixel responses"):
        _run(manager, data)
