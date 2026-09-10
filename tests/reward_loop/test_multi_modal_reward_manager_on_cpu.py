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
import threading
import time
from types import SimpleNamespace
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
        "activate_calls": 0,
        "score_calls": 0,
        "deactivate_calls": 0,
        "finalize_calls": 0,
    }


def activate(state, device):
    """Activate one fake reward on the supplied runtime device."""
    state["activate_calls"] += 1
    state["device"] = device
    if state["invalid_case"] == "activate_error":
        raise RuntimeError("fake activate failure")


def score_batch(state, batch, micro_batch_size, **kwargs):
    """Return deterministic sample-aligned fake scores."""
    if "device" not in state:
        raise RuntimeError("fake reward is not active")
    state["score_calls"] += 1
    if state["invalid_case"] == "score_error":
        raise RuntimeError("fake score failure")
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
    """Deactivate one fake reward and release its device reference."""
    state["deactivate_calls"] += 1
    state.pop("device", None)
    if state["invalid_case"] == "deactivate_error":
        raise RuntimeError("fake deactivate failure")


def finalize(state):
    """Finalize fake reward state."""
    state["finalize_calls"] += 1
    if state["invalid_case"] == "finalize_error":
        raise RuntimeError("fake finalize failure")


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


def _make_config(
    reward_functions=None, component_order=None, aggregation="preserve_components", parallel_groups=None
):
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
    if parallel_groups is not None:
        OmegaConf.update(
            config,
            "reward.native",
            {"parallel_groups": parallel_groups},
            force_add=True,
        )
    return config


def _make_batch(batch_size=3):
    return DataProto.from_dict(
        tensors={"responses": torch.randint(256, (batch_size, 3, 8, 8), dtype=torch.uint8)},
        non_tensors={"sample_uid": np.array([f"sample-{index}" for index in range(batch_size)], dtype=object)},
    )


def _build_manager(
    reward_functions=None, component_order=None, aggregation="preserve_components", parallel_groups=None
):
    return MultiModalRewardManager(
        _make_config(reward_functions, component_order, aggregation, parallel_groups),
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
    assert [entry.state["activate_calls"] for entry in manager._reward_entries] == [1, 1]
    assert [entry.state["score_calls"] for entry in manager._reward_entries] == [1, 1]
    assert [entry.state["deactivate_calls"] for entry in manager._reward_entries] == [1, 1]
    assert all("device" not in entry.state for entry in manager._reward_entries)


def test_run_batch_promotes_scalar_audio_sample_rate_metadata():
    manager = _build_manager()
    data = _make_batch(batch_size=2)
    data.non_tensor_batch["audio_sample_rate"] = np.asarray([24_000, 48_000], dtype=object)
    seen = []

    original_score_batch = manager._reward_entries[0].score_batch

    def score_and_check(state, batch, micro_batch_size):
        seen.append(batch.batch["audio_sample_rate"].clone())
        return original_score_batch(state, batch, micro_batch_size)

    manager._reward_entries[0].score_batch = score_and_check
    _run(manager, data)

    torch.testing.assert_close(seen[0], torch.tensor([24_000, 48_000], dtype=torch.long))


def test_routing_weights_are_reserved_manager_metadata():
    routing_weights = {"video": 1.0, "audio": 0.0}
    manager = _build_manager({"visual": _reward_config(routing_weights=routing_weights)})

    assert manager._reward_entries[0].state["initialize_calls"] == 1
    assert OmegaConf.to_container(manager.config.reward.reward_functions.visual.routing_weights) == routing_weights


def test_run_batch_keeps_only_one_reward_active(monkeypatch):
    reward_functions = {
        "visual": _reward_config(offset=10.0),
        "audio": _reward_config(offset=20.0),
    }
    manager = _build_manager(reward_functions)
    active = set()
    events = []

    monkeypatch.setattr("verl_omni.reward_loop.reward_manager.multimodal.get_device_id", lambda: "cpu:fake")
    for entry in manager._reward_entries:
        original_score_batch = entry.score_batch

        def activate_one(state, device, name=entry.name):
            assert not active
            active.add(name)
            state["device"] = device
            events.append(("activate", name, device))

        def score_one(state, batch, micro_batch_size, name=entry.name, score_hook=original_score_batch):
            assert active == {name}
            events.append(("score", name, len(batch)))
            return score_hook(state, batch, micro_batch_size)

        def deactivate_one(state, name=entry.name):
            assert active == {name}
            active.remove(name)
            state.pop("device", None)
            events.append(("deactivate", name, None))

        entry.activate = activate_one
        entry.score_batch = score_one
        entry.deactivate = deactivate_one

    _run(manager, _make_batch())

    assert not active
    assert events == [
        ("activate", "visual", "cpu:fake"),
        ("score", "visual", 3),
        ("deactivate", "visual", None),
        ("activate", "audio", "cpu:fake"),
        ("score", "audio", 3),
        ("deactivate", "audio", None),
    ]


def test_parallel_group_scores_overlap_and_preserve_component_order(monkeypatch):
    reward_functions = {
        "visual": _reward_config(offset=10.0),
        "audio": _reward_config(offset=20.0),
        "pending": _reward_config(offset=30.0),
    }
    manager = _build_manager(
        reward_functions,
        component_order=["visual", "audio", "pending"],
        parallel_groups={"small": {"rewards": ["audio", "visual"]}},
    )
    events = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)
    sync_calls = []

    class _Accelerator:
        def synchronize(self, device):
            sync_calls.append(device)

    monkeypatch.setattr("verl_omni.reward_loop.reward_manager.multimodal.get_device_id", lambda: "cpu:fake")
    monkeypatch.setattr("verl_omni.reward_loop.reward_manager.multimodal.get_torch_device", lambda: _Accelerator())
    for entry in manager._reward_entries[:2]:
        original_activate = entry.activate
        original_score_batch = entry.score_batch
        original_deactivate = entry.deactivate

        def activate_parallel(state, device, name=entry.name, activate_hook=original_activate):
            with lock:
                events.append(("activate", name))
            activate_hook(state, device)

        def deactivate_parallel(state, name=entry.name, deactivate_hook=original_deactivate):
            with lock:
                events.append(("deactivate", name))
            deactivate_hook(state)

        def score_parallel(state, batch, micro_batch_size, name=entry.name, score_hook=original_score_batch):
            with lock:
                events.append(("score_start", name))
            barrier.wait(timeout=2)
            time.sleep(0.01)
            result = score_hook(state, batch, micro_batch_size)
            with lock:
                events.append(("score_end", name))
            return result

        entry.activate = activate_parallel
        entry.score_batch = score_parallel
        entry.deactivate = deactivate_parallel

    result = _run(manager, _make_batch())

    assert result["reward_names"] == ["visual", "audio", "pending"]
    torch.testing.assert_close(
        result["rm_scores"],
        torch.tensor([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]),
    )
    assert sync_calls == ["cpu:fake"]
    assert [event[0] for event in events] == [
        "activate",
        "activate",
        "score_start",
        "score_start",
        "score_end",
        "score_end",
        "deactivate",
        "deactivate",
    ]
    assert manager._reward_entries[0].state["activate_calls"] == 1
    assert manager._reward_entries[1].state["activate_calls"] == 1
    assert manager._reward_entries[2].state["activate_calls"] == 1


@pytest.mark.parametrize(
    ("parallel_groups", "match"),
    [
        ({"single": {"rewards": ["visual"]}}, "at least two"),
        ({"unknown": {"rewards": ["visual", "missing"]}}, "unknown rewards"),
        ({"duplicate": {"rewards": ["visual", "visual"]}}, "duplicate"),
        (
            {"first": {"rewards": ["visual", "audio"]}, "second": {"rewards": ["audio", "pending"]}},
            "multiple",
        ),
        ({"non_contiguous": {"rewards": ["visual", "pending"]}}, "contiguous"),
    ],
)
def test_invalid_parallel_group_config_fails_closed(parallel_groups, match):
    reward_functions = {
        "visual": _reward_config(),
        "audio": _reward_config(),
        "pending": _reward_config(),
    }
    with pytest.raises(ValueError, match=match):
        _build_manager(reward_functions, parallel_groups=parallel_groups)


@pytest.mark.parametrize("failing_name", ["audio", "visual"])
def test_parallel_group_failure_deactivates_all_started_members_and_stops_dispatch(failing_name):
    reward_functions = {
        "visual": _reward_config(invalid_case="score_error" if failing_name == "visual" else None),
        "audio": _reward_config(invalid_case="activate_error" if failing_name == "audio" else None),
        "pending": _reward_config(),
    }
    manager = _build_manager(
        reward_functions,
        component_order=["visual", "audio", "pending"],
        parallel_groups={"small": {"rewards": ["visual", "audio"]}},
    )

    with pytest.raises(RuntimeError):
        _run(manager, _make_batch())

    visual, audio, pending = manager._reward_entries
    assert visual.state["deactivate_calls"] == 1
    assert audio.state["deactivate_calls"] == 1
    assert pending.state["activate_calls"] == 0
    assert pending.state["score_calls"] == 0


@pytest.mark.parametrize(
    ("invalid_case", "match", "expected_score_calls"),
    [
        ("activate_error", "fake activate failure", 0),
        ("score_error", "fake score failure", 1),
        ("deactivate_error", "fake deactivate failure", 1),
    ],
)
def test_lifecycle_failure_cleans_current_reward_and_stops_dispatch(invalid_case, match, expected_score_calls):
    reward_functions = {
        "failing": _reward_config(invalid_case=invalid_case),
        "pending": _reward_config(),
    }
    manager = _build_manager(reward_functions)

    with pytest.raises(RuntimeError, match=match):
        _run(manager, _make_batch())

    failing, pending = manager._reward_entries
    assert failing.state["activate_calls"] == 1
    assert failing.state["score_calls"] == expected_score_calls
    assert failing.state["deactivate_calls"] == 1
    assert "device" not in failing.state
    assert pending.state["activate_calls"] == 0
    assert pending.state["score_calls"] == 0
    assert pending.state["deactivate_calls"] == 0


def test_run_single_uses_same_batch_contract():
    manager = _build_manager()
    data = _make_batch(batch_size=1)

    result = manager.loop.run_until_complete(manager.run_single(data))

    assert result["rm_scores"].shape == (1, 1)
    assert result["reward_valid_mask"].shape == (1, 1)
    with pytest.raises(ValueError, match="batch size 1"):
        manager.loop.run_until_complete(manager.run_single(_make_batch(batch_size=2)))


def test_shutdown_finalizes_every_reward_once_and_disables_scoring():
    manager = _build_manager({"visual": _reward_config(), "audio": _reward_config()})

    manager.shutdown()
    manager.shutdown()

    assert [entry.state["finalize_calls"] for entry in manager._reward_entries] == [1, 1]
    with pytest.raises(RuntimeError, match="after shutdown"):
        _run(manager, _make_batch())


def test_shutdown_continues_after_finalize_failure():
    manager = _build_manager(
        {
            "failing": _reward_config(invalid_case="finalize_error"),
            "pending": _reward_config(),
        }
    )

    with pytest.raises(RuntimeError, match="fake finalize failure"):
        manager.shutdown()

    assert [entry.state["finalize_calls"] for entry in manager._reward_entries] == [1, 1]
    manager.shutdown()


def test_finalize_hook_is_optional(monkeypatch):
    module = SimpleNamespace(
        initialize=initialize,
        activate=activate,
        score_batch=score_batch,
        deactivate=deactivate,
    )
    monkeypatch.setattr("verl_omni.reward_loop.reward_manager.multimodal.load_module", lambda path: module)
    manager = _build_manager()

    manager.shutdown()

    assert manager._shutdown


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


@pytest.mark.parametrize("invalid_case", ["bad_score_shape", "nonfinite_score", "bad_mask_dtype", "invalid_required"])
def test_run_batch_trusts_reward_hook_result_contract(invalid_case):
    manager = _build_manager({"visual": _reward_config(invalid_case=invalid_case)})

    result = _run(manager, _make_batch())
    assert set(result) == {"rm_scores", "reward_valid_mask", "reward_names", "sample_uid", "reward_extra_info"}


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
