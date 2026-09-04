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
"""Ascend NPU lifecycle smoke test for batch-preserving Native Rewards."""

import asyncio
import os
import time
from copy import deepcopy

import numpy as np
import pytest
import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.device import get_device_id

from verl_omni.reward_loop.multimodal_reward_loop import MultiModalRewardLoopWorker
from verl_omni.reward_loop.reward_manager import MultiModalRewardManager

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Ascend NPU is required",
)

_ALLOCATION_MIB = 64
_MEMORY_TOLERANCE_BYTES = 16 * 1024**2
_HOOK_PATH = os.path.abspath(__file__)


def _memory_snapshot() -> dict[str, int]:
    torch.npu.synchronize()
    allocated = torch.npu.memory_allocated()
    free, total = torch.npu.mem_get_info()
    return {"allocated": allocated, "free": free, "total": total}


def _release_device_memory() -> dict[str, int]:
    torch.npu.synchronize()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    return _memory_snapshot()


def initialize(name, invalid_case=None, allocation_mib=_ALLOCATION_MIB):
    """Create CPU-only state for one synthetic Native Reward."""
    return {
        "name": name,
        "invalid_case": invalid_case,
        "allocation_mib": allocation_mib,
        "events": [],
        "finalize_calls": 0,
    }


def activate(state, device):
    """Allocate a model-sized tensor on the runtime-selected NPU."""
    device_index = int(device)
    torch.npu.set_device(device_index)
    before = _release_device_memory()
    numel = state["allocation_mib"] * 1024**2
    state["model"] = torch.ones(numel, dtype=torch.uint8, device=f"npu:{device_index}")
    active = _memory_snapshot()
    state["events"].append(
        {
            "event": "activate",
            "time_ns": time.monotonic_ns(),
            "device_arg": device_index,
            "current_device": torch.npu.current_device(),
            "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
            "before": before,
            "after": active,
        }
    )


def score_batch(state, batch, micro_batch_size, **kwargs):
    """Run a real NPU operation and return a valid sample-aligned result."""
    checksum = int(state["model"].sum().item())
    state["events"].append(
        {
            "event": "score",
            "time_ns": time.monotonic_ns(),
            "checksum": checksum,
            "memory": _memory_snapshot(),
        }
    )
    if state["invalid_case"] == "score_error":
        raise RuntimeError("synthetic NPU score failure")
    batch_size = len(batch)
    return {
        "scores": torch.arange(batch_size, dtype=torch.float32),
        "valid_mask": torch.ones(batch_size, dtype=torch.bool),
        "metrics": {"micro_batch_size": micro_batch_size, "checksum": checksum},
        "model_revision": "synthetic-npu-v1",
        "definition_version": "synthetic-lifecycle-v1",
    }


def deactivate(state):
    """Drop every device reference before optionally reporting a hook failure."""
    state.pop("model", None)
    after = _release_device_memory()
    state["events"].append({"event": "deactivate", "time_ns": time.monotonic_ns(), "after": after})
    if state["invalid_case"] == "deactivate_error":
        raise RuntimeError("synthetic NPU deactivate failure")


def finalize(state):
    """Release any remaining state during Worker shutdown."""
    state.pop("model", None)
    state["finalize_calls"] += 1
    state["events"].append({"event": "finalize", "time_ns": time.monotonic_ns(), "after": _release_device_memory()})


def _make_config(invalid_case=None, parallel_groups=None):
    config_dir = os.path.abspath("verl_omni/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="diffusion_trainer")
    config.reward.aggregation = "preserve_components"
    config.reward.component_order = ["first", "second"]
    config.reward.reward_functions = OmegaConf.create(
        {
            "first": {
                "path": _HOOK_PATH,
                "required": True,
                "micro_batch_size": 2,
                "name": "first",
                "invalid_case": invalid_case,
                "allocation_mib": _ALLOCATION_MIB,
            },
            "second": {
                "path": _HOOK_PATH,
                "required": True,
                "micro_batch_size": 2,
                "name": "second",
                "allocation_mib": _ALLOCATION_MIB,
            },
        }
    )
    config.reward.reward_model.enable = False
    if parallel_groups is not None:
        OmegaConf.update(
            config,
            "reward.native",
            {"parallel_groups": parallel_groups},
            force_add=True,
        )
    return config


def _make_batch():
    return DataProto.from_dict(
        tensors={"responses": torch.zeros((2, 3, 8, 8), dtype=torch.uint8)},
        non_tensors={"sample_uid": np.asarray(["sample-0", "sample-1"], dtype=object)},
    )


class _LifecycleActor:
    def __init__(self, config):
        manager = MultiModalRewardManager(config, tokenizer=None, compute_score=None)
        self.worker = object.__new__(MultiModalRewardLoopWorker)
        self.worker.reward_manager = manager

    def run(self):
        baseline = _release_device_memory()
        error = None
        output = None
        try:
            output = asyncio.run(self.worker.compute_score_batch(_make_batch()))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return {
            "baseline": baseline,
            "error": error,
            "reward_names": None if output is None else output["reward_names"],
            "device_id": get_device_id(),
            "current_device": torch.npu.current_device(),
            "accelerator_ids": ray.get_runtime_context().get_accelerator_ids(),
            "states": self._states(),
        }

    def shutdown(self):
        self.worker.shutdown()
        return self._states()

    def _states(self):
        return [deepcopy(entry.state) for entry in self.worker.reward_manager._reward_entries]


@pytest.fixture(scope="module", autouse=True)
def _ray_runtime():
    ray.init(num_cpus=2, resources={"NPU": 1}, include_dashboard=False, log_to_driver=True)
    yield
    ray.shutdown()


def _run_scenario(invalid_case=None, parallel_groups=None):
    actor_cls = ray.remote(resources={"NPU": 1})(_LifecycleActor)
    actor = actor_cls.remote(_make_config(invalid_case, parallel_groups))
    result = ray.get(actor.run.remote())
    finalized_states = ray.get(actor.shutdown.remote())
    ray.kill(actor)
    return result, finalized_states


def _assert_released(snapshot, baseline):
    assert snapshot["allocated"] <= baseline["allocated"] + _MEMORY_TOLERANCE_BYTES
    assert snapshot["free"] >= baseline["free"] - _MEMORY_TOLERANCE_BYTES


def _assert_device_identity(result):
    assert result["device_id"] == result["current_device"] == 0
    assert result["accelerator_ids"]["NPU"] == ["0"]
    assert all(not values for name, values in result["accelerator_ids"].items() if name != "NPU")
    assert result["states"][0]["events"][0]["visible_devices"] == "0"


def test_sequential_rewards_release_npu_memory():
    result, finalized_states = _run_scenario()

    assert result["error"] is None
    assert result["reward_names"] == ["first", "second"]
    _assert_device_identity(result)
    first, second = result["states"]
    assert [event["event"] for event in first["events"]] == ["activate", "score", "deactivate"]
    assert [event["event"] for event in second["events"]] == ["activate", "score", "deactivate"]
    assert first["events"][2]["time_ns"] < second["events"][0]["time_ns"]
    for state in (first, second):
        activate_event = state["events"][0]
        allocated_delta = activate_event["after"]["allocated"] - activate_event["before"]["allocated"]
        assert allocated_delta >= (_ALLOCATION_MIB - 1) * 1024**2
        _assert_released(state["events"][2]["after"], result["baseline"])
    assert [state["finalize_calls"] for state in finalized_states] == [1, 1]
    for state in finalized_states:
        _assert_released(state["events"][-1]["after"], result["baseline"])


def test_parallel_rewards_release_npu_memory_after_group_score():
    result, finalized_states = _run_scenario(parallel_groups={"small": {"rewards": ["first", "second"]}})

    assert result["error"] is None
    assert result["reward_names"] == ["first", "second"]
    _assert_device_identity(result)
    first, second = result["states"]
    assert [event["event"] for event in first["events"]] == ["activate", "score", "deactivate"]
    assert [event["event"] for event in second["events"]] == ["activate", "score", "deactivate"]
    activation_times = [state["events"][0]["time_ns"] for state in (first, second)]
    score_times = [state["events"][1]["time_ns"] for state in (first, second)]
    deactivate_times = [state["events"][2]["time_ns"] for state in (first, second)]
    assert max(activation_times) < min(score_times)
    assert max(score_times) < min(deactivate_times)
    for state in (first, second):
        _assert_released(state["events"][2]["after"], result["baseline"])
    assert [state["finalize_calls"] for state in finalized_states] == [1, 1]
    for state in finalized_states:
        _assert_released(state["events"][-1]["after"], result["baseline"])


@pytest.mark.parametrize("invalid_case", ["score_error", "deactivate_error"])
def test_reward_failure_releases_npu_memory_and_stops_dispatch(invalid_case):
    result, finalized_states = _run_scenario(invalid_case)

    assert invalid_case.split("_")[0] in result["error"]
    _assert_device_identity(result)
    first, second = result["states"]
    assert [event["event"] for event in first["events"]] == ["activate", "score", "deactivate"]
    assert second["events"] == []
    _assert_released(first["events"][-1]["after"], result["baseline"])
    assert [state["finalize_calls"] for state in finalized_states] == [1, 1]
    for state in finalized_states:
        _assert_released(state["events"][-1]["after"], result["baseline"])
