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

"""Ascend NPU production-replay test for the batch-native OmniNFT DeSync reward."""

import asyncio
import os

import pytest
import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop.multimodal_reward_loop import MultiModalRewardLoopWorker
from verl_omni.reward_loop.reward_manager import MultiModalRewardManager

pytest.importorskip("timm")
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Ascend NPU is required",
)

_MEMORY_TOLERANCE_BYTES = 128 * 1024**2
_MODEL_REVISION = "zghhui/OmniNFT-Reward-Series@9e30061a1392d03bafdcf717e80a385ddf411b4d"
_SOURCE_REVISION = "fb9237f6e74edf0d0f2a683f4d975b79fde588fe"


def _memory_snapshot():
    torch.npu.synchronize()
    allocated = torch.npu.memory_allocated()
    free, total = torch.npu.mem_get_info()
    return {"allocated": allocated, "free": free, "total": total}


def _release_memory():
    torch.npu.synchronize()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    return _memory_snapshot()


def _make_config(model_path, source_root):
    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer")
    config.reward.aggregation = "preserve_components"
    config.reward.component_order = ["desync"]
    config.reward.reward_functions = OmegaConf.create(
        {
            "desync": {
                "path": "pkg://verl_omni.utils.reward_score.desync_native",
                "required": True,
                "micro_batch_size": 2,
                "model_path": model_path,
                "source_root": source_root,
                "model_revision": _MODEL_REVISION,
                "source_revision": _SOURCE_REVISION,
            }
        }
    )
    config.reward.reward_model.enable = False
    return config


class _DeSyncReplayActor:
    def __init__(self, model_path, source_root):
        torch.npu.set_device(0)
        manager = MultiModalRewardManager(_make_config(model_path, source_root), tokenizer=None, compute_score=None)
        self.worker = object.__new__(MultiModalRewardLoopWorker)
        self.worker.reward_manager = manager

    def run(self, replay_path):
        replay = DataProto.load_from_disk(replay_path)
        if len(replay) != 8:
            raise ValueError(f"Expected the G=8 replay, got {len(replay)} samples.")
        if replay.batch["responses"].shape[1] != 121:
            frame_count = replay.batch["responses"].shape[1]
            raise ValueError(f"Expected the production 121-frame replay, got {frame_count}.")
        if not torch.equal(replay.batch["fps"], torch.full((8,), 24.0, dtype=replay.batch["fps"].dtype)):
            raise ValueError("Expected the production replay at 24 fps.")
        expected_uids = [str(uid) for uid in replay.non_tensor_batch["sample_uid"]]
        asyncio.run(self.worker.compute_score_batch(replay))
        baseline = _release_memory()

        batch_output = asyncio.run(self.worker.compute_score_batch(replay))
        after_batch = _memory_snapshot()
        entry = self.worker.reward_manager._reward_entries[0]
        entry.micro_batch_size = 1
        reference_output = asyncio.run(self.worker.compute_score_batch(replay))
        after_reference = _memory_snapshot()
        self.worker.shutdown()
        after_shutdown = _release_memory()
        return {
            "expected_uids": expected_uids,
            "batch_output": batch_output,
            "reference_output": reference_output,
            "baseline": baseline,
            "after_batch": after_batch,
            "after_reference": after_reference,
            "after_shutdown": after_shutdown,
        }


@pytest.fixture(scope="module", autouse=True)
def _ray_runtime():
    ray.init(num_cpus=2, resources={"NPU": 1}, include_dashboard=False, log_to_driver=True)
    yield
    ray.shutdown()


def _assert_released(snapshot, baseline):
    assert snapshot["allocated"] <= baseline["allocated"] + _MEMORY_TOLERANCE_BYTES
    assert snapshot["free"] >= baseline["free"] - _MEMORY_TOLERANCE_BYTES


def test_real_desync_scores_complete_production_replay_batch():
    model_path = os.environ["DESYNC_MODEL_PATH"]
    source_root = os.environ["DESYNC_SOURCE_ROOT"]
    replay_path = os.environ["OMNIFT_REPLAY_PATH"]
    actor_cls = ray.remote(resources={"NPU": 1})(_DeSyncReplayActor)
    actor = actor_cls.remote(model_path, source_root)
    try:
        result = ray.get(actor.run.remote(replay_path))
    finally:
        ray.kill(actor)

    batch_output = result["batch_output"]
    reference_output = result["reference_output"]
    batch_metrics = batch_output["reward_extra_info"]["desync"]["metrics"]
    reference_metrics = reference_output["reward_extra_info"]["desync"]["metrics"]
    assert batch_output["sample_uid"] == reference_output["sample_uid"] == result["expected_uids"]
    assert batch_output["rm_scores"].shape == reference_output["rm_scores"].shape == (8, 1)
    assert (
        batch_metrics["micro_batch_size"],
        batch_metrics["video_forward_calls"],
        batch_metrics["audio_forward_calls"],
        batch_metrics["compare_forward_calls"],
    ) == (2, 4, 4, 8)
    assert (
        reference_metrics["micro_batch_size"],
        reference_metrics["video_forward_calls"],
        reference_metrics["audio_forward_calls"],
        reference_metrics["compare_forward_calls"],
    ) == (1, 8, 8, 16)
    assert batch_metrics["source_revision"] == reference_metrics["source_revision"] == _SOURCE_REVISION
    assert torch.isfinite(batch_output["rm_scores"]).all()
    assert ((batch_output["rm_scores"] >= 1 / 3) & (batch_output["rm_scores"] <= 1)).all()
    torch.testing.assert_close(batch_output["rm_scores"], reference_output["rm_scores"], rtol=1e-4, atol=1e-5)
    assert result["after_batch"]["allocated"] <= result["baseline"]["allocated"] + _MEMORY_TOLERANCE_BYTES
    _assert_released(result["after_reference"], result["after_batch"])
    _assert_released(result["after_shutdown"], result["after_batch"])
