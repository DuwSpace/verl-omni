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

"""Ascend NPU replay test for the batch-native OmniNFT VideoAlign reward."""

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

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Ascend NPU is required",
)

_MEMORY_TOLERANCE_BYTES = 128 * 1024**2
_MODEL_REVISION = "KlingTeam/VideoReward@4f26600130683e6f1de9f5d463887f28e8ef995c"
_BASE_MODEL_REVISION = "Qwen/Qwen2-VL-2B-Instruct@895c3a49bc3fa70a340399125c650a463535e71c"


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


def _make_config(model_path, base_model_path):
    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer")
    config.reward.aggregation = "preserve_components"
    config.reward.component_order = ["videoalign"]
    config.reward.reward_functions = OmegaConf.create(
        {
            "videoalign": {
                "path": "pkg://verl_omni.utils.reward_score.videoalign_native",
                "required": True,
                "micro_batch_size": 2,
                "model_path": model_path,
                "base_model_path": base_model_path,
                "model_revision": _MODEL_REVISION,
                "base_model_revision": _BASE_MODEL_REVISION,
            }
        }
    )
    config.reward.reward_model.enable = False
    return config


class _VideoAlignReplayActor:
    def __init__(self, model_path, base_model_path):
        torch.npu.set_device(0)
        manager = MultiModalRewardManager(_make_config(model_path, base_model_path), tokenizer=None, compute_score=None)
        self.worker = object.__new__(MultiModalRewardLoopWorker)
        self.worker.reward_manager = manager

    def run(self, replay_path):
        replay = DataProto.load_from_disk(replay_path)
        if len(replay) != 8:
            raise ValueError(f"Expected the G=8 replay, got {len(replay)} samples.")
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


def test_real_videoalign_scores_complete_local_replay_batch():
    model_path = os.environ["VIDEOALIGN_MODEL_PATH"]
    base_model_path = os.environ["VIDEOALIGN_BASE_MODEL_PATH"]
    replay_path = os.environ["OMNIFT_REPLAY_PATH"]
    actor_cls = ray.remote(resources={"NPU": 1})(_VideoAlignReplayActor)
    actor = actor_cls.remote(model_path, base_model_path)
    try:
        result = ray.get(actor.run.remote(replay_path))
    finally:
        ray.kill(actor)

    batch_output = result["batch_output"]
    reference_output = result["reference_output"]
    batch_metrics = batch_output["reward_extra_info"]["videoalign"]["metrics"]
    reference_metrics = reference_output["reward_extra_info"]["videoalign"]["metrics"]
    assert batch_output["sample_uid"] == reference_output["sample_uid"] == result["expected_uids"]
    assert batch_output["rm_scores"].shape == reference_output["rm_scores"].shape == (8, 1)
    assert (batch_metrics["micro_batch_size"], batch_metrics["forward_calls"]) == (2, 4)
    assert (reference_metrics["micro_batch_size"], reference_metrics["forward_calls"]) == (1, 8)
    assert batch_metrics["frames_per_sample"] == reference_metrics["frames_per_sample"] == [8] * 8
    assert batch_metrics["target_fps"] == reference_metrics["target_fps"] == 24.0
    assert torch.isfinite(batch_output["rm_scores"]).all()
    # VideoReward is bfloat16; NPU matmul rounding varies slightly with batch shape.
    torch.testing.assert_close(batch_output["rm_scores"], reference_output["rm_scores"], rtol=2e-2, atol=1e-3)
    assert result["after_batch"]["allocated"] <= result["baseline"]["allocated"] + _MEMORY_TOLERANCE_BYTES
    _assert_released(result["after_reference"], result["after_batch"])
    _assert_released(result["after_shutdown"], result["after_batch"])
