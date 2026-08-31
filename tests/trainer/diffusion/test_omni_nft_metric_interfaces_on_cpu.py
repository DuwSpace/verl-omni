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

"""Replay OmniNFT scored batches through metric interfaces without rollout."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.trainer.diffusion.diffusion_metric_utils import compute_data_metrics_diffusion
from verl_omni.trainer.diffusion.ray_diffusion_trainer import MultiModalDirectPreferenceRayTrainer

REWARD_NAMES = ["video_align", "hpsv3", "audiobox", "clap", "desync"]
_DUMP_ENV = "OMNIFT_REWARD_DUMP"


def _synthetic_reward_batch() -> DataProto:
    scores = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
        ],
        dtype=torch.float32,
    )
    extras = np.array(
        [
            {name: {"metrics": {"batch_size": 2}, "model_revision": "r1"} for name in REWARD_NAMES},
            {name: {"metrics": {"batch_size": 2}, "model_revision": "r1"} for name in REWARD_NAMES},
        ],
        dtype=object,
    )
    return DataProto.from_dict(
        tensors={"rm_scores": scores, "sample_level_rewards": scores.clone()},
        non_tensors={
            "uid": np.array(["uid-0", "uid-1"], dtype=object),
            "sample_uid": np.array(["s0", "s1"], dtype=object),
            "reward_extra_info": extras,
        },
        meta_info={"reward_names": list(REWARD_NAMES), "reward_extra_keys": ["reward_extra_info"]},
    )


def _load_or_synthetic_batch() -> DataProto:
    path = os.environ.get(_DUMP_ENV)
    if path and Path(path).is_file():
        return DataProto.load_from_disk(path)
    return _synthetic_reward_batch()


def _trainer_with_names(names):
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    trainer._last_reward_names = list(names)
    trainer.config = OmegaConf.create({"trainer": {}})
    return trainer


def _assert_finite_floats(metrics: dict):
    assert metrics
    for key, value in metrics.items():
        assert isinstance(value, float), key
        assert np.isfinite(value), key


def test_maybe_dump_stage_is_noop_without_debug_dir(tmp_path):
    trainer = _trainer_with_names(REWARD_NAMES)
    batch = _synthetic_reward_batch()
    trainer._maybe_dump_stage("reward", batch)
    assert list(tmp_path.iterdir()) == []


def test_maybe_dump_stage_writes_reward_batch(tmp_path):
    trainer = _trainer_with_names(REWARD_NAMES)
    trainer.global_steps = 0
    trainer.config = OmegaConf.create({"trainer": {"debug_dump_dir": str(tmp_path)}})
    batch = _synthetic_reward_batch()
    batch.meta_info["validate"] = True

    trainer._maybe_dump_stage("reward", batch)

    dump_path = tmp_path / "reward_val_0.pkl"
    assert dump_path.is_file()
    restored = DataProto.load_from_disk(str(dump_path))
    torch.testing.assert_close(restored.batch["rm_scores"], batch.batch["rm_scores"])
    assert restored.meta_info["reward_names"] == REWARD_NAMES


def test_maybe_load_stage_reuses_dumped_reward_batch(tmp_path):
    trainer = _trainer_with_names(REWARD_NAMES)
    trainer.global_steps = 0
    trainer.config = OmegaConf.create(
        {"trainer": {"debug_dump_dir": str(tmp_path), "reuse_debug_dump": True}}
    )
    batch = _synthetic_reward_batch()
    batch.meta_info["validate"] = True
    trainer._maybe_dump_stage("reward", batch)

    loaded = trainer._maybe_load_stage("reward", batch)
    assert loaded is not None
    torch.testing.assert_close(loaded.batch["rm_scores"], batch.batch["rm_scores"])


def test_metric_interfaces_accept_dumped_or_synthetic_reward_batch():
    batch = _load_or_synthetic_batch()
    names = list(batch.meta_info.get("reward_names") or REWARD_NAMES)
    trainer = _trainer_with_names(names)
    published = trainer._publish_component_reward_scores(batch)

    extras = {
        "reward": published.batch["rm_scores"].sum(dim=-1).detach().cpu().tolist(),
        "reward_extra_info": published.non_tensor_batch.get("reward_extra_info"),
    }
    for name in names:
        extras[name] = published.non_tensor_batch[name].tolist()

    val_metrics = trainer._val_metrics_update(
        np.array(["src"] * len(published), dtype=object),
        [f"uid-{index}" for index in range(len(published))],
        extras,
        [],
    )
    train_metrics = trainer._reward_component_wandb_metrics(published, split="train")
    data_metrics = trainer._compute_data_metrics(published)

    _assert_finite_floats(val_metrics)
    _assert_finite_floats(train_metrics)
    _assert_finite_floats(data_metrics)
    for name in names:
        assert f"val/reward/{name}/mean" in val_metrics
        assert f"train/reward/{name}/mean" in train_metrics
    assert "val/reward/sum/mean" in val_metrics
    assert "train/reward/sum/mean" in train_metrics
    assert not any("reward_extra_info" in key for key in val_metrics)
    assert data_metrics["critic/rewards/mean"] == pytest.approx(train_metrics["train/reward/sum/mean"])
    shared_metrics = compute_data_metrics_diffusion(published)
    assert shared_metrics["critic/rewards/mean"] == pytest.approx(
        published.batch["sample_level_rewards"].mean(dim=1).mean().item()
    )
