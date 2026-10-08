# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Joint trainer routing/seed/cleanup, with no Ray jobs or model weights."""

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.trainer.diffusion import bagel_joint_ray_trainer as joint
from verl_omni.trainer.diffusion.native_ray_trainer import NativeRayDiffusionTrainer
from verl_omni.trainer.main_diffusion import _get_trainer_cls


def _config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"algorithm": "unigrpo_nft", "lora_rank": 0, "lora": {}, "lora_adapter_path": None},
                "actor": {"diffusion_loss": {"loss_mode": "unigrpo_nft"}},
                "rollout": {"name": "vllm_omni", "tensor_model_parallel_size": 1, "n": 8, "seed": 42},
            },
            "algorithm": {"trainer_type": "unigrpo_nft", "sample_source": "online"},
            "trainer": {"nnodes": 1},
        }
    )


def test_joint_route_does_not_replace_accepted_standalone_route():
    config = _config()
    assert _get_trainer_cls(config) is joint.BagelJointRayTrainer
    config.algorithm.trainer_type = "unigrpo"
    assert _get_trainer_cls(config) is NativeRayDiffusionTrainer


def test_full_recipe_binds_reward_workers_to_all_eight_cards(tmp_path):
    # Inspect the real launch arguments without starting Ray or loading weights.
    python = tmp_path / "python3"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    recipe = Path(__file__).resolve().parents[3] / "examples/unigrpo_nft_trainer/bagel/run_bagel_unigrpo_nft_npu.sh"
    result = subprocess.run(
        [shutil.which("bash"), str(recipe)],
        env={"PATH": str(tmp_path), "BAGEL_MODEL_PATH": "/cached/full-bagel"},
        capture_output=True,
        text=True,
        check=True,
    )
    arguments = set(result.stdout.splitlines())
    assert {
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        "reward.num_workers=8",
        "reward.accelerator_workers.enabled=True",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.actor.fsdp_config.offload_policy=True",
        "actor_rollout_ref.rollout.bagel_joint.sync_bucket_size_mb=1024",
        "actor_rollout_ref.rollout.pipeline.height=512",
        "actor_rollout_ref.rollout.pipeline.width=512",
        "actor_rollout_ref.rollout.pipeline.num_inference_steps=26",
    } <= arguments


@pytest.mark.parametrize("cards", [4, 8])
def test_joint_smoke_exercises_training_resume_and_checkpoint_audit(tmp_path, cards):
    python = tmp_path / "python3"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [shutil.which("bash"), "tests/special_e2e/run_unigrpo_nft_bagel.sh"],
        cwd=root,
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "NUM_GPUS": str(cards),
            "BAGEL_JOINT_DEVICE": "npu",
            "BAGEL_JOINT_TRAIN_OUTPUT_DIR": str(tmp_path / "run"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    runs = result.stdout.split("-m\nverl_omni.trainer.main_diffusion\n")[1:]
    assert len(runs) == 2
    for index, run in enumerate(runs):
        arguments = dict(line.split("=", 1) for line in run.splitlines() if "=" in line)
        assert arguments["trainer.n_gpus_per_node"] == str(cards)
        assert arguments["reward.num_workers"] == str(cards)
        assert arguments["reward.accelerator_workers.enabled"] == "True"
        assert arguments["actor_rollout_ref.actor.ppo_mini_batch_size"] == str(cards // 2)
        assert arguments["trainer.total_training_steps"] == str(index + 1)
        assert arguments["trainer.resume_mode"] == ("disable" if index == 0 else "auto")
    assert f"--world-size\n{cards}\n--optimizer-updates\n4\n--policy-version\n1\n" in result.stdout
    assert "--compare-reference\n" in result.stdout


@pytest.mark.parametrize(
    "key,value",
    [
        ("actor_rollout_ref.rollout.name", "native"),
        ("actor_rollout_ref.rollout.tensor_model_parallel_size", 2),
        ("actor_rollout_ref.rollout.n", 1),
        ("actor_rollout_ref.rollout.seed", None),
        ("actor_rollout_ref.model.lora_rank", 8),
        ("actor_rollout_ref.actor.diffusion_loss.loss_mode", "unigrpo"),
        ("algorithm.sample_source", "offline"),
        ("trainer.nnodes", 2),
    ],
)
def test_joint_recipe_rejects_unsupported_or_fallback_layouts(monkeypatch, key, value):
    monkeypatch.setattr(NativeRayDiffusionTrainer, "__init__", lambda *a, **k: None)
    config = _config()
    OmegaConf.update(config, key, value)
    with pytest.raises(ValueError):
        joint.BagelJointRayTrainer(config)


def test_global_sample_seeds_are_unique_and_resume_version_is_explicit(monkeypatch):
    trainer = object.__new__(joint.BagelJointRayTrainer)
    trainer.config, trainer.global_steps = _config(), 3
    trainer.config.actor_rollout_ref.rollout.n = 2
    dispatch = Mock(
        side_effect=lambda batch: DataProto(
            non_tensor_batch={"actual_seeds": batch.non_tensor_batch["_bagel_sample_seed"]}
        )
    )
    monkeypatch.setattr(NativeRayDiffusionTrainer, "_generate_native", dispatch)
    data = DataProto(non_tensor_batch={"uid": np.array(["one", "one", "two", "two"])})
    output = trainer._generate_native(data)
    np.testing.assert_array_equal(data.non_tensor_batch["_bagel_policy_version"], [2] * 4)
    np.testing.assert_array_equal(data.non_tensor_batch["_bagel_sample_seed"], [50, 51, 52, 53])
    np.testing.assert_array_equal(output.non_tensor_batch["actual_seeds"], [50, 51, 52, 53])
    dispatch.assert_called_once()
    np.testing.assert_array_equal(dispatch.call_args.args[0].non_tensor_batch["uid"], ["one", "two", "one", "two"])


def test_full_group_is_spread_across_all_eight_replicas(monkeypatch):
    trainer = object.__new__(joint.BagelJointRayTrainer)
    trainer.config, trainer.global_steps = _config(), 1
    uids = np.repeat(np.arange(8).astype(str), 8)

    def dispatch(self, data):
        for chunk in data.chunk(8):
            assert len(set(chunk.non_tensor_batch["uid"])) == 8
        return DataProto.from_single_dict(
            {
                "responses": torch.from_numpy(data.non_tensor_batch["_bagel_sample_seed"]),
                "actual_uids": data.non_tensor_batch["uid"],
            }
        )

    monkeypatch.setattr(NativeRayDiffusionTrainer, "_generate_native", dispatch)
    data = DataProto(non_tensor_batch={"uid": uids})
    output = trainer._generate_native(data)
    np.testing.assert_array_equal(output.non_tensor_batch["actual_uids"], uids)
    torch.testing.assert_close(output.batch["responses"], torch.arange(42, 106))


@pytest.mark.parametrize("uids", [[], ["one"] * 7, ["one", "two"] * 4])
def test_mixed_or_incomplete_prompt_groups_are_rejected(monkeypatch, uids):
    trainer = object.__new__(joint.BagelJointRayTrainer)
    trainer.config, trainer.global_steps = _config(), 1
    dispatch = Mock()
    monkeypatch.setattr(NativeRayDiffusionTrainer, "_generate_native", dispatch)
    with pytest.raises(ValueError, match="prompt"):
        trainer._generate_native(DataProto(non_tensor_batch={"uid": np.array(uids)}))
    dispatch.assert_not_called()


def test_cleanup_failure_does_not_mask_training_failure(monkeypatch):
    trainer = object.__new__(joint.BagelJointRayTrainer)
    trainer.actor_rollout_wg = Mock()
    trainer.actor_rollout_wg.close_joint_rollout.return_value = ["pending"]
    monkeypatch.setattr(joint.ray, "get", Mock(side_effect=TimeoutError("cleanup timeout")))
    monkeypatch.setattr(NativeRayDiffusionTrainer, "fit", Mock(side_effect=ValueError("training failed")))
    with pytest.raises(ValueError, match="training failed"):
        trainer.fit()
    joint.ray.get.assert_called_once_with(["pending"], timeout=30)
