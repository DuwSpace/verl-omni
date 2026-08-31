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

"""CPU contracts for OmniNFT trainer dispatch and actor hook wiring."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

import verl_omni.pipelines  # noqa: F401
from verl_omni.trainer.diffusion.ray_diffusion_trainer import (
    DirectPreferenceRayTrainer,
    MultiModalDirectPreferenceRayTrainer,
    PolicyGradientRayTrainer,
)
from verl_omni.trainer.main_diffusion import _get_trainer_cls

REPO_ROOT = Path(__file__).parents[3]
DEBUG_RECIPE = REPO_ROOT / "examples/omninft_trainer/ltx2/debug/run_ltx2_3_omninft_reward_only_npu.sh"


def _config(algorithm, trainer_type="direct_preference"):
    return OmegaConf.create(
        {
            "algorithm": {"trainer_type": trainer_type},
            "actor_rollout_ref": {"model": {"algorithm": algorithm}},
        }
    )


def test_debug_recipe_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(DEBUG_RECIPE)], check=True)


def test_get_trainer_cls_dispatches_omni_nft_to_multimodal_subclass():
    assert _get_trainer_cls(_config("omni_nft")) is MultiModalDirectPreferenceRayTrainer
    assert _get_trainer_cls(_config("diffusion_nft")) is DirectPreferenceRayTrainer
    assert _get_trainer_cls(_config("flow_grpo", "policy_gradient")) is PolicyGradientRayTrainer


def test_multimodal_init_reuses_direct_preference_setup(monkeypatch):
    config = object()
    expected_args = ("tokenizer",)
    expected_kwargs = {"resource_pool_manager": "pool"}

    def parent_init(self, received_config, *args, **kwargs):
        assert received_config is config
        assert args == expected_args
        assert kwargs == expected_kwargs
        self._loss_fn = "omni-loss"
        self._has_old_adapter = True

    monkeypatch.setattr(DirectPreferenceRayTrainer, "__init__", parent_init)

    trainer = MultiModalDirectPreferenceRayTrainer(config, *expected_args, **expected_kwargs)

    assert trainer._loss_fn == "omni-loss"
    assert trainer._has_old_adapter is True
    assert trainer.use_rm is True
    assert trainer.reward_batch_coordinator is None


def test_multimodal_colocated_workers_reuse_initialized_actor_path(monkeypatch):
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    expected_pool = object()
    parent_init_workers = MagicMock(return_value=expected_pool)
    monkeypatch.setattr(DirectPreferenceRayTrainer, "_init_colocated_workers", parent_init_workers)

    assert trainer._init_colocated_workers() is expected_pool
    parent_init_workers.assert_called_once_with()


def test_prepare_actor_batch_routes_reward_matrix():
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    trainer.config = object()
    prepared = object()
    trainer._loss_fn = SimpleNamespace(prepare_actor_batch=MagicMock(return_value=prepared))
    scores = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    batch = DataProto.from_dict(tensors={"rm_scores": scores}, meta_info={"reward_names": ["a", "b", "c"]})
    extracted_scores = scores.clone()

    result = trainer._prepare_actor_batch(batch, extracted_scores)

    assert result is prepared
    trainer._loss_fn.prepare_actor_batch.assert_called_once_with(batch, extracted_scores, trainer.config)


def test_post_reward_hooks_delegate_to_direct_preference(monkeypatch):
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    batch = DataProto.from_dict(tensors={"rm_scores": torch.ones(2, 5)})
    actor_output = object()
    ref_output = object()
    parent_update_actor = MagicMock(return_value=actor_output)
    parent_compute_ref = MagicMock(return_value=ref_output)
    parent_update_old = MagicMock(return_value=(True, 0.5, "ema"))
    monkeypatch.setattr(DirectPreferenceRayTrainer, "_update_actor", parent_update_actor)
    monkeypatch.setattr(DirectPreferenceRayTrainer, "_compute_ref_noise_pred", parent_compute_ref)
    monkeypatch.setattr(DirectPreferenceRayTrainer, "_update_old_policy", parent_update_old)

    assert trainer._update_actor(batch) is actor_output
    assert trainer._compute_ref_noise_pred(batch) is ref_output
    assert trainer._update_old_policy() == (True, 0.5, "ema")
    parent_update_actor.assert_called_once_with(batch)
    parent_compute_ref.assert_called_once_with(batch)
    parent_update_old.assert_called_once_with()


def test_reward_colocate_uses_batch_coordinator():
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    expected = DataProto.from_dict(tensors={"rm_scores": torch.ones(2, 5)})
    trainer.reward_batch_coordinator = SimpleNamespace(compute=MagicMock(return_value=expected))
    trainer._maybe_wait = MagicMock()
    batch = DataProto.from_dict(tensors={"responses": torch.zeros(2, 1)})

    result = trainer._compute_reward_colocate(batch)

    assert result is expected
    trainer.reward_batch_coordinator.compute.assert_called_once_with(batch)


@pytest.mark.parametrize("fail", [False, True])
def test_fit_finalizes_multimodal_reward_workers(monkeypatch, fail):
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    manager = SimpleNamespace(shutdown=MagicMock())
    trainer.reward_loop_manager = manager

    def parent_fit(self):
        if fail:
            raise RuntimeError("fit failed")
        return "done"

    monkeypatch.setattr(DirectPreferenceRayTrainer, "fit", parent_fit)

    if fail:
        with pytest.raises(RuntimeError, match="fit failed"):
            trainer.fit()
    else:
        assert trainer.fit() == "done"

    manager.shutdown.assert_called_once_with()


def test_diffusion_trainer_shutdowns_dataloader_workers():
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)

    class Iterator:
        def __init__(self):
            self.shutdown_calls = 0

        def _shutdown_workers(self):
            self.shutdown_calls += 1

    class Loader:
        def __init__(self, iterator):
            self._iterator = iterator

    train_iterator = Iterator()
    val_iterator = Iterator()
    train_loader = Loader(train_iterator)
    val_loader = Loader(val_iterator)
    trainer.train_dataloader = train_loader
    trainer.val_dataloader = val_loader

    trainer._shutdown_dataloaders()

    assert train_loader._iterator is None
    assert val_loader._iterator is None
    assert train_iterator.shutdown_calls == 1
    assert val_iterator.shutdown_calls == 1


def test_init_failure_finalizes_created_multimodal_reward_workers():
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    trainer.is_offline = False
    manager = SimpleNamespace(shutdown=MagicMock())
    trainer.reward_loop_manager = None
    trainer._init_colocated_workers = MagicMock(return_value="actor-pool")

    def fail_after_manager_creation(resource_pool):
        assert resource_pool == "actor-pool"
        trainer.reward_loop_manager = manager
        raise RuntimeError("rollout init failed")

    trainer._init_online_rollout_stack = fail_after_manager_creation

    with pytest.raises(RuntimeError, match="rollout init failed"):
        trainer.init_workers()

    manager.shutdown.assert_called_once_with()


def test_maybe_wait_is_off_by_default(monkeypatch):
    monkeypatch.delenv("OMNIFT_WAIT_BEFORE_GENERATE", raising=False)
    MultiModalDirectPreferenceRayTrainer._maybe_wait("generate")


def test_reward_colocate_publishes_component_score_columns():
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    trainer._maybe_wait = MagicMock()
    scored = DataProto.from_dict(
        tensors={"rm_scores": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        non_tensors={"reward_extra_info": np.array([{"a": {"metrics": {}}}, {"a": {"metrics": {}}}], dtype=object)},
        meta_info={"reward_names": ["video_align", "hpsv3"], "reward_extra_keys": ["reward_extra_info"]},
    )
    trainer.reward_batch_coordinator = SimpleNamespace(compute=MagicMock(return_value=scored))

    result = trainer._compute_reward_colocate(DataProto.from_dict(tensors={"responses": torch.zeros(2, 1)}))

    np.testing.assert_allclose(result.non_tensor_batch["video_align"], np.array([1.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(result.non_tensor_batch["hpsv3"], np.array([2.0, 4.0], dtype=np.float32))
    assert result.meta_info["reward_extra_keys"] == ["reward_extra_info", "video_align", "hpsv3"]
    assert trainer._last_reward_names == ["video_align", "hpsv3"]


def test_val_metrics_update_means_each_reward_and_skips_metadata_dicts():
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    trainer._last_reward_names = ["video_align", "hpsv3"]
    extras = {
        "reward": [3.0, 7.0],
        "video_align": [1.0, 3.0],
        "hpsv3": [2.0, 4.0],
        "reward_extra_info": [
            {"video_align": {"metrics": {"batch_size": 1}, "model_revision": "r1"}},
            {"video_align": {"metrics": {"batch_size": 1}, "model_revision": "r1"}},
        ],
    }

    metrics = MultiModalDirectPreferenceRayTrainer._val_metrics_update(
        trainer,
        np.array(["src", "src"], dtype=object),
        ["uid-0", "uid-1"],
        extras,
        [],
    )

    assert metrics["val/reward/video_align/mean"] == pytest.approx(2.0)
    assert metrics["val/reward/hpsv3/mean"] == pytest.approx(3.0)
    assert metrics["val/reward/sum/mean"] == pytest.approx(5.0)
    assert metrics["val/reward/video_align/min"] == pytest.approx(1.0)
    assert metrics["val/reward/hpsv3/max"] == pytest.approx(4.0)
    assert any("/reward/" in key and "mean@" in key for key in metrics)
    assert not any("reward_extra_info" in key for key in metrics)
    assert all(isinstance(value, float) for value in metrics.values())


def test_update_actor_attaches_train_reward_wandb_metrics(monkeypatch):
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    trainer._last_reward_names = ["video_align", "hpsv3"]
    batch = DataProto.from_dict(
        tensors={"rm_scores": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        meta_info={"reward_names": ["video_align", "hpsv3"]},
    )
    actor_output = DataProto.from_dict(tensors={"dummy": torch.zeros(2)}, meta_info={"metrics": {"actor/loss": [0.5]}})

    monkeypatch.setattr(DirectPreferenceRayTrainer, "_update_actor", lambda self, received: actor_output)

    result = trainer._update_actor(batch)

    assert result is actor_output
    assert result.meta_info["metrics"]["actor/loss"] == [0.5]
    assert result.meta_info["metrics"]["train/reward/video_align/mean"][0] == pytest.approx(2.0)
    assert result.meta_info["metrics"]["train/reward/hpsv3/mean"][0] == pytest.approx(3.0)
    assert result.meta_info["metrics"]["train/reward/sum/mean"][0] == pytest.approx(5.0)
