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

"""CPU contracts for OmniNFT trainer dispatch and post-reward skip."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

import verl_omni.pipelines  # noqa: F401
from verl_omni.pipelines.ltx2_omni_nft.diffusers_training_adapter import LTX23OmniNFT
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.trainer.diffusion.diffusion_trainer_utils import SleepOnlyCheckpointManager
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


def test_omni_nft_training_adapter_is_registered_and_rejects_actor_calls():
    adapter = DiffusionModelBase.get_class_by_name("LTX2Pipeline", "omni_nft")
    assert adapter is LTX23OmniNFT
    assert adapter.prepare_processor_files("/unused") is None
    with pytest.raises(NotImplementedError, match="not implemented"):
        adapter.build_scheduler(SimpleNamespace())


def test_get_trainer_cls_dispatches_omni_nft_to_multimodal_subclass():
    assert _get_trainer_cls(_config("omni_nft")) is MultiModalDirectPreferenceRayTrainer
    assert _get_trainer_cls(_config("diffusion_nft")) is DirectPreferenceRayTrainer
    assert _get_trainer_cls(_config("flow_grpo", "policy_gradient")) is PolicyGradientRayTrainer


def test_prepare_actor_batch_keeps_reward_matrix(capsys):
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    scores = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    batch = DataProto.from_dict(tensors={"rm_scores": scores}, meta_info={"reward_names": ["a", "b", "c"]})

    result = trainer._prepare_actor_batch(batch, scores)

    assert result is batch
    torch.testing.assert_close(result.batch["rm_scores"], scores)
    captured = capsys.readouterr().out
    assert "REWARD_OK" in captured
    assert "scores=[2, 3]" in captured


def test_sleep_only_checkpoint_manager_sleeps_replicas():
    class _Replica:
        def __init__(self):
            self.slept = False

        async def sleep(self):
            self.slept = True

    replica = _Replica()
    SleepOnlyCheckpointManager([replica]).sleep_replicas()
    assert replica.slept


def test_post_reward_hooks_do_not_touch_actor():
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    batch = DataProto.from_dict(tensors={"rm_scores": torch.ones(2, 5)})

    actor_output = trainer._update_actor(batch)
    assert actor_output.meta_info["metrics"] == {}
    assert trainer._compute_ref_noise_pred(batch) is None
    assert trainer._update_old_policy() == (False, 0.0, "none")


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
