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

from unittest.mock import MagicMock, patch

import pytest
import torch
from verl import DataProto

from verl_omni.trainer.diffusion.ray_diffusion_trainer import (
    DirectPreferenceRayTrainer,
    MultiModalDirectPreferenceRayTrainer,
)


@pytest.mark.parametrize("validate", [True, False])
def test_multimodal_reward_uses_current_split(validate):
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    trainer._is_validating = validate
    trainer._last_reward_names = None
    reward_batch = DataProto.from_dict(
        tensors={"rm_scores": torch.tensor([[1.0]])},
        meta_info={"reward_names": ["quality"], "reward_extra_keys": []},
    )
    trainer.reward_batch_coordinator = MagicMock()
    trainer.reward_batch_coordinator.compute.return_value = reward_batch
    batch = DataProto(meta_info={"timing": {"gen": 1.0}})

    assert trainer._compute_reward_colocate(batch) is reward_batch
    assert batch.meta_info == {"timing": {"gen": 1.0}, "validate": validate}


def test_multimodal_fit_releases_reward_workers():
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    manager = MagicMock()
    trainer.reward_loop_manager = manager
    trainer.reward_batch_coordinator = MagicMock()

    with patch.object(DirectPreferenceRayTrainer, "fit", return_value="done"):
        assert trainer.fit() == "done"

    manager.shutdown.assert_called_once_with()
    assert trainer.reward_loop_manager is None
    assert trainer.reward_batch_coordinator is None


def test_multimodal_init_failure_releases_reward_workers():
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    manager = MagicMock()
    trainer.reward_loop_manager = manager
    trainer.reward_batch_coordinator = MagicMock()

    with patch.object(DirectPreferenceRayTrainer, "init_workers", side_effect=RuntimeError("init failed")):
        with pytest.raises(RuntimeError, match="init failed"):
            trainer.init_workers()

    manager.shutdown.assert_called_once_with()
    assert trainer.reward_loop_manager is None
    assert trainer.reward_batch_coordinator is None


def test_multimodal_validation_context_is_reset_on_failure():
    trainer = object.__new__(MultiModalDirectPreferenceRayTrainer)
    trainer._is_validating = False

    def fail_validation():
        assert trainer._is_validating is True
        raise RuntimeError("validation failed")

    with patch.object(DirectPreferenceRayTrainer, "_validate", side_effect=fail_validation):
        with pytest.raises(RuntimeError, match="validation failed"):
            trainer._validate()

    assert trainer._is_validating is False
