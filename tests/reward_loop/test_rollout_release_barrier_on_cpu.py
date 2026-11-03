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

"""Offloaded native rewards release device storage before synchronous rollout."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager
from verl_omni.reward_loop.reward_model import NativeManagedRewardModel
from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1
from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync


def test_release_waits_for_all_native_workers_and_excludes_resident_models():
    events = []

    async def sleep():
        await asyncio.sleep(0)
        events.append("released")

    native = object.__new__(NativeManagedRewardModel)
    native.offload = True
    native.sleep = AsyncMock(side_effect=sleep)
    resident = object.__new__(NativeManagedRewardModel)
    resident.offload = False
    resident.sleep = AsyncMock()
    engine = SimpleNamespace(offload=True, sleep=AsyncMock())
    manager = object.__new__(OmniRewardLoopManager)
    manager._score_lock = asyncio.Lock()
    manager.multi_reward_model_manager = SimpleNamespace(models=dict(native=native, resident=resident, engine=engine))
    manager.prepare_for_rollout()
    assert events == ["released"]
    resident.sleep.assert_not_awaited()
    engine.sleep.assert_not_awaited()
    native.sleep.side_effect = RuntimeError("release failed")
    with pytest.raises(RuntimeError, match="release failed"):
        manager.prepare_for_rollout()


def test_release_precedes_generation_submission(monkeypatch):
    events = []
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1Sync)
    trainer.reward_loop_manager = SimpleNamespace(prepare_for_rollout=lambda: events.append("release"))
    monkeypatch.setattr(PolicyGradientDiffusionTrainerV1, "prepare_step", lambda self: events.append("submit"))
    trainer.prepare_step()
    assert events == ["release", "submit"]
    trainer.on_validate_begin()
    assert events[-1] == "release"
