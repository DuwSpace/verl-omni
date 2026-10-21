# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from types import SimpleNamespace

import pytest

from verl_omni.reward_loop import reward_model_executor as executor_module
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager, OmniRewardLoopWorker
from verl_omni.reward_loop.reward_model import MultiRewardModelManager, NativeManagedRewardModel
from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor
from verl_omni.workers.config.reward import RewardModelSpec


class Model:
    fail_stage = None

    def __init__(self, **kwargs):
        self.events = []
        self.closed = False
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()

    def check_health(self, *, stage, context):
        self.events.append((stage, dict(context)))
        if stage == self.fail_stage:
            raise RuntimeError("probe evidence saved")

    def to(self, device):
        self.events.append(("move", str(device)))

    async def infer(self):
        self.events.append(("infer", {}))
        self.entered.set()
        await self.release.wait()
        return 7.0

    def close(self):
        self.closed = True


@pytest.fixture
def executor(monkeypatch):
    monkeypatch.setattr(executor_module, "_load_native_model", lambda _: Model)
    monkeypatch.setattr(executor_module, "_empty_accelerator_cache", lambda: None)
    monkeypatch.setattr(Model, "fail_stage", None)
    return NativeRewardExecutor(
        RewardModelSpec(
            name="hps",
            backend="native",
            cpu_offload=True,
            executor_config={"model": "fake:Model", "kwargs": {"device": "cpu"}},
        )
    )


@pytest.mark.asyncio
async def test_lifecycle_probes_run_at_boundaries_with_context(executor):
    train = {"step": 14, "phase": "train"}
    validation = {"step": 14, "phase": "validation"}
    await executor.wake_up(context=train)
    model = executor._model
    assert await executor.infer() == 7.0
    await executor.sleep()
    await executor.wake_up(context=validation)
    assert await executor.infer() == 7.0
    await executor.sleep()
    assert model.events == [
        ("after_load", train),
        ("infer", {}),
        ("before_sleep", train),
        ("move", "cpu"),
        ("move", "cpu"),
        ("after_wake", validation),
        ("infer", {}),
        ("before_sleep", validation),
        ("move", "cpu"),
    ]
    await executor.close()
    assert model.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["after_load", "after_wake", "before_sleep"])
async def test_failure_latches_and_cleanup_does_not_retry_probe(executor, monkeypatch, stage):
    if stage != "after_load":
        await executor.wake_up()
        if stage == "after_wake":
            await executor.sleep()
    monkeypatch.setattr(Model, "fail_stage", stage)
    with pytest.raises(RuntimeError, match="evidence saved"):
        if stage == "before_sleep":
            await executor.sleep()
        else:
            await executor.wake_up()
    model = executor._model
    with pytest.raises(RuntimeError, match="failed its health check"):
        await executor.infer()
    with pytest.raises(RuntimeError, match="failed its health check"):
        await executor.wake_up()
    failures = sum(event[0] == stage for event in model.events)
    await executor.sleep()  # finally cleanup may offload, but cannot clear the latch
    await executor.close()
    assert model.closed and sum(event[0] == stage for event in model.events) == failures


@pytest.mark.asyncio
async def test_post_batch_probe_waits_for_inflight_inference(executor):
    await executor.wake_up()
    model = executor._model
    model.release.clear()
    task = asyncio.create_task(executor.infer())
    await model.entered.wait()
    sleeper = asyncio.create_task(executor.sleep())
    await asyncio.sleep(0)
    assert not sleeper.done()
    assert all(event[0] != "before_sleep" for event in model.events)
    model.release.set()
    assert await task == 7.0
    await sleeper
    assert model.events[-2:] == [("before_sleep", {}), ("move", "cpu")]
    await executor.close()


def _manager(executor, *, offload=True):
    worker = object.__new__(OmniRewardLoopWorker)
    worker.native_reward_executors = {"hps": executor}

    class Remote:
        def __init__(self, fn):
            self.remote = fn

    remote = SimpleNamespace(
        **{
            name: Remote(getattr(worker, name))
            for name in ("wake_up_reward_model", "sleep_reward_model", "check_reward_model_health")
        }
    )
    native = NativeManagedRewardModel(
        "hps",
        {
            "backend": "native",
            "placement": {"devices": [0]},
            "offload": offload,
            "cpu_offload": True,
            "executor": {"model": "fake:Model", "kwargs": {"health_check": {"enabled": True}}},
        },
    )
    native.bind_workers([remote])
    models = object.__new__(MultiRewardModelManager)
    models.models = {"hps": native}
    manager = object.__new__(OmniRewardLoopManager)
    manager.multi_reward_model_manager = models
    manager._score_lock = asyncio.Lock()

    async def score(data):
        return await executor.infer()

    manager._compute_named_model_scores = score
    return manager


@pytest.mark.asyncio
async def test_context_crosses_manager_worker_executor_and_resident_batches_are_checked(executor):
    manager = _manager(executor, offload=False)
    context = {"step": 14, "phase": "validation"}
    data = SimpleNamespace(meta_info={"reward_diagnostics_context": context})
    assert await manager.async_compute_rm_score(data) == 7.0
    assert await manager.async_compute_rm_score(data) == 7.0
    assert executor._model.events == [
        ("after_load", context),
        ("infer", {}),
        ("after_batch", context),
        ("before_batch", context),
        ("infer", {}),
        ("after_batch", context),
    ]
    await executor.close()


@pytest.mark.asyncio
async def test_post_batch_failure_prevents_returning_rewards_to_trainer(executor, monkeypatch):
    manager = _manager(executor)
    monkeypatch.setattr(Model, "fail_stage", "before_sleep")
    with pytest.raises(RuntimeError, match="evidence saved"):
        await manager.async_compute_rm_score(SimpleNamespace(meta_info={}))
    assert any(stage == "infer" for stage, _ in executor._model.events)
    await executor.close()


@pytest.mark.asyncio
async def test_pre_batch_failure_prevents_scoring(executor, monkeypatch):
    manager = _manager(executor)
    monkeypatch.setattr(Model, "fail_stage", "after_load")
    with pytest.raises(RuntimeError, match="evidence saved"):
        await manager.async_compute_rm_score(SimpleNamespace(meta_info={}))
    assert all(stage != "infer" for stage, _ in executor._model.events)
    await executor.close()
