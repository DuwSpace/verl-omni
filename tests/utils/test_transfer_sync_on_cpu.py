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
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from verl_omni.utils.transfer_sync import synchronized_device_transfer


@pytest.fixture
def accelerator(monkeypatch):
    events = []
    local = threading.local()

    @contextmanager
    def device(value):
        previous = getattr(local, "device", None)
        local.device = value
        events.append(("enter", str(value)))
        try:
            yield
        finally:
            events.append(("exit", str(value)))
            local.device = previous

    def synchronize(value):
        assert local.device == value
        events.append(("sync", str(value)))

    api = SimpleNamespace(device=device, synchronize=synchronize, current_device=lambda: 4)
    monkeypatch.setattr(torch, "get_device_module", lambda _: api)
    return events, local


@pytest.mark.parametrize("raises", [False, True])
def test_transfer_fences_surround_operation_and_restore_context(accelerator, raises):
    events, local = accelerator

    def run():
        with synchronized_device_transfer("cpu", "cuda:4", "cuda:4"):
            assert local.device == torch.device("cuda:4")
            events.append(("copy", "cpu"))
            if raises:
                raise RuntimeError("copy failed")

    if raises:
        with pytest.raises(RuntimeError, match="copy failed"):
            run()
    else:
        run()
    assert events == [
        ("enter", "cuda:4"),
        ("sync", "cuda:4"),
        ("copy", "cpu"),
        ("sync", "cuda:4"),
        ("exit", "cuda:4"),
    ]
    assert local.device is None


def test_cpu_transfer_does_not_initialize_accelerator(monkeypatch):
    def unexpected(_):
        pytest.fail("CPU transfer must not initialize an accelerator")

    monkeypatch.setattr(torch, "get_device_module", unexpected)
    with synchronized_device_transfer("cpu"):
        pass


def test_joint_offload_fences_unregistered_sidecars_before_replacement(accelerator):
    from verl_omni.pipelines.ltx2_unigrpo_nft.offload import JointRolloutOffload

    events, _ = accelerator

    class Module(torch.nn.Module):
        # Even if registered weights are already on CPU, the native sidecar
        # source device must be synchronized before its storage is replaced.
        lora_a_stacked = (SimpleNamespace(device=torch.device("cuda:4")),)
        lora_b_stacked = ()

        def parameters(self):
            return iter([torch.ones(1)])

        def buffers(self):
            return iter([])

        def modules(self):
            return iter([self])

        def _apply(self, fn, recurse=True):
            events.append(("copy", "cpu"))
            return super()._apply(fn, recurse=recurse)

    JointRolloutOffload._move(Module(), torch.device("cpu"))
    assert events == [
        ("enter", "cuda:4"),
        ("sync", "cuda:4"),
        ("copy", "cpu"),
        ("sync", "cuda:4"),
        ("exit", "cuda:4"),
    ]


@pytest.mark.asyncio
async def test_reward_transfer_selects_own_device_inside_pool_thread(accelerator):
    from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor
    from verl_omni.workers.config.reward import RewardModelSpec

    events, local = accelerator
    parent_thread = threading.get_ident()

    def move(destination):
        assert threading.get_ident() != parent_thread
        assert local.device == torch.device("cuda:4")
        events.append(("copy", str(destination)))

    executor = NativeRewardExecutor(RewardModelSpec(name="hps", backend="native", executor_config={}))
    executor._device = torch.device("cuda:4")
    executor._model = SimpleNamespace(to=move)
    await asyncio.to_thread(executor._move_model, torch.device("cpu"))
    assert events == [
        ("enter", "cuda:4"),
        ("sync", "cuda:4"),
        ("copy", "cpu"),
        ("sync", "cuda:4"),
        ("exit", "cuda:4"),
    ]
