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

from types import SimpleNamespace

import torch
from vllm_omni.diffusion.offloader.sequential_backend import ModelLevelOffloadBackend

from verl_omni.workers.rollout.vllm_rollout import npu_utils
from verl_omni.workers.rollout.vllm_rollout.npu_utils import vLLMOmniNPUColocateWorkerExtension
from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension


class _FakeData:
    def __init__(self, device: str, events: list[str]):
        self.device = torch.device(device)
        self._events = events

    def to(self, device, non_blocking=False):
        del non_blocking
        target = torch.device(device)
        self._events.append(f"move:{self.device.type}->{target.type}")
        return _FakeData(str(target), self._events)

    def pin_memory(self):
        self._events.append("pin")
        return self


class _FakeTensor:
    def __init__(self, device: str, events: list[str]):
        self.data = _FakeData(device, events)

    @property
    def device(self):
        return self.data.device


class _FakePipeline:
    def __init__(self, parameters, buffers=()):
        self._parameters = list(parameters)
        self._buffers = list(buffers)

    def parameters(self):
        return iter(self._parameters)

    def buffers(self):
        return iter(self._buffers)


class _SleepWakeBase:
    def sleep(self, level=1):
        self.events.append(f"allocator_sleep:{level}:{self.accelerator_tensor.device.type}")
        return 123

    def wake_up(self, tags=None):
        self.events.append(f"allocator_wake:{tags}:{self.accelerator_tensor.device.type}")
        return True


class _TestWorker(vLLMOmniColocateWorkerExtension, _SleepWakeBase):
    pass


class _TestNPUWorker(vLLMOmniNPUColocateWorkerExtension, _SleepWakeBase):
    pass


def _make_worker(enable_cpu_offload=True, pin_cpu_memory=False, worker_cls=_TestWorker):
    worker = object.__new__(worker_cls)
    worker.events = []
    worker.accelerator_tensor = _FakeTensor("cuda:3", worker.events)
    worker.cpu_tensor = _FakeTensor("cpu", worker.events)
    worker.model_runner = SimpleNamespace(
        pipeline=_FakePipeline(
            [worker.accelerator_tensor, worker.cpu_tensor, worker.accelerator_tensor],
        ),
        offload_backend=object.__new__(ModelLevelOffloadBackend),
    )
    worker.od_config = SimpleNamespace(
        enable_cpu_offload=enable_cpu_offload,
        pin_cpu_memory=pin_cpu_memory,
    )
    worker._sleep_saved_buffers = {}
    return worker


def test_level_one_sleep_releases_cpu_offload_tensors_and_restores_after_wake():
    worker = _make_worker()

    assert worker.sleep(level=1) == 123
    assert worker.accelerator_tensor.device.type == "cpu"
    assert worker.cpu_tensor.device.type == "cpu"
    assert worker.events == ["move:cuda->cpu", "allocator_sleep:1:cpu"]

    assert worker.wake_up(tags=["weights"]) is True
    assert worker.accelerator_tensor.device == torch.device("cuda:3")
    assert worker.cpu_tensor.device.type == "cpu"
    assert worker.events[-2:] == ["allocator_wake:['weights']:cpu", "move:cpu->cuda"]


def test_cpu_offload_sleep_waits_for_weight_wake_before_restoring_tensors():
    worker = _make_worker()

    worker.sleep(level=1)
    worker.wake_up(tags=["kv_cache"])

    assert worker.accelerator_tensor.device.type == "cpu"
    assert worker.events[-1] == "allocator_wake:['kv_cache']:cpu"

    worker.wake_up(tags=["weights"])
    assert worker.accelerator_tensor.device.type == "cuda"


def test_sleep_does_not_move_tensors_when_model_level_cpu_offload_is_disabled():
    worker = _make_worker(enable_cpu_offload=False)

    worker.sleep(level=1)

    assert worker.accelerator_tensor.device.type == "cuda"
    assert worker.events == ["allocator_sleep:1:cuda"]


def test_npu_sleep_releases_cpu_offload_tensors_and_clears_device_cache(monkeypatch):
    worker = _make_worker(worker_cls=_TestNPUWorker)

    class FakeAllocator:
        def sleep(self, offload_tags):
            worker.events.append(f"npu_allocator_sleep:{offload_tags}:{worker.accelerator_tensor.device.type}")

        def wake_up(self, tags):
            worker.events.append(f"npu_allocator_wake:{tags}:{worker.accelerator_tensor.device.type}")

    fake_npu = SimpleNamespace(
        mem_get_info=lambda: (100, 200),
        empty_cache=lambda: worker.events.append("npu_empty_cache"),
    )
    monkeypatch.setattr(npu_utils, "_is_npu_platform", lambda: True)
    monkeypatch.setattr(npu_utils, "_get_npu_memory_allocator", FakeAllocator)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    assert worker.sleep(level=1) is True
    assert worker.accelerator_tensor.device.type == "cpu"
    assert worker.events == [
        "move:cuda->cpu",
        "npu_allocator_sleep:('weights',):cpu",
        "npu_empty_cache",
    ]

    assert worker.wake_up(tags=["weights"]) is True
    assert worker.accelerator_tensor.device.type == "cuda"
    assert worker.events[-2:] == ["npu_allocator_wake:['weights']:cpu", "move:cpu->cuda"]
