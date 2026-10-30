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

"""CPU retention for synchronous, executor-owned native reward models."""

import torch


class NativeRewardCpuOffload:
    """Retain frozen CPU weights and copy them to the inference device on wake.

    The executor drains requests before sleep and owns cache cleanup and terminal
    cleanup on transfer failure. Subclasses retain their existing close methods.
    Sleep restores CPU parameter storage without copying weights back. Buffers
    still move normally so runtime updates survive. Transfers preserve dtypes.
    """

    supports_cpu_offload = True

    @torch.inference_mode(False)
    @torch.no_grad()
    def _init_cpu_offload(self, device):
        self.device = torch.device(device)
        self._target_device = self.device
        self._offload_asleep = False
        self._offload_closed = False
        self._cpu_weights = []
        seen = set()
        for module in self._offload_modules():
            module.requires_grad_(False)
            for parameter in module.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                if parameter.device.type != "cpu":
                    raise ValueError("Native reward weights must be loaded on CPU before offload initialization")
                self._cpu_weights.append((parameter, parameter.detach()))
        self._move_reward_model(self._target_device)

    def _offload_modules(self):
        return (self.model,)

    def _move_reward_model(self, device):
        for module in self._offload_modules():
            module.to(device)
        self.device = device

    @torch.inference_mode(False)
    @torch.no_grad()
    def sleep(self):
        with self._infer_lock:
            if self._offload_closed or self._offload_asleep:
                return
            # Executor work may run on another thread/stream. Finish outstanding
            # inference before releasing accelerator storage.
            devices = {parameter.device for parameter, _ in self._cpu_weights}
            for module in self._offload_modules():
                devices.update(buffer.device for buffer in module.buffers())
            for device in devices:
                if device.type != "cpu":
                    torch.get_device_module(device).synchronize(device)
            for parameter, host in self._cpu_weights:
                parameter.data = host
            # Parameters are already on CPU; only buffers need a D2H transfer.
            self._move_reward_model(torch.device("cpu"))
            self._offload_asleep = True

    @torch.inference_mode(False)
    @torch.no_grad()
    def wake_up(self):
        with self._infer_lock:
            if self._offload_closed:
                raise RuntimeError(f"{type(self).__name__} is closed")
            if not self._offload_asleep:
                return
            self._move_reward_model(self._target_device)
            self._offload_asleep = False
