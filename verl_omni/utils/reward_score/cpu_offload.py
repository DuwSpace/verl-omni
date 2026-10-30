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
    """Move weights under the inference lock without reconstructing the model.

    The executor drains requests before sleep and owns cache cleanup and terminal
    cleanup on transfer failure. Subclasses retain their existing close methods.
    Transfers change device only, preserving mixed parameter/buffer dtypes.
    """

    supports_cpu_offload = True

    def _init_cpu_offload(self, device):
        self.device = torch.device(device)
        self._target_device = self.device
        self._offload_asleep = False
        self._offload_closed = False

    def _offload_modules(self):
        return (self.model,)

    def _move_reward_model(self, device):
        for module in self._offload_modules():
            module.to(device)
        self.device = device

    def sleep(self):
        with self._infer_lock:
            if self._offload_closed or self._offload_asleep:
                return
            self._move_reward_model(torch.device("cpu"))
            self._offload_asleep = True

    def wake_up(self):
        with self._infer_lock:
            if self._offload_closed:
                raise RuntimeError(f"{type(self).__name__} is closed")
            if not self._offload_asleep:
                return
            self._move_reward_model(self._target_device)
            self._offload_asleep = False
