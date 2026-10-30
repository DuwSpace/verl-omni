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

"""Frozen inference storage retained on CPU across component offload."""

import torch


class FrozenCpuWeights:
    @torch.inference_mode(False)
    @torch.no_grad()
    def __init__(self, module):
        self.parameters = {}
        self.buffers = {}
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                continue
            if parameter.device.type != "cpu":
                raise ValueError(f"Capture frozen CPU weights before accelerator loading: {name}")
            self.parameters[name] = (parameter, parameter.detach())
        for prefix, child in module.named_modules():
            for name, value in child.named_buffers(recurse=False):
                if name in child._non_persistent_buffers_set:
                    continue
                key = f"buffer:{prefix}.{name}"
                if value.device.type != "cpu":
                    raise ValueError(f"Capture frozen CPU buffers before accelerator loading: {key}")
                self.buffers[key] = (child, name, value.detach())

    @torch.inference_mode(False)
    @torch.no_grad()
    def restore_host(self):
        """Rebind to saved CPU storage without D2H; caller must synchronize first."""
        for parameter, cpu_tensor in self.parameters.values():
            parameter.data = cpu_tensor
        for module, attribute, host in self.buffers.values():
            setattr(module, attribute, host)
