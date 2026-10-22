# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rollout residency using blocking Module.to transfers, without CPU masters.

Only the currently parked model storage is retained. Releasing a component
copies its current accelerator weights back to CPU; activation copies them H2D.
Actor/FSDP parameters are outside this controller.
"""

from functools import wraps

import torch

from verl_omni.utils.transfer_sync import synchronized_device_transfer


class JointRolloutOffload:
    """Keep one component on the accelerator, including non-forward entry points.

    This controller is for the single-request, unsharded rollout worker, not
    FSDP training. Transfers are blocking: host tensors may be reused as soon
    as a transition returns. Pinned copies of the entire FP32 Gemma are avoided.
    """

    component_names = ("text_encoder", "connectors", "transformer", "vae", "audio_vae", "vocoder")

    def __init__(self, pipeline, device):
        self.device = torch.device(device)
        self.components = {name: getattr(pipeline, name) for name in self.component_names}
        self.active = None
        self.policy_version = -1
        self.policy_update_in_progress = False
        self.pending_policy_version = None
        self.handles = []
        self.methods = []
        self.release()
        for name, module in self.components.items():
            self.handles.append(module.register_forward_pre_hook(self._pre_forward(name)))
        self._wrap("text_encoder", "generate")
        self._wrap("vae", "decode")
        self._wrap("audio_vae", "decode")

    def activate(self, name):
        """Release other components, then load the requested one."""
        if self.policy_update_in_progress:
            raise RuntimeError("Cannot activate rollout components during an incomplete policy update")
        if self.active == name:
            return
        self.active = None
        for other, module in self.components.items():
            if other != name:
                self._move(module, torch.device("cpu"))
        self._flush()
        self._move(self.components[name], self.device)
        self.active = name

    def release(self):
        """Move every component to CPU; keep activation hooks alive."""
        self.active = None
        for module in self.components.values():
            self._move(module, torch.device("cpu"))
        self._flush()

    def begin_policy_update(self, version):
        if self.policy_update_in_progress or version <= self.policy_version:
            raise ValueError("Rollout adapter policy update must be complete and versions must increase")
        self.policy_update_in_progress = True
        self.pending_policy_version = version
        self.release()

    def commit_policy_update(self, version):
        if not self.policy_update_in_progress or version != self.pending_policy_version:
            raise ValueError("No matching rollout adapter policy update")
        self.policy_version = version
        self.pending_policy_version = None
        self.policy_update_in_progress = False

    @staticmethod
    @torch.inference_mode(False)
    @torch.no_grad()
    def _move(module, device):
        """Synchronize and move current weights, including native LoRA slots."""
        device = torch.device(device)
        with synchronized_device_transfer(*component_devices(module), device):
            move_native_lora_sidecars(module, device)
            module.to(device, non_blocking=False)

    def disable(self):
        """Move components to CPU and remove automatic activation hooks."""
        self.release()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for module, method, previous in reversed(self.methods):
            if previous is None:
                delattr(module, method)
            else:
                setattr(module, method, previous)
        self.methods.clear()

    def _flush(self):
        if self.device.type != "cpu":
            api = torch.get_device_module(self.device)
            with api.device(self.device):
                api.synchronize(self.device)
                api.empty_cache()

    def _pre_forward(self, name):
        def activate(module, args):
            self.activate(name)

        return activate

    def _wrap(self, name, method):
        module = self.components[name]
        original = getattr(module, method)
        previous = module.__dict__.get(method)

        @wraps(original)
        def resident_call(*args, **kwargs):
            self.activate(name)
            return original(*args, **kwargs)

        self.methods.append((module, method, previous))
        setattr(module, method, resident_call)


def component_devices(module):
    """Include native LoRA sidecars, which Module.parameters/buffers do not see."""
    devices = {value.device for value in (*module.parameters(), *module.buffers())}
    for child in module.modules():
        for name in ("lora_a_stacked", "lora_b_stacked"):
            devices.update(value.device for value in getattr(child, name, ()) or ())
    return sorted(devices, key=str)


def move_native_lora_sidecars(module, device):
    """Set residency before Module.to and keep slots writable by the next policy RPC.

    Call under inference_mode(False) and the component's transfer fence.
    The native setter moves the tuples using their current tensor values.
    """
    for child in module.modules():
        setter = getattr(child, "_set_diffusion_lora_buffer_device", None)
        if setter is None:
            continue
        for name in ("lora_a_stacked", "lora_b_stacked"):
            values = getattr(child, name, None)
            if values is not None:
                mutable_values = tuple(value.clone() if torch.is_inference(value) else value for value in values)
                setattr(child, name, mutable_values)
        setter(device)
