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

"""Rollout residency: keep CPU weights, load one component onto the device.

Load:    CPU master -> device copy -> run.
Release: wait for compute -> reuse CPU storage -> discard device copy.

The CPU master stays alive throughout. Release does not copy managed weights
back from the device. Actor/FSDP parameters are outside this controller.
"""

from functools import wraps

import torch

from .transfer_sync import synchronized_device_transfer


class JointRolloutOffload:
    """Keep one component on the accelerator, including non-forward entry points.

    This controller is for the single-request, unsharded rollout worker, not
    FSDP training. Transfers are blocking: host tensors may be reused as soon
    as a transition returns. Pinned copies of the entire Gemma are avoided.
    """

    component_names = ("text_encoder", "connectors", "transformer", "vae", "audio_vae", "vocoder")

    def __init__(self, pipeline, device):
        self.device = torch.device(device)
        self.components = {name: getattr(pipeline, name) for name in self.component_names}
        self.active = None
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
        """Restore every component's CPU storage; keep masters and hooks alive."""
        self.active = None
        for module in self.components.values():
            self._move(module, torch.device("cpu"))
        self._flush()

    @staticmethod
    @torch.inference_mode(False)
    @torch.no_grad()
    def _move(module, device):
        """Restore CPU storage on release; copy weights on device load."""
        device = torch.device(device)
        frozen_weights = getattr(module, "_joint_cpu_weights", None)
        lora_weights = getattr(module, "_joint_adapter_cpu_weights", None)
        # Wait before replacing storage, and finish transfers before returning.
        # Keep new tensors mutable so the next policy update can write LoRA slots.
        with synchronized_device_transfer(*component_devices(module), device):
            if device.type == "cpu":
                # Rebind parameters to existing CPU storage. No weight D2H.
                if frozen_weights is not None:
                    frozen_weights.restore_host()
                if lora_weights is not None:
                    lora_weights.restore_host()
                # Managed weights are already on CPU. Move only remaining state
                # (e.g. dynamic caches), including unregistered native LoRA slots.
                move_native_lora_sidecars(module, device)
                module.to(device, non_blocking=False)
                return

            move_native_lora_sidecars(module, device)
            module.to(device, non_blocking=False)

    def disable(self):
        """Release device copies and remove automatic activation hooks."""
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


class AdapterCpuWeights:
    """Capture after a complete policy load on CPU, never from an actor/FSDP.

    Native sidecars are tuples outside Module buffers. Keep their CPU storage
    and restore the tuple before the native setter runs, avoiding D2H on sleep.
    The next policy load may mutate/resize slots; replace this snapshot only
    after both Gemma and DiT have finished loading that policy.
    """

    @torch.inference_mode(False)
    @torch.no_grad()
    def __init__(self, module, version):
        self.version = version
        self.parameters = {}
        self.sidecars = {}
        for name, parameter in module.named_parameters():
            if "lora_" in name:
                self._require_cpu(name, parameter)
                # detach() shares CPU storage; it does not allocate a backup.
                # This reference keeps that storage alive after module.to(device).
                self.parameters[name] = (parameter, parameter.detach())
        for name, child in module.named_modules():
            for attribute in ("lora_a_stacked", "lora_b_stacked"):
                values = getattr(child, attribute, None)
                if values is None:
                    continue
                hosts = []
                for index, value in enumerate(values):
                    self._require_cpu(f"{name}.{attribute}[{index}]", value)
                    hosts.append(value.clone() if torch.is_inference(value) else value.detach())
                self.sidecars[f"{name}.{attribute}"] = (child, attribute, tuple(hosts))

    @staticmethod
    def _require_cpu(name, value):
        if value.device.type != "cpu":
            raise ValueError(f"Load rollout adapters on CPU before capturing policy: {name}")

    @torch.inference_mode(False)
    @torch.no_grad()
    def restore_host(self):
        """Reuse saved CPU storage instead of copying the device values back."""
        for parameter, cpu_tensor in self.parameters.values():
            parameter.data = cpu_tensor  # Keep Parameter identity; replace its storage.
        for child, attribute, hosts in self.sidecars.values():
            setattr(child, attribute, hosts)


def component_devices(module):
    """Include native LoRA sidecars, which Module.parameters/buffers do not see."""
    devices = {value.device for value in (*module.parameters(), *module.buffers())}
    for child in module.modules():
        for name in ("lora_a_stacked", "lora_b_stacked"):
            devices.update(value.device for value in getattr(child, name, ()) or ())
    return sorted(devices, key=str)


def move_native_lora_sidecars(module, device):
    """Set residency before Module.to and keep slots writable by the next policy RPC.

    Call under inference_mode(False) and the component's transfer fence, after
    restoring adapter CPU masters on offload. The native setter moves the tuples.
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
