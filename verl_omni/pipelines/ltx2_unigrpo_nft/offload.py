# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Request-local, mutually exclusive component residency for joint LTX rollout."""

from functools import wraps

import torch


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
        self.handles = []
        self.methods = []
        self.release()
        for name, module in self.components.items():
            self.handles.append(module.register_forward_pre_hook(self._pre_forward(name)))
        self._wrap("text_encoder", "generate")
        self._wrap("vae", "decode")
        self._wrap("audio_vae", "decode")

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

    @staticmethod
    @torch.inference_mode(False)
    @torch.no_grad()
    def _move(module, device):
        # Native execution enters inference_mode, but the next policy RPC
        # mutates LoRA buffers outside it. Persistent transfers must therefore
        # create normal tensors. Repair previously materialized sidecars too.
        # Native diffusion LoRA stores sidecars in tuples, not registered
        # buffers. Set their target BEFORE Module.to(), whose _apply otherwise
        # restores these tensors to the previously fixed accelerator device.
        for child in module.modules():
            setter = getattr(child, "_set_diffusion_lora_buffer_device", None)
            if setter is not None:
                for name in ("lora_a_stacked", "lora_b_stacked"):
                    values = getattr(child, name, None)
                    if values is not None:
                        setattr(
                            child,
                            name,
                            tuple(value.clone() if torch.is_inference(value) else value for value in values),
                        )
                setter(device)
        module.to(device)

    def _flush(self):
        if self.device.type != "cpu":
            api = torch.get_device_module(self.device)
            api.synchronize()
            api.empty_cache()

    def activate(self, name):
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
        self.active = None
        for module in self.components.values():
            self._move(module, torch.device("cpu"))
        self._flush()

    def disable(self):
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
