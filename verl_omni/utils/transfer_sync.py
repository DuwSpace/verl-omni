# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Device-wide ordering for model residency transitions, including worker threads."""

from contextlib import ExitStack, contextmanager

import torch
from torch.distributed.tensor import DTensor


@contextmanager
def synchronized_device_transfer(*devices):
    """Finish pending work before moving storage, and transfers before returning.

    Explicit device contexts matter for asyncio.to_thread: a pool thread need
    not have the owning worker's current accelerator selected. This orders work
    on the supplied devices within this process; it is not a cross-rank barrier.
    """
    accelerators = {}
    for value in devices:
        device = torch.device(value)
        if device.type == "cpu":
            continue
        api = torch.get_device_module(device)
        if device.index is None:
            device = torch.device(device.type, api.current_device())
        accelerators[device] = api
    with ExitStack() as stack:
        for device, api in accelerators.items():
            stack.enter_context(api.device(device))
        for device, api in accelerators.items():
            api.synchronize(device)
        try:
            yield
        finally:
            for device, api in accelerators.items():
                api.synchronize(device)


@torch.inference_mode(False)
@torch.no_grad()
def copy_tensor_to_device(tensor, device):
    """Copy into zero-initialized destination storage; never retain a master.

    D2H zeros touch the CPU pages before DMA writes them. H2D uses the same
    allocate-then-copy contract. DTensors move only their local shard and keep
    the existing spec: this is storage offload, not mesh redistribution.
    The caller fences source/destination devices around the whole transition.
    """
    device = torch.device(device)
    source = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    local = torch.zeros(source.shape, dtype=source.dtype, device=device)
    local.copy_(source, non_blocking=False)
    if isinstance(tensor, DTensor):
        return DTensor(local, tensor._spec, requires_grad=local.requires_grad)
    return local


@torch.inference_mode(False)
@torch.no_grad()
def move_module(module, device):
    """Move parameters/buffers with zeros + copy_, preserving shared objects."""
    device = torch.device(device)
    sources = {value.device for value in (*module.parameters(), *module.buffers())}
    # Buffers may be registered under multiple names. Reuse their destination
    # tensor within this call, without retaining either side after the transfer.
    moved = {}
    aliases = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)

    def transfer(value):
        key = id(value)
        if key not in moved:
            if value.device == device or value.device.type == device.type == "cpu":
                destination = value.clone() if torch.is_inference(value) else value
            else:
                destination = copy_tensor_to_device(value, device)
            # Keep the source object alive so Python cannot reuse its id.
            moved[key] = (value, destination)
        return moved[key][1]

    with synchronized_device_transfer(*sorted(sources, key=str), device):
        module._apply(transfer)
        # _apply can replace Parameter objects across device types. Restore
        # tied names to the same new Parameter rather than separate wrappers.
        for names in aliases.values():
            if len(names) > 1:
                parameter = module.get_parameter(names[0])
                for name in names[1:]:
                    parent, _, attribute = name.rpartition(".")
                    setattr(module.get_submodule(parent), attribute, parameter)
    return module
