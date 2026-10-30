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

"""Device-wide ordering for model residency transitions, including worker threads."""

from contextlib import ExitStack, contextmanager

import torch


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
