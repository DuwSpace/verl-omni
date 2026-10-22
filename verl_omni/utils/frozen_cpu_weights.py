# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU masters for frozen, unsharded inference parameters (never actor/FSDP)."""

import hashlib
import json
import os
import uuid
from pathlib import Path

import torch

from .transfer_sync import synchronized_device_transfer


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    result = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}".encode())
    result.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return result.hexdigest()


@torch.no_grad()
def first_different_byte_block(value, host, *, chunk_bytes=4 * 1024 * 1024):
    """Compare every byte on the working device, without copying weights D2H.

    A bounded reference chunk is copied H2D independently of the model load.
    Return only complementary all-equal/any-different flags, twice, so a zeroed
    result buffer cannot be interpreted as success. This detects differences;
    it does not prove that the device or both transfer paths are fault-free.
    No floating-point reduction, custom kernel, or full pinned model is used.
    """
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if value.shape != host.shape or value.dtype != host.dtype or host.device.type != "cpu":
        raise ValueError("Working tensor and CPU master must have the same shape and dtype")
    actual = value.detach().contiguous().reshape(-1).view(torch.uint8)
    reference = host.detach().contiguous().reshape(-1).view(torch.uint8)
    for offset in range(0, reference.numel(), chunk_bytes):
        expected = reference[offset : offset + chunk_bytes].to(value.device, non_blocking=False)
        equal = actual[offset : offset + chunk_bytes] == expected
        status = torch.stack((equal.all(), (~equal).any()))
        for _ in range(2):
            if status.to("cpu", copy=True, non_blocking=False).tolist() != [True, False]:
                return offset
    return None


class FrozenCpuWeights:
    """Retain initial CPU storage; discard accelerator copies on offload.

    Capture before adapter injection and before the first device transfer.
    Parameter identities survive PEFT wrapping; later-added LoRA parameters
    are deliberately absent even when inference marks them requires_grad=False.
    Persistent buffers are frozen inference state too. Nonpersistent buffers
    (for example rotary caches) continue using normal Module.to().
    CPU storage is shared with the parked module, not OS write-protected.
    All captured parameters are verified before accelerator activation.
    """

    @torch.inference_mode(False)
    @torch.no_grad()
    def __init__(self, module, *, failure_dir="outputs/weight_integrity"):
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
        self.expected = {name: tensor_digest(host) for name, _, host in self._values()}
        self.failure_dir = Path(failure_dir)
        self.failure = None

    def _values(self):
        for name, (parameter, host) in self.parameters.items():
            yield name, parameter, host
        for name, (module, attribute, host) in self.buffers.items():
            yield name, getattr(module, attribute), host

    def check(self, stage, *, master=False):
        if self.failure is not None:
            raise RuntimeError(self.failure)
        for name, parameter, host in self._values():
            expected = self.expected[name]
            value = host if master else parameter
            block = None
            if value.device.type == "cpu":
                observed = tensor_digest(value)
                if observed == expected:
                    continue
            else:
                block = first_different_byte_block(value, host)
                if block is None:
                    continue
                observed = None
            self._save_failure(name, value, host, stage, master, expected, observed, block)

    def _save_failure(self, name, value, host, stage, master, expected, observed, block):
        """Latch the failure and save evidence; subsequent checks cannot resume execution."""
        directory = self.failure_dir / f"failure-{os.getpid()}-{uuid.uuid4().hex}"
        self.failure = f"Frozen weight integrity failure at {stage}: {name}; evidence: {directory}"
        directory.mkdir(parents=True)
        torch.save(value.detach().cpu(), directory / "observed.pt")
        torch.save(host, directory / "cpu-master.pt")
        report = {
            "parameter": name,
            "stage": stage,
            "master": master,
            "pid": os.getpid(),
            "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
            "device": str(value.device),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "different_block_byte_offset": block,
            "comparison": "cpu_sha256" if value.device.type == "cpu" else "device_bytewise",
        }
        (directory / "failure.json").write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError(self.failure)

    @torch.inference_mode(False)
    @torch.no_grad()
    def restore_host(self):
        """Rebind to saved CPU storage without D2H; caller must synchronize first."""
        if self.failure is not None:
            raise RuntimeError(self.failure)
        for parameter, cpu_tensor in self.parameters.values():
            parameter.data = cpu_tensor
        for module, attribute, host in self.buffers.values():
            setattr(module, attribute, host)

    @torch.inference_mode(False)
    @torch.no_grad()
    def move(self, module, device):
        """Reward residency: restore CPU storage on sleep, copy and verify on wake."""
        device = torch.device(device)
        sources = {value.device for value in (*module.parameters(), *module.buffers())}
        with synchronized_device_transfer(*sorted(sources, key=str), device):
            if device.type == "cpu":
                self.restore_host()
                # Managed weights are already on CPU; only remaining state moves.
                module.to(device, non_blocking=False)
                return

            self.check("before_transfer_cpu_master", master=True)
            module.to(device, non_blocking=False)
            self.check(f"after_transfer:{device}")
