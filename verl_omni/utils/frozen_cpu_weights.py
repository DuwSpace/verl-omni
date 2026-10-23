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

import fcntl
import hashlib
import json
import mmap
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch

from .transfer_sync import synchronized_device_transfer


def _file_identity(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _checkpoint_identity(model_path):
    path = Path(model_path)
    files = {}
    if path.is_dir():
        for entry in sorted(path.rglob("*")):
            if entry.is_file() and entry.suffix in (".json", ".safetensors", ".bin", ".pt"):
                files[str(entry.relative_to(path))] = _file_identity(entry)
    return {"model": str(path.resolve()) if path.exists() else str(model_path), "files": files}


@contextmanager
def shared_cpu_weight_loading(directory, *, model_path, component_names):
    """Load once under a lock; release followers to build from the ready cache.

    Only a successfully initialized publisher writes ready.json. Followers
    check checkpoint/file identities, then release the lock BEFORE constructing
    their engines. They never reload checkpoint tensors or repeat initial hashes.
    Activation-time CPU and device integrity checks remain enabled.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ready = directory / "ready.json"
    identity = _checkpoint_identity(model_path)
    with (directory / "initialization.lock").open("a+b") as lock:
        print(f"LTX_SHARED_CPU waiting for initialization lock: {directory}", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if ready.exists():
                manifest = json.loads(ready.read_text())
                if manifest.get("version") != 1 or manifest.get("checkpoint") != identity:
                    raise RuntimeError(f"Shared CPU cache belongs to a different checkpoint: {directory}")
                expected = {f"{name}.pt" for name in component_names}
                if set(manifest["files"]) != expected:
                    raise RuntimeError(f"Incomplete shared CPU cache: {directory}")
                for name, fingerprint in manifest["files"].items():
                    if _file_identity(directory / name) != fingerprint:
                        raise RuntimeError(f"Shared CPU cache changed after publication: {directory / name}")
                fcntl.flock(lock, fcntl.LOCK_UN)
                yield True
                return

            yield False
            if _checkpoint_identity(model_path) != identity:
                raise RuntimeError("Checkpoint files changed while creating the shared CPU cache")
            manifest = {
                "version": 1,
                "checkpoint": identity,
                "files": {f"{name}.pt": _file_identity(directory / f"{name}.pt") for name in component_names},
            }
            temporary = directory / f".ready-{uuid.uuid4().hex}.json"
            try:
                with temporary.open("w") as stream:
                    json.dump(manifest, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, ready)
            finally:
                temporary.unlink(missing_ok=True)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _storage_layout(values):
    """Record views and aliases as well as shapes; pointer values are process-local."""
    groups = {}
    layout = {}
    for name, value in values.items():
        storage = value.untyped_storage()
        group = groups.setdefault(storage._cdata, len(groups))
        layout[name] = (str(value.dtype), tuple(value.shape), tuple(value.stride()), value.storage_offset(), group)
    return layout


def _map_cpu_weights(path, values, expected):
    """Publish once, then mmap the same inode in all local rollout processes.

    MAP_PRIVATE shares clean file-backed pages, but isolates accidental CPU
    writes with copy-on-write. The existing integrity guard detects such writes.
    torch.save/load preserve storage aliases and strides. Files are scoped to
    the run and retained until the user removes its cache after workers exit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layout = _storage_layout(values)
    with path.with_suffix(".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            storages = {v.untyped_storage()._cdata: v.untyped_storage().nbytes() for v in values.values()}
            required = sum(storages.values()) + 16 * 1024**2
            if shutil.disk_usage(path.parent).free < required:
                raise RuntimeError(f"Insufficient space for shared CPU weights: {path}; need about {required} bytes")
            fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    torch.save({"version": 1, "layout": layout, "digests": expected, "tensors": values}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        with torch.serialization.set_default_mmap_options(mmap.MAP_PRIVATE):
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if payload["version"] != 1 or payload["layout"] != layout or payload["digests"] != expected:
            raise RuntimeError(f"Shared CPU weight cache does not match the loaded model: {path}")
        mapped = payload["tensors"]
        if mapped.keys() != values.keys() or _storage_layout(mapped) != layout:
            raise RuntimeError(f"Invalid shared CPU weight layout: {path}")
        for name, value in mapped.items():
            if tensor_digest(value) != expected[name]:
                raise RuntimeError(f"Corrupt shared CPU weight cache: {path}: {name}")
    print(f"LTX_SHARED_CPU mapped {path} ({path.stat().st_size / 1024**3:.2f} GiB), pid={os.getpid()}", flush=True)
    return mapped


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
    def __init__(self, module, *, failure_dir="outputs/weight_integrity", shared_path=None):
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
        if shared_path is not None:
            mapped = _map_cpu_weights(shared_path, {name: host for name, _, host in self._values()}, self.expected)
            self.parameters = {name: (parameter, mapped[name]) for name, (parameter, _) in self.parameters.items()}
            self.buffers = {
                name: (child, attribute, mapped[name]) for name, (child, attribute, _) in self.buffers.items()
            }
            self.restore_host()

    @classmethod
    @torch.inference_mode(False)
    @torch.no_grad()
    def from_shared(cls, module, shared_path, *, failure_dir="outputs/weight_integrity"):
        """Bind an empty component to a publisher-validated cache, without scans.

        Keep parameter identities and vLLM parameter subclasses/loader metadata
        when replacing meta storage. Nonpersistent buffers are constructed on
        CPU by the normal model constructor and deliberately stay private.
        """
        path = Path(shared_path)
        manifest = json.loads((path.parent / "ready.json").read_text())
        expected_file = manifest["files"][path.name]
        if _file_identity(path) != expected_file:
            raise RuntimeError(f"Shared CPU cache changed after publication: {path}")
        with torch.serialization.set_default_mmap_options(mmap.MAP_PRIVATE):
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        values = payload["tensors"]
        if payload["version"] != 1 or _storage_layout(values) != payload["layout"]:
            raise RuntimeError(f"Invalid shared CPU weight layout: {path}")
        result = cls.__new__(cls)
        result.parameters = {}
        result.buffers = {}
        for name, parameter in module.named_parameters():
            result.parameters[name] = (parameter, values[name])
        for prefix, child in module.named_modules():
            for name, value in child.named_buffers(recurse=False):
                if name not in child._non_persistent_buffers_set:
                    key = f"buffer:{prefix}.{name}"
                    result.buffers[key] = (child, name, values[key])
        if set(values) != set(result.parameters) | set(result.buffers) or set(values) != set(payload["digests"]):
            raise RuntimeError(f"Shared CPU cache does not match component parameters: {path}")
        for name, parameter, host in result._values():
            if parameter.shape != host.shape or parameter.dtype != host.dtype:
                raise RuntimeError(f"Shared CPU cache shape/dtype mismatch: {path}: {name}")
        for parameter, host in result.parameters.values():
            if parameter.is_meta:
                replacement = torch.Tensor._make_subclass(type(parameter), host, False)
                replacement.__dict__.update(parameter.__dict__)
                torch.utils.swap_tensors(parameter, replacement)
            else:
                parameter.data = host
                parameter.requires_grad_(False)
        for child, name, host in result.buffers.values():
            setattr(child, name, host)
        if _file_identity(path) != expected_file:
            raise RuntimeError(f"Shared CPU cache changed during mapping: {path}")
        result.expected = payload["digests"]
        result.failure_dir = Path(failure_dir)
        result.failure = None
        print(f"LTX_SHARED_CPU attached {path} without checkpoint loading, pid={os.getpid()}", flush=True)
        return result

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
