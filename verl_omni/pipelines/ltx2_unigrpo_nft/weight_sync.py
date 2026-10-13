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

"""Publish Gemma/DiT LoRA as one immutable, checksummed policy bundle."""

import hashlib
import json
import os
import re
import shutil
import socket
import tempfile
from pathlib import Path

import torch


@torch.no_grad()
def export_joint_actor_lora(module, directory, *, version, base_model, device, coordination_group=None):
    """Collect the active paired LoRA, including CPU-offloaded FSDP2 shards.

    Call collectively after freezing old and before any optimizer update. All
    ranks enter each DTensor gather in the same order; only global rank zero
    publishes a bundle. The caller supplies a CPU-capable coordination group
    for accelerator jobs. Shared-file transport is restricted to one host.
    """
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    parameters = sorted((name, value) for name, value in module.named_parameters() if value.requires_grad)
    config = module.peft_config["default"]
    identity = {
        "version": version,
        "base_model": str(base_model),
        "directory": str(Path(directory).resolve()),
        "hostname": socket.gethostname(),
        "parameters": [(name, tuple(value.shape), str(value.dtype)) for name, value in parameters],
        "config": config.to_dict(),
    }
    if distributed:
        identities = [None] * dist.get_world_size()
        dist.all_gather_object(identities, identity, group=coordination_group)
        if any(other != identity for other in identities):
            raise ValueError("Joint actor ranks disagree on policy, LoRA layout, destination or host")
    if not parameters or any(
        not name.startswith(("gemma.model.language_model.", "transformer."))
        or not name.endswith((".lora_A.default.weight", ".lora_B.default.weight"))
        for name, _ in parameters
    ):
        raise ValueError("Joint export requires exclusively trainable default Gemma/DiT LoRA matrices")
    weights = {}
    for name, parameter in parameters:
        # HCCL cannot gather offloaded CPU shards. Move the local shard to its
        # compute device first, retaining FP32 precision for Gemma adapters.
        value = parameter.detach().to(device=device)
        if isinstance(value, DTensor):
            value = value.full_tensor()
        if rank == 0:
            weights[name] = value.float().cpu().clone()
        del value
    result = [None]
    if rank == 0:
        try:
            write_joint_lora_bundle(directory, weights, config, version=version, base_model=base_model)
            result[0] = {"directory": identity["directory"]}
        except Exception as error:
            result[0] = {"error": f"{type(error).__name__}: {error}"}
    if distributed:
        dist.broadcast_object_list(result, src=0, group=coordination_group)
    if "error" in result[0]:
        raise RuntimeError(f"Joint actor LoRA export failed: {result[0]['error']}")
    return Path(result[0]["directory"])


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_joint_lora_bundle(directory, weights, lora_config, *, version, base_model):
    """Atomically publish already-gathered CPU default adapters; never publish half a pair."""
    from peft import LoraConfig
    from safetensors.torch import save_file

    directory = Path(directory)
    if type(version) is not int or version < 0 or directory.exists():
        raise ValueError("Joint bundle needs a new directory and nonnegative integer version")
    parts = {"gemma": {}, "dit": {}}
    for name, tensor in weights.items():
        if not any(marker in name for marker in (".lora_A.default.weight", ".lora_B.default.weight")):
            raise ValueError(f"Unexpected non-default-LoRA parameter in bundle: {name}")
        if tensor.device.type != "cpu" or tensor.ndim != 2 or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Bundle requires finite CPU LoRA matrices: {name}")
        if name.startswith("gemma.model.language_model."):
            key, part = name.removeprefix("gemma."), "gemma"
        elif name.startswith("transformer."):
            key, part = "base_model.model." + name, "dit"
        else:
            raise ValueError(f"Unexpected joint LoRA component: {name}")
        key = key.replace(".default.weight", ".weight")
        parts[part][key] = tensor.detach().float().contiguous().clone()
    if not all(parts.values()):
        raise ValueError("Joint bundle requires both Gemma and DiT LoRA")
    if lora_config.use_dora or lora_config.use_rslora or lora_config.rank_pattern or lora_config.alpha_pattern:
        raise ValueError("Initial joint transport requires uniform ordinary LoRA rank and alpha")
    for state in parts.values():
        modules = {name.rsplit(".lora_", 1)[0] for name in state}
        for name in modules:
            if set((name + ".lora_A.weight", name + ".lora_B.weight")) - state.keys():
                raise ValueError(f"Missing paired LoRA matrix: {name}")
            if (
                state[name + ".lora_A.weight"].shape[0] != lora_config.r
                or state[name + ".lora_B.weight"].shape[1] != lora_config.r
            ):
                raise ValueError(f"LoRA matrix rank differs from config: {name}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".joint-policy-", dir=directory.parent))
    try:
        manifest = {"schema_version": 1, "policy_version": version, "base_model": str(base_model), "components": {}}
        for part, state in parts.items():
            target = temporary / part
            target.mkdir()
            modules = sorted({name.rsplit(".lora_", 1)[0].removeprefix("base_model.model.") for name in state})
            config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                # Native DiT scans every module with every list entry. Large
                # exact-name lists overflow its regex cache and repeatedly
                # compile thousands of expressions. One anchored alternation
                # preserves the exact target set and compiles only once.
                target_modules="^(?:" + "|".join(re.escape(name) for name in modules) + ")$"
                if part == "dit"
                else modules,
                bias="none",
                lora_dropout=0.0,
                inference_mode=True,
            )
            config.save_pretrained(target)
            save_file(state, target / "adapter_model.safetensors")
            manifest["components"][part] = {
                "keys": sorted(state),
                "weights_sha256": file_sha256(target / "adapter_model.safetensors"),
                "config_sha256": file_sha256(target / "adapter_config.json"),
            }
        (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))
        os.rename(temporary, directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def validate_joint_lora_bundle(directory):
    """Validate both components before mutating either live rollout model."""
    from safetensors import safe_open

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or set(manifest.get("components", {})) != {"gemma", "dit"}:
        raise ValueError("Invalid paired policy manifest")
    if type(manifest.get("policy_version")) is not int or manifest["policy_version"] < 0:
        raise ValueError("Invalid paired policy version")
    for part, expected in manifest["components"].items():
        path = directory / part
        if file_sha256(path / "adapter_model.safetensors") != expected["weights_sha256"]:
            raise ValueError(f"Joint policy weight checksum mismatch: {part}")
        if file_sha256(path / "adapter_config.json") != expected["config_sha256"]:
            raise ValueError(f"Joint policy config checksum mismatch: {part}")
        with safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu") as stream:
            if sorted(stream.keys()) != expected["keys"] or not expected["keys"]:
                raise ValueError(f"Joint policy tensor manifest mismatch: {part}")
    return manifest
