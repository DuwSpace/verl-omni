# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in, same-run R / R-prime / T evidence; never reconstruct a failed worker.

R is the original generate score, R-prime is immediate native teacher forcing,
and T is actor old-policy teacher forcing. Full snapshots are rank-local on
FSDP and include nonpersistent buffers. They must not be loaded as full models.
"""

import hashlib
import json
import math
import os
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from verl_omni.utils.weight_integrity import tensor_digest


@dataclass(frozen=True)
class TextDiagnosticConfig:
    """Explicit diagnostic settings transported identically to actor and native worker."""

    enabled: bool = False
    output_dir: str = ""
    mean_abs_threshold: float = 1e-3
    max_abs_threshold: float = 0.05
    stop_on_mismatch: bool = True
    capture_step: int = 0

    def __post_init__(self):
        if any(type(value) is not bool for value in (self.enabled, self.stop_on_mismatch)):
            raise ValueError("Text diagnostic switches must be booleans")
        if type(self.capture_step) is not int or self.capture_step < 0:
            raise ValueError("Text diagnostic capture_step must be a nonnegative integer")
        if self.capture_step and not self.enabled:
            raise ValueError("Text diagnostic capture_step requires text diagnostics")
        if self.enabled and (not self.output_dir or not Path(self.output_dir).is_absolute()):
            raise ValueError("Enabled text diagnostics require an absolute output_dir")
        for value in (self.mean_abs_threshold, self.max_abs_threshold):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Text diagnostic thresholds must be finite and positive")


def local_cpu(value):
    """Clone a local tensor without triggering a DTensor gather."""
    if isinstance(value, DTensor):
        value = value.to_local()
    return value.detach().cpu().contiguous().clone()


def tensor_hash(value):
    """Hash shape, dtype and exact local bytes, including nonpersistent buffers."""
    return tensor_digest(local_cpu(value))


def input_identity(prompt, response, *, version, seed, temperature):
    """Pair original token IDs and sampling semantics across the transport boundary."""
    fields = {
        "prompt_sha256": tensor_hash(prompt),
        "response_sha256": tensor_hash(response),
        "policy_version": int(version),
        "seed": int(seed),
        "temperature": float(temperature),
    }
    return {**fields, "input_sha256": hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()}


def difference(left, right):
    """Summarize aligned token logprobs without treating nonfinite results as a pass."""
    if left.ndim != 1 or left.shape != right.shape or not left.numel():
        raise ValueError("Text comparison requires matching nonempty token vectors")
    delta = left.detach().cpu().double() - right.detach().cpu().double()
    finite = bool(torch.isfinite(delta).all())
    if not finite:
        return {"finite": False, "tokens": delta.numel(), "mean_abs": None, "max_abs": None, "ratio_outside_1pct": None}
    return {
        "finite": True,
        "tokens": delta.numel(),
        "mean_abs": float(delta.abs().mean()),
        "max_abs": float(delta.abs().max()),
        "ratio_outside_1pct": float(((delta < math.log(0.99)) | (delta > math.log(1.01))).double().mean()),
    }


def mismatched(metrics, config):
    """Apply explicit configurable thresholds to one paired comparison."""
    return (
        not metrics["finite"]
        or metrics["mean_abs"] > config.mean_abs_threshold
        or metrics["max_abs"] > config.max_abs_threshold
    )


def collective_errors(error):
    """Every actor rank must call, including ranks without a local failure."""
    if not dist.is_initialized():
        return [error] if error else []
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    return [f"rank {rank}: {value}" for rank, value in enumerate(errors) if value]


def stop_collectively(error):
    """Raise on every participating rank when any rank reports a failure."""
    errors = collective_errors(error)
    if errors:
        raise RuntimeError("LTX text diagnostic stopped all actor ranks: " + "; ".join(errors))


def model_evidence(model, *, include_base=False, tensors=False):
    """Name-keyed hashes; preserve DTensor layout instead of comparing shards as replicas."""
    result = {"parameters": {}, "buffers": {}, "adapters": {}}
    values = {"parameters": {}, "buffers": {}}
    for name, parameter in sorted(model.named_parameters()):
        if not include_base and "lora_" not in name:
            continue
        local = local_cpu(parameter)
        entry = {"sha256": tensor_hash(local), "dtype": str(local.dtype), "local_shape": list(local.shape)}
        if isinstance(parameter, DTensor):
            entry.update(
                global_shape=list(parameter.shape),
                placements=[str(p) for p in parameter.placements],
                mesh=parameter.device_mesh.mesh.cpu().tolist(),
                coordinate=parameter.device_mesh.get_coordinate(),
            )
        result["parameters"][name] = entry
        if tensors:
            values["parameters"][name] = local
    for name, buffer in sorted(model.named_buffers()):
        local = local_cpu(buffer)
        result["buffers"][name] = {"sha256": tensor_hash(local), "dtype": str(local.dtype), "shape": list(local.shape)}
        values["buffers"][name] = local
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            result["adapters"][name] = {
                "active": list(module.active_adapters),
                "disabled": module.disable_adapters,
                "merged": module.merged,
                "scaling": dict(module.scaling),
            }
    result["tensor_values"] = values
    return result


class TextDiagnostics:
    """Own immutable per-process evidence and a bounded first-failure snapshot."""

    def __init__(self, config=None, *, side, source_rank=None):
        self.config = TextDiagnosticConfig(**dict(config or {}))
        self.side = side
        self.source_rank = source_rank
        self.directory = None
        self._policies = set()
        self._failure_snapshot = None
        self._version = None
        if self.config.enabled:
            name = f"{side}-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            self.directory = Path(self.config.output_dir) / name
            self.directory.mkdir(parents=True, exist_ok=False)

    @property
    def enabled(self):
        """Whether this process must participate in text diagnosis."""
        return self.config.enabled

    def location(self, model):
        """Record process/device identity and precision-relevant execution settings."""
        return {
            "side": self.side,
            "source_rank": self.source_rank,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "device": str(next(model.parameters()).device),
            "visible_devices": {
                k: os.environ.get(k)
                for k in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
            },
            "training": model.training,
            "torch_version": str(torch.__version__),
            "parameter_dtype": str(next(model.parameters()).dtype),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "model_config": json.loads(model.config.to_json_string(use_diff=False))
            if hasattr(getattr(model, "config", None), "to_json_string")
            else None,
            "attention_implementation": getattr(getattr(model, "config", None), "_attn_implementation", None),
        }

    def save(self, name, metadata, tensors):
        """Write tensors before their completion metadata; propagate any I/O failure."""
        path = self.directory / name
        torch.save(tensors, path.with_suffix(".pt"))
        metadata = {"schema_version": 1, "complete": True, "tensor_path": str(path.with_suffix(".pt")), **metadata}
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return str(path.with_suffix(".pt"))

    def policy(self, model, version, expected=None):
        """Verify native loaded LoRA, save every version and baseline frozen hashes."""
        if not self.enabled:
            return
        self._version = version
        if expected is not None:
            parameters = dict(model.named_parameters())
            for name, value in expected.items():
                key = name.replace(".lora_A.weight", ".lora_A.joint.weight").replace(
                    ".lora_B.weight", ".lora_B.joint.weight"
                )
                actual = local_cpu(parameters[key])
                if actual.dtype != value.dtype or not torch.equal(actual, value.cpu()):
                    self.snapshot(model, f"load-mismatch-policy-{version}")
                    raise RuntimeError(f"Actual native Gemma LoRA differs from exported policy {version}: {key}")
            for name, module in model.named_modules():
                if hasattr(module, "lora_A") and (
                    module.disable_adapters or module.merged or list(module.active_adapters) != ["joint"]
                ):
                    self.snapshot(model, f"adapter-mismatch-policy-{version}")
                    raise RuntimeError(f"Native Gemma adapter is not active: {name}")
                if hasattr(module, "lora_A"):
                    config = model.peft_config["joint"]
                    if module.scaling["joint"] != config.lora_alpha / config.r:
                        self.snapshot(model, f"scaling-mismatch-policy-{version}")
                        raise RuntimeError(f"Native Gemma adapter scaling differs from transport config: {name}")
        if version not in self._policies:
            evidence = model_evidence(model, include_base=not self._policies, tensors=False)
            # Full base is hashed only on the first version; LoRA and all buffers are retained each version.
            weights = {n: local_cpu(p) for n, p in model.named_parameters() if "lora_" in n}
            values = evidence.pop("tensor_values")
            values["parameters"] = weights
            self.save(f"policy-{version}", {**self.location(model), "policy_version": version, **evidence}, values)
            self._policies.add(version)

    def snapshot(self, model, reason, *, force=False):
        """Capture before restoration/offload. Caller ensures FSDP shards are materialized locally."""
        if self._failure_snapshot is not None and not force:
            return self._failure_snapshot
        evidence = model_evidence(model, include_base=True, tensors=True)
        values = evidence.pop("tensor_values")
        self._failure_snapshot = self.save(
            f"failure-{uuid.uuid4().hex}",
            {**self.location(model), "policy_version": self._version, "reason": reason, **evidence},
            values,
        )
        return self._failure_snapshot

    @torch.no_grad()
    def rollout(self, model, prompt, response, rollout, *, version, seed, temperature, request_id):
        """Immediately replay original tokens, before condition encoding and AV execution."""
        if not self.enabled:
            return None
        self._version = version
        from .model import LTXJointModel

        device = next(model.parameters()).device
        ids = torch.cat((prompt, response)).to(device).unsqueeze(0)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[:, prompt.numel() :] = True
        inputs = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids, dtype=torch.bool),
            "response_mask": mask,
            "temperature": temperature,
        }
        repeated = LTXJointModel.recaption_logprobs(SimpleNamespace(gemma=model), **inputs)[mask[:, 1:]].float().cpu()
        identity = input_identity(prompt, response, version=version, seed=seed, temperature=temperature)
        sample_id = uuid.uuid4().hex
        metrics = difference(repeated, rollout)
        failed = mismatched(metrics, self.config)
        evidence = model_evidence(model)
        values = evidence.pop("tensor_values")
        metadata = {
            **self.location(model),
            **identity,
            "sample_id": sample_id,
            "request_id": request_id,
            "Rprime_vs_R": metrics,
            "mismatch": failed,
            "model": evidence,
            "rollout_sha256": tensor_hash(rollout),
            "recomputed_sha256": tensor_hash(repeated),
        }
        tensors = {
            **values,
            **{k: local_cpu(v) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()},
            "R": local_cpu(rollout),
            "Rprime": repeated.clone(),
        }
        artifact = self.save(f"sample-{sample_id}", metadata, tensors)
        if failed:
            self.snapshot(model, f"Rprime_vs_R sample={sample_id} policy={version}")
            if self.config.stop_on_mismatch or not metrics["finite"]:
                raise RuntimeError(f"Native Rprime/R mismatch; evidence saved at {artifact}")
        return {**metadata, "artifact": artifact, "recomputed_logprobs": repeated.clone()}

    def actor(self, model, record, train, inputs):
        """Validate original identity and save all three token vectors, before backward."""
        diagnostic = record.text_diagnostic
        if not diagnostic:
            raise ValueError("Text diagnostics enabled but native Rprime evidence is missing")
        identity = input_identity(
            record.prompt_token_ids,
            record.response_token_ids,
            version=record.policy_version,
            seed=record.seed,
            temperature=record.temperature,
        )
        for key, expected in identity.items():
            if diagnostic.get(key) != expected:
                raise ValueError(f"Unpaired text diagnostic: {key}")
        repeated = diagnostic["recomputed_logprobs"]
        if diagnostic["rollout_sha256"] != tensor_hash(record.rollout_logprobs) or diagnostic[
            "recomputed_sha256"
        ] != tensor_hash(repeated):
            raise ValueError("Text diagnostic scores changed during transport")
        comparisons = {
            "T_vs_R": difference(train, record.rollout_logprobs),
            "T_vs_Rprime": difference(train, repeated),
            "Rprime_vs_R": difference(repeated, record.rollout_logprobs),
        }
        failed = any(mismatched(v, self.config) for v in comparisons.values())
        evidence = model_evidence(model)
        values = evidence.pop("tensor_values")
        artifact = self.save(
            f"sample-{diagnostic['sample_id']}-{uuid.uuid4().hex[:8]}",
            {
                **self.location(model),
                **identity,
                "sample_id": diagnostic["sample_id"],
                "rollout_source_rank": diagnostic["source_rank"],
                "rollout_pid": diagnostic["pid"],
                "rollout_artifact": diagnostic["artifact"],
                "comparisons": comparisons,
                "mismatch": failed,
                "model": evidence,
            },
            {
                **values,
                **{k: local_cpu(v) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()},
                "R": local_cpu(record.rollout_logprobs),
                "Rprime": local_cpu(repeated),
                "T": local_cpu(train),
            },
        )
        if any(not value["finite"] for value in comparisons.values()):
            raise RuntimeError(f"Nonfinite text comparison; evidence saved at {artifact}")
        return failed, artifact
