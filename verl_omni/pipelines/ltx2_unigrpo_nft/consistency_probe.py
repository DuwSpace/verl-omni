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

"""Opt-in same-input native/actor AV collection using the public MSProbe API."""

import hashlib
import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def probe_selected(version):
    """Training version 2*s-1 identifies global step s; even versions are evaluation."""
    if os.environ.get("DUMP_ON", "0") != "1":
        return False
    if type(version) is not int or version < 0:
        raise ValueError("Probe requires a scalar integer joint policy version")
    target = int(os.environ["MSPROBE_DUMP_GLOBAL_STEP"])
    if target < 1:
        raise ValueError("Probe global step must be positive")
    phase = os.environ.get("DUMP_PHASE", "update_actor")
    if phase not in ("update_actor", "log_prob", "all"):
        raise ValueError("Invalid DUMP_PHASE")
    step = (version + 1) // 2 if version % 2 else None
    selected = step == target and phase in ("update_actor", "all")
    print(
        json.dumps(
            {
                "ltx_probe_window": {
                    "source": "joint_policy_version",
                    "raw_type": type(version).__name__,
                    "version": version,
                    "global_step": step,
                    "target": target,
                    "selected": selected,
                }
            }
        ),
        flush=True,
    )
    return selected


def probe_identity(record, timestep):
    tokens = record.condition_token_ids.contiguous().numpy().tobytes()
    return {
        "sample_id": hashlib.sha256(record.original_prompt.encode() + tokens + str(record.seed).encode()).hexdigest(),
        "policy_version": record.policy_version,
        "global_step": (record.policy_version + 1) // 2,
        "seed": record.seed,
        "condition_tokens_sha256": hashlib.sha256(tokens).hexdigest(),
        "condition_token_count": record.condition_token_ids.numel(),
        "timestep": float(timestep),
        "timestep_units": "model_time_0_to_1000",
        "branch": "old_cond",
        "logical_batch_size": 1,
        "sample_axis": 0,
        "stage": "omninft_forward_process_replay",
    }


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return type(value)(to_cpu(item) for item in value)
    return value


@contextmanager
def capture_forward(model, *, side, identity):
    """One actual forward per process; tensor boundaries supplement module dumps.

    Initialization and collection errors propagate: a missing side is a failed
    diagnostic, never an accepted consistency result.
    """
    from msprobe.pytorch import PrecisionDebugger

    if side not in ("actor", "generate"):
        raise ValueError("Invalid probe side")
    directory = Path(os.environ["DUMP_PATH"]) / side / f"{socket.gethostname()}-{os.getpid()}"
    directory.mkdir(parents=True, exist_ok=False)
    config = os.environ[f"MSPROBE_CONFIG_{side.upper()}"]
    debugger = PrecisionDebugger(config_path=config, dump_path=str(directory / "msprobe"))
    boundaries = {}
    handles = []
    modules = dict(model.named_modules())
    for name in ("connectors", "transformer"):

        def save_boundary(module, args, kwargs, output, name=name):
            boundaries[name] = {"args": to_cpu(args), "kwargs": to_cpu(kwargs), "output": to_cpu(output)}

        handles.append(modules[name].register_forward_hook(save_boundary, with_kwargs=True))
    complete = False
    started = False
    empty_dump = False
    try:
        debugger.start(model=model)
        started = True
        yield directory
        if set(boundaries) != {"connectors", "transformer"}:
            raise RuntimeError("Both real connector and DiT forwards must be captured")
        torch.save(boundaries, directory / "boundaries.pt")
        complete = True
    finally:
        for handle in handles:
            handle.remove()
        if started:
            debugger.stop()
            debugger.step()
        if complete and not any(
            json.loads(path.read_text()).get("data") for path in (directory / "msprobe").rglob("dump.json")
        ):
            complete = False
            empty_dump = True
        metadata = {
            **identity,
            "side": side,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "msprobe_local_step": 0,
            "complete": complete,
            "directory": str(directory),
        }
        (directory / "correlation.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if empty_dump:
            raise RuntimeError("MSProbe did not write any module tensor records")


def compare_boundaries(actor_directory, generate_directory):
    """Validate pairing, then report tensor differences without inventing a tolerance.

    Files must be trusted outputs of this collector. This boundary report does
    not replace MSProbe module mapping or the root-cause analysis workflow.
    """
    directories = [Path(actor_directory), Path(generate_directory)]
    metadata = [json.loads((directory / "correlation.json").read_text()) for directory in directories]
    for side, item in zip(("actor", "generate"), metadata, strict=True):
        if item.get("side") != side or item.get("complete") is not True:
            raise ValueError(f"Incomplete or incorrect {side} collection")
    identity_fields = (
        "sample_id",
        "policy_version",
        "global_step",
        "seed",
        "condition_tokens_sha256",
        "condition_token_count",
        "timestep",
        "timestep_units",
        "branch",
        "logical_batch_size",
        "sample_axis",
        "stage",
    )
    for key in identity_fields:
        if key not in metadata[0] or metadata[0][key] != metadata[1].get(key):
            raise ValueError(f"Unpaired collections: {key}")
    if metadata[0]["branch"] != "old_cond" or metadata[0]["logical_batch_size"] != 1:
        raise ValueError("Only one matching old conditional sample is supported")
    data = [torch.load(directory / "boundaries.pt", map_location="cpu", weights_only=True) for directory in directories]
    actor, native = [item["transformer"]["kwargs"] for item in data]
    shared_inputs = (
        "hidden_states",
        "audio_hidden_states",
        "timestep",
        "audio_timestep",
        "sigma",
        "audio_sigma",
        "num_frames",
        "height",
        "width",
        "fps",
        "audio_num_frames",
    )
    for key in shared_inputs:
        if key not in actor or key not in native:
            raise ValueError(f"Missing forward input: {key}")
        left, right = actor[key], native[key]
        if isinstance(left, torch.Tensor):
            equal = isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
        else:
            equal = type(left) is type(right) and left == right
        if not equal:
            raise ValueError(f"Different forward inputs: {key}")
    for item in data:
        if len(item["connectors"]["args"]) < 2:
            raise ValueError("Missing connector inputs")
    if not torch.equal(data[0]["connectors"]["args"][1], data[1]["connectors"]["args"][1]):
        raise ValueError("Different condition attention masks")

    def difference(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Different boundary shapes: {left.shape}, {right.shape}")
        left64, right64 = left.double(), right.double()
        if not torch.isfinite(left64).all() or not torch.isfinite(right64).all():
            raise ValueError("Nonfinite boundary tensor")
        delta = (left64 - right64).abs().flatten()
        norm = float(left64.norm())
        return {
            "shape": list(left.shape),
            "actor_dtype": str(left.dtype),
            "native_dtype": str(right.dtype),
            "exact": torch.equal(left, right),
            "abs_mean": float(delta.mean()),
            # Stacked LTX Gemma hidden states exceed torch.quantile's 2**24
            # element limit. NumPy computes the same linear quantile on CPU.
            "abs_p99": float(np.quantile(delta.numpy(), 0.99)),
            "abs_max": float(delta.max()),
            "relative_l2": float(delta.norm()) / norm if norm else (0.0 if not delta.any() else None),
        }

    values = {
        "gemma_stacked_embeddings": difference(data[0]["connectors"]["args"][0], data[1]["connectors"]["args"][0])
    }
    for key in ("encoder_hidden_states", "audio_encoder_hidden_states"):
        values[key] = difference(actor[key], native[key])
    for key in ("encoder_attention_mask", "audio_encoder_attention_mask"):
        if not torch.equal(actor[key], native[key]):
            raise ValueError(f"Different connector output masks: {key}")
    for index, modality in enumerate(("video", "audio")):
        values[f"{modality}_prediction"] = difference(
            data[0]["transformer"]["output"][index], data[1]["transformer"]["output"][index]
        )
    return {
        "identity": {key: metadata[0][key] for key in identity_fields},
        "inputs_exact": True,
        "boundaries": values,
        "acceptance": "Requires an observed baseline and module-level interpretation",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check pairing and compare trusted LTX diagnostic boundaries")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--generate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare_boundaries(args.actor, args.generate)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
