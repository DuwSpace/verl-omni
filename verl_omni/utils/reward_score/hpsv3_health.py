# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fixed-input HPS diagnostics. References are prepared offline, never learned on wake."""

import hashlib
import json
import logging
import math
import os
import time
import uuid
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class HPSHealthCheckError(RuntimeError):
    """Fixed-input inference changed; evidence has been saved before returning rewards."""


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(repr((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_fingerprints(model):
    """Copy tensors for hashing without moving or changing the live model."""
    return {
        **{f"parameter:{name}": tensor_digest(tensor) for name, tensor in model.named_parameters()},
        **{f"buffer:{name}": tensor_digest(tensor) for name, tensor in model.named_buffers()},
    }


def _validate_sample(sample):
    frames, prompts = sample["frames"], sample["prompts"]
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[1] != 3 or min(frames.shape) <= 0:
        raise ValueError("HPS health frames must be nonempty uint8 [N,3,H,W]")
    if len(prompts) != len(frames) or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("HPS health probes require one nonempty prompt per frame")


@torch.inference_mode()
def evaluate_probe(inferencer, sample, observed=None):
    """Use the normal processor/forward, retaining inputs including explicit decoder positions."""
    frames = sample["frames"]
    images = [Image.fromarray(frame.permute(1, 2, 0).numpy().copy()) for frame in frames]
    return evaluate_images(inferencer, images, list(sample["prompts"]), observed)


@torch.no_grad()
def evaluate_images(inferencer, images, prompts, observed=None):
    """Capture the actual forward without rerunning preprocessing or inference."""
    inputs = inferencer.prepare_batch(images, prompts)
    captured = {name: value.detach().cpu().clone() for name, value in inputs.items()}
    observed = {} if observed is None else observed
    observed["inputs"] = captured

    def capture_positions(module, args, kwargs):
        positions = kwargs.get("position_ids")
        if positions is not None:
            captured["effective_position_ids"] = positions.detach().cpu().clone()

    inner = getattr(inferencer.model, "model", None)
    handle = inner.register_forward_pre_hook(capture_positions, with_kwargs=True) if inner is not None else None
    try:
        logits = inferencer.model(return_dict=True, **inputs)["logits"].detach().cpu().clone()
    finally:
        if handle is not None:
            handle.remove()
    observed["logits"] = logits
    return observed


def build_reference(inferencer, samples, model_identity):
    """Explicit offline calibration; callers must validate the model and preserve this file."""
    if not samples or any(module.training for module in inferencer.model.modules()):
        raise ValueError("Reference creation requires samples and an eval-mode model")
    references = []
    for sample in samples:
        _validate_sample(sample)
        sample = {"frames": sample["frames"].detach().cpu().clone(), "prompts": list(sample["prompts"])}
        observed = evaluate_probe(inferencer, sample)
        if observed["logits"].shape != (len(sample["frames"]), 2) or not torch.isfinite(observed["logits"]).all():
            raise ValueError("Reference frame logits must be finite [N,2]")
        references.append({**sample, **observed})
    return {
        "schema_version": 1,
        "model_identity": model_identity,
        "samples": references,
        "model_fingerprints": model_fingerprints(inferencer.model),
    }


class HPSHealthProbe:
    def __init__(self, config, model_identity):
        unknown = set(config) - {
            "enabled",
            "baseline_path",
            "output_dir",
            "atol",
            "rtol",
            "dump_model_state",
            "input_stats",
        }
        if unknown:
            raise ValueError(f"Unknown HPS health settings: {sorted(unknown)}")
        self.atol, self.rtol = float(config.get("atol", 1e-3)), float(config.get("rtol", 0.0))
        if any(not math.isfinite(value) or value < 0 for value in (self.atol, self.rtol)):
            raise ValueError("HPS health tolerances must be finite and nonnegative")
        self.dump_model_state = config.get("dump_model_state", False)
        if type(self.dump_model_state) is not bool:
            raise ValueError("dump_model_state must be a boolean")
        if not config.get("baseline_path") or not config.get("output_dir"):
            raise ValueError("Enabled HPS health checks require baseline_path and output_dir")
        baseline = Path(config["baseline_path"])
        self.baseline_id = hashlib.sha256(baseline.read_bytes()).hexdigest()
        self.reference = torch.load(baseline, map_location="cpu", weights_only=True)
        if self.reference.get("schema_version") != 1 or self.reference.get("model_identity") != model_identity:
            raise ValueError("HPS health reference schema/model identity mismatch; calibrate explicitly")
        if not self.reference.get("samples") or not self.reference.get("model_fingerprints"):
            raise ValueError("HPS health reference requires samples and model fingerprints")
        for sample in self.reference["samples"]:
            _validate_sample(sample)
            logits = sample["logits"]
            if logits.shape != (len(sample["frames"]), 2) or not torch.isfinite(logits).all():
                raise ValueError("HPS health reference logits must be finite [N,2]")
            if not sample.get("inputs") or not all(
                isinstance(value, torch.Tensor) for value in sample["inputs"].values()
            ):
                raise ValueError("HPS health reference requires the actual processor tensors")
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.failure = None
        stats_config = config.get("input_stats") or {}
        if type(stats_config.get("enabled", True)) is not bool:
            raise ValueError("HPS input_stats.enabled must be a boolean")
        self.input_stats = None
        if stats_config.get("enabled", True):
            from .hpsv3_input_stats import HPSInputStats

            self.input_stats = HPSInputStats(stats_config, self.output_dir)

    @torch.inference_mode()
    def check(self, inferencer, *, stage, context):
        if self.failure is not None:
            raise HPSHealthCheckError(self.failure)
        record = {
            "time": time.time(),
            "pid": os.getpid(),
            "device": str(inferencer.device),
            "stage": stage,
            "context": dict(context),
            "baseline_id": self.baseline_id,
            "atol": self.atol,
            "rtol": self.rtol,
            "samples": [],
            "parameter_dtypes": sorted({str(value.dtype) for value in inferencer.model.parameters()}),
            "torch_version": str(torch.__version__),
        }
        observed_samples = []
        reason = None
        try:
            training = [name for name, module in inferencer.model.named_modules() if module.training]
            record["training_modules"] = training
            if training:
                raise ValueError("HPS model is no longer entirely in eval mode")
            for reference in self.reference["samples"]:
                observed = {}
                observed_samples.append(observed)
                evaluate_probe(inferencer, reference, observed)
                expected_hashes = {key: tensor_digest(value) for key, value in reference["inputs"].items()}
                actual_hashes = {key: tensor_digest(value) for key, value in observed["inputs"].items()}
                actual, expected = observed["logits"], reference["logits"]
                same_shape = actual.shape == expected.shape
                finite = bool(torch.isfinite(actual).all())
                diff = float((actual.float() - expected.float()).abs().max()) if same_shape and finite else None
                matches = (
                    same_shape
                    and finite
                    and torch.allclose(actual.float(), expected.float(), atol=self.atol, rtol=self.rtol)
                )
                record["samples"].append(
                    {
                        "frames_sha256": tensor_digest(reference["frames"]),
                        "prompts_sha256": hashlib.sha256(json.dumps(reference["prompts"]).encode()).hexdigest(),
                        "expected_input_hashes": expected_hashes,
                        "actual_input_hashes": actual_hashes,
                        "max_logit_diff": diff,
                        "logits_match": matches,
                    }
                )
                if actual_hashes != expected_hashes or not matches:
                    raise ValueError("HPS fixed-input preprocessing or frame logits changed")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"

        record["ok"] = reason is None
        if reason is not None:
            record["reason"] = reason
            destination = self.output_dir / f"failure-{os.getpid()}-{uuid.uuid4().hex}"
            destination.mkdir()
            self.failure = f"HPS health check failed at {stage}: {reason}; evidence: {destination}"
            # Save small input/output evidence first, before expensive model copies.
            (destination / "failure.json").write_text(json.dumps(record, indent=2))
            torch.save({"reference": self.reference, "observed": observed_samples}, destination / "probes.pt")
            if self.input_stats is not None:
                try:
                    self.input_stats.dump(reason=self.failure, destination=destination)
                except Exception as exc:
                    record["input_capture_error"] = repr(exc)
            try:
                fingerprints = model_fingerprints(inferencer.model)
                expected = self.reference["model_fingerprints"]
                record["changed_model_tensors"] = sorted(
                    key for key in set(expected) | set(fingerprints) if expected.get(key) != fingerprints.get(key)
                )
                (destination / "model_fingerprints.json").write_text(json.dumps(fingerprints, indent=2))
                torch.save(
                    {key: value.detach().cpu().clone() for key, value in inferencer.model.named_buffers()},
                    destination / "buffers.pt",
                )
                head = getattr(inferencer.model, "rm_head", None)
                if head is not None:
                    torch.save(
                        {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
                        destination / "reward_head.pt",
                    )
                if self.dump_model_state:
                    torch.save(
                        {key: value.detach().cpu().clone() for key, value in inferencer.model.state_dict().items()},
                        destination / "model_state.pt",
                    )
            except Exception as exc:
                record["model_capture_error"] = repr(exc)
            (destination / "failure.json").write_text(json.dumps(record, indent=2))

        with (self.output_dir / f"checks-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        logger.log(
            logging.INFO if record["ok"] else logging.ERROR,
            "HPS health stage=%s context=%s ok=%s diffs=%s",
            stage,
            context,
            record["ok"],
            [sample["max_logit_diff"] for sample in record["samples"]],
        )
        if self.failure is not None:
            raise HPSHealthCheckError(self.failure)
