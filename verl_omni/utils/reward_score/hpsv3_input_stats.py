# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Actual HPS request statistics and bounded, failure-only tensor capture."""

import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def tensor_statistics(tensor):
    from .hpsv3_health import tensor_digest

    value = tensor.detach().cpu().contiguous()
    minimum, maximum, finite_count = None, None, 0
    # Bound temporary float/mask allocations even for full raw videos.
    for chunk in value.reshape(-1).split(1024 * 1024):
        finite = torch.isfinite(chunk)
        finite_count += int(finite.sum())
        if finite.any():
            valid = chunk[finite].float()
            low, high = float(valid.min()), float(valid.max())
            minimum = low if minimum is None else min(minimum, low)
            maximum = high if maximum is None else max(maximum, high)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(tensor.device),
        "min": minimum,
        "max": maximum,
        "finite_count": finite_count,
        "numel": value.numel(),
        "sha256": tensor_digest(value),
    }


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, tuple | list):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class HPSInputStats:
    """All callers share one lock; no device/model mutation or extra inference."""

    def __init__(self, config, output_dir):
        defaults = {
            "enabled": True,
            "max_samples": 16,
            "max_batches": 2,
            "max_bytes": 21474836480,
            "max_anomaly_dumps": 2,
            "score_drop_threshold": 3.0,
            "min_batch_samples": 4,
        }
        unknown = set(config) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown HPS input_stats settings: {sorted(unknown)}")
        settings = {**defaults, **config}
        for key in ("max_samples", "max_batches", "max_bytes", "max_anomaly_dumps", "min_batch_samples"):
            if type(settings[key]) is not int or settings[key] <= 0:
                raise ValueError(f"HPS input_stats.{key} must be a positive integer")
            setattr(self, key, settings[key])
        self.score_drop_threshold = settings["score_drop_threshold"]
        if self.score_drop_threshold is not None:
            self.score_drop_threshold = float(self.score_drop_threshold)
            if not math.isfinite(self.score_drop_threshold) or self.score_drop_threshold <= 0:
                raise ValueError("HPS input_stats.score_drop_threshold must be positive or null")
        self.output_dir = Path(output_dir)
        self._lock = threading.RLock()
        self._recent = OrderedDict()
        self._cached_bytes = 0
        self._batch_id = 0
        self._context = {}
        self._batch = None
        self._previous = {}
        self._anomaly_dumps = 0

    def _write(self, kind, record):
        with (self.output_dir / f"{kind}-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(_json_safe(record), allow_nan=False) + "\n")

    def begin_batch(self, context):
        with self._lock:
            self._batch_id += 1
            self._context = dict(context)
            self._batch = {"count": 0, "errors": 0, "scores": []}
            self._trim()

    def _trim(self):
        while self._recent:
            first = next(iter(self._recent.values()))
            if (
                len(self._recent) <= self.max_samples
                and self._cached_bytes <= self.max_bytes
                and first["record"]["batch_id"] > self._batch_id - self.max_batches
            ):
                break
            _, removed = self._recent.popitem(last=False)
            self._cached_bytes -= _tensor_bytes(removed)

    def begin_request(self, video, prompt, *, uid, data_source, settings, seed=None):
        stats = tensor_statistics(video)
        with self._lock:
            if self._batch is None:
                self.begin_batch({})
            record = {
                "request_id": uuid.uuid4().hex,
                "uid": None if uid is None else str(uid),
                "sample_seed": None if seed is None else str(seed),
                "time": time.time(),
                "pid": os.getpid(),
                "batch_id": self._batch_id,
                "context": dict(self._context),
                "data_source": str(data_source),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_chars": len(prompt),
                "raw_video": stats,
                "settings": dict(settings),
                "calls": [],
            }
            # Oversize samples still get complete statistics; do not allocate a cached copy.
            if video.numel() * video.element_size() <= self.max_bytes:
                entry = {"record": record, "prompt": prompt, "video": video.detach().cpu().clone(), "calls": []}
                self._recent[record["request_id"]] = entry
                self._cached_bytes += _tensor_bytes(entry)
                self._trim()
            return record

    def record_call(self, record, frames, observed, *, frame_offset):
        call = {
            "frame_offset": frame_offset,
            "frame_count": len(frames),
            "model_device": observed.get("device"),
            "training_modules": observed.get("training_modules"),
            "frames": tensor_statistics(frames),
            "inputs": {key: tensor_statistics(value) for key, value in observed.get("inputs", {}).items()},
        }
        if "logits" in observed:
            call["logits"] = observed["logits"].float().tolist()
        with self._lock:
            record["calls"].append(call)
            entry = self._recent.get(record["request_id"])
            if entry is not None:
                payload = {"frame_offset": frame_offset, "frames": frames, **observed}
                entry["calls"].append(payload)
                self._cached_bytes += _tensor_bytes(payload)
                self._trim()

    def finish_request(self, record, *, result=None, error=None):
        with self._lock:
            record["ok"] = error is None
            record["duration_s"] = time.time() - record["time"]
            record["result"] = result
            record["error"] = repr(error) if error is not None else None
            record["raw_input_cached"] = record["request_id"] in self._recent
            active_batch = self._batch is not None and record["batch_id"] == self._batch_id
            record["late_completion"] = not active_batch
            self._write("requests", record)
            if active_batch:
                self._batch["count"] += 1
            if error is not None:
                if active_batch:
                    self._batch["errors"] += 1
                self.dump(reason=f"request {record['request_id']}: {error}")
            elif active_batch:
                self._batch["scores"].append(float(result["score"]))

    def finish_batch(self):
        with self._lock:
            if self._batch is None:
                return
            scores = self._batch["scores"]
            record = {
                "time": time.time(),
                "pid": os.getpid(),
                "batch_id": self._batch_id,
                "context": dict(self._context),
                "count": self._batch["count"],
                "errors": self._batch["errors"],
                "mean": sum(scores) / len(scores) if scores else None,
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
                "cached_samples": len(self._recent),
                "cached_tensor_bytes": self._cached_bytes,
            }
            phase = self._context.get("phase", "unknown")
            previous = self._previous.get(phase)
            drop = None
            if len(scores) >= self.min_batch_samples and previous is not None:
                drop = previous["mean"] - record["mean"]
            record["previous_same_phase"] = previous
            record["mean_drop"] = drop
            record["drop_alert"] = (
                drop is not None and self.score_drop_threshold is not None and drop >= self.score_drop_threshold
            )
            if len(scores) >= self.min_batch_samples and not record["errors"]:
                self._previous[phase] = {"mean": record["mean"], "context": dict(self._context)}
            if record["drop_alert"]:
                record["evidence"] = self.dump(reason=f"same-phase batch mean dropped by {drop}")
            self._write("batches", record)
            logger.log(
                logging.WARNING if record["drop_alert"] or record["errors"] else logging.INFO,
                "HPS inputs context=%s count=%s errors=%s mean=%s range=[%s,%s] drop=%s evidence=%s",
                self._context,
                record["count"],
                record["errors"],
                record["mean"],
                record["min"],
                record["max"],
                drop,
                record.get("evidence"),
            )
            self._batch = None

    def dump(self, *, reason, destination=None):
        with self._lock:
            if destination is None:
                if self._anomaly_dumps >= self.max_anomaly_dumps:
                    return None
                self._anomaly_dumps += 1
                destination = self.output_dir / f"input-anomaly-{os.getpid()}-{uuid.uuid4().hex}"
            destination = Path(destination)
            destination.mkdir(exist_ok=True)
            path = destination / "recent_requests.pt"
            torch.save(
                {
                    "reason": reason,
                    "context": dict(self._context),
                    "tensor_bytes": self._cached_bytes,
                    "max_bytes": self.max_bytes,
                    "requests": list(self._recent.values()),
                },
                path,
            )
            logger.warning("HPS actual-input evidence: %s (%s)", path, reason)
            return str(path)
