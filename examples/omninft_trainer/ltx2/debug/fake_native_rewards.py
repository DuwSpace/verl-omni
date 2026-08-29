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

"""Deterministic fake Native Reward hooks for OmniNFT trainer wiring."""

import torch


def initialize(name, offset=0.0, model_revision="fake-model-v1", definition_version="fake-def-v1", **kwargs):
    """Create CPU-only state for one named fake reward."""
    del kwargs
    return {
        "name": name,
        "offset": float(offset),
        "model_revision": model_revision,
        "definition_version": definition_version,
    }


def activate(state, device):
    """Record the runtime device without allocating a model."""
    state["device"] = device


def score_batch(state, batch, micro_batch_size, **kwargs):
    """Return finite sample-aligned scores shaped `[B]`."""
    del micro_batch_size, kwargs
    if "device" not in state:
        raise RuntimeError(f"fake reward {state['name']!r} is not active")
    batch_size = len(batch)
    return {
        "scores": torch.arange(batch_size, dtype=torch.float32) + state["offset"],
        "valid_mask": torch.ones(batch_size, dtype=torch.bool),
        "metrics": {"batch_size": batch_size, "name": state["name"]},
        "model_revision": state["model_revision"],
        "definition_version": state["definition_version"],
    }


def deactivate(state):
    """Drop the runtime device reference."""
    state.pop("device", None)


def finalize(state):
    """Drop leftover fake state."""
    state.clear()
