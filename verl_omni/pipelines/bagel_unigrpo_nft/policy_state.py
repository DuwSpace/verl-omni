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

"""Durable full-weight old/reference policies over rank-local FSDP2 shards."""

from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class BagelPolicyState:
    """Keep the initial reference fixed and old policy fixed throughout an update batch.

    Snapshots are FP32 CPU rank-local tensors, not replicated full models.
    Checkpoints include both policies, version, and sharding layout. A different
    world size fails closed rather than silently constructing a new reference.
    """

    def __init__(self, module):
        self.module = module
        self.version = 0
        self.reference = self._snapshot()
        self.old = {name: tensor.clone() for name, tensor in self.reference.items()}

    def _reshard(self):
        for module in self.module.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

    def _parameters(self):
        return {name: parameter for name, parameter in self.module.named_parameters() if parameter.requires_grad}

    def _snapshot(self):
        self._reshard()
        parameters = self._parameters()
        if not parameters:
            raise ValueError("Joint policy state requires trainable parameters")
        return {
            name: _local(parameter).detach().to(device="cpu", dtype=torch.float32).clone()
            for name, parameter in parameters.items()
        }

    def begin_rollout(self, *, version: int):
        """Copy current shards to old once, before syncing that policy to vllm-omni."""
        if version < self.version:
            raise ValueError("Old-policy versions must not go backwards")
        self.old = self._snapshot()
        self.version = int(version)

    @contextmanager
    def use(self, name: str):
        """Temporarily select frozen old/reference shards; restore on every exit path."""
        if name not in {"old", "reference"}:
            raise ValueError("Policy must be old or reference")
        snapshot = getattr(self, name)
        current = self._snapshot()
        parameters = self._parameters()
        try:
            with torch.no_grad():
                for key, parameter in parameters.items():
                    _local(parameter).copy_(snapshot[key].to(device=_local(parameter).device, dtype=parameter.dtype))
            yield
        finally:
            self._reshard()
            with torch.no_grad():
                for key, parameter in parameters.items():
                    _local(parameter).copy_(current[key].to(device=_local(parameter).device, dtype=parameter.dtype))

    def _layout(self):
        self._reshard()
        return {
            name: {
                "global_shape": tuple(parameter.shape),
                "local_shape": tuple(_local(parameter).shape),
                "placements": tuple(str(p) for p in parameter.placements) if isinstance(parameter, DTensor) else (),
            }
            for name, parameter in self._parameters().items()
        }

    def state_dict(self):
        """Return both frozen policies and enough metadata to reject incompatible resume."""
        return {
            "schema_version": 1,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "rank": dist.get_rank() if dist.is_initialized() else 0,
            "layout": self._layout(),
            "policy_version": self.version,
            "reference": self.reference,
            "old": self.old,
        }

    def load_state_dict(self, state):
        """Restore policies without resetting reference to the resumed trainable weights."""
        expected = self.state_dict()
        for key in ("schema_version", "world_size", "rank", "layout"):
            if state.get(key) != expected[key]:
                raise ValueError(f"Incompatible BAGEL policy checkpoint: {key}")
        if not isinstance(state.get("policy_version"), int) or state["policy_version"] < 0:
            raise ValueError("Invalid policy checkpoint version")
        validated = {}
        for name in ("old", "reference"):
            snapshot = state.get(name)
            if not isinstance(snapshot, dict) or set(snapshot) != set(self.reference):
                raise ValueError(f"Missing/extra named parameters in {name} policy checkpoint")
            for key, tensor in snapshot.items():
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.dtype != torch.float32
                    or tuple(tensor.shape) != expected["layout"][key]["local_shape"]
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise ValueError(f"Invalid {name} policy tensor: {key}")
            validated[name] = {key: tensor.detach().cpu().clone() for key, tensor in snapshot.items()}
        self.reference, self.old = validated["reference"], validated["old"]
        self.version = state["policy_version"]
