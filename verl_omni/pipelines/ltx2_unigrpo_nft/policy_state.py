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

"""Paired Gemma/DiT LoRA snapshots with immutable rollout versions."""

import hashlib
from contextlib import contextmanager

import torch

from verl_omni.pipelines.trainable_policy_state import TrainablePolicyState


class LTXJointPolicyState(TrainablePolicyState):
    """Snapshot both default adapters together; ref retains the initial LoRA state.

    Old/ref live as rank-local CPU snapshots, not PEFT adapter names. The active
    module always uses its default adapters. Frozen forwards temporarily install
    one paired snapshot under no_grad, then restore current parameters, including
    on exceptions. Callers must finish all frozen forwards before building a
    current autograd graph; swapping live parameters between forward/backward
    would invalidate saved tensors.
    """

    def __init__(self, module):
        self.rollout_started = False
        super().__init__(module)
        self._frozen_digests = self._snapshot_digests()

    def _snapshot_digests(self):
        digests = {}
        for policy in ("old", "reference"):
            digest = hashlib.sha256()
            for name, tensor in sorted(getattr(self, policy).items()):
                digest.update(repr((name, tuple(tensor.shape), str(tensor.dtype))).encode())
                digest.update(tensor.contiguous().numpy().tobytes())
            digests[policy] = digest.hexdigest()
        return digests

    def assert_frozen(self):
        """Check rank-local CPU snapshots across accumulation and optimizer steps."""
        for name, digest in self._snapshot_digests().items():
            if digest != self._frozen_digests[name]:
                raise RuntimeError(f"Frozen joint {name} policy was modified during rollout/update")

    def _parameters(self):
        parameters = super()._parameters()
        for name in parameters:
            if not name.startswith(("gemma.model.language_model.", "transformer.")) or not any(
                marker in name for marker in (".lora_A.default.", ".lora_B.default.")
            ):
                raise ValueError(f"Joint policy may train only default Gemma/DiT LoRA matrices: {name}")
        if not all(any(name.startswith(prefix) for name in parameters) for prefix in ("gemma.", "transformer.")):
            raise ValueError("Joint policy snapshot requires both Gemma and DiT LoRA")
        return parameters

    def begin_rollout(self, *, version):
        self.assert_frozen()
        if type(version) is not int or version < 0 or (self.rollout_started and version <= self.version):
            raise ValueError("Each joint rollout requires a new strictly increasing nonnegative integer version")
        super().begin_rollout(version=version)
        self.rollout_started = True
        self._frozen_digests = self._snapshot_digests()

    @contextmanager
    def use(self, name):
        if name not in {"old", "ref"}:
            raise ValueError("Frozen joint policy must be old or ref")
        if name == "old" and not self.rollout_started:
            raise ValueError("Freeze a rollout version before using old")
        with torch.no_grad(), super().use("reference" if name == "ref" else "old"):
            yield

    def state_dict(self):
        self.assert_frozen()
        state = super().state_dict()
        # Return independent snapshots, so serialization/export cannot mutate a
        # policy that must remain fixed across every mini-batch in this rollout.
        for policy in ("old", "reference"):
            state[policy] = {name: value.clone() for name, value in state[policy].items()}
        state["joint_policy_kind"] = "ltx_gemma_dit_lora"
        state["rollout_started"] = self.rollout_started
        return state

    def load_state_dict(self, state):
        if state.get("joint_policy_kind") != "ltx_gemma_dit_lora" or type(state.get("rollout_started")) is not bool:
            raise ValueError("Checkpoint does not contain a complete paired LTX policy")
        if type(state.get("policy_version")) is not int:
            raise ValueError("Joint checkpoint policy version must be an integer")
        super().load_state_dict(state)
        self.rollout_started = state["rollout_started"]
        self._frozen_digests = self._snapshot_digests()
