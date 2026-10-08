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

"""Frozen policy lifecycle and resume tests for the full-weight hybrid algorithm."""

import copy
import io

import pytest
import torch

from verl_omni.pipelines.bagel_unigrpo_nft.policy_state import BagelPolicyState


def _module():
    module = torch.nn.Linear(3, 2, bias=False)
    module.weight.data.fill_(1)
    return module


def test_reference_is_fixed_old_is_frozen_and_live_weights_are_restored():
    module = _module()
    policies = BagelPolicyState(module)
    module.weight.data.fill_(2)
    policies.begin_rollout(version=1)
    module.weight.data.fill_(3)
    with policies.use("old"):
        torch.testing.assert_close(module.weight, torch.full((2, 3), 2.0))
    with policies.use("reference"):
        torch.testing.assert_close(module.weight, torch.ones(2, 3))
    torch.testing.assert_close(module.weight, torch.full((2, 3), 3.0))


def test_exception_during_frozen_forward_restores_current_weights():
    module = _module()
    policies = BagelPolicyState(module)
    module.weight.data.fill_(3)
    with pytest.raises(RuntimeError, match="forward failed"), policies.use("reference"):
        raise RuntimeError("forward failed")
    torch.testing.assert_close(module.weight, torch.full((2, 3), 3.0))


def test_checkpoint_roundtrip_keeps_initial_reference_not_resume_weights():
    module = _module()
    policies = BagelPolicyState(module)
    module.weight.data.fill_(2)
    policies.begin_rollout(version=4)
    module.weight.data.fill_(3)
    buffer = io.BytesIO()
    torch.save(policies.state_dict(), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    restored = BagelPolicyState(module)
    restored.load_state_dict(checkpoint)
    assert restored.version == 4
    with restored.use("reference"):
        torch.testing.assert_close(module.weight, torch.ones(2, 3))
    with restored.use("old"):
        torch.testing.assert_close(module.weight, torch.full((2, 3), 2.0))
    torch.testing.assert_close(module.weight, torch.full((2, 3), 3.0))


@pytest.mark.parametrize("key,value", [("world_size", 8), ("rank", 3), ("schema_version", 2), ("policy_version", -1)])
def test_checkpoint_rejects_incompatible_or_invalid_metadata(key, value):
    policies = BagelPolicyState(_module())
    checkpoint = copy.deepcopy(policies.state_dict())
    checkpoint[key] = value
    with pytest.raises(ValueError):
        policies.load_state_dict(checkpoint)


@pytest.mark.parametrize("mode", ["missing", "shape", "nonfinite"])
def test_checkpoint_validates_all_policy_tensors_before_mutation(mode):
    policies = BagelPolicyState(_module())
    checkpoint = copy.deepcopy(policies.state_dict())
    if mode == "missing":
        del checkpoint["old"]["weight"]
    elif mode == "shape":
        checkpoint["reference"]["weight"] = torch.zeros(1)
    else:
        checkpoint["reference"]["weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        policies.load_state_dict(checkpoint)
    torch.testing.assert_close(policies.reference["weight"], torch.ones(2, 3))


def test_policy_versions_cannot_go_backwards():
    policies = BagelPolicyState(_module())
    policies.begin_rollout(version=3)
    with pytest.raises(ValueError, match="backwards"):
        policies.begin_rollout(version=2)
