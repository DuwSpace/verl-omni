# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Atomic policy manifest/version admission; no real rollout model or devices."""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.workers.rollout.bagel_joint_worker_extension import BagelJointWorkerExtension


class _Pipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bagel = SimpleNamespace()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def load_weights(self, weights):
        for name, tensor in weights:
            assert tensor.device.type == "cpu"
            self._joint_loaded_actor_keys.add(name)
        return {"loaded"}


def _worker():
    worker = BagelJointWorkerExtension()
    worker.model_runner = SimpleNamespace(model=_Pipeline())
    worker.device = torch.device("cpu")
    return worker


def test_sync_requires_all_declared_tensor_buckets_before_version_commit(tmp_path):
    worker = _worker()
    keys = ["transformer.vae2llm.weight", "transformer.llm2vae.weight"]
    worker.begin_joint_policy_sync(3, keys)
    bucket = tmp_path / "weights.pt"
    torch.save({keys[0]: torch.ones(1)}, bucket)
    worker.load_joint_weight_file(str(bucket))
    with pytest.raises(RuntimeError, match="missing tensors"):
        worker.commit_joint_policy_sync()
    assert worker.model_runner.model._joint_sync_in_progress
    assert worker.joint_model_info()["policy_version"] == 0
    torch.save({keys[1]: torch.ones(1)}, bucket)
    worker.load_joint_weight_file(str(bucket))
    assert worker.commit_joint_policy_sync() == {"version": 3, "loaded_tensors": 2}
    assert worker.joint_model_info()["policy_version"] == 3


def test_undeclared_tensor_does_not_count_as_successful_policy_sync(tmp_path):
    worker = _worker()
    worker.begin_joint_policy_sync(1, ["transformer.vae2llm.weight"])
    bucket = tmp_path / "weights.pt"
    torch.save({"transformer.llm2vae.weight": torch.ones(1)}, bucket)
    with pytest.raises(ValueError, match="Undeclared"):
        worker.load_joint_weight_file(str(bucket))
    assert not worker.model_runner.model._joint_loaded_actor_keys


@pytest.mark.parametrize("keys", [[], ["a", "a"]])
def test_invalid_manifest_is_rejected_before_invalidation(keys):
    worker = _worker()
    with pytest.raises(ValueError, match="manifest"):
        worker.begin_joint_policy_sync(1, keys)
    assert not getattr(worker.model_runner.model, "_joint_sync_in_progress", False)


def test_cannot_load_or_commit_without_starting_sync():
    worker = _worker()
    with pytest.raises(RuntimeError, match="Begin"):
        worker.load_joint_weight_file("unused.pt")
    with pytest.raises(RuntimeError, match="No joint"):
        worker.commit_joint_policy_sync()


def test_reentrant_and_backwards_version_sync_are_rejected():
    worker = _worker()
    worker.begin_joint_policy_sync(3, ["a"])
    with pytest.raises(RuntimeError, match="already"):
        worker.begin_joint_policy_sync(4, ["a"])
    worker.model_runner.model._joint_loaded_actor_keys.add("a")
    worker.commit_joint_policy_sync()
    with pytest.raises(ValueError, match="older"):
        worker.begin_joint_policy_sync(2, ["a"])
