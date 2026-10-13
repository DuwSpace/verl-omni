# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real two-rank DTensor export, paired publication and collective failures."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig
from safetensors.torch import load_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from verl_omni.pipelines.ltx2_unigrpo_nft.weight_sync import export_joint_actor_lora, validate_joint_lora_bundle


def _export_worker(rank, root):
    root = Path(root)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=f"file://{root / 'rendezvous'}", timeout=timedelta(seconds=90)
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        weights = {}
        expected = {}
        for prefix in ("gemma.model.language_model.layers.0.self_attn.q_proj", "transformer.attn.to_q"):
            for letter, shape in (("A", (2, 4)), ("B", (4, 2))):
                name = f"{prefix}.lora_{letter}.default.weight"
                value = torch.arange(8, dtype=torch.float32).reshape(shape) + 1.0001
                expected[name] = value
                weights[name] = torch.nn.Parameter(distribute_tensor(value, mesh, [Shard(0)]))
        # Frozen base weights must not enter the bundle or a collective gather.
        weights["frozen_base.weight"] = torch.nn.Parameter(torch.ones(1024), requires_grad=False)
        module = SimpleNamespace(named_parameters=lambda: weights.items(), peft_config={"default": LoraConfig(r=2)})
        destination = root / "policy-4"
        returned = export_joint_actor_lora(module, destination, version=4, base_model="fixture", device="cpu")
        assert returned == destination
        assert validate_joint_lora_bundle(destination)["policy_version"] == 4
        for component in ("gemma", "dit"):
            saved = load_file(destination / component / "adapter_model.safetensors")
            assert len(saved) == 2
            for name, value in expected.items():
                if component == "gemma" and name.startswith("gemma."):
                    key = name.removeprefix("gemma.").replace(".default.weight", ".weight")
                elif component == "dit" and name.startswith("transformer."):
                    key = "base_model.model." + name.replace(".default.weight", ".weight")
                else:
                    continue
                torch.testing.assert_close(saved[key], value, rtol=0, atol=0)
        # Root-only publication failures must be communicated to both ranks.
        with pytest.raises(RuntimeError, match="export failed"):
            export_joint_actor_lora(module, destination, version=4, base_model="fixture", device="cpu")
        with pytest.raises(ValueError, match="disagree"):
            export_joint_actor_lora(module, root / "mismatch", version=5 + rank, base_model="fixture", device="cpu")
    finally:
        dist.destroy_process_group()


def test_real_sharded_joint_lora_export_and_collective_failures(tmp_path):
    torch.multiprocessing.spawn(_export_worker, args=(str(tmp_path),), nprocs=2, join=True)


def test_actor_export_commits_pair_before_releasing_previous_files(monkeypatch):
    from verl.utils import device as device_utils

    from verl_omni.workers.rollout.ltx_joint_backend import LTXJointVllmBackend

    monkeypatch.setattr(device_utils, "get_device_name", lambda: "cpu")
    parameters = {}
    for prefix in ("gemma.model.language_model.layers.0.self_attn.q_proj", "transformer.attn.to_q"):
        for letter, shape in (("A", (2, 4)), ("B", (4, 2))):
            parameters[f"{prefix}.lora_{letter}.default.weight"] = torch.nn.Parameter(torch.ones(shape))
    module = SimpleNamespace(named_parameters=lambda: parameters.items(), peft_config={"default": LoraConfig(r=2)})
    loaded = []

    def rpc(method, args, timeout):
        assert method == "load_joint_lora_bundle"
        path = Path(args[0])
        manifest = validate_joint_lora_bundle(path)
        # The old pair must still exist while the new pair is being committed.
        if loaded:
            assert loaded[-1].exists()
        loaded.append(path)
        return [{"policy_version": manifest["policy_version"], "components": manifest["components"]}]

    backend = object.__new__(LTXJointVllmBackend)
    backend.engine = SimpleNamespace(od_config=SimpleNamespace(model="fixture"), collective_rpc=rpc, close=lambda: None)
    backend.timeout = 10
    backend._actor_policy_root = backend._coordination_group = None
    backend.policy_version = backend.lora_request = None
    try:
        backend.load_actor_policy(module, version=0)
        assert backend.policy_version == 0
        backend.load_actor_policy(module, version=1)
        assert backend.policy_version == 1 and not loaded[0].exists() and loaded[1].exists()

        def failure(*args, **kwargs):
            raise RuntimeError("injected worker failure")

        backend.engine.collective_rpc = failure
        with pytest.raises(RuntimeError, match="not committed"):
            backend.load_actor_policy(module, version=2)
        assert backend.policy_version is None and loaded[1].exists()
    finally:
        root = backend._actor_policy_root
        backend.close()
        assert not root.exists()
