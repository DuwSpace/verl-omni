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

"""Branch gradient statistics preserve gradients and reduce FSDP shards globally."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from verl_omni.pipelines.ltx2_unigrpo_nft.training.update import branch_grad_norms


def _distributed_norm_worker(rank, rendezvous):
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=rendezvous)
    try:
        mesh = init_device_mesh("cpu", (2,))
        model = torch.nn.Module()
        model.gemma = torch.nn.Module()
        model.transformer = torch.nn.Module()
        for branch, name, values, placement in (
            (model.gemma, "sharded", [3.0, 4.0, 0.0, 0.0], Shard(0)),
            (model.gemma, "replicated", [0.0, 12.0], Replicate()),
            (model.transformer, "sharded", [0.0, 0.0, 5.0, 12.0], Shard(0)),
        ):
            gradient = distribute_tensor(torch.tensor(values), mesh, [placement])
            parameter = torch.nn.Parameter(torch.zeros_like(gradient))
            parameter.grad = gradient
            branch.register_parameter(name, parameter)
        before = [p.grad.to_local().clone() for p in model.parameters()]
        norms = branch_grad_norms(model, torch.device("cpu"))
        assert norms == pytest.approx({"text/grad_norm": 13.0, "dit/grad_norm": 13.0})
        for p, expected in zip(model.parameters(), before, strict=True):
            torch.testing.assert_close(p.grad.to_local(), expected)
    finally:
        dist.destroy_process_group()


def test_branch_norms_reduce_shards_without_double_counting_replicas(tmp_path):
    mp.spawn(_distributed_norm_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
