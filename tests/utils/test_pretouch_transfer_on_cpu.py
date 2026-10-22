# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Destination initialization, local DTensor copies, and model alias handling."""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from verl_omni.utils.transfer_sync import copy_tensor_to_device, move_module


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
@pytest.mark.parametrize("shape", [(), (0,), (7, 9)])
def test_copy_writes_preinitialized_independent_storage(monkeypatch, dtype, shape):
    source = torch.full(shape, 7, dtype=dtype)
    original_copy = torch.Tensor.copy_
    destinations = []

    def checked_copy(destination, value, *, non_blocking=False):
        assert non_blocking is False
        assert destination.shape == source.shape and destination.dtype == source.dtype
        assert torch.count_nonzero(destination).item() == 0
        destinations.append(destination)
        return original_copy(destination, value, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "copy_", checked_copy)
    with torch.inference_mode():
        result = copy_tensor_to_device(source, "cpu")
    assert len(destinations) == 1 and result is destinations[0]
    assert not torch.is_inference(result)
    torch.testing.assert_close(result, source, rtol=0, atol=0)
    result.fill_(3)
    torch.testing.assert_close(source, torch.full(shape, 7, dtype=dtype), rtol=0, atol=0)


def test_noncontiguous_copy_preserves_values():
    source = torch.arange(24).reshape(4, 6).t()
    assert not source.is_contiguous()
    torch.testing.assert_close(copy_tensor_to_device(source, "cpu"), source, rtol=0, atol=0)


@pytest.fixture
def cpu_mesh(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("placement", [Replicate(), Shard(0)])
def test_dtensor_moves_local_storage_without_redistribution(cpu_mesh, placement, monkeypatch):
    source = DTensor.from_local(torch.arange(12).reshape(3, 4), cpu_mesh, [placement])

    def no_gather(*args, **kwargs):
        raise AssertionError("Offload must not gather or redistribute the global tensor")

    monkeypatch.setattr(DTensor, "full_tensor", no_gather)
    monkeypatch.setattr(DTensor, "redistribute", no_gather)
    copied = copy_tensor_to_device(source, "cpu")
    assert isinstance(copied, DTensor)
    assert copied._spec is source._spec
    assert copied.to_local().data_ptr() != source.to_local().data_ptr()
    torch.testing.assert_close(copied.to_local(), source.to_local(), rtol=0, atol=0)


def test_module_transfer_preserves_tied_parameters_and_buffer_aliases(monkeypatch):
    # meta exercises a real device transition on CPU-only CI. Track the actual
    # destination objects to check aliases; value copies are covered above.
    model = torch.nn.Module()
    model.a = torch.nn.Linear(3, 2, bias=False)
    model.b = torch.nn.Linear(3, 2, bias=False)
    model.b.weight = model.a.weight
    model.register_buffer("first", torch.ones(3))
    model.register_buffer("second", model.first)
    from contextlib import nullcontext

    import verl_omni.utils.transfer_sync as transfer

    monkeypatch.setattr(transfer, "synchronized_device_transfer", lambda *devices: nullcontext())
    move_module(model, "meta")
    assert model.a.weight is model.b.weight
    assert model.first is model.second
    assert all(value.device.type == "meta" for value in model.parameters())


def test_failed_copy_is_propagated(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(torch.Tensor, "copy_", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        copy_tensor_to_device(torch.ones(2), "cpu")
