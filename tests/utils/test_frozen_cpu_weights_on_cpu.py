# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from types import SimpleNamespace

import pytest
import torch

from verl_omni.utils.frozen_cpu_weights import FrozenCpuWeights, first_different_byte_block


def test_discard_corrupt_working_copy_preserves_master_and_new_policy():
    from peft import LoraConfig, inject_adapter_in_model

    from verl_omni.pipelines.ltx2_unigrpo_nft.offload import JointRolloutOffload

    model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False)).requires_grad_(False)
    expected = model[0].weight.detach().clone()
    original_parameter = model[0].weight
    master = FrozenCpuWeights(model)
    model._joint_cpu_weights = master
    inject_adapter_in_model(LoraConfig(r=1, target_modules=["0"]), model, adapter_name="joint")
    model.requires_grad_(False)
    assert model[0].base_layer.weight is original_parameter
    lora = model[0].lora_B["joint"].weight
    with torch.no_grad():
        lora.fill_(7)
        # Separate CPU storage simulates a device allocation without an NPU.
        original_parameter.data = original_parameter.detach().clone()
        original_parameter.zero_()
    torch.testing.assert_close(master.parameters["0.weight"][1], expected, rtol=0, atol=0)
    JointRolloutOffload._move(model, torch.device("cpu"))
    torch.testing.assert_close(model[0].base_layer.weight, expected, rtol=0, atol=0)
    torch.testing.assert_close(lora, torch.full_like(lora, 7), rtol=0, atol=0)
    assert original_parameter.data_ptr() == master.parameters["0.weight"][1].data_ptr()


@pytest.mark.parametrize("damage_master", [False, True])
def test_corruption_stops_and_saves_only_on_failure(tmp_path, damage_master):
    model = torch.nn.Linear(3, 2, bias=False).requires_grad_(False)
    with torch.no_grad():
        model.weight.fill_(2)
    master = FrozenCpuWeights(model, failure_dir=tmp_path)
    master.move(model, "cpu")
    assert not list(tmp_path.iterdir())
    if not damage_master:
        model.weight.data = model.weight.detach().clone()
    with torch.no_grad():
        model.weight[0, 0] = 0
    with pytest.raises(RuntimeError, match="integrity failure"):
        master.check("before_activation", master=damage_master)
    report = json.loads(next(tmp_path.glob("failure-*/failure.json")).read_text())
    assert report["master"] is damage_master
    assert report["expected_sha256"] != report["observed_sha256"]
    assert next(tmp_path.glob("failure-*/observed.pt")).exists()
    if not damage_master:
        torch.testing.assert_close(master.parameters["weight"][1], torch.full((2, 3), 2.0), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="integrity failure"):
        master.move(model, "cpu")
    assert len(list(tmp_path.iterdir())) == 1


def test_tied_parameters_dtypes_dynamic_buffers_and_inference_mode():
    model = torch.nn.Module()
    model.a = torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16)
    model.b = torch.nn.Linear(2, 2, bias=False)
    model.b.weight = model.a.weight
    model.requires_grad_(False)
    model.register_buffer("cache", torch.ones(1), persistent=False)
    original = model.a.weight.detach().clone()
    master = FrozenCpuWeights(model)
    model.a.weight.data = original.clone()
    model.cache = torch.ones(3)
    with torch.inference_mode():
        master.move(model, "cpu")
    assert model.a.weight is model.b.weight
    assert model.a.weight.dtype == torch.bfloat16
    assert model.cache.shape == (3,)
    torch.testing.assert_close(model.a.weight, original, rtol=0, atol=0)


def test_capture_requires_cpu():
    with pytest.raises(ValueError, match="before accelerator"):
        FrozenCpuWeights(torch.nn.Linear(2, 2, device="meta").requires_grad_(False))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64])
def test_byte_comparison_covers_all_chunks_and_detects_small_corruption(dtype):
    reference = torch.arange(40, dtype=dtype).reshape(5, 8)
    actual = reference.clone()
    assert first_different_byte_block(actual, reference, chunk_bytes=16) is None
    actual.view(-1)[-1] = 0
    assert first_different_byte_block(actual, reference, chunk_bytes=16) is not None


def test_byte_comparison_checks_bits_not_float_equality():
    reference = torch.tensor([float("nan"), 0.0])
    actual = reference.clone()
    assert first_different_byte_block(actual, reference) is None
    actual[1] = -0.0
    assert first_different_byte_block(actual, reference) == 0


def test_all_parameters_are_checked_before_activation(tmp_path):
    model = torch.nn.Linear(3, 2).requires_grad_(False)
    master = FrozenCpuWeights(model, failure_dir=tmp_path)
    model.bias.data = model.bias.detach().clone()
    with torch.no_grad():
        model.bias[0] += 1
    with pytest.raises(RuntimeError, match="bias"):
        master.check("after_transfer")


def test_zeroed_status_transfer_cannot_pass(monkeypatch):
    original = torch.Tensor.to

    def zero_status(tensor, *args, **kwargs):
        result = original(tensor, *args, **kwargs)
        if tensor.dtype == torch.bool and tensor.numel() == 2:
            result.zero_()
        return result

    monkeypatch.setattr(torch.Tensor, "to", zero_status)
    value = torch.arange(8, dtype=torch.float32)
    assert first_different_byte_block(value.clone(), value) == 0


def test_persistent_buffers_use_masters_but_dynamic_caches_are_not_reset(tmp_path):
    model = torch.nn.BatchNorm1d(2).eval().requires_grad_(False)
    model.register_buffer("cache", torch.ones(1), persistent=False)
    master = FrozenCpuWeights(model, failure_dir=tmp_path)
    original = model.running_var.clone()
    model.running_var = torch.zeros_like(original)
    model.cache = torch.ones(4)
    master.restore_host()
    torch.testing.assert_close(model.running_var, original, rtol=0, atol=0)
    assert model.cache.shape == (4,)
    model.running_var = torch.zeros_like(original)
    with pytest.raises(RuntimeError, match="running_var"):
        master.check("after_transfer")


def test_hps_captures_before_device_transfer_and_discards_working_copy(monkeypatch):
    from verl_omni.utils.reward_score import hpsv3_reward

    model = torch.nn.Linear(2, 2).requires_grad_(False)
    expected = model.weight.detach().clone()
    inferencer = SimpleNamespace(model=model, device="cpu")

    def load(**kwargs):
        assert kwargs["device"] == "cpu"
        return inferencer

    monkeypatch.setattr(hpsv3_reward, "_HPSv3Inferencer", load)
    wrapper = hpsv3_reward.HPSv3Model("fake", "cpu")
    model.weight.data = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.zero_()
    wrapper.to("cpu")
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
    wrapper.close()
    assert wrapper._cpu_weights is None and wrapper._inferencer is None


def _mapped_test_model():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.full((1024, 1024), 2.0), requires_grad=False)
    model.tied = model.weight
    model.register_buffer("view", model.weight.detach().t(), persistent=True)
    model.register_buffer("cache", torch.ones(2), persistent=False)
    return model


def _mapped_child(path, pipe):
    # Spawn a truly independent process: no inherited Python tensors or mapping.
    from pathlib import Path

    torch.set_num_threads(1)
    model = _mapped_test_model()
    master = FrozenCpuWeights(model, shared_path=path)
    mapped = str(Path(path).resolve())
    maps = [line for line in Path("/proc/self/maps").read_text().splitlines() if mapped in line]
    pipe.send((float(model.weight.sum()), maps))
    assert pipe.recv() == "change"
    # A CPU write is private to this process, while unmodified pages are shared.
    with torch.no_grad():
        model.weight[0, 0] = 17
    assert model.view[0, 0].item() == 17
    assert model.weight is model.tied
    pipe.send(model.weight[0, 0].item())
    assert pipe.recv() == "close"
    del master, model
    pipe.close()


def test_independent_processes_map_same_inode_and_isolate_writes(tmp_path):
    import multiprocessing
    from pathlib import Path

    torch.set_num_threads(1)
    path = tmp_path / "gemma.pt"
    model = _mapped_test_model()
    master = FrozenCpuWeights(model, shared_path=path)
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    worker = ctx.Process(target=_mapped_child, args=(str(path), child))
    worker.start()
    child.close()
    try:
        assert parent.poll(60), "Shared-weight child did not initialize"
        total, mappings = parent.recv()
        assert total == 2 * 1024 * 1024
        inode = str(path.stat().st_ino)
        assert mappings and all(line.split()[4] == inode for line in mappings)
        own_maps = [line for line in Path("/proc/self/maps").read_text().splitlines() if str(path) in line]
        assert own_maps and all(line.split()[4] == inode for line in own_maps)
        # Both processes have read the same file-backed pages. Linux reports
        # physical sharing (whether the pages are currently clean or dirty).
        smaps = Path("/proc/self/smaps").read_text().splitlines()
        active = False
        shared_kib = 0
        for line in smaps:
            if "-" in line.split()[0]:
                active = str(path) in line
            elif active and line.startswith(("Shared_Clean:", "Shared_Dirty:")):
                shared_kib += int(line.split()[1])
        assert shared_kib >= 1024
        parent.send("change")
        assert parent.poll(30)
        assert parent.recv() == 17
        assert model.weight[0, 0].item() == 2
        master.check("other_process_write", master=True)
        parent.send("close")
        worker.join(30)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(10)
        parent.close()


def test_shared_cache_preserves_aliases_and_leaves_adapter_private(tmp_path):
    from peft import LoraConfig, inject_adapter_in_model

    path = tmp_path / "base.pt"
    model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False)).requires_grad_(False)
    model[0].weight.data.fill_(2)
    original = model[0].weight
    master = FrozenCpuWeights(model, shared_path=path)
    inject_adapter_in_model(LoraConfig(r=1, target_modules=["0"]), model, adapter_name="joint")
    adapter = model[0].lora_B["joint"].weight
    with torch.no_grad():
        adapter.fill_(9)
    original.data = torch.zeros_like(original)
    master.restore_host()
    assert model[0].base_layer.weight is original
    torch.testing.assert_close(original, torch.full_like(original, 2), rtol=0, atol=0)
    torch.testing.assert_close(adapter, torch.full_like(adapter, 9), rtol=0, atol=0)
    assert set(master.parameters) == {"0.weight"}
    # Same run/model opens the cache without writing another copy.
    second = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False)).requires_grad_(False)
    second[0].weight.data.fill_(2)
    modified = path.stat().st_mtime_ns
    FrozenCpuWeights(second, shared_path=path)
    assert path.stat().st_mtime_ns == modified


def test_mapped_views_dynamic_buffers_and_offload_restore(tmp_path):
    model = _mapped_test_model()
    with torch.inference_mode():
        master = FrozenCpuWeights(model, shared_path=tmp_path / "base.pt")
    assert model.weight is model.tied
    assert model.weight.untyped_storage().data_ptr() == model.view.untyped_storage().data_ptr()
    assert model.view.stride() == (1, 1024)
    model.cache = torch.ones(7)
    model.weight.data = model.weight.detach().clone().zero_()
    master.restore_host()
    assert model.cache.shape == (7,)
    assert model.weight.untyped_storage().data_ptr() == model.view.untyped_storage().data_ptr()
    master.check("restored", master=True)


def test_shared_cache_rejects_another_checkpoint_and_corruption(tmp_path):
    path = tmp_path / "base.pt"
    model = torch.nn.Linear(3, 2, bias=False).requires_grad_(False)
    model.weight.data.fill_(2)
    FrozenCpuWeights(model, shared_path=path)
    other = torch.nn.Linear(3, 2, bias=False).requires_grad_(False)
    other.weight.data.fill_(3)
    with pytest.raises(RuntimeError, match="does not match"):
        FrozenCpuWeights(other, shared_path=path)
    # Keep the metadata intact but corrupt the actual serialized payload.
    payload = torch.load(path, weights_only=True)
    payload["tensors"]["weight"].zero_()
    path.unlink()
    torch.save(payload, path)
    other.weight.data.fill_(2)
    with pytest.raises(RuntimeError, match="Corrupt shared"):
        FrozenCpuWeights(other, shared_path=path)


def test_partial_shared_cache_is_not_published_and_can_retry(tmp_path, monkeypatch):
    path = tmp_path / "base.pt"
    model = torch.nn.Linear(3, 2, bias=False).requires_grad_(False)
    save = torch.save

    def fail_save(payload, stream):
        stream.write(b"partial")
        raise OSError("injected write failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="injected"):
        FrozenCpuWeights(model, shared_path=path)
    assert not path.exists()
    assert not list(tmp_path.glob(".base.pt.*"))
    monkeypatch.setattr(torch, "save", save)
    master = FrozenCpuWeights(model, shared_path=path)
    master.check("retry", master=True)
