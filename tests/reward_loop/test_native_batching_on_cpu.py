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
"""Cross-request native batching, output routing and lifecycle ownership."""

import asyncio
import threading

import pytest
import torch

from verl_omni.reward_loop import reward_model_executor as executor_module
from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor
from verl_omni.workers.config.reward import NativeRewardModelExecutorConfig, RewardModelSpec


class Model:
    batch_input_names = ("a", "b")
    supports_cpu_offload = True

    def __init__(self, kind="video", **kwargs):
        self.kind = kind
        self.batch_output_dim = 1 if kind == "desync" else 0
        self.batch_output_key = "predictions" if kind == "audiobox" else None
        self.calls = []
        self.fail = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.asleep = False
        self.closed = False

    def infer(self, a, b):
        assert not self.asleep and not self.closed
        self.calls.append(len(a))
        self.entered.set()
        assert self.release.wait(timeout=5)
        if self.fail:
            raise RuntimeError("forward failed")
        x = torch.as_tensor(a, dtype=torch.float32).reshape(len(a), -1).sum(dim=1)
        y = torch.as_tensor(b, dtype=torch.float32).reshape(len(b), -1).sum(dim=1)
        value = torch.stack((x, y), dim=1)
        if self.kind == "desync":
            return torch.stack((value, value + 10), dim=0)
        if self.kind == "clap":
            return {"audio_embeds": value, "text_embeds": value + 10}
        if self.kind == "audiobox":
            return {"predictions": {"CE": x, "CU": y}, "target_transform": {"CE": (0.0, 1.0)}}
        return value

    def sleep(self):
        self.asleep = True

    def wake_up(self):
        self.asleep = False

    def close(self):
        self.closed = True


def make_executor(monkeypatch, kind="video", size=32, wait=10):
    monkeypatch.setattr(executor_module, "_load_native_model", lambda _: Model)
    return NativeRewardExecutor(
        RewardModelSpec(
            name="test",
            backend="native",
            device_type="cpu",
            executor_config={
                "model": "test:Model",
                "kwargs": {"kind": kind},
                "max_batch_size": size,
                "batch_wait_ms": wait,
            },
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["video", "hpsv3", "clap", "audiobox", "desync"])
@pytest.mark.parametrize("size", [16, 32])
async def test_batch_and_tail_match_individual_inference(monkeypatch, kind, size):
    executor = make_executor(monkeypatch, kind, size)
    await executor.wake_up()
    model = executor._model
    reference = Model(kind)

    # Lists retain per-sample inputs; audio tensors concatenate along dim zero.
    def inputs(index):
        values = ([[index, index + 1]], [[index + 2]])
        return tuple(torch.tensor(value) for value in values) if kind in ("audiobox", "desync") else values

    results = await asyncio.wait_for(asyncio.gather(*(executor.infer(*inputs(i)) for i in range(size + 3))), timeout=5)
    assert model.calls == [size, 3]
    for i, result in enumerate(results):
        torch.testing.assert_close(result, reference.infer(*inputs(i)))
    await executor.sleep()
    assert model.asleep
    await executor.wake_up()
    torch.testing.assert_close(await executor.infer(*inputs(9)), reference.infer(*inputs(9)))
    await executor.close()
    assert executor._batcher is None and model.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["video", "clap", "audiobox", "desync"])
async def test_oversized_request_is_chunked_and_reassembled(monkeypatch, kind):
    executor = make_executor(monkeypatch, kind)
    await executor.wake_up()
    a = torch.arange(70).reshape(-1, 1)
    b = a + 1
    result = await executor.infer(a=a, b=b)
    torch.testing.assert_close(result, Model(kind).infer(a, b))
    assert executor._model.calls == [32, 32, 6]
    await executor.close()


@pytest.mark.asyncio
async def test_unequal_request_sizes_preserve_order(monkeypatch):
    executor = make_executor(monkeypatch, size=4)
    await executor.wake_up()
    requests = [([i] * size, [i + 1] * size) for i, size in enumerate([3, 1, 2, 5, 1])]
    results = await asyncio.gather(*(executor.infer(*values) for values in requests))
    for values, result in zip(requests, results, strict=True):
        torch.testing.assert_close(result, Model().infer(*values))
    assert max(executor._model.calls) <= 4
    await executor.close()


@pytest.mark.asyncio
async def test_forward_failure_reaches_all_requests_and_queue_recovers(monkeypatch):
    executor = make_executor(monkeypatch, size=4)
    await executor.wake_up()
    executor._model.fail = True
    results = await asyncio.gather(*(executor.infer([i], [i]) for i in range(4)), return_exceptions=True)
    assert all(isinstance(result, RuntimeError) and "forward failed" in str(result) for result in results)
    assert executor._inflight == 0
    executor._model.fail = False
    torch.testing.assert_close(await executor.infer([2], [3]), torch.tensor([[2.0, 3.0]]))
    await executor.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["sleep", "close"])
async def test_cancelled_request_and_lifecycle_drain_accepted_batch(monkeypatch, transition):
    executor = make_executor(monkeypatch, size=2)
    await executor.wake_up()
    model = executor._model
    model.release.clear()
    requests = [asyncio.create_task(executor.infer([i], [i + 1])) for i in range(2)]
    try:
        for _ in range(1000):
            if model.entered.is_set():
                break
            await asyncio.sleep(0.001)
        assert model.entered.is_set()
        requests[0].cancel()
        lifecycle = asyncio.create_task(getattr(executor, transition)())
        await asyncio.sleep(0.01)
        assert not lifecycle.done()
        assert not model.asleep and not model.closed
    finally:
        model.release.set()
    results = await asyncio.gather(*requests, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    torch.testing.assert_close(results[1], torch.tensor([[1.0, 2.0]]))
    await lifecycle
    assert executor._inflight == 0
    await executor.close()


@pytest.mark.asyncio
async def test_invalid_input_does_not_poison_queue(monkeypatch):
    executor = make_executor(monkeypatch)
    await executor.wake_up()
    with pytest.raises(ValueError, match="matching, non-empty"):
        await executor.infer([1], [2, 3])
    assert executor._inflight == 0
    torch.testing.assert_close(await executor.infer([1], [2]), torch.tensor([[1.0, 2.0]]))
    await executor.close()


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_batch_size_rejected(size):
    with pytest.raises(ValueError):
        NativeRewardModelExecutorConfig(model="test:Model", max_batch_size=size)


@pytest.mark.parametrize("wait", [-1, True, float("nan"), float("inf"), "10"])
def test_invalid_wait_rejected(wait):
    with pytest.raises(ValueError):
        NativeRewardModelExecutorConfig(model="test:Model", batch_wait_ms=wait)


def test_recipe_batch_settings_reach_executor_spec(monkeypatch):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from verl_omni.reward_loop.reward_model import NativeManagedRewardModel

    root = Path(__file__).resolve().parents[2]
    for key in ("LTX_MODEL_PATH", "LTX_REWARD_ROOT", "LTX_OUTPUT_DIR"):
        monkeypatch.setenv(key, "/tmp/unused")
    with initialize_config_dir(config_dir=str(root / "examples/unigrpo_nft_trainer/ltx2"), version_base=None):
        cfg = compose(config_name="joint_train")
    for name, entry in cfg.reward.models.items():
        spec = NativeManagedRewardModel(name, entry).executor_spec
        config = NativeRewardModelExecutorConfig.from_mapping(name, spec.executor_config)
        assert config.max_batch_size == (16 if name == "video_align" else 32)
        assert config.batch_wait_ms == 10
        assert config.deterministic


def test_deterministic_mode_is_strict_and_restores_seed(monkeypatch):
    from unittest.mock import Mock

    from verl_omni.reward_loop import reward_reproducibility as module

    seeded = Mock()
    deterministic = Mock()
    monkeypatch.setattr(module.random, "seed", seeded)
    monkeypatch.setattr(module.np.random, "seed", seeded)
    monkeypatch.setattr(module.torch, "manual_seed", seeded)
    monkeypatch.setattr(module.torch, "use_deterministic_algorithms", deterministic)
    monkeypatch.setattr(module.torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(module.torch.backends.cudnn, "benchmark", True)
    for key in (
        "CUBLAS_WORKSPACE_CONFIG",
        "FLASH_ATTENTION_DETERMINISTIC",
        "HCCL_DETERMINISTIC",
        "LCCL_DETERMINISTIC",
        "CLOSE_MATMUL_K_SHIFT",
    ):
        monkeypatch.setenv(key, "initial")
    module.configure_reward_reproducibility(deterministic=False)
    deterministic.assert_not_called()
    module.configure_reward_reproducibility(deterministic=True)
    deterministic.assert_called_once_with(True, warn_only=False)
    assert seeded.call_count == 3
    assert all(call.args == (0,) for call in seeded.call_args_list)
    assert module.os.environ["CLOSE_MATMUL_K_SHIFT"] == "1"


@pytest.mark.asyncio
async def test_determinism_is_configured_before_model_construction(monkeypatch):
    calls = []
    executor = make_executor(monkeypatch)
    executor.spec.executor_config["deterministic"] = True
    monkeypatch.setattr(executor_module, "configure_reward_reproducibility", lambda **kwargs: calls.append(kwargs))

    def load_model(**kwargs):
        assert calls == [{"deterministic": True}]
        return Model(**kwargs)

    monkeypatch.setattr(executor_module, "_load_native_model", lambda _: load_model)
    await executor.wake_up()
    await executor.close()
