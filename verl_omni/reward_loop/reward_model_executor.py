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
"""Worker-side executors for named reward models."""

from __future__ import annotations

import asyncio
import gc
import importlib
import inspect
from enum import Enum
from typing import Any

import torch
from verl.utils.device import get_device_id, get_device_name

from verl_omni.workers.config.reward import RewardModelSpec, is_engine_backend

from .async_utils import gather_complete

__all__ = [
    "EngineRewardExecutor",
    "NativeRewardExecutor",
    "NativeRewardModelState",
    "build_engine_reward_executors",
    "build_native_reward_executors",
]


class EngineRouterClient:
    """Expose one engine router to a configured reward function."""

    def __init__(self, router_address: str, model_path: str):
        self.router_address = router_address
        self.model_path = model_path

    def reward_kwargs(self) -> dict[str, str]:
        return {
            "reward_router_address": self.router_address,
            "model_name": self.model_path,
        }


class EngineRewardExecutor:
    """Worker-side request contract for one engine-backed reward model."""

    def __init__(self, spec: RewardModelSpec):
        self.spec = spec
        self._client = EngineRouterClient(
            router_address=spec.router_address,
            model_path=spec.model_path,
        )

    def reward_kwargs(self) -> dict[str, str]:
        return self._client.reward_kwargs()


class NativeRewardModelState(str, Enum):
    """Explicit placement state for one worker-local native model."""

    CLOSED = "closed"
    CPU = "cpu"
    DEVICE = "device"


class NativeRewardExecutor:
    """Own native model placement and inference, but never reward semantics."""

    def __init__(self, spec: RewardModelSpec):
        if spec.backend != "native":
            raise ValueError(f"NativeRewardExecutor requires a native spec, got {spec.backend!r}")
        if spec.offload_mode not in {"recreate", "cpu"}:
            raise ValueError(f"Unsupported native reward offload_mode: {spec.offload_mode!r}")
        self.spec = spec
        self._model: Any | None = None
        self._state = NativeRewardModelState.CLOSED
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> NativeRewardModelState:
        return self._state

    @property
    def _device(self) -> torch.device:
        return torch.device(get_device_name(), get_device_id())

    def _build_model(self, device: torch.device):
        model_cls = _load_native_model(self.spec.executor_config["model"])
        kwargs = dict(self.spec.executor_config.get("kwargs", {}))
        if self.spec.model_path is not None:
            kwargs.setdefault("model_path", self.spec.model_path)
        kwargs.setdefault("device", device)
        return model_cls(**kwargs)

    def _validate_cpu_lifecycle(self) -> None:
        """Fail before activation when a CPU-resident adapter is incomplete."""
        required_methods = ("activate", "offload_to_cpu", "close")
        missing = [name for name in required_methods if not callable(getattr(self._model, name, None))]
        if missing:
            model_type = type(self._model).__name__
            self._model = None
            self._state = NativeRewardModelState.CLOSED
            gc.collect()
            raise TypeError(
                f"Native reward model {model_type!r} must define {', '.join(f'{name}()' for name in missing)} "
                "for offload_mode='cpu'"
            )

    async def _call_model_method(self, name: str, *args, required: bool) -> Any:
        if self._model is None:
            raise RuntimeError(f"Native reward model {self.spec.name!r} is closed")
        method = getattr(self._model, name, None)
        if method is None:
            if required:
                raise TypeError(
                    f"Native reward model {type(self._model).__name__!r} must define {name}() "
                    f"for offload_mode={self.spec.offload_mode!r}"
                )
            return None

        async def invoke():
            result = (
                await method(*args) if inspect.iscoroutinefunction(method) else await asyncio.to_thread(method, *args)
            )
            return await result if inspect.isawaitable(result) else result

        return (await gather_complete([invoke()]))[0]

    async def _close_locked(self) -> None:
        """Finalize and forget the model while holding ``_lock``."""
        error = None
        try:
            if self._model is not None:
                await self._call_model_method("close", required=self.spec.offload_mode == "cpu")
        except BaseException as exc:
            error = exc
        self._model = None
        self._state = NativeRewardModelState.CLOSED
        gc.collect()
        _empty_accelerator_cache()
        if error is not None:
            raise error

    async def _activate_locked(self) -> None:
        """Move a CPU-resident model to its assigned device."""
        try:
            await self._call_model_method("activate", self._device, required=True)
        except asyncio.CancelledError:
            # ``gather_complete`` has drained activate(). Move the potentially
            # activated model back before propagating cancellation.
            try:
                await self._call_model_method("offload_to_cpu", required=True)
            finally:
                self._state = NativeRewardModelState.CPU
                gc.collect()
                _empty_accelerator_cache()
            raise
        except BaseException:
            await self._close_locked()
            raise
        self._state = NativeRewardModelState.DEVICE

    async def wake_up(self) -> None:
        async with self._lock:
            if self._state is NativeRewardModelState.DEVICE:
                return
            if self.spec.offload_mode == "cpu":
                if self._state is NativeRewardModelState.CLOSED:
                    self._model = self._build_model(torch.device("cpu"))
                    self._state = NativeRewardModelState.CPU
                    self._validate_cpu_lifecycle()
                await self._activate_locked()
                return
            self._model = self._build_model(self._device)
            self._state = NativeRewardModelState.DEVICE

    def reward_kwargs(self) -> dict[str, Any]:
        return {"reward_model": self}

    async def infer(self, *args, **kwargs):
        """Run model inference while protecting the wake/sleep boundary."""
        async with self._lock:
            if self._model is None or self._state is not NativeRewardModelState.DEVICE:
                raise RuntimeError(f"Native reward model {self.spec.name!r} is not awake")
            self._inflight += 1
            self._idle.clear()
        try:
            infer_fn = getattr(self._model, "infer", None)
            if infer_fn is None:
                raise TypeError(f"Native reward model {type(self._model).__name__!r} must define infer()")

            async def invoke():
                if inspect.iscoroutinefunction(infer_fn):
                    result = await infer_fn(*args, **kwargs)
                else:
                    result = await asyncio.to_thread(infer_fn, *args, **kwargs)
                return await result if inspect.isawaitable(result) else result

            # Cancelling an asyncio.to_thread waiter does not stop the native
            # inference thread. Drain it before lowering the inflight count so
            # sleep cannot close the model while kernels still use it.
            return (await gather_complete([invoke()]))[0]
        finally:
            async with self._lock:
                self._inflight -= 1
                if self._inflight == 0:
                    self._idle.set()

    async def sleep(self) -> None:
        """Release device memory, retaining the instance only in CPU mode."""
        while True:
            await self._idle.wait()
            async with self._lock:
                if self._inflight:
                    continue
                if self._state is not NativeRewardModelState.DEVICE:
                    return
                if self.spec.offload_mode == "cpu":
                    cancellation = None
                    try:
                        await self._call_model_method("offload_to_cpu", required=True)
                    except asyncio.CancelledError as exc:
                        cancellation = exc
                    self._state = NativeRewardModelState.CPU
                    gc.collect()
                    _empty_accelerator_cache()
                    if cancellation is not None:
                        raise cancellation
                    return
                await self._close_locked()
                return

    async def close(self) -> None:
        """Wait for real inference completion, then release the whole instance."""
        while True:
            await self._idle.wait()
            async with self._lock:
                if self._inflight:
                    continue
                await self._close_locked()
                return


def build_engine_reward_executors(specs: dict[str, RewardModelSpec]) -> dict[str, EngineRewardExecutor]:
    """Build worker-side router clients for engine-backed model specs."""
    return {name: EngineRewardExecutor(spec) for name, spec in specs.items() if is_engine_backend(spec.backend)}


def build_native_reward_executors(specs: dict[str, RewardModelSpec]) -> dict[str, NativeRewardExecutor]:
    """Build worker-local executors for native model specs."""
    return {name: NativeRewardExecutor(spec) for name, spec in specs.items() if spec.backend == "native"}


def _load_native_model(model_path: str):
    module_path, class_name = model_path.rsplit(":", 1)
    if module_path.startswith("pkg://"):
        module_path = module_path[len("pkg://") :].replace("/", ".")
    if "/" not in module_path and not module_path.endswith(".py"):
        return getattr(importlib.import_module(module_path), class_name)

    from verl.utils.import_utils import load_extern_object

    return load_extern_object(module_path=module_path, object_name=class_name)


def _empty_accelerator_cache() -> None:
    accelerator = getattr(torch, get_device_name(), None)
    empty_cache = getattr(accelerator, "empty_cache", None)
    if callable(empty_cache) and getattr(accelerator, "is_available", lambda: False)():
        empty_cache()
