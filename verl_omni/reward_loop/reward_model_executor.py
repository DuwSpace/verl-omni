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
    """Own one worker-local reward model and its placement transitions.

    States are CLOSED (no instance), CPU (retained offloaded instance), and
    DEVICE (available for inference). A lock serializes placement transitions
    and inference admission; admitted inference calls may run concurrently.
    The inflight count and idle event prevent movement during those calls.
    CPU offload retains the instance; recreate sleep closes it. Adapter hooks
    own actual tensor movement and reference release.
    """

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
        """Construct synchronously, defaulting model_path/device unless kwargs override them."""
        model_cls = _load_native_model(self.spec.executor_config["model"])
        kwargs = dict(self.spec.executor_config.get("kwargs", {}))
        if self.spec.model_path is not None:
            kwargs.setdefault("model_path", self.spec.model_path)
        kwargs.setdefault("device", device)
        return model_cls(**kwargs)

    def _validate_cpu_lifecycle(self) -> None:
        """Require CPU lifecycle hooks, dropping an incomplete instance before activation.

        Missing callable activate/offload_to_cpu/close hooks reset state to CLOSED
        and raise TypeError. This validation does not invoke the adapter's close.
        """
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
        """Invoke a lifecycle hook and drain it before propagating errors/cancellation.

        Async hooks run on the event loop; synchronous hooks run in a thread.
        Await a returned awaitable as well. Callers own transition locking.
        A closed model raises RuntimeError; a missing required hook raises
        TypeError, while an absent optional hook returns None.
        """
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
        """Close under the caller's lock after inference has drained.

        Call the adapter's close hook (required in CPU mode), then drop the
        instance, mark CLOSED, collect garbage, and empty the accelerator cache.
        A hook error or cancellation is re-raised after clearing the state.
        Live references elsewhere may keep allocations alive.
        """
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
        """Activate under the caller's lock, publishing DEVICE only on success.

        On cancellation, wait for activate to finish, offload back to CPU, and
        re-raise. Failed activation or failed cancellation cleanup closes the
        instance; cleanup failures may supersede the original exception.
        """
        try:
            await self._call_model_method("activate", self._device, required=True)
        except asyncio.CancelledError:
            try:
                await self._call_model_method("offload_to_cpu", required=True)
            except BaseException:
                await self._close_locked()
                raise
            self._state = NativeRewardModelState.CPU
            gc.collect()
            _empty_accelerator_cache()
            raise
        except BaseException:
            await self._close_locked()
            raise
        self._state = NativeRewardModelState.DEVICE

    async def wake_up(self) -> None:
        """Make the model available for inference while holding the transition lock.

        DEVICE is a no-op. CPU mode constructs with CPU as the default device
        when closed, validates lifecycle hooks, then activates the retained
        instance. Recreate mode defaults to the assigned device. Explicit
        constructor kwargs can override either default. Construction is
        synchronous; activation drains before cancellation handling completes.
        """
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

    def metadata(self) -> dict[str, Any]:
        """Copy reported definition/revision labels from an awake adapter.

        This does not verify weight identity or acquire a lifecycle lock. Raise
        RuntimeError when not in DEVICE state and TypeError for a missing hook
        or non-dict result.
        """
        if self._model is None or self._state is not NativeRewardModelState.DEVICE:
            raise RuntimeError(f"Native reward model {self.spec.name!r} is not awake")
        metadata_fn = getattr(self._model, "metadata", None)
        if not callable(metadata_fn):
            raise TypeError(f"Native reward model {type(self._model).__name__!r} must define metadata()")
        metadata = metadata_fn()
        if not isinstance(metadata, dict):
            raise TypeError("Native reward model metadata() must return a dict")
        return dict(metadata)

    async def infer(self, *args, **kwargs):
        """Run inference on an awake instance, returning the adapter's result unchanged.

        Admit the call and increment inflight under the lock, then release the
        lock while executing. Async infer runs on the loop; synchronous infer
        runs in a thread, and any returned awaitable is also awaited. Drain the
        actual work before decrementing inflight, including after cancellation.
        The idle event is set when the last admitted call finishes. Sleep/close
        can then transition placement. A non-awake model raises RuntimeError;
        a missing infer method raises TypeError. Other errors follow gather_complete.
        """
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

            # Cancelling the waiter cannot stop a native inference thread.
            return (await gather_complete([invoke()]))[0]
        finally:
            async with self._lock:
                self._inflight -= 1
                if self._inflight == 0:
                    self._idle.set()

    async def sleep(self) -> None:
        """Wait for inflight work, then offload or close under the transition lock.

        Recheck inflight after taking the lock. CPU mode calls offload_to_cpu,
        retains the instance, publishes CPU, and clears unused cache; recreate
        mode closes and publishes CLOSED. Non-DEVICE states are a no-op.
        Cancellation while waiting for idle/lock may exit without transition;
        once offload starts it drains and publishes CPU before re-raising
        cancellation. Offload errors trigger close instead of publishing CPU.
        """
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
                    except BaseException:
                        # A partial move cannot safely be advertised as CPU-resident.
                        await self._close_locked()
                        raise
                    self._state = NativeRewardModelState.CPU
                    gc.collect()
                    _empty_accelerator_cache()
                    if cancellation is not None:
                        raise cancellation
                    return
                await self._close_locked()
                return

    async def close(self) -> None:
        """Wait for idle, then close and forget the instance under the lock.

        Recheck inflight before closing. This may be called between scoring
        phases as well as at shutdown; the next wake rebuilds the model.
        Cancellation before the close hook starts may leave it open; once
        started, the hook drains and CLOSED is published before propagation.
        Reference release and cache cleanup do not guarantee all memory is freed.
        """
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
