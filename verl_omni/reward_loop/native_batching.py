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
"""Bounded cross-request batching for synchronous native reward inference."""

import asyncio
import inspect
import logging
from collections import deque
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class _Request:
    inputs: tuple
    size: int
    future: asyncio.Future


class NativeRewardBatcher:
    """Merge aligned list/tensor inputs and split tensor-tree outputs in order.

    Models opt in with ``batch_input_names``. Outputs are tensors or dictionaries
    of tensors, with ``batch_output_dim`` selecting their batch axis. A model may
    set ``batch_output_key`` to keep other dictionary entries as shared metadata.
    The owning executor counts queued requests as inflight and drains them before
    sleep/close; cancellation of a caller must not cancel an accepted request.
    """

    def __init__(self, model, max_batch_size: int, batch_wait_ms: float):
        self.model = model
        self.max_batch_size = max_batch_size
        self.wait_seconds = batch_wait_ms / 1000
        self.input_names = getattr(model, "batch_input_names", ())
        if not self.input_names or inspect.iscoroutinefunction(model.infer):
            raise ValueError("Native batching requires a synchronous model declaring batch_input_names")
        self.signature = inspect.signature(model.infer)
        self.output_dim = getattr(model, "batch_output_dim", 0)
        self.output_key = getattr(model, "batch_output_key", None)
        self.pending = deque()
        self.pending_size = 0
        self.changed = asyncio.Event()
        self.runner = None

    async def infer(self, args, kwargs):
        bound = self.signature.bind(*args, **kwargs)
        if set(bound.arguments) != set(self.input_names):
            raise ValueError("Batched inference requires exactly the declared batch inputs")
        inputs = tuple(bound.arguments[name] for name in self.input_names)
        if any(not isinstance(value, list | torch.Tensor) for value in inputs):
            raise TypeError("Native batch inputs must be lists or tensors with a leading batch dimension")
        size = len(inputs[0])
        if size < 1 or any(len(value) != size for value in inputs):
            raise ValueError("Native batch inputs must have matching, non-empty batch dimensions")
        future = asyncio.get_running_loop().create_future()
        self.pending.append(_Request(inputs, size, future))
        self.pending_size += size
        self.changed.set()
        if self.runner is None:
            self.runner = asyncio.create_task(self._run())
        return await future

    async def _run(self):
        try:
            while self.pending:
                deadline = asyncio.get_running_loop().time() + self.wait_seconds
                while self.pending_size < self.max_batch_size:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    self.changed.clear()
                    try:
                        await asyncio.wait_for(self.changed.wait(), remaining)
                    except TimeoutError:
                        break
                requests = []
                size = 0
                while self.pending:
                    candidate = self.pending[0]
                    if requests and size + candidate.size > self.max_batch_size:
                        break
                    requests.append(self.pending.popleft())
                    size += candidate.size
                    self.pending_size -= candidate.size
                    if size >= self.max_batch_size:
                        break
                try:
                    results = await asyncio.to_thread(self._infer_batch, requests)
                except Exception as exc:
                    for request in requests:
                        request.future.set_exception(exc)
                else:
                    for request, result in zip(requests, results, strict=True):
                        request.future.set_result(result)
        finally:
            self.runner = None

    def _infer_batch(self, requests):
        inputs = []
        for column in zip(*(request.inputs for request in requests), strict=True):
            inputs.append(torch.cat(column, dim=0) if isinstance(column[0], torch.Tensor) else sum(column, []))
        size = sum(request.size for request in requests)
        outputs = []
        shared = None
        for start in range(0, size, self.max_batch_size):
            stop = min(start + self.max_batch_size, size)
            batch_inputs = {name: value[start:stop] for name, value in zip(self.input_names, inputs, strict=True)}
            output = self.model.infer(**batch_inputs)
            logger.debug("%s inferred batch_size=%d", type(self.model).__name__, stop - start)
            batched = output if self.output_key is None else output[self.output_key]
            _map_tensors(batched, lambda tensor, count=stop - start: self._validate_output(tensor, count))
            outputs.append(batched)
            if self.output_key is not None:
                shared = {key: value for key, value in output.items() if key != self.output_key}
        combined = _concat_outputs(outputs, self.output_dim)
        results = []
        start = 0
        for request in requests:
            result = _map_tensors(
                combined, lambda tensor, offset=start, count=request.size: tensor.narrow(self.output_dim, offset, count)
            )
            results.append(result if self.output_key is None else {**shared, self.output_key: result})
            start += request.size
        return results

    def _validate_output(self, tensor, size):
        if tensor.ndim <= self.output_dim or tensor.shape[self.output_dim] != size:
            raise ValueError(f"Native reward output must have batch dimension {self.output_dim} of size {size}")
        return tensor


def _map_tensors(value, transform):
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, transform) for key, item in value.items()}
    raise TypeError("Native batched outputs must be tensors or dictionaries of tensors")


def _concat_outputs(values, dim):
    if isinstance(values[0], torch.Tensor):
        return torch.cat(values, dim=dim)
    return {key: _concat_outputs([value[key] for value in values], dim) for key in values[0]}
