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

"""Whole-round AR preparation barrier over the existing TQ rollout workers."""

import asyncio
import logging
from collections import defaultdict
from time import perf_counter

import ray
from tensordict.tensorclass import NonTensorData
from verl.trainer.ppo.v1 import AgentLoopManagerTQ
from verl.workers.rollout.llm_server import LLMServerClient

from verl_omni.agent_loop.diffusion_agent_loop_tq import DiffusionAgentLoopWorkerTQ

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RolloutRound:
    """Own fixed server leases and a global preparation barrier for one round."""

    def __init__(self, client, timeout):
        logging.basicConfig(level=logging.INFO)
        self.client = client
        self.timeout = timeout
        self.finished = asyncio.Event()
        self.finished.set()
        self.error = None
        self.entries = {}

    async def begin(self, expected):
        """Wait for the preceding round before admitting new requests."""
        await asyncio.wait_for(self.finished.wait(), self.timeout)
        if self.error is not None:
            raise RuntimeError("Previous staged rollout failed") from self.error
        if expected <= 0:
            raise ValueError("A rollout round must contain requests")
        self.expected = expected
        self.entries = {}
        self.submitted = set()
        self.preparation_error = None
        self.servers = {}
        self.completed = set()
        self.arrived = asyncio.Event()
        self.ready = asyncio.Event()
        self.finished.clear()
        self.preparation = asyncio.create_task(self._prepare())

    async def submit(self, request_id, payload):
        """Assign a stable owner and wait until all replicas finish stage one."""
        if self.finished.is_set() or self.error is not None:
            raise RuntimeError("No active staged rollout round")
        if request_id in self.submitted or len(self.submitted) >= self.expected:
            raise ValueError("Duplicate or excess staged rollout request")
        self.submitted.add(request_id)
        server_id, server = await self.client._acquire_server(request_id)
        if self.finished.is_set():
            self.client._release_server(server_id)
            raise RuntimeError("Staged rollout expired while acquiring a server")
        self.entries[request_id] = (server_id, server, payload)
        self.servers[server_id] = server
        if len(self.entries) == self.expected:
            self.arrived.set()
        await self.wait_prepared()
        return server

    async def _prepare(self):
        try:
            await asyncio.wait_for(self.arrived.wait(), self.timeout)
            grouped = defaultdict(list)
            for server_id, _, payload in self.entries.values():
                grouped[server_id].append(payload)
            logger.info("Rollout stage 1: preparing %d requests on %d replicas", self.expected, len(grouped))
            started = perf_counter()
            # Await every worker, including when a peer fails, before clearing CPU caches.
            results = await asyncio.wait_for(
                asyncio.gather(
                    *[self.servers[key].prepare_rollout_requests.remote(payloads) for key, payloads in grouped.items()],
                    return_exceptions=True,
                ),
                self.timeout,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            self.video_started = perf_counter()
            logger.info(
                "Rollout stage 1 complete in %.2fs; admitting stage 2 for %d requests",
                self.video_started - started,
                self.expected,
            )
        except Exception as error:
            self.error = self.preparation_error = error
            await self._clear()
            for server_id, _, _ in self.entries.values():
                self.client._release_server(server_id)
            self.finished.set()
        finally:
            self.ready.set()

    async def wait_prepared(self):
        """Expose preparation failures to the synchronous training driver."""
        await self.ready.wait()
        if self.preparation_error is not None:
            raise RuntimeError("Rollout text preparation failed") from self.preparation_error

    async def complete(self, request_id, failed=False):
        """Release exactly one lease and clear remaining caches after the last AV request."""
        if self.finished.is_set():
            return
        if request_id in self.completed:
            raise ValueError("Duplicate rollout completion")
        server_id, _, _ = self.entries[request_id]
        self.client._release_server(server_id)
        self.completed.add(request_id)
        if failed:
            self.error = RuntimeError(f"Video generation failed for {request_id}")
        if len(self.completed) == self.expected:
            logger.info(
                "Rollout stage 2 complete in %.2fs for %d requests", perf_counter() - self.video_started, self.expected
            )
            try:
                await self._clear()
            finally:
                self.finished.set()

    async def _clear(self):
        results = await asyncio.gather(
            *[server.clear_prepared_rollout.remote() for server in self.servers.values()], return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                self.error = result


class StagedRolloutClient(LLMServerClient):
    """Use the same server and request identity for preparation and AV generation."""

    def __init__(self, client, coordinator):
        super().__init__(client.config, load_balancer_handle=client._load_balancer)
        self.coordinator = coordinator

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        request_id = self._vllm_request_id(request_id)
        payload = dict(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params={**sampling_params, "prepared_request_id": request_id},
            **kwargs,
        )
        server = await self.coordinator.submit.remote(request_id, payload)
        failed = True
        try:
            output = await server.generate.remote(**payload)
            global_steps = output.extra_fields.get("global_steps")
            output.extra_fields.setdefault("min_global_steps", global_steps)
            output.extra_fields.setdefault("max_global_steps", global_steps)
            failed = False
            return output
        finally:
            await self.coordinator.complete.remote(request_id, failed)


class LTXStagedAgentLoopManager(AgentLoopManagerTQ):
    """Add a preparation barrier while reusing ordinary diffusion TQ workers."""

    def __init__(self, config, llm_client, **kwargs):
        if config.trainer.v1.trainer_mode != "sync":
            raise ValueError("Two-stage LTX rollout requires the synchronous trainer")
        timeout = config.actor_rollout_ref.rollout.pipeline.stage_timeout_s
        self.coordinator = ray.remote(RolloutRound).options(max_concurrency=100000).remote(llm_client, timeout)
        super().__init__(config=config, llm_client=StagedRolloutClient(llm_client, self.coordinator), **kwargs)
        self.agent_loop_workers_class = DiffusionAgentLoopWorkerTQ

    def generate_sequences(self, prompts):
        """Prepare all expanded samples before allowing the native AV scheduler to run."""
        validate = prompts.get("validate", False)
        if isinstance(validate, NonTensorData):
            validate = validate.data
        config = self.config.actor_rollout_ref.rollout
        count = config.val_kwargs.n if validate else config.n
        ray.get(self.coordinator.begin.remote(len(prompts) * count))
        super().generate_sequences(prompts)
        ray.get(self.coordinator.wait_prepared.remote())
