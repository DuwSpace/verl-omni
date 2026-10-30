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

"""Exercise the global AR barrier and stable routing without model weights."""

import asyncio
from types import SimpleNamespace

import pytest

from verl_omni.pipelines.ltx2_unigrpo_nft.rollout.stages import RolloutRound, StagedRolloutClient


class Remote:
    def __init__(self, function):
        self.remote = function


class Server:
    def __init__(self, events, gate=None, failure=False):
        self.events = events
        self.gate = gate
        self.failure = failure
        self.prepare_rollout_requests = Remote(self.prepare)
        self.clear_prepared_rollout = Remote(self.clear)
        self.generate = Remote(self.video)
        self.requests = set()

    async def prepare(self, payloads):
        self.events.append("prepare")
        if self.gate:
            await self.gate.wait()
        if self.failure:
            raise ValueError("preparation failed")
        self.requests.update(payload["request_id"] for payload in payloads)
        self.events.append("prepared")

    async def clear(self):
        self.requests.clear()
        self.events.append("clear")

    async def video(self, **payload):
        assert payload["sampling_params"]["prepared_request_id"] == payload["request_id"]
        self.requests.remove(payload["request_id"])
        self.events.append("video")
        return SimpleNamespace(extra_fields={"global_steps": 3})


class Client:
    def __init__(self, servers):
        self.servers = servers
        self.assigned = 0
        self.released = []
        self.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(full_determinism=True)))
        self._load_balancer = None

    async def _acquire_server(self, request_id):
        index = self.assigned % len(self.servers)
        self.assigned += 1
        return str(index), self.servers[index]

    def _release_server(self, server_id):
        self.released.append(server_id)


def proxy(coordinator):
    return SimpleNamespace(submit=Remote(coordinator.submit), complete=Remote(coordinator.complete))


@pytest.mark.asyncio
async def test_global_barrier_keeps_routes_and_releases_after_all_videos():
    events, gate = [], asyncio.Event()
    servers = [Server(events), Server(events, gate)]
    base = Client(servers)
    round = RolloutRound(base, 5)
    client = StagedRolloutClient(base, proxy(round))
    await round.begin(4)
    tasks = [asyncio.create_task(client.generate(str(i), prompt_ids=[i], sampling_params={})) for i in range(4)]
    # A slow replica must hold back AV on the fast replica too.
    for _ in range(100):
        if events.count("prepare") == 2:
            break
        await asyncio.sleep(0.001)
    assert events.count("prepare") == 2
    assert events.count("prepared") == 1
    assert "video" not in events
    gate.set()
    outputs = await asyncio.gather(*tasks)
    assert events.index("video") > max(i for i, value in enumerate(events) if value == "prepared")
    assert events.count("video") == 4
    assert sorted(base.released) == ["0", "0", "1", "1"]
    assert all(not server.requests for server in servers)
    assert all(output.extra_fields["min_global_steps"] == 3 for output in outputs)
    await round.begin(1)
    await client.generate("next-round", prompt_ids=[1], sampling_params={})


@pytest.mark.asyncio
async def test_preparation_failure_clears_all_replicas_without_video():
    events = []
    servers = [Server(events), Server(events, failure=True)]
    base = Client(servers)
    round = RolloutRound(base, 5)
    client = StagedRolloutClient(base, proxy(round))
    await round.begin(2)
    results = await asyncio.gather(
        *[client.generate(str(i), prompt_ids=[i], sampling_params={}) for i in range(2)], return_exceptions=True
    )
    assert all(isinstance(result, RuntimeError) for result in results)
    assert "video" not in events
    assert events.count("clear") == 2
    assert sorted(base.released) == ["0", "1"]
    assert all(not server.requests for server in servers)
    with pytest.raises(RuntimeError, match="Previous staged rollout failed"):
        await round.begin(1)


@pytest.mark.asyncio
async def test_missing_request_times_out_and_releases_existing_lease():
    events = []
    base = Client([Server(events)])
    round = RolloutRound(base, 0.02)
    await round.begin(2)
    with pytest.raises(RuntimeError, match="text preparation failed"):
        await round.submit("only-one", {"request_id": "only-one"})
    assert base.released == ["0"]
    assert "prepare" not in events
