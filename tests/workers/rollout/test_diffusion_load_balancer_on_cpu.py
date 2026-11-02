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

"""Exercise diffusion routing through the upstream manager without Ray actors."""

import asyncio
from collections import Counter
from types import SimpleNamespace

import pytest
from verl.workers.rollout import llm_server

from verl_omni.workers.config import DiffusionRolloutConfig
from verl_omni.workers.rollout.replica import DiffusionLLMServerManager, DiffusionRequestLoadBalancer


def _manager(monkeypatch, *, replicas=8, deterministic=True, **kwargs):
    rollout = DiffusionRolloutConfig(name="vllm_omni", nnodes=1, full_determinism=deterministic, seed=42)
    config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout, model=None))
    manager = DiffusionLLMServerManager(config, **kwargs)
    manager.server_addresses = [f"replica-{i}" for i in range(replicas)]
    manager.server_handles = [object() for _ in range(replicas)]
    monkeypatch.setattr(llm_server.ray, "remote", lambda cls: SimpleNamespace(remote=lambda **kw: cls(**kw)))
    asyncio.run(manager._init_global_load_balancer())
    return manager


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("replicas,requests", [(1, 3), (8, 3), (8, 16), (3, 19)])
def test_colliding_hashes_remain_balanced(monkeypatch, deterministic, replicas, requests):
    manager = _manager(monkeypatch, replicas=replicas, deterministic=deterministic)
    balancer = manager.global_load_balancer
    assigned = Counter({server: 0 for server in manager.server_addresses})
    candidate = 0
    for _ in range(requests):
        # All IDs would select replica zero under the previous hash routing.
        while hash(f"request-{candidate}") % replicas:
            candidate += 1
        server, handle = balancer.acquire_server(f"request-{candidate}")
        candidate += 1
        assigned[server] += 1
        assert handle is manager.server_handles[manager.server_addresses.index(server)]
    assert max(assigned.values()) - min(assigned.values()) <= 1
    assert balancer.get_status()["servers"] == dict(assigned)
    assert manager.rollout_config.full_determinism is deterministic
    assert manager.rollout_config.seed == 42


def test_completed_requests_free_capacity_and_retries_keep_affinity(monkeypatch):
    balancer = _manager(monkeypatch, replicas=2).global_load_balancer
    first, _ = balancer.acquire_server("first")
    second, _ = balancer.acquire_server("second")
    assert first != second
    balancer.release_server(first)
    assert balancer.acquire_server("new")[0] == first
    assert balancer.acquire_server("first")[0] == first
    assert balancer.acquire_server("next")[0] == second
    for server in (first, first, second, second):
        balancer.release_server(server)
    assert balancer.get_status()["total_inflight"] == 0


def test_added_and_removed_replicas_participate_in_balancing(monkeypatch):
    balancer = _manager(monkeypatch, replicas=1).global_load_balancer
    original, _ = balancer.acquire_server("first")
    added_handle = object()
    balancer.add_servers({"added": added_handle})
    assert balancer.acquire_server("second") == ("added", added_handle)
    balancer.remove_servers([original])
    assert balancer.acquire_server("first") == ("added", added_handle)
    balancer.release_server(original)
    assert balancer.get_status()["active_servers"] == 1


def test_explicit_balancer_override_is_preserved(monkeypatch):
    class CustomBalancer(DiffusionRequestLoadBalancer):
        pass

    manager = _manager(monkeypatch, load_balancer_cls=CustomBalancer)
    assert isinstance(manager.global_load_balancer, CustomBalancer)
