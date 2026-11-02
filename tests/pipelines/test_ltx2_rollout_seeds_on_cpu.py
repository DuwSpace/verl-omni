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

"""Legacy joint RNG assignments survive TQ worker splitting and request order."""

import asyncio
from types import MethodType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl_omni.agent_loop import diffusion_agent_loop_tq as worker_module
from verl_omni.pipelines.ltx2_unigrpo_nft.rollout import stages
from verl_omni.pipelines.ltx2_unigrpo_nft.rollout.seeds import assign_legacy_rollout_seeds


def batch(step, indices, validate=False):
    return TensorDict(
        {
            "index": torch.tensor(indices),
            "uid": NonTensorStack(*[f"prompt-{i}" for i in indices]),
            "global_steps": NonTensorData(step),
            "rollout_seed": NonTensorData(42 + step - 1),
            "validate": NonTensorData(validate),
        },
        batch_size=[len(indices)],
    )


@pytest.mark.parametrize("step,start", [(1, 298), (2, 810), (10, 4906)])
def test_training_seeds_match_old_policy_versions_including_resume(step, start):
    prompts = batch(step, range(32))
    assign_legacy_rollout_seeds(prompts, base_seed=42, n=8, validate=False)
    torch.testing.assert_close(prompts["_rollout_seeds"].flatten(), torch.arange(start, start + 256))


def test_validation_seeds_use_global_indices_across_batches_and_steps():
    for step in (0, 1, 10):
        for indices in ([0, 1], [2, 3]):
            prompts = batch(step, indices, True)
            assign_legacy_rollout_seeds(prompts, base_seed=73, n=2, validate=True)
            torch.testing.assert_close(
                prompts["_rollout_seeds"].flatten(), torch.arange(73 + indices[0] * 2, 73 + (indices[-1] + 1) * 2)
            )


@pytest.mark.parametrize("step", [0, -1, True, 1.5])
def test_invalid_training_step_rejected(step):
    with pytest.raises(ValueError, match="global_steps"):
        assign_legacy_rollout_seeds(batch(step, [0]), base_seed=42, n=8, validate=False)


@pytest.mark.parametrize("validate", [False, True])
def test_manager_to_workers_preserves_explicit_seeds_after_chunk_reordering(monkeypatch, validate):
    captured = []
    rollout = SimpleNamespace(
        seed=42,
        n=8,
        pipeline=OmegaConf.create({"rollout_mode": "flat", "videos_per_recaption": 1}),
        algo={},
        calculate_log_probs=False,
        agent=SimpleNamespace(default_agent_loop="diffusion_single_turn_agent"),
        val_kwargs=SimpleNamespace(
            n=2,
            seed=999,
            seed_offset_by_index=False,
            pipeline=OmegaConf.create({"rollout_mode": "flat", "videos_per_recaption": 1}),
            algo={},
        ),
    )
    manager = object.__new__(stages.LTXStagedAgentLoopManager)
    manager.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout))
    rounds = []
    manager.coordinator = SimpleNamespace(
        begin=SimpleNamespace(remote=lambda count: rounds.append(count)),
        wait_prepared=SimpleNamespace(remote=lambda: None),
    )
    monkeypatch.setattr(stages.ray, "get", lambda value: value)
    monkeypatch.setattr(stages.AgentLoopManagerTQ, "generate_sequences", lambda self, prompts: captured.append(prompts))
    prompts = batch(2, range(32), validate)
    manager.generate_sequences(prompts)
    assert rounds == [32 * (2 if validate else 8)]

    worker_cls = worker_module.DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.rollout_config = rollout
    worker.background_tasks = set()
    actual = {}

    async def fake_put(**kwargs):
        pass

    async def fake_trajectory(step, indices, validation):
        return [{"validate": validation} for _ in indices]

    async def fake_run(self, sampling_params, *, session_id, uid, **kwargs):
        assert "_rollout_seeds" not in kwargs
        actual[(uid, session_id)] = sampling_params["seed"]

    async def fake_write(self, *args, **kwargs):
        pass

    monkeypatch.setattr(worker_module.tq, "async_kv_put", fake_put)
    monkeypatch.setattr(worker_module, "get_trajectory_info", fake_trajectory)
    monkeypatch.setattr(worker_module, "_config_to_sampling_dict", lambda config: {})
    worker._run_agent_loop = MethodType(fake_run, worker)
    worker._write_trajectories_to_tq = MethodType(fake_write, worker)

    async def dispatch():
        for chunk in reversed(captured[0].chunk(8)):
            await worker.generate_sequences(chunk)
        await asyncio.gather(*worker.background_tasks)

    asyncio.run(dispatch())
    n = 2 if validate else 8
    start = 42 if validate else 810
    assert actual == {(f"prompt-{i}", j): start + i * n + j for i in range(32) for j in range(n)}


def test_tree_manager_does_not_override_existing_seed_assignment(monkeypatch):
    manager = object.__new__(stages.LTXStagedAgentLoopManager)
    config = SimpleNamespace(
        seed=42,
        n=8,
        pipeline=OmegaConf.create({"rollout_mode": "tree", "videos_per_recaption": 2}),
        val_kwargs=SimpleNamespace(n=1, pipeline=OmegaConf.create({"rollout_mode": "flat", "videos_per_recaption": 1})),
    )
    manager.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=config))
    manager.coordinator = SimpleNamespace(
        begin=SimpleNamespace(remote=lambda count: None), wait_prepared=SimpleNamespace(remote=lambda: None)
    )
    monkeypatch.setattr(stages.ray, "get", lambda value: value)
    captured = []
    monkeypatch.setattr(stages.AgentLoopManagerTQ, "generate_sequences", lambda self, prompts: captured.append(prompts))
    for validate in (False, True):
        manager.generate_sequences(batch(1, [0, 1], validate))
    assert all("_rollout_seeds" not in prompts for prompts in captured)
