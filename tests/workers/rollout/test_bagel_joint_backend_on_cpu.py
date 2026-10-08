# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Original prompt, config and bounded transport contracts without accelerators."""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from verl_omni.workers.config.diffusion.rollout import BagelJointRolloutConfig
from verl_omni.workers.rollout import bagel_joint_backend as backend


@pytest.mark.parametrize(
    "prompt",
    [
        "original text",
        [{"role": "user", "content": "original text"}],
        np.array([{"role": "user", "content": [{"type": "text", "text": "original text"}]}], dtype=object),
    ],
)
def test_original_prompt_is_preserved(prompt):
    assert backend.original_prompt_text(prompt) == "original text"


@pytest.mark.parametrize(
    "prompt",
    [
        [1, 2],
        [{"role": "assistant", "content": "no"}],
        [{"role": "user", "content": [{"type": "image", "image": "no"}]}],
        [],
    ],
)
def test_prompt_cannot_silently_drop_nontext_or_chat_turns(prompt):
    with pytest.raises(ValueError):
        backend.original_prompt_text(prompt)


@pytest.mark.parametrize(
    "values",
    [{"max_think_tokens": 1}, {"text_temperature": float("nan")}, {"sync_bucket_size_mb": 0}, {"cfg_text_scale": -1}],
)
def test_native_worker_knobs_validate(values):
    with pytest.raises(ValueError):
        BagelJointRolloutConfig(**values)


def test_inference_port_does_not_inherit_training_rendezvous(monkeypatch):
    monkeypatch.setenv("MASTER_PORT", "32123")
    with backend.independent_rollout_port():
        assert "MASTER_PORT" not in os.environ
    assert os.environ["MASTER_PORT"] == "32123"
    with pytest.raises(RuntimeError), backend.independent_rollout_port():
        raise RuntimeError("configuration failure")
    assert os.environ["MASTER_PORT"] == "32123"


def test_full_policy_export_is_bounded_and_commits_only_after_loading(monkeypatch):
    instance = object.__new__(backend.BagelJointVllmBackend)
    instance.rank, instance.group = 0, None
    instance.config = SimpleNamespace(sync_timeout_seconds=10, sync_bucket_size_mb=1)
    monkeypatch.setattr(backend, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(backend, "get_device_id", lambda: None)
    monkeypatch.setattr(backend.dist, "broadcast_object_list", lambda *a, **k: None)
    monkeypatch.setattr(backend.dist, "barrier", lambda *a, **k: None)
    loaded, calls, paths = {}, [], []

    class Worker:
        def collective_rpc(self, name, timeout, args=()):
            calls.append(name)
            if name == "begin_joint_policy_sync":
                assert args[0] == 3
            elif name == "load_joint_weight_file":
                paths.append(Path(args[0]))
                loaded.update(torch.load(args[0], weights_only=True))
            elif name == "commit_joint_policy_sync":
                assert args == ()
                assert len(loaded) == 4
                return [{"version": 3}]

    instance.engine = Worker()
    model = torch.nn.Sequential(torch.nn.Linear(600, 600), torch.nn.Linear(600, 600))
    instance.load_actor_policy(model, version=3)
    assert calls[0] == "begin_joint_policy_sync" and calls[-1] == "commit_joint_policy_sync"
    assert len(paths) == 2
    assert not any(path.exists() or path.parent.exists() for path in paths)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded["transformer." + name], tensor.to(torch.bfloat16))


def test_native_sync_queue_generation_works_inside_ray_event_loop(monkeypatch):
    instance = object.__new__(backend.BagelJointVllmBackend)
    instance.rank = 0
    instance.config = BagelJointRolloutConfig(max_think_tokens=8)
    instance.geometry = SimpleNamespace(height=8, width=8, num_inference_steps=5)
    requests = []

    class Worker:
        def add_req_and_wait_for_response(self, request):
            requests.append(request)
            return request.sampling_params.extra_args["policy_version"]

        def postprocess_output(self, request, output):
            return [SimpleNamespace(images=[Image.new("RGB", (8, 8))], multimodal_output={"metadata": {"rl": output}})]

    instance.engine = Worker()
    monkeypatch.setattr(
        backend,
        "replay_from_metadata",
        lambda value: SimpleNamespace(policy_version=value, response_token_ids=torch.tensor([5, 6]), eos_token_id=6),
    )

    async def ray_actor():
        for version in (0, 1):
            images, records = instance.generate(["original prompt"], [42 + version], version=version)
            assert images.shape == (1, 3, 8, 8) and images.dtype == torch.uint8
            assert records[0].policy_version == version

    asyncio.run(ray_actor())
    assert requests[0].prompt["prompt"] == "original prompt"
    assert requests[1].sampling_params.seed == 43
