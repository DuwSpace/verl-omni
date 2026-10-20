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

"""Original-task reward projection and sequential model lifecycle contracts."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl_omni.pipelines.ltx2_unigrpo_nft import rewards


def _sample():
    return {
        "replay": SimpleNamespace(
            validate=lambda: None,
            original_prompt="A bell rings.",
            recaption="An easier unrelated task.",
            generation_params={"frame_rate": 24.0},
        ),
        "video": torch.full((4, 3, 8, 8), 0.5),
        "audio": torch.ones(2, 480),
        "audio_sample_rate": 48000,
        "fps": 24.0,
    }


def test_reward_projection_preserves_original_task_and_actual_audio_rate():
    item = rewards.reward_sample(_sample())[0]
    assert item.non_tensor_batch["reward_model"]["ground_truth"] == "A bell rings."
    assert item.non_tensor_batch["reward_inputs"]["text"] == {"audio": "A bell rings.", "video": "A bell rings."}
    assert item.batch["audio"].shape == (2, 480)
    assert item.batch["audio_sample_rate"].item() == 48000
    assert item.batch["responses"].dtype == torch.uint8
    assert item.batch["responses"].unique().tolist() == [128]


@pytest.mark.parametrize("field,value", [("video", float("nan")), ("video", 1.5), ("audio", float("inf"))])
def test_reward_projection_rejects_invalid_media(field, value):
    sample = _sample()
    sample[field].flatten()[0] = value
    with pytest.raises(ValueError):
        rewards.reward_sample(sample)


async def _scorer(ground_truth, batch, reward_model, **kwargs):
    assert ground_truth == "A bell rings."
    assert batch[0].non_tensor_batch["reward_inputs"]["text"]["audio"] == ground_truth
    assert "routing_weights" not in kwargs
    return {"score": await reward_model.infer()}


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_reward_runner_retains_columns_and_releases_models(monkeypatch, fail):
    names = ["video_align", "hpsv3", "audiobox", "clap", "desync"]
    calls = []
    created = []

    class Executor:
        def __init__(self, spec):
            self.name = spec.name
            created.append(self)

        async def wake_up(self):
            calls.append((self.name, "load"))

        async def sleep(self):
            calls.append((self.name, "release"))

        def reward_kwargs(self):
            return {"reward_model": self}

        async def infer(self):
            calls.append((self.name, "infer"))
            if fail:
                raise RuntimeError("required model failed")
            return names.index(self.name) + 1.0

    monkeypatch.setattr(rewards, "NativeRewardExecutor", Executor)
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"pipeline": {"output_type": "video"}}},
            "reward": {
                "aggregation": "preserve_components",
                "models": {name: {"backend": "native", "model_path": "/unused", "executor": {}} for name in names},
                "reward_functions": {
                    name: {
                        "path": __file__,
                        "name": "_scorer",
                        "model": name,
                        "required": True,
                        "weight": 1.0,
                        "routing_weights": {"video": 1.0, "audio": 1.0},
                    }
                    for name in names
                },
            },
        }
    )
    runner = rewards.LTXJointRewardRunner(config)
    if fail:
        with pytest.raises(RuntimeError, match="required model failed"):
            await runner.score([_sample()])
        assert calls == [("audiobox", event) for event in ("load", "infer", "release")]
    else:
        scores, order, infos, metrics = await runner.score([_sample(), _sample()])
        assert order == sorted(names)
        torch.testing.assert_close(scores, torch.tensor([[3.0, 4.0, 5.0, 2.0, 1.0]] * 2))
        assert calls == [(name, event) for name in order for event in ("load", "infer", "infer", "release")]
        assert infos[0]["reward/video_align"] == 1.0
        assert metrics["reward/total_time"] > 0
        await runner.score([_sample()])
        assert len(created) == 5  # reuse each executor across batches
