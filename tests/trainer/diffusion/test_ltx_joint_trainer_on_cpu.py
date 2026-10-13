# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Joint LTX dispatch and mini-batch semantics without Ray or model weights."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.ltx_joint_ray_trainer import LTXJointRayTrainer
from verl_omni.trainer.diffusion.native_ray_trainer import NativeRayDiffusionTrainer
from verl_omni.trainer.main_diffusion import _get_trainer_cls


def _config():
    return OmegaConf.create(
        {
            "algorithm": {"trainer_type": "direct_preference", "sample_source": "online"},
            "trainer": {"nnodes": 1, "n_gpus_per_node": 2},
            "data": {"train_batch_size": 4},
            "actor_rollout_ref": {
                "model": {"algorithm": "ltx2_unigrpo_nft", "lora_rank": 8, "policy_state_adapters": ["default"]},
                "actor": {
                    "diffusion_loss": {"loss_mode": "ltx2_unigrpo_nft"},
                    "ppo_micro_batch_size_per_gpu": 1,
                    "ppo_mini_batch_size": 4,
                    "ppo_epochs": 2,
                    "data_loader_seed": 7,
                    "shuffle": False,
                },
                "rollout": {"name": "vllm_omni", "tensor_model_parallel_size": 1, "n": 2, "seed": 42},
            },
        }
    )


def test_ltx_route_and_grouped_dispatch_preserve_seed_and_version(monkeypatch):
    config = _config()
    LTXJointRayTrainer.validate_joint_config(config)
    assert _get_trainer_cls(config) is LTXJointRayTrainer
    trainer = object.__new__(LTXJointRayTrainer)
    trainer.config, trainer.global_steps = config, 3

    def generate(self, batch):
        assert list(batch.non_tensor_batch["uid"]) == ["one", "two", "one", "two"]
        return DataProto(non_tensor_batch={"seeds": batch.non_tensor_batch["_ltx_sample_seed"]})

    monkeypatch.setattr(NativeRayDiffusionTrainer, "_generate_native", generate)
    batch = DataProto(non_tensor_batch={"uid": np.array(["one", "one", "two", "two"])})
    result = trainer._generate_native(batch)
    assert list(batch.non_tensor_batch["_ltx_policy_version"]) == [5] * 4
    assert list(result.non_tensor_batch["seeds"]) == [62, 63, 64, 65]


def test_fixed_evaluation_uses_updated_pair_and_stable_seeds_without_optimizer(monkeypatch, tmp_path):
    from verl_omni.pipelines.ltx2_unigrpo_nft import artifacts

    trainer = object.__new__(LTXJointRayTrainer)
    trainer.config = _config()
    trainer.config.trainer.rollout_data_dir = str(tmp_path)
    trainer.config.reward = {"reward_functions": {f"reward{i}": {} for i in range(5)}}
    trainer.val_dataloader = [{"raw_prompt": np.array(["one", "two"], dtype=object)}]
    observed = []

    def generate(self, batch):
        observed.append(
            (batch.non_tensor_batch["_ltx_policy_version"].copy(), batch.non_tensor_batch["_ltx_sample_seed"].copy())
        )
        assert list(batch.non_tensor_batch["raw_prompt"]) == ["one", "two", "one", "two"]
        return DataProto.from_dict(tensors={"responses": torch.zeros(4, 1)})

    monkeypatch.setattr(NativeRayDiffusionTrainer, "_generate_native", generate)
    trainer._compute_reward_colocate = lambda batch: DataProto.from_dict(tensors={"rm_scores": torch.ones(4, 5)})

    def prepare(batch, reward):
        assert list(batch.non_tensor_batch["uid"]) == ["eval-0", "eval-0", "eval-1", "eval-1"]
        batch.batch["text_reward_prob"] = torch.full((4,), 0.5)
        batch.batch["reward_prob"] = torch.full((4, 1, 2), 0.5)
        return batch

    trainer._prepare_actor_batch = prepare
    saved = []

    def save(batch, directory, names):
        saved.append(directory)
        assert len(names) == 5
        return [{"ended_with_eos": True} for _ in range(len(batch))]

    monkeypatch.setattr(artifacts, "save_joint_samples", save)
    for step in [0, 1, 2]:
        trainer.global_steps = step
        metrics = trainer._validate()
        assert metrics["val/joint/policy_version"] == 2 * step
        assert metrics["val/joint/samples"] == 4
        assert not trainer._evaluating
        trainer.global_steps += 1
        assert trainer._rollout_version() == 2 * step + 1
    assert [list(item[0]) for item in observed] == [[0] * 4, [2] * 4, [4] * 4]
    assert all(list(item[1]) == [42, 44, 43, 45] for item in observed)
    assert len(set(saved)) == 3

    def fail(batch):
        raise RuntimeError("evaluation generation failed")

    trainer._generate_native = fail
    with pytest.raises(RuntimeError, match="evaluation generation"):
        trainer._validate()
    assert not trainer._evaluating


@pytest.mark.parametrize(
    "key,value",
    [
        ("actor_rollout_ref.rollout.name", "native"),
        ("actor_rollout_ref.model.lora_rank", 0),
        ("actor_rollout_ref.actor.ppo_mini_batch_size", 3),
        ("actor_rollout_ref.rollout.n", 1),
        ("algorithm.trainer_type", "policy_gradient"),
    ],
)
def test_invalid_joint_layouts_fail_before_launch(key, value):
    config = _config()
    OmegaConf.update(config, key, value)
    with pytest.raises(ValueError):
        LTXJointRayTrainer.validate_joint_config(config)


def test_ppo_minibatch_counts_samples_without_multiplying_rollout_n(tmp_path):
    trainer = object.__new__(LTXJointRayTrainer)
    trainer.config = _config()
    trainer.config.trainer.default_local_dir = str(tmp_path)
    trainer.global_steps = 3

    def update(data):
        assert tu.get(data, "global_batch_size") == tu.get(data, "mini_batch_size") == 4
        assert tu.get(data, "epochs") == 2
        assert "responses" not in data
        assert data["audio_latents_clean"].shape == (8, 2, 4)
        result = TensorDict({}, batch_size=[])
        tu.assign_non_tensor(result, metrics={"joint/total_loss": [1.0]})
        return result

    trainer.actor_rollout_wg = SimpleNamespace(update_actor=update)
    batch = DataProto.from_dict(tensors={"responses": torch.zeros(8, 1), "audio_latents_clean": torch.zeros(8, 2, 4)})
    output = trainer._update_actor(batch)
    assert output.meta_info["metrics"]["actor/joint/total_loss"] == [1.0]
    saved = torch.load(tmp_path / "update_metrics/step-000003.pt", weights_only=True)
    assert saved["global_step"] == 3 and saved["rollout_policy_version"] == 5
    assert saved["phase"] == "actor_update_complete"
    assert saved["metrics"] == output.meta_info["metrics"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_joint_wall_time_does_not_sum_parallel_stage_times(monkeypatch):
    import verl_omni.trainer.diffusion.ltx_joint_ray_trainer as joint

    clock = iter([20.0, 30.0])
    monkeypatch.setattr(joint, "perf_counter", lambda: next(clock))
    trainer = object.__new__(LTXJointRayTrainer)
    trainer.config = _config()
    trainer._joint_reward_time = 5.0
    result = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(result, metrics={})
    trainer.actor_rollout_wg = SimpleNamespace(update_actor=lambda data: result)
    timings = [
        {
            "rank": rank,
            "sync_started_at": 5.0 + rank,
            "rollout/sync_time": 2.0,
            "rollout/model_load_time": 50.0,
            "rollout/recaption_time": 1.0 + rank,
            "rollout/generation_time": 3.0,
        }
        for rank in [0, 1, 0, 1]
    ]
    batch = DataProto.from_dict(
        tensors={"audio_latents_clean": torch.zeros(4, 2, 4)},
        non_tensors={
            "ltx_rollout_metrics": np.array(timings, dtype=object),
            "ltx_joint_replay": np.array([SimpleNamespace(response_token_ids=torch.ones(3))] * 4, dtype=object),
        },
    )
    metrics = trainer._update_actor(batch).meta_info["metrics"]
    assert metrics["step/total_time"] == 25.0
    assert metrics["train/total_time"] == 10.0 and metrics["reward/total_time"] == 5.0
    assert metrics["rollout/recaption_time"] == 4.0 and metrics["rollout/generation_time"] == 6.0
    assert metrics["rollout/model_load_time"] == 50.0
    assert metrics["rollout/recaption_tokens_per_second"] == 3.0


def test_real_recipe_assigns_all_five_native_models_to_disjoint_devices(monkeypatch):
    from verl_omni.reward_loop.reward_model import MultiRewardModelManager
    from verl_omni.workers.config.reward import parse_reward_model_config

    monkeypatch.setenv("LTX_REWARD_ROOT", "/reward-assets")
    recipe = Path(__file__).resolve().parents[3] / "examples/unigrpo_nft_trainer/ltx2"
    config = OmegaConf.merge(OmegaConf.load(recipe / "rewards.yaml"), OmegaConf.load(recipe / "joint_train.yaml"))
    entries = [(name, parse_reward_model_config(name, value)) for name, value in config.reward.models.items()]
    assignments = MultiRewardModelManager._validate_native_device_assignments(entries)
    assert set(assignments) == {"video_align", "hpsv3", "audiobox", "clap", "desync"}
    assert sorted(device for devices in assignments.values() for device in devices) == list(range(5))
    assert all(config.reward.reward_functions[name].required for name in assignments)


def test_composed_recipe_prepares_joint_actor_batch_without_dataclass_instantiation(monkeypatch):
    from hydra import compose, initialize_config_dir

    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss

    for key in ("MODEL_PATH", "REWARD_ROOT", "TRAIN_FILE", "VAL_FILE", "OUTPUT_DIR"):
        monkeypatch.setenv(f"LTX_{key}", f"/unused/{key}")
    recipe = Path(__file__).resolve().parents[3] / "examples/unigrpo_nft_trainer/ltx2"
    with initialize_config_dir(config_dir=str(recipe), version_base=None):
        config = compose(config_name="joint_train")
    names = sorted(config.reward.reward_functions)
    scores = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    batch = DataProto.from_dict(
        tensors={
            "rm_scores": scores,
            "video_latents_clean": torch.zeros(4, 2, 4),
            "audio_latents_clean": torch.zeros(4, 2, 4),
            "train_timesteps": torch.linspace(1000, 1, 30).repeat(4, 1),
        },
        non_tensors={
            "uid": np.array(["one", "one", "two", "two"]),
            "ltx_joint_replay": np.array([SimpleNamespace(validate=lambda: None, policy_version=0)] * 4),
        },
        meta_info={"reward_names": names},
    )
    result = LTXJointLoss.prepare_actor_batch(batch, scores, config)
    assert result.batch["reward_prob"].shape == (4, 3, 2)
    assert result.batch["video_forward_noise"].shape == (4, 3, 2, 4)
    torch.testing.assert_close(result.batch["text_advantages"], torch.tensor([-1.0, 1.0, -1.0, 1.0]), atol=1e-4, rtol=0)
