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


@pytest.mark.parametrize("evaluating,phase,version", [(False, "train", 27), (True, "validation", 28)])
def test_reward_health_context_identifies_step_and_phase(monkeypatch, evaluating, phase, version):
    trainer = object.__new__(LTXJointRayTrainer)
    trainer.global_steps, trainer._evaluating = 14, evaluating
    batch = DataProto.from_dict(tensors={"responses": torch.zeros(1, 1)})

    def score(self, data):
        assert data.meta_info["reward_diagnostics_context"] == {"step": 14, "phase": phase, "policy_version": version}
        return "scored"

    monkeypatch.setattr(NativeRayDiffusionTrainer, "_compute_reward_colocate", score)
    assert trainer._compute_reward_colocate(batch) == "scored"


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
    trainer.config.actor_rollout_ref.rollout.val_kwargs = {"n": 2}
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

    def save(batch, directory, names, *, save_tensors):
        assert save_tensors is False
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
    trainer.config.algorithm.old_policy_decay_schedule = "linear_to_0_5"
    events = []

    def update(data):
        events.append("actor")
        assert tu.get(data, "global_batch_size") == tu.get(data, "mini_batch_size") == 4
        assert tu.get(data, "epochs") == 2
        assert "responses" not in data
        assert data["audio_latents_clean"].shape == (8, 2, 4)
        result = TensorDict({}, batch_size=[])
        tu.assign_non_tensor(result, metrics={"joint/total_loss": [1.0]})
        return result

    def ema(**kwargs):
        assert events == ["actor"]
        assert kwargs == {"source": "default", "target": "old", "decay": 0.003}
        assert not (tmp_path / "update_metrics/step-000003.pt").exists()
        events.append("ema")

    trainer.actor_rollout_wg = SimpleNamespace(update_actor=update, ema_update_adapter=ema)
    batch = DataProto.from_dict(tensors={"responses": torch.zeros(8, 1), "audio_latents_clean": torch.zeros(8, 2, 4)})
    output = trainer._update_actor(batch)
    assert output.meta_info["metrics"]["actor/joint/total_loss"] == [1.0]
    saved = torch.load(tmp_path / "update_metrics/step-000003.pt", weights_only=True)
    assert saved["global_step"] == 3 and saved["rollout_policy_version"] == 5
    assert events == ["actor", "ema"]
    assert saved["metrics"]["old_policy/decay"] == 0.003
    assert saved["phase"] == "actor_update_complete"
    assert saved["metrics"] == output.meta_info["metrics"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_joint_wall_time_does_not_sum_parallel_stage_times(monkeypatch):
    import verl_omni.trainer.diffusion.ltx_joint_ray_trainer as joint

    clock = iter([20.0, 30.0])
    monkeypatch.setattr(joint, "perf_counter", lambda: next(clock))
    trainer = object.__new__(LTXJointRayTrainer)
    trainer.config = _config()
    trainer.global_steps = 1
    trainer._joint_reward_time = 5.0
    result = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(result, metrics={})
    trainer.actor_rollout_wg = SimpleNamespace(
        update_actor=lambda data: result, ema_update_adapter=lambda **kwargs: None
    )
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
            "ltx_joint_replay": np.array(
                [SimpleNamespace(validate=lambda: None, policy_version=0, seed=42 + i) for i in range(4)]
            ),
        },
        meta_info={"reward_names": names},
    )
    result = LTXJointLoss.prepare_actor_batch(batch, scores, config)
    selected_steps = max(1, int(30 * config.algorithm.timestep_fraction))
    assert result.batch["reward_prob"].shape == (4, selected_steps, 2)
    assert result.batch["video_forward_noise"].shape == (4, selected_steps, 2, 4)
    torch.testing.assert_close(result.batch["text_advantages"], torch.tensor([-1.0, 1.0, -1.0, 1.0]), atol=1e-4, rtol=0)


def test_train_eval_train_sampling_settings_are_request_scoped(monkeypatch):
    trainer = object.__new__(LTXJointRayTrainer)
    trainer.config, trainer.global_steps = _config(), 1
    rollout = trainer.config.actor_rollout_ref.rollout
    rollout.pipeline = dict(
        height=256,
        width=384,
        num_frames=121,
        frame_rate=24.0,
        num_inference_steps=20,
        max_sequence_length=1024,
        video_cfg_scale=1.5,
        audio_cfg_scale=3.0,
    )
    rollout.val_kwargs = {
        "pipeline": dict(
            num_inference_steps=40,
            video_cfg_scale=3.0,
            audio_cfg_scale=7.0,
            video_modality_scale=3.0,
            audio_modality_scale=3.0,
            video_rescale_scale=0.7,
            audio_rescale_scale=0.7,
        )
    }
    seen = []

    def generate(self, batch):
        settings = list(batch.non_tensor_batch["_ltx_sampling_params"])
        assert all(item == settings[0] for item in settings)
        seen.append(settings[0])
        return DataProto(non_tensor_batch={"uid": batch.non_tensor_batch["uid"]})

    monkeypatch.setattr(NativeRayDiffusionTrainer, "_generate_native", generate)
    for evaluating in (False, True, False):
        trainer._evaluating = evaluating
        trainer._generate_native(DataProto(non_tensor_batch={"uid": np.array(["a", "a"])}))
    assert [item["num_inference_steps"] for item in seen] == [20, 40, 20]
    assert [item["extra_args"]["audio_cfg_scale"] for item in seen] == [3.0, 7.0, 3.0]
    assert [item["extra_args"]["video_rescale_scale"] for item in seen] == [0.0, 0.7, 0.0]
    assert seen[0] == seen[2]
    assert rollout.pipeline.num_inference_steps == 20


@pytest.mark.parametrize("prompt_count,val_n,world", [(1, 1, 8), (3, 1, 2), (2, 3, 4)])
def test_validation_unpads_before_reward_and_saving(monkeypatch, tmp_path, prompt_count, val_n, world):
    from verl_omni.pipelines.ltx2_unigrpo_nft import artifacts

    trainer = object.__new__(LTXJointRayTrainer)
    trainer.config, trainer.global_steps = _config(), 1
    trainer.config.trainer.n_gpus_per_node = world
    trainer.config.actor_rollout_ref.rollout.val_kwargs = {"n": val_n}
    trainer.config.trainer.rollout_data_dir = str(tmp_path)
    trainer.config.reward = {"reward_functions": {f"reward{i}": {} for i in range(5)}}
    trainer.val_dataloader = [{"raw_prompt": np.array([f"p{i}" for i in range(prompt_count)], dtype=object)}]
    count = prompt_count * val_n
    observed = []

    def generate(self, batch):
        assert len(batch) % world == 0
        assert len(batch) == ((count + world - 1) // world) * world
        observed.append(list(batch.non_tensor_batch["_ltx_sample_seed"]))
        # Unique sentinel for each physical request makes accidental inclusion
        # of padded outputs visible to scoring and artifact assertions.
        return DataProto.from_dict(tensors={"responses": torch.arange(len(batch)).view(-1, 1)})

    def score(batch):
        assert len(batch) == count
        assert int(batch.batch["responses"].max()) < count
        assert list(batch.non_tensor_batch["_ltx_sample_seed"]) == list(range(42, 42 + count))
        return DataProto.from_dict(tensors={"rm_scores": torch.ones(count, 5)})

    def prepare(batch, reward):
        batch.batch["text_reward_prob"] = torch.full((count,), 0.5)
        batch.batch["reward_prob"] = torch.full((count, 1, 2), 0.5)
        return batch

    def save(batch, directory, names, *, save_tensors):
        assert save_tensors is False
        assert len(batch) == count
        return [{"ended_with_eos": True} for _ in range(count)]

    monkeypatch.setattr(NativeRayDiffusionTrainer, "_generate_native", generate)
    monkeypatch.setattr(artifacts, "save_joint_samples", save)
    trainer._compute_reward_colocate, trainer._prepare_actor_batch = score, prepare
    for step in (1, 2):
        trainer.global_steps = step
        result = trainer._validate()
        assert result["val/joint/samples"] == count
        assert not trainer._evaluating
    assert observed[0] == observed[1]
    assert trainer.config.actor_rollout_ref.rollout.n == 2


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_validation_candidate_count(value):
    trainer = object.__new__(LTXJointRayTrainer)
    trainer.config = _config()
    trainer.config.actor_rollout_ref.rollout.val_kwargs = {"n": value}
    with pytest.raises(ValueError, match="val_kwargs.n"):
        trainer._validation_n()


@pytest.mark.parametrize("step,expected", [(1, False), (9, False), (10, True), (11, False), (20, True)])
def test_joint_samples_follow_ten_step_save_frequency(monkeypatch, step, expected):
    from verl_omni.pipelines.ltx2_unigrpo_nft.loss import LTXJointLoss

    trainer = object.__new__(LTXJointRayTrainer)
    trainer.config = _config()
    trainer.config.trainer.rollout_data_dir = "/unused"
    trainer.config.trainer.rollout_data_save_freq = 10
    trainer.global_steps = step
    monkeypatch.setattr(LTXJointLoss, "prepare_actor_batch", lambda batch, reward, config: batch)
    calls = []

    def save(batch, info, timings, root):
        calls.append(root)
        timings["dump_rollout_generations"] = 0.1

    trainer._log_rollout_data = save
    trainer._prepare_actor_batch(DataProto.from_dict(tensors={"value": torch.ones(1)}), torch.ones(1))
    assert bool(calls) == expected


@pytest.mark.parametrize("evaluating", [False, True])
def test_dataset_modality_reward_targets_survive_native_dispatch(monkeypatch, evaluating):
    trainer = object.__new__(LTXJointRayTrainer)
    trainer.config, trainer.global_steps = _config(), 3
    trainer._evaluating = evaluating
    trainer.config.actor_rollout_ref.rollout.val_kwargs = {"n": 2}
    targets = np.array([{"text": {"video": f"video {i}", "audio": f"audio {i}"}} for i in range(4)], dtype=object)

    def generate(self, batch):
        return DataProto(
            non_tensor_batch={
                "reward_inputs": np.array(
                    [{"text": {"video": "AV fallback", "audio": "AV fallback"}}] * len(batch), dtype=object
                )
            }
        )

    monkeypatch.setattr(NativeRayDiffusionTrainer, "_generate_native", generate)
    batch = DataProto(non_tensor_batch={"uid": np.array(["a", "a", "b", "b"]), "reward_inputs": targets})
    output = trainer._generate_native(batch)
    assert output.non_tensor_batch["reward_inputs"].tolist() == targets.tolist()
    assert "reward_inputs" not in batch.non_tensor_batch


@pytest.mark.parametrize(
    "step,interval,decay,update",
    [
        (1, 1, 0.001, True),
        (3, 1, 0.003, True),
        (800, 1, 0.5, True),
        (3, 2, 0.0, False),
    ],
)
def test_ema_schedule_updates_at_current_completed_step(step, interval, decay, update):
    config = _config()
    config.algorithm.old_policy_decay_schedule = "linear_to_0_5"
    config.algorithm.old_policy_update_interval = interval
    trainer = object.__new__(LTXJointRayTrainer)
    trainer.config, trainer.global_steps = config, step
    calls = []
    trainer.actor_rollout_wg = SimpleNamespace(ema_update_adapter=lambda **kwargs: calls.append(kwargs))
    result = trainer._update_old_policy()
    assert result == (update, decay, "ema" if update else "none")
    assert calls == ([{"source": "default", "target": "old", "decay": decay}] if update else [])


def test_full_batch_256_with_four_samples_per_rank_microbatch():
    config = _config()
    config.data.train_batch_size = 32
    config.trainer.n_gpus_per_node = 8
    config.actor_rollout_ref.rollout.n = 8
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 256
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 4
    LTXJointRayTrainer.validate_joint_config(config)
    for invalid in (0, -1, 3):
        config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = invalid
        with pytest.raises(ValueError, match="micro-batch"):
            LTXJointRayTrainer.validate_joint_config(config)
