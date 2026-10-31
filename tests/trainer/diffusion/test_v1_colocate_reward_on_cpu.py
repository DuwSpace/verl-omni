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
"""CPU test for the colocated reward-model step of the v1 diffusion trainer."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from verl import DataProto
from verl.utils import tensordict_utils as tu

import verl_omni

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")


def compose_cfg(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=overrides)


class _Stop(Exception):
    pass


def test_colocate_reward_keeps_trajectory_fields(monkeypatch):
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    trainer = PolicyGradientDiffusionTrainerV1Sync(compose_cfg(["reward.reward_model.enable=true"]))
    assert trainer.use_rm

    data = DataProto.from_tensordict(tu.get_tensordict({"all_timesteps": torch.zeros(2, 4)}))
    reward = DataProto.from_tensordict(tu.get_tensordict({"rm_scores": torch.ones(2, 1)}))
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    checkpoint_calls = []
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: checkpoint_calls.append("sleep"),
        update_weights=lambda step: checkpoint_calls.append(("update_weights", step)),
    )
    trainer.global_steps = 1
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.diffusion_tq_batch_to_dataproto",
        lambda meta, pad_token_id: data,
    )
    monkeypatch.setattr(trainer, "_compute_reward_colocate", lambda d: reward)
    captured = {}

    def stop_at_balance(d, metrics):
        captured["data"] = d
        raise _Stop

    monkeypatch.setattr(trainer, "_balance_batch", stop_at_balance)

    with pytest.raises(_Stop):
        trainer._train_sampled_batch({}, {}, object())

    assert "all_timesteps" in captured["data"].batch
    assert "rm_scores" in captured["data"].batch
    assert checkpoint_calls == []


def test_colocate_reward_keeps_rollout_asleep_through_actor_update(monkeypatch):
    """In sync mode the colocated-RM block must not wake the rollout engine.

    ``update_weights`` resumes the rollout replicas' GPU weights (~55GB for
    Qwen-Image at rollout TP=1). If it runs between the reward phase and the
    actor update, every training phase (old log-prob, ref, advantage, actor
    update) executes next to the resident rollout weights and OOMs. In sync
    mode, waking and weight-syncing belongs to ``on_step_end``, after the
    actor update.
    """
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    trainer = PolicyGradientDiffusionTrainerV1Sync(compose_cfg(["reward.reward_model.enable=true"]))
    assert trainer.use_rm

    data = DataProto.from_tensordict(tu.get_tensordict({"all_timesteps": torch.zeros(2, 4)}))
    reward = DataProto.from_tensordict(tu.get_tensordict({"rm_scores": torch.ones(2, 1)}))
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    calls: list[str] = []
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: calls.append("sleep"),
        update_weights=lambda step: calls.append("update_weights"),
    )
    trainer.global_steps = 1
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.diffusion_tq_batch_to_dataproto",
        lambda meta, pad_token_id: data,
    )
    monkeypatch.setattr(trainer, "_compute_reward_colocate", lambda d: (calls.append("reward"), reward)[1])
    monkeypatch.setattr(trainer, "_balance_batch", lambda d, metrics: d)
    old_log_prob = DataProto.from_tensordict(tu.get_tensordict({"old_log_probs": torch.zeros(2, 4)}))
    monkeypatch.setattr(trainer, "_compute_old_log_prob", lambda d: (calls.append("old_log_prob"), old_log_prob)[1])
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.compute_rollout_corr_metrics_from_batch",
        lambda data, bypass_mode: {},
    )
    monkeypatch.setattr(trainer, "_compute_advantage", lambda d: (calls.append("advantage"), d)[1])

    def stop_at_update_actor(data):
        calls.append("update_actor")
        raise _Stop

    monkeypatch.setattr(trainer, "_update_actor", stop_at_update_actor)

    with pytest.raises(_Stop):
        trainer._train_sampled_batch({}, {}, object())

    assert calls == ["reward", "old_log_prob", "advantage", "update_actor"]


def test_colocate_reward_wakes_rollout_mid_cycle_in_async_mode(monkeypatch):
    """Async modes keep the mid-cycle ``update_weights`` after the colocated RM.

    separate_async's ``on_step_end`` only weight-syncs the standalone rollout;
    the colocated replicas are woken by ``switch_to_rollout``, which is not
    guaranteed to fire (``should_switch_to_rollout`` is still a TODO and the
    direct ``sleep_replicas`` here does not set ``_colocated_slept``). Without
    the mid-cycle wake+sync, colocated generation stalls on stale weights.
    The sync subclass is used as a lightweight CPU harness for the shared
    base; only ``trainer_mode`` differs.
    """
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    trainer = PolicyGradientDiffusionTrainerV1Sync(compose_cfg(["reward.reward_model.enable=true"]))
    assert trainer.use_rm
    trainer.trainer_mode = "separate_async"

    data = DataProto.from_tensordict(tu.get_tensordict({"all_timesteps": torch.zeros(2, 4)}))
    reward = DataProto.from_tensordict(tu.get_tensordict({"rm_scores": torch.ones(2, 1)}))
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    calls: list[str] = []
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: calls.append("sleep"),
        update_weights=lambda step: calls.append("update_weights"),
    )
    trainer.global_steps = 1
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.diffusion_tq_batch_to_dataproto",
        lambda meta, pad_token_id: data,
    )
    monkeypatch.setattr(trainer, "_compute_reward_colocate", lambda d: (calls.append("reward"), reward)[1])

    def stop_at_balance(d, metrics):
        calls.append("balance")
        raise _Stop

    monkeypatch.setattr(trainer, "_balance_batch", stop_at_balance)

    with pytest.raises(_Stop):
        trainer._train_sampled_batch({}, {}, object())

    assert calls == ["sleep", "reward", "update_weights", "balance"]


@pytest.mark.parametrize("direct_preference", [False, True])
def test_colocate_reward_logs_component_means_before_actor_preparation(monkeypatch, direct_preference):
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    trainer = PolicyGradientDiffusionTrainerV1Sync(compose_cfg(["reward.reward_model.enable=true"]))
    trainer._is_direct_preference = direct_preference
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    data = DataProto.from_dict(tensors={"all_timesteps": torch.zeros(2, 4)})
    components = {
        "reward/video_align": np.array([1.0, 3.0]),
        "reward/hpsv3": np.array([6.0, 8.0]),
        "reward/audiobox": np.array([0.1, 0.3]),
        "reward/clap": np.array([0.5, 0.7]),
        "reward/desync": np.array([0.7, 0.9]),
    }
    reward = DataProto.from_dict(
        tensors={"rm_scores": torch.ones(2, 5)},
        non_tensors=components,
        meta_info={"reward_extra_keys": list(components)},
    )
    monkeypatch.setattr(trainer, "_compute_reward_colocate", lambda data: reward)

    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(trainer, "_balance_batch", stop)
    monkeypatch.setattr(trainer, "_train_direct_preference_batch", stop)
    metrics = {}
    with pytest.raises(_Stop):
        trainer._train_sampled_batch(metrics, {}, object(), data=data)
    for name, values in components.items():
        assert metrics[f"critic/{name}/mean"] == pytest.approx(values.mean())
        assert metrics[f"critic/{name}/std"] == pytest.approx(values.std(ddof=0))


def test_reward_std_combines_local_batches_by_sample_count():
    from verl.trainer.ppo.v1.utils import MetricsAggregator

    from verl_omni.trainer.diffusion.diffusion_metric_utils import (
        compute_reward_extra_metrics_diffusion,
        finalize_reward_extra_metrics_diffusion,
    )

    aggregator = MetricsAggregator()
    chunks = [np.array([1.0, 3.0]), np.array([9.0])]
    for values in chunks:
        aggregator.add_step_metrics(
            compute_reward_extra_metrics_diffusion({"reward/quality": values}, include_second_moment=True),
            sample_count=len(values),
        )
    metrics = aggregator.get_aggregated_metrics()
    finalize_reward_extra_metrics_diffusion(metrics)
    values = np.concatenate(chunks)
    assert metrics["critic/reward/quality/mean"] == pytest.approx(values.mean())
    assert metrics["critic/reward/quality/std"] == pytest.approx(values.std(ddof=0))
    assert not any("_second_moment" in key for key in metrics)


def test_reward_stats_skip_metadata_and_handle_singleton():
    from verl_omni.trainer.diffusion.diffusion_metric_utils import compute_reward_extra_metrics_diffusion

    metrics = compute_reward_extra_metrics_diffusion(
        {"reward/quality": [np.float32(2.0)], "metadata": ["text"], "empty": []}
    )
    assert metrics == {"critic/reward/quality/mean": 2.0, "critic/reward/quality/std": 0.0}
