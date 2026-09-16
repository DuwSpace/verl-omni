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
"""Lightweight integration coverage for the OmniNFT reward-to-loss path."""

import asyncio
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.pipelines.ltx2_omni_nft.config import OmniNFTLossConfig
from verl_omni.pipelines.ltx2_omni_nft.vllm_omni_rollout_adapter import LTX23OmniNFTPipeline
from verl_omni.reward_loop.reward_components import assemble_component_rewards, compute_component_rewards
from verl_omni.reward_loop.reward_manager.multimodal import MultiModalRewardManager
from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor, NativeRewardModelState
from verl_omni.trainer.diffusion.diffusion_algos import DiffusionNFTLoss, OmniNFTLoss
from verl_omni.trainer.diffusion.modality_advantage import ModalityAdvantageRouter
from verl_omni.trainer.diffusion.ray_diffusion_trainer import DirectPreferenceRayTrainer
from verl_omni.trainer.main_diffusion import _get_trainer_cls
from verl_omni.utils import fsdp_utils
from verl_omni.utils.reward_score import (
    audiobox_native,
    clap_native,
    desync_native,
    hpsv3_native,
    videoalign_native,
)
from verl_omni.workers.config.reward import RewardModelSpec
from verl_omni.workers.engine.fsdp import diffusers_impl
from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine, OmniNFTDiffusersFSDPEngine


def _worker_output(sample_uids, reward_names, scores):
    return {
        "rm_scores": torch.tensor(scores, dtype=torch.float32),
        "reward_valid_mask": torch.ones((len(sample_uids), len(reward_names)), dtype=torch.bool),
        "reward_names": reward_names,
        "sample_uid": np.asarray(sample_uids, dtype=object),
        "reward_definitions": {
            name: {
                "model": name,
                "model_revision": f"{name}-v1",
                "definition_version": "test-v1",
            }
            for name in reward_names
        },
    }


def _prepare_actor_batch():
    scoring_batch = DataProto.from_dict(
        tensors={"responses": torch.zeros((4, 1), dtype=torch.uint8)},
        non_tensors={"sample_uid": np.asarray(["s0", "s1", "s2", "s3"], dtype=object)},
    )
    chunks = scoring_batch.chunk(2)
    reward_batch = assemble_component_rewards(
        scoring_batch,
        [
            (
                "video",
                chunks[0],
                _worker_output(["s1", "s0"], ["video_quality"], [[3.0], [1.0]]),
            ),
            (
                "video",
                chunks[1],
                _worker_output(["s3", "s2"], ["video_quality"], [[2.0], [4.0]]),
            ),
            (
                "audio",
                scoring_batch,
                _worker_output(
                    ["s2", "s0", "s3", "s1"],
                    ["audio_quality"],
                    [[40.0], [10.0], [20.0], [30.0]],
                ),
            ),
        ],
        {"video_quality", "audio_quality"},
    )
    rollout_batch = DataProto.from_dict(
        tensors={
            "video_latents_clean": torch.zeros((4, 2, 2)),
            "audio_latents_clean": torch.zeros((4, 2, 2)),
            "train_timesteps": torch.arange(3).expand(4, -1).clone(),
        },
        non_tensors={"uid": np.asarray(["p0", "p0", "p1", "p1"], dtype=object)},
    )
    actor_batch = rollout_batch.union(reward_batch)
    loss_config = OmniNFTLossConfig(adv_clip_max=2.0, mix_beta=1.0)
    config = SimpleNamespace(
        algorithm=SimpleNamespace(
            norm_adv_by_std_in_grpo=False,
            global_std=False,
            adv_mode="continuous",
            timestep_fraction=1.0,
        ),
        actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(diffusion_loss=loss_config, data_loader_seed=42)),
        reward=SimpleNamespace(
            reward_functions={
                "video_quality": {"routing_weights": {"video": 1.0, "audio": 0.0}},
                "audio_quality": {"routing_weights": {"video": 0.0, "audio": 1.0}},
            },
        ),
    )
    return OmniNFTLoss.prepare_actor_batch(actor_batch, actor_batch.batch["rm_scores"], config), config


def test_worker_rewards_are_reordered_and_routed_to_modalities():
    actor_batch, _ = _prepare_actor_batch()

    torch.testing.assert_close(
        actor_batch.batch["rm_scores"],
        torch.tensor([[10.0, 1.0], [30.0, 3.0], [40.0, 4.0], [20.0, 2.0]]),
    )
    expected_video = torch.tensor([0.25, 0.75, 0.75, 0.25]).unsqueeze(1).expand(-1, 3)
    expected_audio = torch.tensor([0.0, 1.0, 1.0, 0.0]).unsqueeze(1).expand(-1, 3)
    torch.testing.assert_close(actor_batch.batch["video_reward_prob"], expected_video)
    torch.testing.assert_close(actor_batch.batch["audio_reward_prob"], expected_audio)
    assert actor_batch.meta_info["reward_extra_keys"] == ["reward/audio_quality", "reward/video_quality"]
    np.testing.assert_array_equal(actor_batch.non_tensor_batch["reward/video_quality"], [1.0, 3.0, 4.0, 2.0])


def test_component_reward_manager_uses_named_model_executor_contract():
    async def score(batch, reward_model, *, micro_batch_size):
        assert micro_batch_size == 2
        scores = await reward_model.infer(batch)
        return {
            "scores": scores,
            "valid_mask": torch.ones(len(batch), dtype=torch.bool),
            "model_revision": "test-model",
            "definition_version": "test-definition",
        }

    config = OmegaConf.create(
        {
            "reward": {
                "aggregation": "preserve_components",
                "models": {"video_model": {"backend": "native"}},
                "reward_functions": {
                    "video_quality": {
                        "path": "pkg://test.reward",
                        "name": "compute_score_batch",
                        "model": "video_model",
                        "weight": 1.0,
                        "required": True,
                        "micro_batch_size": 2,
                        "routing_weights": {"video": 1.0, "audio": 0.0},
                    }
                },
            }
        }
    )
    with patch(
        "verl_omni.reward_loop.reward_manager.multimodal.load_extern_object",
        return_value=score,
    ):
        manager = MultiModalRewardManager(config, tokenizer=None, compute_score=MagicMock())

    assert manager.config is config
    assert manager.component_names == ["video_quality"]
    assert not hasattr(manager, "_reward_entries")

    manager.set_reward_executors({}, {"video_model": _FakeNativeRewardHandle(torch.tensor([0.25, 0.75]))})
    data = DataProto.from_dict(
        tensors={"responses": torch.zeros((2, 1), dtype=torch.uint8)},
        non_tensors={"sample_uid": np.asarray(["s0", "s1"], dtype=object)},
    )
    with patch("verl_omni.reward_loop.reward_manager.multimodal._validate_visual_response"):
        result = asyncio.run(manager.run_batch(data))
    torch.testing.assert_close(result["rm_scores"], torch.tensor([[0.25], [0.75]]))
    assert result["reward_names"] == ["video_quality"]


class _FakeNativeRewardHandle:
    def __init__(self, output):
        self.output = output

    async def infer(self, *args):
        del args
        return self.output

    def reward_kwargs(self):
        return {"reward_model": self}


def test_video_and_hps_component_score_definitions_use_raw_executor_outputs():
    video_batch = [None]
    with patch.object(
        videoalign_native,
        "_extract_inputs",
        return_value=([torch.empty(1)], ["prompt"], [[0, 1]], [24.0]),
    ):
        video = asyncio.run(
            videoalign_native.compute_score_batch(
                video_batch,
                _FakeNativeRewardHandle(torch.tensor([[3.6757, 0.0, 2.8105]])),
                micro_batch_size=1,
            )
        )
    torch.testing.assert_close(video["scores"], torch.zeros(1), atol=1e-6, rtol=0)

    frame_scores = torch.tensor([1.0, 10.0, 20.0, 5.0, 14.0])
    hps_logits = torch.stack((frame_scores, torch.zeros_like(frame_scores)), dim=1)
    with patch.object(
        hpsv3_native,
        "_extract_inputs",
        return_value=([[object()] * 5], ["prompt"]),
    ):
        hps = asyncio.run(
            hpsv3_native.compute_score_batch(
                [None],
                _FakeNativeRewardHandle(hps_logits),
                micro_batch_size=1,
            )
        )
    # The source definition caps each frame at 15 and averages the top two of five.
    torch.testing.assert_close(hps["scores"], torch.tensor([14.5]))


def test_audio_component_score_definitions_keep_pairing_and_window_weights():
    with patch.object(
        clap_native,
        "_extract_inputs",
        return_value=([np.zeros(1), np.zeros(1)], ["first", "second"], [48_000, 48_000]),
    ):
        clap = asyncio.run(
            clap_native.compute_score_batch(
                [None, None],
                _FakeNativeRewardHandle(
                    {
                        "audio_embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                        "text_embeddings": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
                    }
                ),
                micro_batch_size=2,
            )
        )
    torch.testing.assert_close(clap["scores"], torch.tensor([1.0, 0.5]))

    output = {
        "predictions": {
            "CE": torch.tensor([40.0, 80.0]),
            "CU": torch.zeros(2),
            "PC": torch.zeros(2),
            "PQ": torch.zeros(2),
        },
        "target_transform": {axis: (0.0, 1.0) for axis in ("CE", "CU", "PC", "PQ")},
    }
    with (
        patch.object(audiobox_native, "_extract_inputs", return_value=([torch.zeros(1)], [16_000])),
        patch.object(
            audiobox_native,
            "_make_windows",
            return_value=(torch.zeros((2, 1)), torch.ones((2, 1)), [0, 0], [0.25, 0.75]),
        ),
    ):
        audiobox = asyncio.run(
            audiobox_native.compute_score_batch(
                [None],
                _FakeNativeRewardHandle(output),
                micro_batch_size=1,
            )
        )
    torch.testing.assert_close(audiobox["scores"], torch.tensor([1.75]))


def test_desync_component_score_definition_averages_both_comparison_windows():
    logits = torch.zeros((2, 1, 21))
    logits[0, 0, 10] = 1.0  # offset 0
    logits[1, 0, 20] = 1.0  # offset 2
    with patch.object(
        desync_native,
        "_extract_inputs",
        return_value=([torch.zeros(1)], [torch.zeros(1)], [25.0], [16_000]),
    ):
        result = asyncio.run(
            desync_native.compute_score_batch(
                [None],
                _FakeNativeRewardHandle(logits),
                micro_batch_size=1,
            )
        )
    torch.testing.assert_close(result["scores"], torch.tensor([0.5]))


def test_desync_infer_scopes_mha_compatibility_to_the_forward():
    model = object.__new__(desync_native.DeSyncNativeModel)
    model._state = object()
    original = torch.backends.mha.get_fastpath_enabled()

    def infer(state, video, audio):
        assert state is model._state
        assert video.shape == audio.shape == (1,)
        assert not torch.backends.mha.get_fastpath_enabled()
        return torch.tensor([1.0])

    with patch.object(desync_native, "_infer_micro_batch", side_effect=infer):
        torch.testing.assert_close(model.infer(torch.zeros(1), torch.zeros(1)), torch.ones(1))
    assert torch.backends.mha.get_fastpath_enabled() is original

    with (
        patch.object(desync_native, "_infer_micro_batch", side_effect=RuntimeError("forward failed")),
        pytest.raises(RuntimeError, match="forward failed"),
    ):
        model.infer(torch.zeros(1), torch.zeros(1))
    assert torch.backends.mha.get_fastpath_enabled() is original


def _movable_module():
    module = MagicMock()
    module.to.return_value = module
    module.eval.return_value = module
    return module


@pytest.mark.parametrize(
    ("adapter_cls", "state", "owned_fields"),
    [
        (
            videoalign_native.VideoAlignNativeModel,
            SimpleNamespace(model=_movable_module(), processor=object(), device=torch.device("cpu")),
            ("model", "processor"),
        ),
        (
            hpsv3_native.HPSv3NativeModel,
            SimpleNamespace(model=_movable_module(), processor=object(), device=torch.device("cpu")),
            ("model", "processor"),
        ),
        (
            audiobox_native.AudioBoxNativeModel,
            SimpleNamespace(model=_movable_module(), target_transform={"mean": 1.0}, device=torch.device("cpu")),
            ("model",),
        ),
        (
            clap_native.CLAPNativeModel,
            SimpleNamespace(model=_movable_module(), processor=object(), device=torch.device("cpu")),
            ("model", "processor"),
        ),
        (
            desync_native.DeSyncNativeModel,
            SimpleNamespace(model=_movable_module(), mel=_movable_module(), device=torch.device("cpu")),
            ("model", "mel"),
        ),
    ],
)
def test_omnift_native_adapters_move_all_owned_modules_and_close(adapter_cls, state, owned_fields):
    adapter = object.__new__(adapter_cls)
    adapter._state = state
    modules = [getattr(state, field) for field in owned_fields if hasattr(getattr(state, field), "to")]

    adapter.activate(torch.device("cuda", 2))
    assert state.device == torch.device("cuda", 2)
    for module in modules:
        module.to.assert_called_with(torch.device("cuda", 2))

    adapter.offload_to_cpu()
    assert state.device == torch.device("cpu")
    for module in modules:
        module.to.assert_called_with(torch.device("cpu"))

    adapter.close()
    assert state.device is None
    for field in owned_fields:
        assert getattr(state, field) is None


def test_routed_rewards_drive_both_omnift_loss_branches():
    actor_batch, config = _prepare_actor_batch()
    video_prediction = torch.full((4, 2, 2), 0.25, requires_grad=True)
    audio_prediction = torch.full((4, 2, 2), -0.25, requires_grad=True)

    def branch_inputs(forward_prediction):
        return {
            "forward_prediction": forward_prediction,
            "old_prediction": torch.zeros_like(forward_prediction),
            "ref_forward_prediction": torch.full_like(forward_prediction, 0.1),
            "x0": torch.zeros_like(forward_prediction),
            "xt": torch.ones_like(forward_prediction),
            "t_expanded": torch.full_like(forward_prediction, 0.5),
        }

    loss, metrics = OmniNFTLoss.compute_loss(
        **{f"video_{key}": value for key, value in branch_inputs(video_prediction).items()},
        video_reward_prob=actor_batch.batch["video_reward_prob"],
        **{f"audio_{key}": value for key, value in branch_inputs(audio_prediction).items()},
        audio_reward_prob=actor_batch.batch["audio_reward_prob"],
        config=config.actor_rollout_ref.actor,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert video_prediction.grad is not None and torch.isfinite(video_prediction.grad).all()
    assert audio_prediction.grad is not None and torch.isfinite(audio_prediction.grad).all()
    assert {"actor/video/policy_loss", "actor/audio/policy_loss", "actor/total_loss"} <= metrics.keys()


def _scalar_branch(forward_value, *, old_value=0.0, ref_value=0.25):
    forward = torch.tensor([[forward_value]], requires_grad=True)
    old = torch.tensor([[old_value]], requires_grad=True)
    reference = torch.tensor([[ref_value]], requires_grad=True)
    return (
        forward,
        old,
        reference,
        {
            "forward_prediction": forward,
            "old_prediction": old,
            "ref_forward_prediction": reference,
            "x0": torch.zeros((1, 1)),
            "xt": torch.ones((1, 1)),
            "t_expanded": torch.full((1, 1), 0.5),
        },
    )


def _scalar_omnift_loss(video_probability, audio_probability):
    video_forward, video_old, video_reference, video = _scalar_branch(0.25)
    audio_forward, audio_old, audio_reference, audio = _scalar_branch(0.25)
    loss_config = OmniNFTLossConfig(
        adv_clip_max=2.0,
        mix_beta=1.0,
        video_weight=1.0,
        audio_weight=3.0,
        video_ref_kl_coef=0.4,
        audio_ref_kl_coef=0.6,
    )
    loss, _ = OmniNFTLoss.compute_loss(
        **{f"video_{key}": value for key, value in video.items()},
        video_reward_prob=torch.tensor([video_probability]),
        **{f"audio_{key}": value for key, value in audio.items()},
        audio_reward_prob=torch.tensor([audio_probability]),
        config=SimpleNamespace(diffusion_loss=loss_config),
    )
    loss.backward()
    return loss.detach(), (video_forward, video_old, video_reference), (audio_forward, audio_old, audio_reference)


def test_omnift_loss_matches_scalar_reference_and_stops_old_reference_gradients():
    loss, video, audio = _scalar_omnift_loss(video_probability=0.8, audio_probability=0.2)

    # For x0=0, xt=1, t=.5, beta=1 and prediction=.25, the detached
    # adaptive denominators make positive/negative losses .875 and 1.125.
    # Applying adv_clip_max=2 gives video=1.85 and audio=2.15; modality
    # weights 1:3 yield 2.075. Reference terms are zero at this fixture.
    torch.testing.assert_close(loss, torch.tensor(2.075), rtol=0, atol=1e-6)
    torch.testing.assert_close(video[0].grad, torch.tensor([[-0.3]]), rtol=0, atol=1e-6)
    torch.testing.assert_close(audio[0].grad, torch.tensor([[0.9]]), rtol=0, atol=1e-6)
    assert video[1].grad is None and video[2].grad is None
    assert audio[1].grad is None and audio[2].grad is None


def test_changing_one_modality_reward_changes_only_that_modality_gradient():
    _, baseline_video, baseline_audio = _scalar_omnift_loss(video_probability=0.8, audio_probability=0.2)
    _, changed_video, changed_audio = _scalar_omnift_loss(video_probability=0.1, audio_probability=0.2)

    assert not torch.equal(baseline_video[0].grad, changed_video[0].grad)
    torch.testing.assert_close(baseline_audio[0].grad, changed_audio[0].grad, rtol=0, atol=0)


def test_noncollinear_rewards_use_per_column_global_std_and_weighted_shared_routing():
    scores = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [2.0, 4.0, 0.0],
            [10.0, 1.0, 3.0],
            [14.0, 5.0, 7.0],
        ]
    )
    advantages = ModalityAdvantageRouter.compute_reward_advantages(
        scores,
        uid=["p0", "p0", "p1", "p1"],
        norm_by_std=True,
        global_std=True,
    )
    centered = torch.tensor(
        [
            [-1.0, -2.0, 1.0],
            [1.0, 2.0, -1.0],
            [-2.0, -2.0, -2.0],
            [2.0, 2.0, 2.0],
        ]
    )
    expected = centered / (torch.tensor([32.75, 4.25, 6.5]).sqrt() + 1e-4)
    torch.testing.assert_close(advantages, expected)

    routing = torch.tensor([[2.0, 0.0], [0.0, 0.5], [1.5, 3.0]])
    routed = ModalityAdvantageRouter.route(advantages, routing)
    torch.testing.assert_close(routed, expected @ routing)
    # The third reward is deliberately shared, with distinct non-unit weights.
    assert torch.allclose(routed[:, 0], 2.0 * expected[:, 0] + 1.5 * expected[:, 2])
    assert torch.allclose(routed[:, 1], 0.5 * expected[:, 1] + 3.0 * expected[:, 2])


def test_omnift_uses_standard_direct_preference_trainer_and_reward_lifecycle():
    config = SimpleNamespace(
        algorithm=SimpleNamespace(trainer_type="direct_preference"),
        actor_rollout_ref=SimpleNamespace(model=SimpleNamespace(algorithm="omni_nft")),
    )
    assert _get_trainer_cls(config) is DirectPreferenceRayTrainer

    trainer = object.__new__(DirectPreferenceRayTrainer)
    trainer.reward_loop_manager = MagicMock()
    manager = trainer.reward_loop_manager
    trainer.shutdown()
    trainer.shutdown()

    manager.shutdown.assert_called_once_with()
    assert trainer.reward_loop_manager is None


def test_required_reward_failure_prevents_actor_update(monkeypatch):
    """The online direct-preference step must fail closed before actor mutation."""

    class _Tracking:
        def __init__(self, **kwargs):
            del kwargs

    monkeypatch.setattr("verl.utils.tracking.Tracking", _Tracking)

    trainer = object.__new__(DirectPreferenceRayTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "required-reward-failure",
                "logger": ["console"],
                "val_before_train": False,
                "total_epochs": 1,
            },
            "actor_rollout_ref": {"rollout": {"n": 1, "seed": None}},
            "global_profiler": {"steps": [], "profile_continuous_steps": False},
        }
    )
    trainer.total_training_steps = 1
    trainer.train_dataloader = [{"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}]
    trainer._has_old_adapter = False
    trainer.is_offline = False
    trainer.use_rm = True
    trainer.enable_agent_reward_loop = False
    trainer.actor_rollout_wg = SimpleNamespace()
    trainer._load_checkpoint = lambda: None
    trainer._start_profiling = lambda enabled: None
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda step: None, sleep_replicas=lambda: None)
    trainer._get_gen_batch = lambda batch: DataProto.from_dict(
        tensors={"prompts": torch.zeros((len(batch), 1), dtype=torch.long)}
    )

    def generate_sequences(data):
        data.meta_info["timing"] = {}
        return data.union(DataProto.from_dict(tensors={"responses": torch.zeros((len(data), 1), dtype=torch.uint8)}))

    trainer.async_rollout_manager = SimpleNamespace(generate_sequences=generate_sequences)
    trainer._compute_reward_colocate = MagicMock(side_effect=RuntimeError("required component failed"))
    trainer._update_actor = MagicMock()

    with pytest.raises(RuntimeError, match="required component failed"):
        trainer.fit()

    trainer._compute_reward_colocate.assert_called_once()
    trainer._update_actor.assert_not_called()


def test_omnift_scopes_gradient_checkpoint_dtype_workaround_to_its_engine():
    standard_module = MagicMock()
    DiffusersFSDPEngine._enable_gradient_checkpointing(None, standard_module)
    standard_module.enable_gradient_checkpointing.assert_called_once_with()

    omnift_module = MagicMock()
    omnift_module._keep_in_fp32_modules = None
    engine = object.__new__(OmniNFTDiffusersFSDPEngine)
    engine.engine_config = SimpleNamespace(mixed_precision={"param_dtype": "bf16"})
    engine.model_config = object()
    engine._enable_gradient_checkpointing(omnift_module)

    kwargs = omnift_module.enable_gradient_checkpointing.call_args.kwargs
    assert callable(kwargs["gradient_checkpointing_func"])


def test_affine_free_rms_norm_fallback_preserves_dtype_and_matches_formula():
    rms_norm = SimpleNamespace(eps=1e-6)
    hidden_states = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16)

    actual = OmniNFTDiffusersFSDPEngine._npu_affine_free_rms_norm_forward(rms_norm, hidden_states)
    variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
    expected = (hidden_states * torch.rsqrt(variance + rms_norm.eps)).to(hidden_states.dtype)

    assert actual.dtype == hidden_states.dtype
    torch.testing.assert_close(actual, expected)


class _RemoteComponentCall:
    def __init__(self, calls, reward_name, offset):
        self.calls = calls
        self.reward_name = reward_name
        self.offset = offset

    async def remote(self, chunk):
        sample_uids = list(reversed(chunk.non_tensor_batch["sample_uid"].tolist()))
        self.calls.append((self.reward_name, sample_uids))
        scores = [[float(uid[1:]) + self.offset] for uid in sample_uids]
        return _worker_output(sample_uids, [self.reward_name], scores)


class _ComponentWorker:
    def __init__(self, calls, reward_name, offset):
        self.compute_score_components = _RemoteComponentCall(calls, reward_name, offset)


def test_component_rewards_merge_independent_models_and_non_divisible_replicas():
    data = DataProto.from_dict(
        tensors={"responses": torch.zeros((8, 1), dtype=torch.uint8)},
        non_tensors={"sample_uid": np.asarray([f"s{index}" for index in range(8)], dtype=object)},
    )
    calls = []
    worker_groups = {
        "video_model": [_ComponentWorker(calls, "video_quality", 0.0) for _ in range(3)],
        "audio_model": [_ComponentWorker(calls, "audio_quality", 100.0)],
    }

    result = asyncio.run(compute_component_rewards(worker_groups, data, {"video_quality", "audio_quality"}))

    video_calls = [uids for name, uids in calls if name == "video_quality"]
    audio_calls = [uids for name, uids in calls if name == "audio_quality"]
    assert sorted(uid for call in video_calls for uid in call) == [f"s{index}" for index in range(8)]
    assert sorted(len(call) for call in video_calls) == [2, 3, 3]
    assert len(audio_calls) == 1 and len(audio_calls[0]) == 8
    np.testing.assert_array_equal(result.non_tensor_batch["sample_uid"], [f"s{index}" for index in range(8)])
    assert result.meta_info["reward_names"] == ["audio_quality", "video_quality"]
    torch.testing.assert_close(result.batch["rm_scores"][:, 0], torch.arange(8, dtype=torch.float32) + 100)
    torch.testing.assert_close(result.batch["rm_scores"][:, 1], torch.arange(8, dtype=torch.float32))


def _native_executor(model):
    executor = NativeRewardExecutor(RewardModelSpec(name="test", backend="native", executor_config={}))
    executor._model = model
    executor._state = NativeRewardModelState.DEVICE
    return executor


def test_native_executor_cancellation_waits_for_live_inference_before_close():
    async def exercise():
        events = []
        started = threading.Event()
        release = threading.Event()

        class Model:
            def infer(self):
                events.append("infer_start")
                started.set()
                assert release.wait(timeout=5)
                events.append("infer_end")
                return 1

            def close(self):
                events.append("close")

        executor = _native_executor(Model())
        task = asyncio.create_task(executor.infer())
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        sleep_task = asyncio.create_task(executor.sleep())
        await asyncio.sleep(0.02)
        assert "close" not in events
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await sleep_task
        assert events.index("infer_end") < events.index("close")

    asyncio.run(exercise())


@pytest.mark.parametrize("raises", [False, True])
def test_native_executor_completion_and_error_allow_ordered_close(raises):
    async def exercise():
        events = []

        class Model:
            def infer(self):
                events.append("infer")
                if raises:
                    raise RuntimeError("scoring failed")
                return 1

            def close(self):
                events.append("close")

        executor = _native_executor(Model())
        if raises:
            with pytest.raises(RuntimeError, match="scoring failed"):
                await executor.infer()
        else:
            assert await executor.infer() == 1
        await executor.sleep()
        assert events == ["infer", "close"]

    asyncio.run(exercise())


def test_train_timestep_selection_preserves_fractional_grid_and_integer_callers():
    fractional = torch.tensor([[999.5, 501.25, 123.75], [888.5, 444.25, 11.75]], dtype=torch.float32)
    selected = DiffusionNFTLoss._select_train_timesteps(fractional, timestep_fraction=1.0, seed=7)
    assert selected.dtype == fractional.dtype
    torch.testing.assert_close(selected.sort(dim=1).values, fractional.sort(dim=1).values)
    assert torch.any(selected != selected.round())

    integer = torch.tensor([[9, 5, 1]], dtype=torch.long)
    selected_integer = DiffusionNFTLoss._select_train_timesteps(integer, timestep_fraction=1.0, seed=7)
    assert selected_integer.dtype == torch.long
    torch.testing.assert_close(selected_integer.sort(dim=1).values, integer.sort(dim=1).values)


@pytest.mark.parametrize("fails", [False, True])
def test_ltx_rollout_releases_captured_state_on_success_and_error(fails):
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    retained = torch.ones(2)

    def implementation(req, **kwargs):
        del req, kwargs
        pipeline._omni_nft_prompt_context = retained
        pipeline._omni_nft_clean_state = retained
        pipeline._omni_nft_forward_context = retained
        if fails:
            raise RuntimeError("rollout failed")
        return "result"

    pipeline._forward_impl = implementation
    if fails:
        with pytest.raises(RuntimeError, match="rollout failed"):
            pipeline.forward(object())
    else:
        assert pipeline.forward(object()) == "result"
    assert pipeline._omni_nft_prompt_context is None
    assert pipeline._omni_nft_clean_state is None
    assert pipeline._omni_nft_forward_context is None


def test_checkpoint_cast_follows_fp32_preservation_policy():
    module = MagicMock()
    module._keep_in_fp32_modules = ["sensitive"]
    engine = object.__new__(OmniNFTDiffusersFSDPEngine)
    engine.engine_config = SimpleNamespace(mixed_precision={"param_dtype": "bf16"})
    engine.model_config = object()
    model_cls = SimpleNamespace(preserve_fp32_modules=lambda: True)

    with (
        patch.object(diffusers_impl.DiffusionModelBase, "get_class", return_value=model_cls),
        patch.object(
            diffusers_impl,
            "_fsdp2_gradient_checkpointing_with_cast_func",
            return_value="checkpoint",
        ) as build,
    ):
        engine._enable_gradient_checkpointing(module)

    build.assert_called_once_with(None)
    module.enable_gradient_checkpointing.assert_called_once_with(gradient_checkpointing_func="checkpoint")


def test_checkpoint_cast_converts_only_floating_tensors_when_enabled():
    observed = {}

    class Module(torch.nn.Module):
        def forward(self, floating, integral):
            observed["dtypes"] = (floating.dtype, integral.dtype)
            return floating

    def eager_checkpoint(function, *args, **kwargs):
        kwargs.pop("use_reentrant")
        return function(*args, **kwargs)

    with patch.object(diffusers_impl, "checkpoint", side_effect=eager_checkpoint):
        checkpoint_func = diffusers_impl._fsdp2_gradient_checkpointing_with_cast_func(torch.bfloat16)
        checkpoint_func(Module(), torch.ones(2, dtype=torch.float32), torch.ones(2, dtype=torch.long))
    assert observed["dtypes"] == (torch.bfloat16, torch.long)


def test_omnift_loss_config_rejects_zero_total_modality_weight():
    with pytest.raises(ValueError, match="At least one"):
        OmniNFTLossConfig(video_weight=0.0, audio_weight=0.0)


class _LoRAHolder(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.lora_A = torch.nn.Linear(1, 1, bias=False)
        self.lora_A.weight.data.fill_(value)


def _fake_peft_state_dict(peft_model, state_dict, adapter_name):
    del peft_model, adapter_name
    return {name: parameter for name, parameter in state_dict.items() if "lora_" in name}


def test_layered_lora_collection_visits_nested_fsdp_units_exactly_once():
    root = torch.nn.Module()
    root.transformer_blocks = torch.nn.ModuleList([torch.nn.Module()])
    parent = root.transformer_blocks[0]
    parent.direct = _LoRAHolder(1.0)
    parent.nested = torch.nn.Module()
    parent.nested.inner = _LoRAHolder(2.0)
    fsdp_units = {id(parent), id(parent.nested)}

    with (
        patch.object(fsdp_utils, "fsdp_version", side_effect=lambda module: 2 if id(module) in fsdp_units else 0),
        patch.object(fsdp_utils, "get_peft_model_state_dict", side_effect=_fake_peft_state_dict),
        patch.object(fsdp_utils, "_param_to_cpu", side_effect=lambda parameter: parameter.detach().clone()),
        patch("verl.utils.device.get_torch_device", return_value=SimpleNamespace(empty_cache=lambda: None)),
    ):
        result = fsdp_utils._layered_summon_lora_params_diffusers(root)

    assert set(result) == {
        "transformer_blocks.0.direct.lora_A.weight",
        "transformer_blocks.0.nested.inner.lora_A.weight",
    }
    assert result["transformer_blocks.0.direct.lora_A.weight"].shape == (1, 1)
    assert result["transformer_blocks.0.nested.inner.lora_A.weight"].item() == 2.0


def test_layered_lora_collection_does_not_skip_fsdp1_flat_parameter_unit():
    class FlatBeforeSummon(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.full_lora = torch.nn.Parameter(torch.tensor([[3.0]]))
            self.flat = torch.nn.Parameter(torch.tensor([0.0]))
            self.summoned = False

        def named_parameters(self, *args, **kwargs):
            del args, kwargs
            if self.summoned:
                yield "proj.lora_A.weight", self.full_lora
            else:
                yield "_flat_param", self.flat

    root = torch.nn.Module()
    root.transformer_blocks = torch.nn.ModuleList([FlatBeforeSummon()])
    unit = root.transformer_blocks[0]

    @contextmanager
    def summon_full_params(module, **kwargs):
        assert kwargs["recurse"] is False
        module.summoned = True
        try:
            yield
        finally:
            module.summoned = False

    with (
        patch.object(fsdp_utils, "fsdp_version", side_effect=lambda module: 1 if module is unit else 0),
        patch.object(fsdp_utils, "get_peft_model_state_dict", side_effect=_fake_peft_state_dict),
        patch.object(fsdp_utils, "_param_to_cpu", side_effect=lambda parameter: parameter.detach().clone()),
        patch(
            "torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params",
            side_effect=summon_full_params,
        ),
        patch("verl.utils.device.get_torch_device", return_value=SimpleNamespace(empty_cache=lambda: None)),
    ):
        result = fsdp_utils._layered_summon_lora_params_diffusers(root)

    assert set(result) == {"transformer_blocks.0.proj.lora_A.weight"}
    assert result["transformer_blocks.0.proj.lora_A.weight"].shape == (1, 1)
    assert result["transformer_blocks.0.proj.lora_A.weight"].item() == 3.0
