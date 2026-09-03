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
"""CPU contract tests for the staged LTX-2.3 OmniNFT recipe."""

import asyncio
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics
from verl.protocol import DataProto


REPO_ROOT = Path(__file__).parents[2]
RECIPE = REPO_ROOT / "examples/omnift_trainer/ltx2/run_ltx2_3_omninft_lora_npu.sh"


def _run_recipe_launcher(tmp_path, reward_parallel_groups=None):
    ascend_home = tmp_path / "ascend-toolkit"
    nnal_home = tmp_path / "nnal" / "atb"
    fake_bin = tmp_path / "bin"
    for directory in (ascend_home, nnal_home, fake_bin):
        directory.mkdir(parents=True)
    for env_script in (ascend_home / "set_env.sh", nnal_home / "set_env.sh"):
        env_script.write_text("", encoding="utf-8")

    captured_args = tmp_path / "python-args.txt"
    python_stub = fake_bin / "python3"
    python_stub.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$CAPTURED_ARGS"\n', encoding="utf-8")
    python_stub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "ASCEND_HOME_PATH": str(ascend_home),
            "CAPTURED_ARGS": str(captured_args),
            "OUTPUT_DIR": str(tmp_path / "outputs"),
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
        }
    )
    if reward_parallel_groups is None:
        env.pop("REWARD_PARALLEL_GROUPS", None)
    else:
        env["REWARD_PARALLEL_GROUPS"] = reward_parallel_groups

    result = subprocess.run([str(RECIPE)], cwd=REPO_ROOT, env=env, text=True, capture_output=True)
    args = captured_args.read_text(encoding="utf-8").splitlines() if captured_args.exists() else []
    return result, args


def test_recipe_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(RECIPE)], check=True)


def test_recipe_defaults_to_sequential_native_rewards(tmp_path):
    result, args = _run_recipe_launcher(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "reward.component_order=[video_align,hpsv3,audiobox,clap,desync]" in args
    assert not any("reward.native.parallel_groups" in arg for arg in args)


def test_recipe_opt_in_adds_only_audiobox_clap_parallel_group(tmp_path):
    result, args = _run_recipe_launcher(tmp_path, reward_parallel_groups="audiobox_clap")

    assert result.returncode == 0, result.stderr
    assert "reward.component_order=[video_align,hpsv3,audiobox,clap,desync]" in args
    assert [arg for arg in args if "reward.native.parallel_groups" in arg] == [
        "+reward.native.parallel_groups.audiobox_clap.rewards=[audiobox,clap]"
    ]


def test_recipe_rejects_unapproved_parallel_group_values(tmp_path):
    result, args = _run_recipe_launcher(tmp_path, reward_parallel_groups="video_align_hpsv3")

    assert result.returncode == 2
    assert args == []
    assert "expected empty or audiobox_clap" in result.stderr


def test_omninft_worker_reuses_shared_run_and_postprocess():
    from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopWorker
    from verl_omni.pipelines.ltx2_omni_nft.agent_loop import LTX2OmniNFTAgentLoopWorker

    assert LTX2OmniNFTAgentLoopWorker._run_agent_loop is DiffusionAgentLoopWorker._run_agent_loop
    assert LTX2OmniNFTAgentLoopWorker._agent_loop_postprocess is DiffusionAgentLoopWorker._agent_loop_postprocess


def test_omninft_worker_assigns_unique_sample_uid_after_prompt_group_expansion(monkeypatch):
    from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopWorker
    from verl_omni.pipelines.ltx2_omni_nft.agent_loop import LTX2OmniNFTAgentLoopWorker

    async def fake_generate(self, batch):
        return batch

    monkeypatch.setattr(DiffusionAgentLoopWorker, "generate_sequences", fake_generate)
    batch = DataProto.from_dict(
        tensors={"placeholder": torch.zeros(8)},
        non_tensors={"uid": np.array(["prompt-0"] * 8, dtype=object)},
    )
    worker = object.__new__(LTX2OmniNFTAgentLoopWorker)
    result = asyncio.run(worker.generate_sequences(batch))

    sample_uids = result.non_tensor_batch["sample_uid"]
    assert sample_uids.shape == (8,)
    assert len(set(sample_uids.tolist())) == 8
    assert set(result.non_tensor_batch["uid"].tolist()) == {"prompt-0"}


def test_omninft_worker_rejects_duplicate_sample_uid(monkeypatch):
    from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopWorker
    from verl_omni.pipelines.ltx2_omni_nft.agent_loop import LTX2OmniNFTAgentLoopWorker

    async def fake_generate(self, batch):
        return batch

    monkeypatch.setattr(DiffusionAgentLoopWorker, "generate_sequences", fake_generate)
    batch = DataProto.from_dict(
        tensors={"placeholder": torch.zeros(2)},
        non_tensors={"sample_uid": np.array(["duplicate", "duplicate"], dtype=object)},
    )
    worker = object.__new__(LTX2OmniNFTAgentLoopWorker)

    with pytest.raises(ValueError, match="must be unique"):
        asyncio.run(worker.generate_sequences(batch))


def test_g8_agent_loop_output_round_trips_without_identity_or_tensor_drift(tmp_path):
    from verl_omni.agent_loop.diffusion_agent_loop import (
        DiffusionAgentLoopWorker,
        _InternalDiffusionAgentLoopOutput,
    )

    tensor_specs = {
        "audio": ((1, 2, 8), torch.float32),
        "video_latents_clean": ((1, 5, 4), torch.float32),
        "audio_latents_clean": ((1, 7, 4), torch.float32),
        "prompt_embeds": ((1, 3, 4), torch.float32),
        "audio_prompt_embeds": ((1, 3, 4), torch.float32),
        "prompt_embeds_mask": ((1, 3), torch.long),
        "negative_prompt_embeds": ((1, 3, 4), torch.float32),
        "negative_audio_prompt_embeds": ((1, 3, 4), torch.float32),
        "negative_prompt_embeds_mask": ((1, 3), torch.long),
        "video_latent_shape": ((1, 2), torch.long),
        "audio_latent_shape": ((1, 2), torch.long),
        "video_seq_len": ((1,), torch.long),
        "audio_seq_len": ((1,), torch.long),
        "fps": ((1,), torch.float32),
        "audio_sample_rate": ((1,), torch.long),
        "train_timesteps": ((1, 2), torch.float32),
    }
    outputs = []
    for index in range(8):
        extra_fields = {
            key: torch.full(shape, index + field_index, dtype=dtype)
            for field_index, (key, (shape, dtype)) in enumerate(tensor_specs.items())
        }
        extra_fields["raw_prompt"] = [{"role": "user", "content": "a joint prompt"}]
        outputs.append(
            _InternalDiffusionAgentLoopOutput(
                prompt_ids=torch.tensor([[index, index + 1]], dtype=torch.long),
                response_diffusion_output=torch.full((1, 9, 3, 2, 2), index, dtype=torch.uint8),
                metrics=AgentLoopMetrics(),
                extra_fields=extra_fields,
            )
        )

    sample_uids = np.array([f"sample-{index}" for index in range(8)], dtype=object)
    worker = object.__new__(DiffusionAgentLoopWorker)
    replay = worker._postprocess(
        outputs,
        input_non_tensor_batch={
            "uid": np.array(["prompt-0"] * 8, dtype=object),
            "sample_uid": sample_uids,
        },
    )
    artifact_path = tmp_path / "replay.pkl"
    replay.save_to_disk(artifact_path)
    restored = DataProto.load_from_disk(artifact_path)

    assert "rollout_log_probs" not in replay.batch
    assert set(restored.batch.keys()) == set(replay.batch.keys())
    for key in replay.batch.keys():
        assert replay.batch[key].device.type == "cpu"
        assert restored.batch[key].device.type == "cpu"
        assert restored.batch[key].dtype == replay.batch[key].dtype
        torch.testing.assert_close(restored.batch[key], replay.batch[key])
    assert restored.batch["responses"][:, 0, 0, 0, 0].tolist() == list(range(8))
    assert restored.batch["responses"].dtype == torch.uint8
    assert restored.batch["prompt_embeds"].dtype == torch.float32
    assert restored.batch["prompt_embeds_mask"].dtype == torch.long
    assert restored.batch["audio_sample_rate"].dtype == torch.long
    assert restored.non_tensor_batch["uid"].tolist() == ["prompt-0"] * 8
    assert restored.non_tensor_batch["sample_uid"].tolist() == sample_uids.tolist()
    assert len(set(restored.non_tensor_batch["sample_uid"].tolist())) == 8
    assert restored.non_tensor_batch["raw_prompt"].tolist() == [
        [{"role": "user", "content": "a joint prompt"}]
    ] * 8
    assert restored.meta_info == replay.meta_info


def test_recipe_training_prerequisites_are_registered():
    from verl_omni.pipelines.ltx2_flow_grpo.diffusers_training_adapter import LTX23FlowGRPO
    from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase
    from verl_omni.trainer.diffusion.diffusion_algos import OmniNFTLoss, get_diffusion_loss_fn

    training_adapter = DiffusionModelBase.get_class_by_name("LTX2Pipeline", "omni_nft")
    rollout_adapter = VllmOmniPipelineBase.get_class("LTX2Pipeline", "omni_nft")

    assert training_adapter.__name__ == "LTX23OmniNFT"
    assert issubclass(training_adapter, DiffusionModelBase)
    assert not issubclass(training_adapter, LTX23FlowGRPO)
    assert rollout_adapter is not None
    assert isinstance(get_diffusion_loss_fn("omni_nft"), OmniNFTLoss)


def test_omninft_training_adapter_builds_one_shot_joint_av_inputs():
    from tensordict import TensorDict

    from verl_omni.pipelines.ltx2_omni_nft.diffusers_training_adapter import LTX23OmniNFT

    batch_size = 2
    micro_batch = TensorDict(
        {
            "audio_prompt_embeds": torch.randn(batch_size, 6, 4),
            "video_seq_len": torch.full((batch_size,), 5),
        },
        batch_size=batch_size,
    )
    model_config = SimpleNamespace(
        pipeline=SimpleNamespace(
            num_frames=81,
            height=256,
            width=384,
            frame_rate=24.0,
            guidance_scale=1.0,
        )
    )
    latents = torch.randn(batch_size, 8, 4)
    timesteps = torch.tensor([900.0, 500.0])
    prompt_embeds = torch.randn(batch_size, 6, 4)
    prompt_mask = torch.ones(batch_size, 6)

    inputs, negative_inputs = LTX23OmniNFT.prepare_model_inputs(
        module=None,
        model_config=model_config,
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_mask,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=micro_batch,
        step=3,
    )

    torch.testing.assert_close(inputs["hidden_states"], latents[:, :5])
    torch.testing.assert_close(inputs["audio_hidden_states"], latents[:, 5:])
    torch.testing.assert_close(inputs["timestep"], timesteps)
    assert negative_inputs is None


def test_omninft_training_adapter_applies_independent_video_audio_cfg_scales():
    from verl_omni.pipelines.ltx2_omni_nft.diffusers_training_adapter import LTX23OmniNFT

    video_sample = torch.zeros(1, 2, 1)
    audio_sample = torch.zeros(1, 1, 1)
    video_positive = torch.full_like(video_sample, 2.0)
    audio_positive = torch.full_like(audio_sample, 4.0)
    video_negative = torch.full_like(video_sample, 1.0)
    audio_negative = torch.full_like(audio_sample, 1.0)
    module = Mock(side_effect=[(video_positive, audio_positive), (video_negative, audio_negative)])
    model_config = SimpleNamespace(
        pipeline=SimpleNamespace(guidance_scale=None, video_cfg_scale=2.0, audio_cfg_scale=3.0)
    )
    model_inputs = {
        "hidden_states": video_sample,
        "audio_hidden_states": audio_sample,
        "timestep": torch.tensor([500.0]),
    }

    video_prediction, audio_prediction = LTX23OmniNFT.forward(
        module,
        model_config,
        model_inputs,
        negative_model_inputs={},
    )

    torch.testing.assert_close(video_prediction, torch.full_like(video_sample, 3.0))
    torch.testing.assert_close(audio_prediction, torch.full_like(audio_sample, 10.0))
    assert module.call_count == 2


def test_omninft_training_adapter_rejects_reverse_transition_api():
    from verl_omni.pipelines.ltx2_omni_nft.diffusers_training_adapter import LTX23OmniNFT

    with pytest.raises(NotImplementedError, match="does not sample reverse transitions"):
        LTX23OmniNFT.forward_and_sample_previous_step(None, None, None, {}, None, None, 0)


def test_omninft_engine_rejects_non_fsdp2_and_context_parallelism():
    from verl_omni.workers.engine.fsdp.diffusers_impl import _validate_omni_nft_fsdp2_config

    _validate_omni_nft_fsdp2_config(SimpleNamespace(strategy="fsdp2", ulysses_sequence_parallel_size=1))
    with pytest.raises(NotImplementedError, match="only actor.strategy=fsdp2"):
        _validate_omni_nft_fsdp2_config(SimpleNamespace(strategy="fsdp", ulysses_sequence_parallel_size=1))
    with pytest.raises(NotImplementedError, match="does not implement Ulysses/context parallelism"):
        _validate_omni_nft_fsdp2_config(SimpleNamespace(strategy="fsdp2", ulysses_sequence_parallel_size=2))
