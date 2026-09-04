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
"""NPU smoke for OmniNFT request-level batching through the trainer generate path.

Uses ``s5_2_data_1k`` parquet and the official
``async_rollout_manager.generate_sequences(gen_batch_output)`` call.

Example:

    docker exec -e PYTHONPATH=/repo -e OMNIFT_NPU_DATA_DIR=/repo/outputs/s5_2_data_1k \
      -e MODEL_PATH=/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee8edcf067e838b843f926ec4d4cc9b2be1aaf \
      -e NUM_GPUS=8 -e ROLLOUT_TP=8 -e MAX_NUM_SEQS=2 -e ROLLOUT_N=2 \
      verl-omni-omnift-rollout-debug bash -lc \
      'cd /repo && python -m pytest -s tests/trainer/diffusion/test_ltx2_omni_nft_generate_sequences_on_npu.py'
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.device import auto_set_device

from verl_omni.trainer.diffusion.ray_diffusion_trainer import DirectPreferenceRayTrainer
from verl_omni.trainer.main_diffusion import TaskRunner
from verl_omni.utils.diffusion_attention import fallback_fa3_if_unavailable, validate_attention_consistency

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Ascend NPU is required",
)

_DEFAULT_DATA_DIRS = (
    "/repo/outputs/s5_2_data_1k",
    "/home/c00987196/verl-omni/outputs/s5_2_data_1k",
)
_DEFAULT_MODEL = "/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee8edcf067e838b843f926ec4d4cc9b2be1aaf"
_LORA_TARGETS = (
    "['attn1.to_q','attn1.to_k','attn1.to_v','attn1.to_out.0','attn2.to_q','attn2.to_k','attn2.to_v',"
    "'attn2.to_out.0','audio_attn1.to_q','audio_attn1.to_k','audio_attn1.to_v','audio_attn1.to_out.0',"
    "'audio_attn2.to_q','audio_attn2.to_k','audio_attn2.to_v','audio_attn2.to_out.0',"
    "'audio_to_video_attn.to_q','audio_to_video_attn.to_k','audio_to_video_attn.to_v',"
    "'audio_to_video_attn.to_out.0','video_to_audio_attn.to_q','video_to_audio_attn.to_k',"
    "'video_to_audio_attn.to_v','video_to_audio_attn.to_out.0','ff.net.0.proj','ff.net.2',"
    "'audio_ff.net.0.proj','audio_ff.net.2']"
)
_RL_KEYS = (
    "audio",
    "video_latents_clean",
    "audio_latents_clean",
    "train_timesteps",
    "video_latent_shape",
    "audio_latent_shape",
    "video_seq_len",
    "audio_seq_len",
    "fps",
    "audio_sample_rate",
)
_PROMPT_KEYS = (
    "prompt_embeds",
    "audio_prompt_embeds",
    "prompt_embeds_mask",
    "negative_prompt_embeds",
    "negative_audio_prompt_embeds",
    "negative_prompt_embeds_mask",
)


def _data_dir() -> Path:
    candidates = [os.environ.get("OMNIFT_NPU_DATA_DIR"), *_DEFAULT_DATA_DIRS]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if (path / "train.parquet").is_file() and (path / "test.parquet").is_file():
            return path
    pytest.skip(f"s5_2_data_1k parquet not found in {candidates}")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _compose_config():
    data_dir = _data_dir()
    model_path = os.environ.get("MODEL_PATH", _DEFAULT_MODEL)
    if not Path(model_path).exists():
        pytest.skip(f"LTX checkpoint missing at {model_path}")
    reward_root = os.environ.get("REWARD_ROOT", "/hub/omnift-rewards")
    desync_root = os.environ.get("DESYNC_SOURCE_ROOT", f"{reward_root}/OmniNFT-reference")
    num_gpus = _env_int("NUM_GPUS", 8)
    rollout_tp = _env_int("ROLLOUT_TP", 8)
    max_num_seqs = _env_int("MAX_NUM_SEQS", 2)
    rollout_n = _env_int("ROLLOUT_N", 2)
    height = _env_int("HEIGHT", 256)
    width = _env_int("WIDTH", 384)
    num_frames = _env_int("NUM_FRAMES", 121)
    num_inference_steps = _env_int("NUM_INFERENCE_STEPS", 2)
    frame_rate = _env_float("FRAME_RATE", 24.0)
    enable_sleep_mode = _env_bool("ENABLE_SLEEP_MODE", True)

    config_dir = os.path.abspath("verl_omni/trainer/config")
    overrides = [
        "trainer.device=npu",
        f"data.train_files={data_dir / 'train.parquet'}",
        f"data.val_files={data_dir / 'test.parquet'}",
        "data.return_multi_modal_inputs=False",
        "data.train_batch_size=1",
        "data.val_max_samples=8",
        "data.max_prompt_length=1024",
        "data.truncation=error",
        "data.seed=42",
        "algorithm.trainer_type=direct_preference",
        "algorithm.sample_source=online",
        "algorithm.paired_preference=false",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.algorithm=omni_nft",
        "actor_rollout_ref.model.model_type=omni_nft_model",
        "actor_rollout_ref.model.attn_backend=native",
        "actor_rollout_ref.model.lora_rank=32",
        "actor_rollout_ref.model.lora_alpha=64",
        'actor_rollout_ref.model.policy_state_adapters=["default","old"]',
        f"actor_rollout_ref.model.target_modules={_LORA_TARGETS}",
        "actor_rollout_ref.model.fsdp_layer_prefixes=['transformer_blocks.']",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LTX2VideoTransformerBlock]",
        "actor_rollout_ref.actor.ppo_mini_batch_size=1",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.diffusion_loss.loss_mode=omni_nft",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.fsdp_config.offload_policy=True",
        "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.name=vllm_omni",
        "actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout_tp}",
        f"actor_rollout_ref.rollout.n={rollout_n}",
        "actor_rollout_ref.rollout.seed=42",
        f"actor_rollout_ref.rollout.agent.num_workers={max(1, num_gpus // rollout_tp)}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=ltx2_omni_nft_single_turn_agent",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.layered_summon=True",
        f"+actor_rollout_ref.rollout.enable_sleep_mode={str(enable_sleep_mode).lower()}",
        "actor_rollout_ref.rollout.calculate_log_probs=False",
        "actor_rollout_ref.rollout.rollout_adapter=old",
        f"actor_rollout_ref.rollout.pipeline.height={height}",
        f"actor_rollout_ref.rollout.pipeline.width={width}",
        f"actor_rollout_ref.rollout.pipeline.num_frames={num_frames}",
        f"actor_rollout_ref.rollout.pipeline.frame_rate={frame_rate}",
        f"actor_rollout_ref.rollout.pipeline.num_inference_steps={num_inference_steps}",
        "actor_rollout_ref.rollout.pipeline.guidance_scale=4.0",
        "actor_rollout_ref.rollout.pipeline.max_sequence_length=1024",
        "+actor_rollout_ref.rollout.pipeline.output_type=pt",
        f"actor_rollout_ref.rollout.val_kwargs.pipeline.height={height}",
        f"actor_rollout_ref.rollout.val_kwargs.pipeline.width={width}",
        f"actor_rollout_ref.rollout.val_kwargs.pipeline.num_frames={num_frames}",
        f"actor_rollout_ref.rollout.val_kwargs.pipeline.frame_rate={frame_rate}",
        f"actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps={num_inference_steps}",
        "+actor_rollout_ref.rollout.val_kwargs.pipeline.output_type=pt",
        f"reward.num_workers={_env_int('REWARD_NUM_WORKERS', num_gpus)}",
        "reward.reward_model.enable=False",
        "reward.reward_model.enable_resource_pool=False",
        "reward.reward_manager.name=MultiModalRewardManager",
        "reward.aggregation=preserve_components",
        "reward.component_order=[video_align,hpsv3,audiobox,clap,desync]",
        "+reward.reward_functions.video_align.path=pkg://verl_omni.utils.reward_score.videoalign_native",
        "+reward.reward_functions.video_align.required=true",
        "+reward.reward_functions.video_align.micro_batch_size=2",
        f"+reward.reward_functions.video_align.model_path={reward_root}/VideoReward/checkpoint-11352/model.pth",
        f"+reward.reward_functions.video_align.base_model_path={reward_root}/Qwen2-VL-2B-Instruct",
        "+reward.reward_functions.hpsv3.path=pkg://verl_omni.utils.reward_score.hpsv3_native",
        "+reward.reward_functions.hpsv3.required=true",
        "+reward.reward_functions.hpsv3.micro_batch_size=8",
        f"+reward.reward_functions.hpsv3.model_path={reward_root}/HPSv3/HPSv3.safetensors",
        f"+reward.reward_functions.hpsv3.base_model_path={reward_root}/Qwen2-VL-7B-Instruct",
        "+reward.reward_functions.audiobox.path=pkg://verl_omni.utils.reward_score.audiobox_native",
        "+reward.reward_functions.audiobox.required=true",
        "+reward.reward_functions.audiobox.micro_batch_size=8",
        f"+reward.reward_functions.audiobox.model_path={reward_root}/audiobox-aesthetics",
        "+reward.reward_functions.clap.path=pkg://verl_omni.utils.reward_score.clap_native",
        "+reward.reward_functions.clap.required=true",
        "+reward.reward_functions.clap.micro_batch_size=8",
        f"+reward.reward_functions.clap.model_path={reward_root}/checkpoints/clap-htsat-unfused",
        "+reward.reward_functions.desync.path=pkg://verl_omni.utils.reward_score.desync_native",
        "+reward.reward_functions.desync.required=true",
        "+reward.reward_functions.desync.micro_batch_size=2",
        f"+reward.reward_functions.desync.model_path={reward_root}/synchformer/synchformer_state_dict.pth",
        f"+reward.reward_functions.desync.source_root={desync_root}",
        "trainer.logger=[console]",
        "trainer.resume_mode=disable",
        "trainer.val_before_train=False",
        "trainer.log_val_generations=0",
        f"trainer.n_gpus_per_node={num_gpus}",
        "trainer.nnodes=1",
        "trainer.total_training_steps=1",
    ]
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="diffusion_trainer", overrides=overrides)
    with open_dict(config):
        config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni = {
            "max_num_seqs": max_num_seqs,
            "request_batch_max_wait_ms": _env_int("REQUEST_BATCH_MAX_WAIT_MS", 200),
        }
    return config


def _init_ray(config, extra_env: dict[str, str] | None = None) -> None:
    extra_env = extra_env or {}
    for key, value in extra_env.items():
        os.environ[key] = value
    if ray.is_initialized():
        return
    default_runtime_env = get_ppo_ray_runtime_env()
    ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    runtime_env_container = OmegaConf.to_container(runtime_env, resolve=True)
    if not isinstance(runtime_env_container, dict):
        runtime_env_container = {}
    env_vars = dict(runtime_env_container.get("env_vars") or {})
    env_vars.update(extra_env)
    runtime_env_container["env_vars"] = env_vars
    init_kwargs = OmegaConf.to_container(OmegaConf.create(ray_init_kwargs), resolve=True) or {}
    init_kwargs["runtime_env"] = runtime_env_container
    ray.init(**init_kwargs)


def _run_official_generate(trainer) -> DataProto:
    """Wake after init sleep, generate, then sleep — same order as ``DirectPreferenceRayTrainer.fit``."""
    trainer.global_steps = max(int(getattr(trainer, "global_steps", 0)), 1)
    batch_dict = next(iter(trainer.train_dataloader))
    batch = DataProto.from_single_dict(batch_dict)
    if "uid" not in batch.non_tensor_batch:
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)

    gen_batch = trainer._get_gen_batch(batch)
    gen_batch.meta_info["global_steps"] = trainer.global_steps
    rollout_seed_cfg = trainer.config.actor_rollout_ref.rollout.get("seed")
    if rollout_seed_cfg is not None:
        gen_batch.meta_info["rollout_seed"] = int(rollout_seed_cfg) + trainer.global_steps - 1

    gen_batch_output = gen_batch.repeat(
        repeat_times=trainer.config.actor_rollout_ref.rollout.n,
        interleave=True,
    )
    gen_batch_output.non_tensor_batch["_rollout_seed_global_idx"] = np.arange(len(gen_batch_output), dtype=np.int64)
    trainer.checkpoint_manager.update_weights(trainer.global_steps)
    gen_batch_output = trainer.async_rollout_manager.generate_sequences(gen_batch_output)
    trainer.checkpoint_manager.sleep_replicas()
    return gen_batch_output


def _assert_generate_contract(output: DataProto, *, n: int, num_frames: int, height: int, width: int) -> None:
    assert len(output) == n
    assert "rollout_log_probs" not in output.batch
    responses = output.batch["responses"]
    assert responses.dtype == torch.uint8
    assert responses.ndim == 5
    assert tuple(responses.shape) == (n, num_frames, 3, height, width)
    assert responses.device.type == "cpu"
    for key in (*_RL_KEYS, *_PROMPT_KEYS):
        assert key in output.batch, f"missing rollout field {key}"
        value = output.batch[key]
        if isinstance(value, torch.Tensor):
            assert value.shape[0] == n
            assert value.device.type == "cpu"
    sample_uids = [str(uid) for uid in output.non_tensor_batch["sample_uid"]]
    assert len(sample_uids) == n
    assert len(set(sample_uids)) == n
    uids = [str(uid) for uid in output.non_tensor_batch["uid"]]
    assert len(set(uids)) == 1


def _assert_packed_parallel(log_path: Path, *, max_num_seqs: int, n: int) -> None:
    if not log_path.is_file():
        raise AssertionError("pipeline.forward did not write OMNIFT_REQUEST_BATCH_LOG; packing cannot be verified.")
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        rows.append((int(parts[0]), int(parts[1])))
    assert rows, f"OMNIFT_REQUEST_BATCH_LOG is empty: {log_path}"
    packed = [(num_reqs, captured) for num_reqs, captured in rows if num_reqs > 1]
    assert packed, (
        f"pipeline.forward only ran serial batches {rows}; requests were not packed into one transformer batch."
    )
    expected = min(int(max_num_seqs), int(n))
    assert any(num_reqs == captured == expected for num_reqs, captured in packed), (
        f"no fused batch of {expected}: {rows}"
    )


def test_omninft_generate_sequences_from_s5_2_data_1k() -> None:
    config = _compose_config()
    auto_set_device(config)
    OmegaConf.resolve(config)
    fallback_fa3_if_unavailable(config)
    validate_attention_consistency(config)
    max_num_seqs = int(config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs)
    rollout_n = int(config.actor_rollout_ref.rollout.n)
    assert rollout_n >= 2 and max_num_seqs >= 2

    log_file = tempfile.NamedTemporaryFile(prefix="omnift_request_batch_", suffix=".log", delete=False)
    log_path = Path(log_file.name)
    log_file.close()
    log_path.unlink(missing_ok=True)
    _init_ray(config, extra_env={"OMNIFT_REQUEST_BATCH_LOG": str(log_path)})

    captured: dict[str, DataProto] = {}
    original_fit = DirectPreferenceRayTrainer.fit

    def generate_only(trainer) -> None:
        captured["output"] = _run_official_generate(trainer)

    DirectPreferenceRayTrainer.fit = generate_only
    try:
        TaskRunner().run(config)
    finally:
        DirectPreferenceRayTrainer.fit = original_fit
        if ray.is_initialized():
            ray.shutdown()

    output = captured["output"]
    pipeline = config.actor_rollout_ref.rollout.pipeline
    _assert_generate_contract(
        output,
        n=rollout_n,
        num_frames=int(pipeline.num_frames),
        height=int(pipeline.height),
        width=int(pipeline.width),
    )
    _assert_packed_parallel(log_path, max_num_seqs=max_num_seqs, n=rollout_n)
