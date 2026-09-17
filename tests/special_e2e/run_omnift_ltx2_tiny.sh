#!/usr/bin/env bash
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
# LTX-2.3 OmniNFT end-to-end smoke with a local tiny-random checkpoint.
# The dummy scorer intentionally keeps reward-model lifecycle and inference on
# the production native-executor path while avoiding external reward weights.
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-4}
MODEL_PATH=${MODEL_PATH:-}
OMNIFT_TINY_MODEL_DIR=${OMNIFT_TINY_MODEL_DIR:-${HOME}/models/tiny-random/LTX-2.3-OmniNFT}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_ltx2_omnift}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/test.parquet}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}

if [[ -z "${MODEL_PATH}" ]]; then
    python3 tests/special_e2e/build_ltx2_omnift_tiny_random.py --output-dir "${OMNIFT_TINY_MODEL_DIR}"
    MODEL_PATH=${OMNIFT_TINY_MODEL_DIR}
fi
readonly MODEL_PATH
python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size 1 \
    --val_size 1

ltx_lora_targets='["attn1.to_q","attn1.to_k","attn1.to_v","attn1.to_out.0","attn2.to_q","attn2.to_k","attn2.to_v","attn2.to_out.0","audio_attn1.to_q","audio_attn1.to_k","audio_attn1.to_v","audio_attn1.to_out.0","audio_attn2.to_q","audio_attn2.to_k","audio_attn2.to_v","audio_attn2.to_out.0","audio_to_video_attn.to_q","audio_to_video_attn.to_k","audio_to_video_attn.to_v","audio_to_video_attn.to_out.0","video_to_audio_attn.to_q","video_to_audio_attn.to_k","video_to_audio_attn.to_v","video_to_audio_attn.to_out.0","ff.net.0.proj","ff.net.2","audio_ff.net.0.proj","audio_ff.net.2"]'

python3 -m verl_omni.trainer.main_diffusion \
    trainer.device=cuda \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.return_multi_modal_inputs=False \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.val_max_samples=1 \
    data.max_prompt_length=64 \
    data.truncation=error \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.paired_preference=false \
    algorithm.timestep_fraction=1 \
    algorithm.old_policy_decay=0.5 \
    algorithm.old_policy_update_interval=1 \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.global_std=true \
    algorithm.adv_mode=continuous \
    actor_rollout_ref.model.pipeline._target_=verl_omni.workers.config.diffusion.rollout.LTXDiffusionPipelineConfig \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.algorithm=omni_nft \
    actor_rollout_ref.model.model_type=omni_nft_model \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.lora_rank=2 \
    actor_rollout_ref.model.lora_alpha=2 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules="${ltx_lora_targets}" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['transformer_blocks.']" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LTX2VideoTransformerBlock]' \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.diffusion_loss._target_=verl_omni.workers.config.diffusion.actor.OmniNFTLossConfig \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=omni_nft \
    +actor_rollout_ref.actor.diffusion_loss.video_weight=1.0 \
    +actor_rollout_ref.actor.diffusion_loss.audio_weight=1.0 \
    actor_rollout_ref.actor.diffusion_loss.mix_beta=1.0 \
    +actor_rollout_ref.actor.diffusion_loss.video_ref_kl_coef=1e-4 \
    +actor_rollout_ref.actor.diffusion_loss.audio_ref_kl_coef=1e-4 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n="${NUM_GPUS}" \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.agent.num_workers="${NUM_GPUS}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=ltx2_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.pipeline._target_=verl_omni.workers.config.diffusion.rollout.LTXDiffusionPipelineConfig \
    actor_rollout_ref.rollout.pipeline.height=32 \
    actor_rollout_ref.rollout.pipeline.width=32 \
    actor_rollout_ref.rollout.pipeline.num_frames=9 \
    actor_rollout_ref.rollout.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=2 \
    +actor_rollout_ref.rollout.pipeline.video_cfg_scale=1.0 \
    +actor_rollout_ref.rollout.pipeline.audio_cfg_scale=1.0 \
    +actor_rollout_ref.rollout.pipeline.video_modality_scale=1.0 \
    +actor_rollout_ref.rollout.pipeline.audio_modality_scale=1.0 \
    +actor_rollout_ref.rollout.pipeline.video_rescale_scale=0.0 \
    +actor_rollout_ref.rollout.pipeline.audio_rescale_scale=0.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=16 \
    +actor_rollout_ref.rollout.pipeline.output_type=pt \
    reward.num_workers="${NUM_GPUS}" \
    reward.reward_model.enable=False \
    reward.reward_model.enable_resource_pool=False \
    reward.accelerator_workers.enabled=False \
    reward.reward_manager.name=MultiModalRewardManager \
    reward.aggregation=preserve_components \
    +reward.models.tiny_omni.backend=native \
    +reward.models.tiny_omni.offload=True \
    +reward.models.tiny_omni.offload_mode=cpu \
    +reward.models.tiny_omni.placement.devices='[0]' \
    +reward.models.tiny_omni.executor.model=tests.special_e2e.omnift_dummy_reward:TinyOmniRewardModel \
    +reward.reward_functions.video_quality.path=pkg://tests.special_e2e.omnift_dummy_reward \
    +reward.reward_functions.video_quality.name=compute_score_batch \
    +reward.reward_functions.video_quality.model=tiny_omni \
    +reward.reward_functions.video_quality.weight=1.0 \
    +reward.reward_functions.video_quality.required=true \
    +reward.reward_functions.video_quality.channel=video \
    +reward.reward_functions.video_quality.routing_weights.video=1.0 \
    +reward.reward_functions.video_quality.routing_weights.audio=0.0 \
    +reward.reward_functions.audio_quality.path=pkg://tests.special_e2e.omnift_dummy_reward \
    +reward.reward_functions.audio_quality.name=compute_score_batch \
    +reward.reward_functions.audio_quality.model=tiny_omni \
    +reward.reward_functions.audio_quality.weight=1.0 \
    +reward.reward_functions.audio_quality.required=true \
    +reward.reward_functions.audio_quality.channel=audio \
    +reward.reward_functions.audio_quality.routing_weights.video=0.0 \
    +reward.reward_functions.audio_quality.routing_weights.audio=1.0 \
    +reward.reward_functions.sync_quality.path=pkg://tests.special_e2e.omnift_dummy_reward \
    +reward.reward_functions.sync_quality.name=compute_score_batch \
    +reward.reward_functions.sync_quality.model=tiny_omni \
    +reward.reward_functions.sync_quality.weight=1.0 \
    +reward.reward_functions.sync_quality.required=true \
    +reward.reward_functions.sync_quality.channel=sync \
    +reward.reward_functions.sync_quality.routing_weights.video=0.5 \
    +reward.reward_functions.sync_quality.routing_weights.audio=0.5 \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=omnift-ltx2-tiny-e2e \
    trainer.default_local_dir="${DATA_DIR}/checkpoints" \
    trainer.resume_mode=disable \
    trainer.log_val_generations=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "$@"

echo "LTX-2.3 OmniNFT tiny end-to-end smoke passed."
