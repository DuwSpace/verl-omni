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
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/omninft/vggsound/verl_omni}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/test.parquet}

export WANDB_MODE=${WANDB_MODE:-online}
export OMNIFT_ROLLOUT_PROGRESS=${OMNIFT_ROLLOUT_PROGRESS:-1}
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}
set +u
source "$ASCEND_HOME_PATH/set_env.sh"
source "$ASCEND_HOME_PATH/../nnal/atb/set_env.sh"
set -u

MODEL_REVISION=8eee8edcf067e838b843f926ec4d4cc9b2be1aaf
MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/outputs}
default_model_path=$MODEL_ROOT/models--diffusers--LTX-2.3-Diffusers/snapshots/$MODEL_REVISION
default_reward_root=$MODEL_ROOT/omnift-rewards

# Prefer the paths populated by download_models.sh, while retaining the
# existing /hub layout when running inside the established training image.
if [[ ! -d "$default_model_path" && -d "/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/$MODEL_REVISION" ]]; then
    default_model_path=/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/$MODEL_REVISION
fi
if [[ ! -d "$default_reward_root" && -d /hub/omnift-rewards ]]; then
    default_reward_root=/hub/omnift-rewards
fi
MODEL_PATH=${MODEL_PATH:-$default_model_path}
REWARD_ROOT=${REWARD_ROOT:-$default_reward_root}
DESYNC_SOURCE_ROOT=${DESYNC_SOURCE_ROOT:-$REWARD_ROOT/OmniNFT-reference}
REWARD_OFFLOAD_MODE=${REWARD_OFFLOAD_MODE:-cpu}
NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_N=${ROLLOUT_N:-8}
# LTX OmniNFT uses request-level batching.  Cap each engine at the number of
# samples one prompt can produce instead of inheriting the generic 1024 limit.
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-$ROLLOUT_N}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
VIDEOALIGN_DEVICES=${VIDEOALIGN_DEVICES:-'[0,1]'}
HPSV3_DEVICES=${HPSV3_DEVICES:-'[2,3]'}
AUDIOBOX_DEVICES=${AUDIOBOX_DEVICES:-'[4]'}
CLAP_DEVICES=${CLAP_DEVICES:-'[5]'}
DESYNC_DEVICES=${DESYNC_DEVICES:-'[6,7]'}
ltx_lora_targets='["attn1.to_q","attn1.to_k","attn1.to_v","attn1.to_out.0","attn2.to_q","attn2.to_k","attn2.to_v","attn2.to_out.0","audio_attn1.to_q","audio_attn1.to_k","audio_attn1.to_v","audio_attn1.to_out.0","audio_attn2.to_q","audio_attn2.to_k","audio_attn2.to_v","audio_attn2.to_out.0","audio_to_video_attn.to_q","audio_to_video_attn.to_k","audio_to_video_attn.to_v","audio_to_video_attn.to_out.0","video_to_audio_attn.to_q","video_to_audio_attn.to_k","video_to_audio_attn.to_v","video_to_audio_attn.to_out.0","ff.net.0.proj","ff.net.2","audio_ff.net.0.proj","audio_ff.net.2"]'

script_name=$(basename "$0" .sh)
output_dir=${OUTPUT_DIR:-$REPO_ROOT/outputs/$script_name}
checkpoint_dir=$output_dir/checkpoints
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log
WANDB_DIR=$output_dir

mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1

# Keep the full-data recipe parameters explicit so individual runs can
# override them with trailing Hydra arguments.
python3 -m verl_omni.trainer.main_diffusion \
    trainer.device=npu \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.return_multi_modal_inputs=False \
    data.train_batch_size=1 \
    data.val_max_samples=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=1024 \
    data.truncation=error \
    data.seed=42 \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.paired_preference=false \
    algorithm.timestep_fraction=1 \
    algorithm.old_policy_decay_schedule=linear_to_0_5 \
    algorithm.old_policy_update_interval=1 \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.global_std=true \
    algorithm.adv_mode=continuous \
    actor_rollout_ref.model.pipeline._target_=verl_omni.pipelines.ltx2_omni_nft.config.LTXDiffusionPipelineConfig \
    +actor_rollout_ref.model.pipeline.video_cfg_scale=1 \
    +actor_rollout_ref.model.pipeline.audio_cfg_scale=1 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.algorithm=omni_nft \
    actor_rollout_ref.model.model_type=omni_nft_model \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules="$ltx_lora_targets" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['transformer_blocks.']" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LTX2VideoTransformerBlock]' \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.optim.weight_decay=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.diffusion_loss._target_=verl_omni.pipelines.ltx2_omni_nft.config.OmniNFTLossConfig \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=omni_nft \
    +actor_rollout_ref.actor.diffusion_loss.video_weight=1.0 \
    +actor_rollout_ref.actor.diffusion_loss.audio_weight=1.0 \
    actor_rollout_ref.actor.diffusion_loss.mix_beta=1.0 \
    +actor_rollout_ref.actor.diffusion_loss.video_ref_kl_coef=1e-4 \
    +actor_rollout_ref.actor.diffusion_loss.audio_ref_kl_coef=1e-4 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.agent.default_agent_loop=ltx2_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.pipeline._target_=verl_omni.pipelines.ltx2_omni_nft.config.LTXDiffusionPipelineConfig \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=384 \
    actor_rollout_ref.rollout.pipeline.num_frames=121 \
    actor_rollout_ref.rollout.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=20 \
    +actor_rollout_ref.rollout.pipeline.video_cfg_scale=1.5 \
    +actor_rollout_ref.rollout.pipeline.audio_cfg_scale=3.0 \
    +actor_rollout_ref.rollout.pipeline.video_modality_scale=1.0 \
    +actor_rollout_ref.rollout.pipeline.audio_modality_scale=1.0 \
    +actor_rollout_ref.rollout.pipeline.video_rescale_scale=0.0 \
    +actor_rollout_ref.rollout.pipeline.audio_rescale_scale=0.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=1024 \
    +actor_rollout_ref.rollout.pipeline.output_type=pt \
    actor_rollout_ref.rollout.val_kwargs.pipeline._target_=verl_omni.pipelines.ltx2_omni_nft.config.LTXDiffusionPipelineConfig \
    actor_rollout_ref.rollout.val_kwargs.pipeline.height=512 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.width=768 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_frames=121 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=1 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.video_cfg_scale=3.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.audio_cfg_scale=7.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.video_modality_scale=3.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.audio_modality_scale=3.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.video_rescale_scale=0.7 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.audio_rescale_scale=0.7 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.output_type=pt \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.num_workers="$NUM_GPUS" \
    reward.reward_model.enable=False \
    reward.reward_model.enable_resource_pool=False \
    reward.accelerator_workers.enabled=False \
    reward.reward_manager.name=MultiModalRewardManager \
    reward.aggregation=preserve_components \
    +reward.models.video_align.backend=native \
    +reward.models.video_align.offload=True \
    +reward.models.video_align.offload_mode="$REWARD_OFFLOAD_MODE" \
    +reward.models.video_align.model_path="$REWARD_ROOT/VideoReward/checkpoint-11352/model.pth" \
    +reward.models.video_align.placement.devices="$VIDEOALIGN_DEVICES" \
    +reward.models.video_align.executor.model=verl_omni.utils.reward_score.videoalign_native:VideoAlignNativeModel \
    +reward.models.video_align.executor.kwargs.base_model_path="$REWARD_ROOT/Qwen2-VL-2B-Instruct" \
    +reward.models.hpsv3.backend=native \
    +reward.models.hpsv3.offload=True \
    +reward.models.hpsv3.offload_mode="$REWARD_OFFLOAD_MODE" \
    +reward.models.hpsv3.model_path="$REWARD_ROOT/HPSv3/HPSv3.safetensors" \
    +reward.models.hpsv3.placement.devices="$HPSV3_DEVICES" \
    +reward.models.hpsv3.executor.model=verl_omni.utils.reward_score.hpsv3_native:HPSv3NativeModel \
    +reward.models.hpsv3.executor.kwargs.base_model_path="$REWARD_ROOT/Qwen2-VL-7B-Instruct" \
    +reward.models.audiobox.backend=native \
    +reward.models.audiobox.offload=True \
    +reward.models.audiobox.offload_mode="$REWARD_OFFLOAD_MODE" \
    +reward.models.audiobox.model_path="$REWARD_ROOT/audiobox-aesthetics" \
    +reward.models.audiobox.placement.devices="$AUDIOBOX_DEVICES" \
    +reward.models.audiobox.executor.model=verl_omni.utils.reward_score.audiobox_native:AudioBoxNativeModel \
    +reward.models.clap.backend=native \
    +reward.models.clap.offload=True \
    +reward.models.clap.offload_mode="$REWARD_OFFLOAD_MODE" \
    +reward.models.clap.model_path="$REWARD_ROOT/checkpoints/clap-htsat-unfused" \
    +reward.models.clap.placement.devices="$CLAP_DEVICES" \
    +reward.models.clap.executor.model=verl_omni.utils.reward_score.clap_native:CLAPNativeModel \
    +reward.models.desync.backend=native \
    +reward.models.desync.offload=True \
    +reward.models.desync.offload_mode="$REWARD_OFFLOAD_MODE" \
    +reward.models.desync.model_path="$REWARD_ROOT/synchformer/synchformer_state_dict.pth" \
    +reward.models.desync.placement.devices="$DESYNC_DEVICES" \
    +reward.models.desync.executor.model=verl_omni.utils.reward_score.desync_native:DeSyncNativeModel \
    +reward.models.desync.executor.kwargs.source_root="$DESYNC_SOURCE_ROOT" \
    +reward.reward_functions.video_align.path=pkg://verl_omni.utils.reward_score.videoalign_native \
    +reward.reward_functions.video_align.name=compute_score_batch \
    +reward.reward_functions.video_align.model=video_align \
    +reward.reward_functions.video_align.weight=1.0 \
    +reward.reward_functions.video_align.required=true \
    +reward.reward_functions.video_align.micro_batch_size=2 \
    +reward.reward_functions.video_align.routing_weights.video=1.0 \
    +reward.reward_functions.video_align.routing_weights.audio=0.0 \
    +reward.reward_functions.hpsv3.path=pkg://verl_omni.utils.reward_score.hpsv3_native \
    +reward.reward_functions.hpsv3.name=compute_score_batch \
    +reward.reward_functions.hpsv3.model=hpsv3 \
    +reward.reward_functions.hpsv3.weight=1.0 \
    +reward.reward_functions.hpsv3.required=true \
    +reward.reward_functions.hpsv3.micro_batch_size=8 \
    +reward.reward_functions.hpsv3.routing_weights.video=1.5 \
    +reward.reward_functions.hpsv3.routing_weights.audio=0.0 \
    +reward.reward_functions.audiobox.path=pkg://verl_omni.utils.reward_score.audiobox_native \
    +reward.reward_functions.audiobox.name=compute_score_batch \
    +reward.reward_functions.audiobox.model=audiobox \
    +reward.reward_functions.audiobox.weight=1.0 \
    +reward.reward_functions.audiobox.required=true \
    +reward.reward_functions.audiobox.micro_batch_size=8 \
    +reward.reward_functions.audiobox.routing_weights.video=0.0 \
    +reward.reward_functions.audiobox.routing_weights.audio=0.5 \
    +reward.reward_functions.clap.path=pkg://verl_omni.utils.reward_score.clap_native \
    +reward.reward_functions.clap.name=compute_score_batch \
    +reward.reward_functions.clap.model=clap \
    +reward.reward_functions.clap.weight=1.0 \
    +reward.reward_functions.clap.required=true \
    +reward.reward_functions.clap.micro_batch_size=8 \
    +reward.reward_functions.clap.routing_weights.video=0.0 \
    +reward.reward_functions.clap.routing_weights.audio=1.0 \
    +reward.reward_functions.desync.path=pkg://verl_omni.utils.reward_score.desync_native \
    +reward.reward_functions.desync.name=compute_score_batch \
    +reward.reward_functions.desync.model=desync \
    +reward.reward_functions.desync.weight=1.0 \
    +reward.reward_functions.desync.required=true \
    +reward.reward_functions.desync.micro_batch_size=2 \
    +reward.reward_functions.desync.routing_weights.video=1.0 \
    +reward.reward_functions.desync.routing_weights.audio=1.0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=omni_nft \
    trainer.experiment_name=sample_1_lr_3e-5_timestep_random_kl_loss_m_rank_32_alpha_64_rollout_cfg_1.5/3 \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.resume_mode=disable \
    trainer.log_val_generations=0 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    "$@"
