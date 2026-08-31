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
VAL_FILE=${VAL_FILE:-$DATA_DIR/train.parquet}

export WANDB_MODE=${WANDB_MODE:-offline}
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}
set +u
source "$ASCEND_HOME_PATH/set_env.sh"
source "$ASCEND_HOME_PATH/../nnal/atb/set_env.sh"
set -u

MODEL_PATH=${MODEL_PATH:-dg845/LTX-2.3-Diffusers}
REWARD_ROOT=${REWARD_ROOT:-/hub/omnift-rewards}
DESYNC_SOURCE_ROOT=${DESYNC_SOURCE_ROOT:-$REWARD_ROOT/OmniNFT-reference}
NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-8}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-$NUM_GPUS}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-500}
ltx_lora_targets="['attn1.to_q','attn1.to_k','attn1.to_v','attn1.to_out.0','attn2.to_q','attn2.to_k','attn2.to_v','attn2.to_out.0','audio_attn1.to_q','audio_attn1.to_k','audio_attn1.to_v','audio_attn1.to_out.0','audio_attn2.to_q','audio_attn2.to_k','audio_attn2.to_v','audio_attn2.to_out.0','audio_to_video_attn.to_q','audio_to_video_attn.to_k','audio_to_video_attn.to_v','audio_to_video_attn.to_out.0','video_to_audio_attn.to_q','video_to_audio_attn.to_k','video_to_audio_attn.to_v','video_to_audio_attn.to_out.0','ff.net.0.proj','ff.net.2','audio_ff.net.0.proj','audio_ff.net.2']"

script_path=$(readlink -f "$0")
script_name=$(basename "$script_path" .sh)
repo_root=$(dirname "$script_path")
while [[ "$repo_root" != "/" && ! -f "$repo_root/LICENSE" ]]; do
    repo_root=$(dirname "$repo_root")
done
if [[ ! -f "$repo_root/LICENSE" ]]; then
    echo "Unable to locate repo root from $script_path: no LICENSE found" >&2
    exit 1
fi

output_dir=${OUTPUT_DIR:-$repo_root/outputs/$script_name}
checkpoint_dir=$output_dir/checkpoints
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log
rollout_data_dir=$output_dir/logs/$run_timestamp/rollout_videos
mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1

# Match the official OmniNFT LTX recipe. The lower timestep fraction and
# conservative LoRA learning rate keep this small G=8 recipe stable.
python3 -m verl_omni.trainer.main_diffusion \
    trainer.device=npu \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.return_multi_modal_inputs=False \
    data.train_batch_size=1 \
    data.val_max_samples=8 \
    data.max_prompt_length=1024 \
    data.truncation=error \
    data.seed=42 \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.paired_preference=false \
    algorithm.timestep_fraction=0.4 \
    algorithm.old_policy_decay_schedule=linear_to_0_5 \
    algorithm.old_policy_update_interval=1 \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.global_std=true \
    algorithm.adv_mode=continuous \
    actor_rollout_ref.model.pipeline.guidance_scale=1.0 \
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
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=omni_nft \
    actor_rollout_ref.actor.diffusion_loss.video_weight=1.0 \
    actor_rollout_ref.actor.diffusion_loss.audio_weight=1.0 \
    actor_rollout_ref.actor.diffusion_loss.mix_beta=1.0 \
    actor_rollout_ref.actor.diffusion_loss.video_ref_kl_coef=1e-4 \
    actor_rollout_ref.actor.diffusion_loss.audio_ref_kl_coef=1e-4 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.agent.default_agent_loop=ltx2_omni_nft_single_turn_agent \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=384 \
    actor_rollout_ref.rollout.pipeline.num_frames=121 \
    actor_rollout_ref.rollout.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=30 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=1024 \
    +actor_rollout_ref.rollout.pipeline.output_type=pt \
    actor_rollout_ref.rollout.val_kwargs.pipeline.height=256 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.width=384 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_frames=121 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=30 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.guidance_scale=4.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.output_type=pt \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.num_workers="$REWARD_NUM_WORKERS" \
    reward.reward_model.enable=False \
    reward.reward_model.enable_resource_pool=False \
    reward.reward_manager.name=MultiModalRewardManager \
    reward.aggregation=preserve_components \
    'reward.component_order=[video_align,hpsv3,audiobox,clap,desync]' \
    +reward.reward_functions.video_align.path=pkg://verl_omni.utils.reward_score.videoalign_native \
    +reward.reward_functions.video_align.required=true \
    +reward.reward_functions.video_align.micro_batch_size=2 \
    +reward.reward_functions.video_align.model_path="$REWARD_ROOT/VideoReward/checkpoint-11352/model.pth" \
    +reward.reward_functions.video_align.base_model_path="$REWARD_ROOT/Qwen2-VL-2B-Instruct" \
    +reward.reward_functions.video_align.routing_weights.video=1.0 \
    +reward.reward_functions.video_align.routing_weights.audio=0.0 \
    +reward.reward_functions.hpsv3.path=pkg://verl_omni.utils.reward_score.hpsv3_native \
    +reward.reward_functions.hpsv3.required=true \
    +reward.reward_functions.hpsv3.micro_batch_size=8 \
    +reward.reward_functions.hpsv3.model_path="$REWARD_ROOT/HPSv3/HPSv3.safetensors" \
    +reward.reward_functions.hpsv3.base_model_path="$REWARD_ROOT/Qwen2-VL-7B-Instruct" \
    +reward.reward_functions.hpsv3.routing_weights.video=1.5 \
    +reward.reward_functions.hpsv3.routing_weights.audio=0.0 \
    +reward.reward_functions.audiobox.path=pkg://verl_omni.utils.reward_score.audiobox_native \
    +reward.reward_functions.audiobox.required=true \
    +reward.reward_functions.audiobox.micro_batch_size=8 \
    +reward.reward_functions.audiobox.model_path="$REWARD_ROOT/audiobox-aesthetics" \
    +reward.reward_functions.audiobox.routing_weights.video=0.0 \
    +reward.reward_functions.audiobox.routing_weights.audio=0.5 \
    +reward.reward_functions.clap.path=pkg://verl_omni.utils.reward_score.clap_native \
    +reward.reward_functions.clap.required=true \
    +reward.reward_functions.clap.micro_batch_size=8 \
    +reward.reward_functions.clap.model_path="$REWARD_ROOT/checkpoints/clap-htsat-unfused" \
    +reward.reward_functions.clap.routing_weights.video=0.0 \
    +reward.reward_functions.clap.routing_weights.audio=1.0 \
    +reward.reward_functions.desync.path=pkg://verl_omni.utils.reward_score.desync_native \
    +reward.reward_functions.desync.required=true \
    +reward.reward_functions.desync.micro_batch_size=2 \
    +reward.reward_functions.desync.model_path="$REWARD_ROOT/synchformer/synchformer_state_dict.pth" \
    +reward.reward_functions.desync.source_root="$DESYNC_SOURCE_ROOT" \
    +reward.reward_functions.desync.routing_weights.video=1.0 \
    +reward.reward_functions.desync.routing_weights.audio=1.0 \
    trainer.logger='["console"]' \
    trainer.project_name=omni_nft \
    trainer.experiment_name=ltx2_3_omninft_lora_npu \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.validation_data_dir=$rollout_data_dir \
    trainer.validation_data_max_samples=8 \
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
