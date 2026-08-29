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
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
DATA_FILE=${DATA_FILE:-$REPO_ROOT/data/omninft/vggsound/train_metadata_20k.jsonl}
FAKE_REWARD_PATH=$REPO_ROOT/examples/omninft_trainer/ltx2/debug/fake_native_rewards.py
MODEL_PATH=${MODEL_PATH:-/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee8edcf067e838b843f926ec4d4cc9b2be1aaf}
NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-8}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/ltx2_3_omninft_reward_only}

export WANDB_MODE=${WANDB_MODE:-offline}
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}
source "$ASCEND_HOME_PATH/set_env.sh"
source "$ASCEND_HOME_PATH/../nnal/atb/set_env.sh"

python3 -m verl_omni.trainer.main_diffusion \
    trainer.device=npu \
    data.train_files="$DATA_FILE" \
    data.val_files="$DATA_FILE" \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_nft_dataset \
    data.custom_cls.name=OmniNFTPromptDataset \
    data.custom_cls.collate_fn=collate_omni_nft_prompt_groups \
    data.return_multi_modal_inputs=False \
    data.train_batch_size=1 \
    data.max_prompt_length=128 \
    data.truncation=error \
    data.seed=42 \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.paired_preference=false \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.tokenizer_path="$MODEL_PATH/tokenizer" \
    actor_rollout_ref.model.algorithm=omni_nft \
    actor_rollout_ref.model.model_type=omni_nft_model \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=omni_nft \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node="$NUM_GPUS" \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=ltx2_omni_nft_single_turn_agent \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.pipeline.height=128 \
    actor_rollout_ref.rollout.pipeline.width=192 \
    actor_rollout_ref.rollout.pipeline.num_frames=9 \
    actor_rollout_ref.rollout.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=2 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=128 \
    +actor_rollout_ref.rollout.pipeline.output_type=pt \
    +actor_rollout_ref.rollout.engine_kwargs.init_timeout=3600 \
    +actor_rollout_ref.rollout.engine_kwargs.stage_init_timeout=3600 \
    reward.num_workers=1 \
    reward.reward_model.enable=false \
    reward.reward_model.enable_resource_pool=false \
    reward.reward_manager.name=MultiModalRewardManager \
    reward.aggregation=preserve_components \
    'reward.component_order=[video_align,hpsv3,audiobox,clap,desync]' \
    +reward.reward_functions.video_align.path="$FAKE_REWARD_PATH" \
    +reward.reward_functions.video_align.required=true \
    +reward.reward_functions.video_align.micro_batch_size=1 \
    +reward.reward_functions.video_align.name=video_align \
    +reward.reward_functions.video_align.offset=0.0 \
    +reward.reward_functions.hpsv3.path="$FAKE_REWARD_PATH" \
    +reward.reward_functions.hpsv3.required=true \
    +reward.reward_functions.hpsv3.micro_batch_size=1 \
    +reward.reward_functions.hpsv3.name=hpsv3 \
    +reward.reward_functions.hpsv3.offset=10.0 \
    +reward.reward_functions.audiobox.path="$FAKE_REWARD_PATH" \
    +reward.reward_functions.audiobox.required=true \
    +reward.reward_functions.audiobox.micro_batch_size=1 \
    +reward.reward_functions.audiobox.name=audiobox \
    +reward.reward_functions.audiobox.offset=20.0 \
    +reward.reward_functions.clap.path="$FAKE_REWARD_PATH" \
    +reward.reward_functions.clap.required=true \
    +reward.reward_functions.clap.micro_batch_size=1 \
    +reward.reward_functions.clap.name=clap \
    +reward.reward_functions.clap.offset=30.0 \
    +reward.reward_functions.desync.path="$FAKE_REWARD_PATH" \
    +reward.reward_functions.desync.required=true \
    +reward.reward_functions.desync.micro_batch_size=1 \
    +reward.reward_functions.desync.name=desync \
    +reward.reward_functions.desync.offset=40.0 \
    trainer.logger='["console"]' \
    trainer.project_name=omni_nft \
    trainer.experiment_name=ltx2_3_omninft_reward_only \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
    trainer.resume_mode=disable \
    trainer.log_val_generations=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    "$@"
