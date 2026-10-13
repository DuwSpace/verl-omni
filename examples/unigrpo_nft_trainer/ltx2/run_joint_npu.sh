#!/usr/bin/env bash
# Load the CANN environment before invoking this launcher.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${LTX_MODEL_PATH:?Set the local LTX-2.3 Diffusers snapshot}"
: "${LTX_TRAIN_FILE:?Set the training parquet path}"
: "${LTX_VAL_FILE:?Set the fixed evaluation parquet path}"
: "${LTX_REWARD_ROOT:?Set the five-reward asset directory}"
: "${LTX_OUTPUT_DIR:?Set a new output directory, or override trainer.resume_mode}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${LTX_CPU_THREADS:-4}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export LTX_ACTOR_NPU_MEMORY_FRACTION="${LTX_ACTOR_NPU_MEMORY_FRACTION:-0.8}"
ltx_devices="${LTX_NUM_DEVICES:-8}"
if [[ ! "$ltx_devices" =~ ^[1-9][0-9]*$ ]] || ((ltx_devices < 5)); then
    echo 'This five-model native reward recipe requires at least 5 devices with disjoint placements' >&2
    exit 2
fi
python3 -m verl_omni.trainer.main_diffusion \
    --config-path "$script_dir" --config-name joint_train \
    trainer.n_gpus_per_node="$ltx_devices" \
    actor_rollout_ref.rollout.n="$ltx_devices" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$ltx_devices" \
    reward.num_workers="$ltx_devices" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.LTX_ACTOR_NPU_MEMORY_FRACTION=\"$LTX_ACTOR_NPU_MEMORY_FRACTION\"" \
    "$@"
