#!/usr/bin/env bash
# Initialize the NPU runtime environment before invoking this launcher.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${LTX_MODEL_PATH:?Set the local LTX-2.3 Diffusers snapshot}"
export LTX_TRAIN_FILE="${LTX_TRAIN_FILE:-$script_dir/../../../data/omninft/vggsound/ltx_joint_full/train.parquet}"
export LTX_VAL_FILE="${LTX_VAL_FILE:-$script_dir/../../../data/omninft/vggsound/ltx_joint_full/test.parquet}"
for data_file in "$LTX_TRAIN_FILE" "$LTX_VAL_FILE"; do
    if [[ ! -f "$data_file" ]]; then
        echo "Missing dataset: $data_file; run $script_dir/prepare_omninft_data.py first" >&2
        exit 2
    fi
done
: "${LTX_REWARD_ROOT:?Set the five-reward asset directory}"
: "${LTX_OUTPUT_DIR:?Set a new output directory, or override trainer.resume_mode}"
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR:-$LTX_OUTPUT_DIR}"
mkdir -p "$LTX_OUTPUT_DIR" "$WANDB_DIR"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${LTX_CPU_THREADS:-4}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
memory_overrides=()
if [[ -n "${LTX_ACTOR_NPU_MEMORY_FRACTION:-}" ]]; then
    memory_overrides+=("++ray_kwargs.ray_init.runtime_env.env_vars.LTX_ACTOR_NPU_MEMORY_FRACTION=\"$LTX_ACTOR_NPU_MEMORY_FRACTION\"")
fi
ltx_devices="${LTX_NUM_DEVICES:-8}"
if [[ ! "$ltx_devices" =~ ^[1-9][0-9]*$ ]] || ((ltx_devices < 5)); then
    echo 'This five-model native reward recipe requires at least 5 devices with disjoint placements' >&2
    exit 2
fi
echo "输出目录：$LTX_OUTPUT_DIR"
python3 -m verl_omni.trainer.main_diffusion \
    --config-path "$script_dir" --config-name joint_train \
    trainer.n_gpus_per_node="$ltx_devices" \
    reward.num_workers="$ltx_devices" \
    actor_rollout_ref.rollout.ltx_joint.cpu_offload=true \
    reward.models.hpsv3.executor.kwargs.health_check.enabled=true \
    reward.models.hpsv3.executor.kwargs.health_check.baseline_path=/repo/outputs/hps-health-check-20261022-deterministic/reference.pt \
    reward.models.hpsv3.executor.kwargs.health_check.output_dir=/repo/outputs/ltx-joint-20261019-084556/hps_health \
    "${memory_overrides[@]}" \
    "$@" 2>&1 | tee "$LTX_OUTPUT_DIR/train.log"
