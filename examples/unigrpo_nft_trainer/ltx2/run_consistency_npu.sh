#!/usr/bin/env bash
# Same-input old AV diagnostic. Normal rollout remains native, serial and cond-only.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${DUMP_PATH:?Set an unused shared diagnostic directory}"
export DUMP_ON=1
export DUMP_PHASE=update_actor
export MSPROBE_DUMP_GLOBAL_STEP="${MSPROBE_DUMP_GLOBAL_STEP:-1}"
export MSPROBE_CONFIG_ACTOR="${MSPROBE_CONFIG_ACTOR:-$script_dir/msprobe_av.json}"
export MSPROBE_CONFIG_GENERATE="${MSPROBE_CONFIG_GENERATE:-$script_dir/msprobe_av.json}"
exec bash "$script_dir/run_joint_npu.sh" \
    data.train_batch_size=1 \
    trainer.val_before_train=false trainer.test_freq=-1 \
    "trainer.total_training_steps=$MSPROBE_DUMP_GLOBAL_STEP" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    '++ray_kwargs.ray_init.runtime_env.env_vars.DUMP_ON="1"' \
    '++ray_kwargs.ray_init.runtime_env.env_vars.DUMP_PHASE="update_actor"' \
    "++ray_kwargs.ray_init.runtime_env.env_vars.MSPROBE_DUMP_GLOBAL_STEP=\"$MSPROBE_DUMP_GLOBAL_STEP\"" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.DUMP_PATH=\"$DUMP_PATH\"" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.MSPROBE_CONFIG_ACTOR=\"$MSPROBE_CONFIG_ACTOR\"" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.MSPROBE_CONFIG_GENERATE=\"$MSPROBE_CONFIG_GENERATE\"" \
    "$@"
