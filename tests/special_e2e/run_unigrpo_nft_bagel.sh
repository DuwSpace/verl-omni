#!/usr/bin/env bash
# Joint full-weight AR-GRPO/image NFT smoke, with native MP rollout and resume.
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
if (( NUM_GPUS < 2 || NUM_GPUS % 2 != 0 )); then
    echo "Joint smoke requires an even accelerator count >= 2" >&2
    exit 1
fi
task_output_dir=${BAGEL_JOINT_TRAIN_OUTPUT_DIR:-$(mktemp -d -t bagel-joint-smoke.XXXXXX)}
mkdir -p "$task_output_dir"
if [[ -e "$task_output_dir/checkpoints" ]]; then
    echo "Refusing to overwrite existing smoke checkpoints: $task_output_dir/checkpoints" >&2
    exit 1
fi
export BAGEL_MODEL_PATH=${MODEL_PATH:-$task_output_dir/models/bagel}
export PICKSCORE_PATH=${PICKSCORE_PATH:-$task_output_dir/models/pickscore}
task_device=${BAGEL_JOINT_DEVICE:-$(python3 -c 'from verl.utils.device import get_device_name; print(get_device_name())')}

python3 tests/special_e2e/build_bagel_tiny_random.py --output-dir "$BAGEL_MODEL_PATH" --untied-lm-head
python3 tests/special_e2e/build_pickscore_tiny_random.py --output-dir "$PICKSCORE_PATH"
python3 tests/special_e2e/create_dummy_bagel_pickscore_data.py \
    --local_save_dir "$task_output_dir/data" --model_path "$BAGEL_MODEL_PATH" \
    --train_size "$((NUM_GPUS * 2))" --val_size 4 --max_prompt_length 64

task_common_args=(
    "data.train_files=$task_output_dir/data/train.parquet"
    "data.val_files=$task_output_dir/data/test.parquet"
    "data.train_batch_size=$NUM_GPUS"
    "data.max_prompt_length=64"
    "trainer.device=$task_device"
    "trainer.n_gpus_per_node=$NUM_GPUS"
    "trainer.max_actor_ckpt_to_keep=2"
    "trainer.default_local_dir=$task_output_dir/checkpoints"
    "reward.num_workers=$NUM_GPUS"
    "reward.custom_reward_function.path=tests/special_e2e/bagel_pickscore_reward.py"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$((NUM_GPUS / 2))"
    "actor_rollout_ref.rollout.n=2"
    "actor_rollout_ref.rollout.pipeline.height=64"
    "actor_rollout_ref.rollout.pipeline.width=64"
    "actor_rollout_ref.rollout.pipeline.num_inference_steps=5"
    "actor_rollout_ref.rollout.bagel_joint.max_think_tokens=8"
    "actor_rollout_ref.rollout.bagel_joint.text_temperature=10"
)
bash examples/unigrpo_nft_trainer/bagel/run_bagel_unigrpo_nft_npu.sh \
    "${task_common_args[@]}" "$@" trainer.total_training_steps=1 trainer.resume_mode=disable
bash examples/unigrpo_nft_trainer/bagel/run_bagel_unigrpo_nft_npu.sh \
    "${task_common_args[@]}" "$@" trainer.total_training_steps=2 trainer.resume_mode=auto
python3 tests/special_e2e/verify_bagel_joint_checkpoint.py \
    --checkpoint "$task_output_dir/checkpoints/global_step_2" --world-size "$NUM_GPUS" \
    --optimizer-updates 4 --policy-version 1 \
    --compare-reference "$task_output_dir/checkpoints/global_step_1"
echo "BAGEL joint trainer and durable resume passed: $task_output_dir"
