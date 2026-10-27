#!/usr/bin/env bash
# Host launcher plus its container entry point. Text checks are enabled by default.
set -eo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ${1:-} == --inside || ${1:-} == --preflight ]]; then
    mode="$1"
    shift
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
        source /usr/local/Ascend/nnal/atb/set_env.sh
    fi
    set -u
    cd /repo
    export PYTHONPATH="/repo${PYTHONPATH:+:$PYTHONPATH}"
    export VLLM_ASCEND_ENABLE_NZ=0
    export LTX_TRAIN_FILE="${LTX_TRAIN_FILE:-/repo/data/omninft/vggsound/ltx_joint_full/train.parquet}"
    export LTX_VAL_FILE="${LTX_VAL_FILE:-/repo/data/omninft/vggsound/ltx_joint_full/test.parquet}"
    # Older containers only mount /repo; keep their outputs on the host too.
    if [[ -z ${LTX_OUTPUT_ROOT:-} ]]; then
        export LTX_OUTPUT_ROOT=/repo/outputs
        [[ ! -d /outputs ]] || export LTX_OUTPUT_ROOT=/outputs
    fi
    : "${LTX_MODEL_PATH:?Set LTX_MODEL_PATH when creating the container}"
    : "${LTX_REWARD_ROOT:?Set LTX_REWARD_ROOT when creating the container}"
    for path in "$LTX_MODEL_PATH/model_index.json" "${LTX_TRAIN_FILE:?}" "${LTX_VAL_FILE:?}" \
        "$LTX_REWARD_ROOT/HPSv3/HPSv3.safetensors" \
        "$LTX_REWARD_ROOT/VideoReward/checkpoint-11352/model.pth" \
        "$LTX_REWARD_ROOT/synchformer/synchformer_state_dict.pth"; do
        [[ -f $path ]] || { echo "Missing asset: $path" >&2; exit 2; }
    done
    for path in Qwen2-VL-2B-Instruct Qwen2-VL-7B-Instruct audiobox-aesthetics checkpoints/clap-htsat-unfused OmniNFT-reference; do
        [[ -d $LTX_REWARD_ROOT/$path ]] || { echo "Missing reward directory: $LTX_REWARD_ROOT/$path" >&2; exit 2; }
    done
    if [[ $mode == --preflight ]]; then
        [[ ! -e ${LTX_OUTPUT_ROOT:-/outputs}/${LTX_RUN_NAME:?} ]] || { echo 'Run directory already exists' >&2; exit 2; }
        flock -n /tmp/ltx-training.lock true || { echo 'An LTX training run is already active' >&2; exit 2; }
        printf '%s\n' "$LTX_OUTPUT_ROOT/$LTX_RUN_NAME"
        exit 0
    fi
    exec 9>/tmp/ltx-training.lock
    flock -n 9 || { echo 'An LTX training run is already active in this container' >&2; exit 2; }
    export LTX_OUTPUT_DIR="${LTX_OUTPUT_ROOT:-/outputs}/${LTX_RUN_NAME:?}"
    mkdir -p "$(dirname -- "$LTX_OUTPUT_DIR")"
    mkdir "$LTX_OUTPUT_DIR"  # A new run must not overwrite earlier logs.
    exec > >(tee "$LTX_OUTPUT_DIR/launcher.log") 2>&1
    echo "Started $(date -Is), code=/repo, output=$LTX_OUTPUT_DIR"
    launcher=run_text_diagnostics.sh
    [[ ${LTX_TEXT_CHECK:-1} != 0 ]] || launcher=run_train.sh
    overrides=(trainer.resume_mode=disable "ablation=${LTX_ABLATION:-joint}")
    if [[ ${LTX_SMOKE:-0} == 1 ]]; then
        overrides+=(trainer.total_training_steps=2 trainer.test_freq=-1 trainer.save_freq=-1 \
            'trainer.logger=[console]' trainer.rollout_data_save_freq=0 data.train_batch_size=4 \
            actor_rollout_ref.rollout.n=2 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1)
    fi
    set +e
    bash "$script_dir/$launcher" "${overrides[@]}" "$@"
    status=$?
    printf '%s\n' "$status" > "$LTX_OUTPUT_DIR/exit_code"
    exit "$status"
fi
set -u
if [[ $# == 0 || ${1:-} == --help ]]; then
    echo "Usage: bash $0 CONTAINER [--name RUN] [--ablation joint|av_only|ar_only] [--foreground] [--smoke] [--no-text-check] [-- HYDRA_OVERRIDES...]"
    echo 'Default: background training with Text checks; no HPS baseline is needed.'
    exit 0
fi
container="$1"
shift
run_name=""
ablation=joint
foreground=0
smoke=0
text_check=1
while (($#)); do
    case "$1" in
        --name) run_name="${2:?Missing run name}"; shift 2 ;;
        --ablation) ablation="${2:?Missing ablation name}"; shift 2 ;;
        --foreground) foreground=1; shift ;;
        --smoke) smoke=1; shift ;;
        --no-text-check) text_check=0; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done
case "$ablation" in
    joint|av_only|ar_only) ;;
    *) echo 'Ablation must be joint, av_only or ar_only' >&2; exit 2 ;;
esac
run_name="${run_name:-ltx-$ablation-$(date +%Y%m%d-%H%M%S)}"
[[ $run_name =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'Invalid run name' >&2; exit 2; }
entry=/repo/examples/unigrpo_nft_trainer/ltx2/start_train.sh
env_args=(-e "LTX_RUN_NAME=$run_name" -e "LTX_SMOKE=$smoke" -e "LTX_TEXT_CHECK=$text_check" -e "LTX_ABLATION=$ablation")
run_dir=$(docker exec "${env_args[@]}" "$container" bash "$entry" --preflight)
if ((foreground)); then
    docker exec "${env_args[@]}" "$container" bash "$entry" --inside "$@"
else
    docker exec -d "${env_args[@]}" "$container" bash "$entry" --inside "$@"
fi
echo "Run: $run_name"
echo "Logs: docker exec $container tail -f $run_dir/launcher.log"
echo "Exit code after completion: $run_dir/exit_code"
