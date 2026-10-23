#!/usr/bin/env bash
# Create the eight-NPU container on the host, then install recipe dependencies.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"
if [[ ${1:-} == --help ]]; then
    echo "Usage: bash $0 [server.env]"
    echo "Copy server.env.example, edit host paths, then pass the copy here."
    exit 0
fi
if (($#)); then
    [[ $# == 1 && -f $1 ]] || { echo 'Expected one existing server.env file' >&2; exit 2; }
    set -a
    source "$1"
    set +a
fi
: "${LTX_HUB_HOST:?Set the host model cache root containing model snapshots and reward assets}"
LTX_IMAGE="${LTX_IMAGE:-crpi-gwfvpzl8eqw3jv8c.cn-shanghai.personal.cr.aliyuncs.com/verl_omni/npu:npu-a2-main-8d2a9fd}"
LTX_CONTAINER_NAME="${LTX_CONTAINER_NAME:-ltx23-unigrpo}"
LTX_DATA_HOST="${LTX_DATA_HOST:-$repo_dir/data/omninft/vggsound/ltx_joint_full}"
LTX_OUTPUT_HOST="${LTX_OUTPUT_HOST:-$repo_dir/outputs}"
LTX_MODEL_PATH="${LTX_MODEL_PATH:-/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee8edcf067e838b843f926ec4d4cc9b2be1aaf}"
LTX_REWARD_ROOT="${LTX_REWARD_ROOT:-/hub/omnift-rewards}"
for directory in "$LTX_HUB_HOST" "$LTX_DATA_HOST"; do
    [[ -d $directory && $directory == /* ]] || { echo "Missing absolute host directory: $directory" >&2; exit 2; }
done
[[ $LTX_OUTPUT_HOST == /* ]] || { echo 'LTX_OUTPUT_HOST must be absolute' >&2; exit 2; }
mkdir -p "$LTX_OUTPUT_HOST"
docker image inspect "$LTX_IMAGE" >/dev/null
if docker container inspect "$LTX_CONTAINER_NAME" >/dev/null 2>&1; then
    echo "Container already exists: $LTX_CONTAINER_NAME. Reuse it with start_train.sh or choose a new name." >&2
    exit 2
fi
devices=()
for device in /dev/davinci{0..7} /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
    [[ -e $device ]] || { echo "Missing NPU device: $device" >&2; exit 2; }
    devices+=(--device "$device:$device")
done
mounts=(-v "$repo_dir:/repo" -v "$LTX_HUB_HOST:/hub:ro" -v "$LTX_DATA_HOST:/ltx-data:ro" -v "$LTX_OUTPUT_HOST:/outputs")
for path in /usr/local/Ascend/driver /usr/local/dcmi /usr/local/sbin/npu-smi /etc/ascend_install.info; do
    [[ ! -e $path ]] || mounts+=(-v "$path:$path:ro")
done
docker run -d --name "$LTX_CONTAINER_NAME" --network host --shm-size "${LTX_SHM_SIZE:-16g}" \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${devices[@]}" "${mounts[@]}" --workdir /repo \
    -e PYTHONPATH=/repo -e LTX_NUM_DEVICES=8 \
    -e "LTX_MODEL_PATH=$LTX_MODEL_PATH" -e "LTX_REWARD_ROOT=$LTX_REWARD_ROOT" \
    -e "LTX_TRAIN_FILE=/ltx-data/${LTX_TRAIN_NAME:-train.parquet}" \
    -e "LTX_VAL_FILE=/ltx-data/${LTX_VAL_NAME:-test.parquet}" \
    -e LTX_OUTPUT_ROOT=/outputs \
    --entrypoint /bin/bash "$LTX_IMAGE" -lc 'exec sleep infinity'
if [[ ${LTX_INSTALL_DEPS:-1} == 1 ]]; then
    pip_env=()
    [[ -z ${PIP_INDEX_URL:-} ]] || pip_env+=(-e "PIP_INDEX_URL=$PIP_INDEX_URL")
    docker exec "${pip_env[@]}" "$LTX_CONTAINER_NAME" bash /repo/examples/unigrpo_nft_trainer/ltx2/setup_npu_env.sh
fi
echo "Ready: $LTX_CONTAINER_NAME"
echo "Start: bash $script_dir/start_train.sh $LTX_CONTAINER_NAME"
