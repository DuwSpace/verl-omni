#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export LTX_TRAIN_FILE="${LTX_TRAIN_FILE:-$REPO_ROOT/data/omninft/vggsound/ltx_joint_full/train.parquet}"
export LTX_VAL_FILE="${LTX_VAL_FILE:-$REPO_ROOT/data/omninft/vggsound/ltx_joint_full/test.parquet}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python3 -m verl_omni.trainer.main_diffusion_v1 \
  --config-path "$REPO_ROOT/examples/unigrpo_nft_trainer/ltx2" \
  --config-name joint_train "$@"
