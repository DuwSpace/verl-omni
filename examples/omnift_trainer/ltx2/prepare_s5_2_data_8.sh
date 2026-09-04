#!/usr/bin/env bash
# Regenerate outputs/s5_2_data_8 from data/omninft/vggsound/train_metadata_20k.jsonl
# train = first 8 records, test = records 1001-1008 (val_file = tail after first 1000 lines)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE="${REPO_ROOT}/data/omninft/vggsound/train_metadata_20k.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/s5_2_data_8}"
TRAIN_SIZE=8
VAL_SIZE=8
VAL_SKIP=1000

VAL_FILE="$(mktemp /tmp/val_tail.XXXXXX.jsonl)"
trap 'rm -f "${VAL_FILE}"' EXIT

tail -n +"$((VAL_SKIP + 1))" "${SOURCE}" > "${VAL_FILE}"

python3 "${REPO_ROOT}/examples/omnift_trainer/ltx2/prepare_data.py" \
  --train_file "${SOURCE}" \
  --val_file "${VAL_FILE}" \
  --train_size "${TRAIN_SIZE}" \
  --val_size "${VAL_SIZE}" \
  --output_dir "${OUTPUT_DIR}"
