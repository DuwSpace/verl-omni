#!/usr/bin/env bash
# Regenerate outputs/s5_2_data from data/omninft/vggsound/train_metadata_20k.jsonl.
# train = first record; test = record 1001.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SOURCE="${REPO_ROOT}/data/omninft/vggsound/train_metadata_20k.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/s5_2_data}"
VAL_SKIP=1000

VAL_FILE="$(mktemp /tmp/omninft_val_tail.XXXXXX.jsonl)"
trap 'rm -f "${VAL_FILE}"' EXIT

tail -n +"$((VAL_SKIP + 1))" "${SOURCE}" > "${VAL_FILE}"

python3 "${SCRIPT_DIR}/prepare_data.py" \
  --train_file "${SOURCE}" \
  --val_file "${VAL_FILE}" \
  --train_size 1 \
  --val_size 1 \
  --output_dir "${OUTPUT_DIR}"
