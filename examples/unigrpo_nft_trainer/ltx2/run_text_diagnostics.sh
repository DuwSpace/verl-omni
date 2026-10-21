#!/usr/bin/env bash
# Keep the original model, dataset, device placement and parallelism.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${LTX_OUTPUT_DIR:?Set a new absolute output directory for the diagnostic run}"
ltx_text_dump_dir="${LTX_TEXT_DIAGNOSTIC_DIR:-$LTX_OUTPUT_DIR/text_diagnostics}"
if [[ "$ltx_text_dump_dir" != /* ]]; then
    echo 'LTX_TEXT_DIAGNOSTIC_DIR must be an absolute path visible to every worker' >&2
    exit 2
fi
exec bash "$script_dir/run_train.sh" \
    actor_rollout_ref.rollout.ltx_joint.text_diagnostics.enabled=true \
    "actor_rollout_ref.rollout.ltx_joint.text_diagnostics.output_dir=\"$ltx_text_dump_dir\"" \
    actor_rollout_ref.rollout.ltx_joint.text_diagnostics.stop_on_mismatch=true \
    "$@"
