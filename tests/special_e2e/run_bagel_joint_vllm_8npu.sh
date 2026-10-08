#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Eight independent native MP engines, one physical NPU per replica.
# Tiny backend integration check, NOT standalone UniGRPO reproduction training.
set -euo pipefail
task_output_dir=${BAGEL_JOINT_OUTPUT_DIR:-reproduction/joint-vllm-8npu}
mkdir -p "$task_output_dir"
task_pids=()
for task_card in 0 1 2 3 4 5 6 7; do
    ASCEND_RT_VISIBLE_DEVICES="$task_card" OMP_NUM_THREADS=4 \
        python tests/special_e2e/bagel_joint_vllm_npu.py \
        --output "$task_output_dir/card-$task_card" "$@" \
        > "$task_output_dir/card-$task_card.log" 2>&1 &
    task_pids+=("$!")
done
task_result=0
for task_pid in "${task_pids[@]}"; do
    if ! wait "$task_pid"; then
        task_result=1
    fi
done
if [[ "$task_result" == 0 ]]; then
    printf 'BAGEL_JOINT_ALL_8_NPU_OK: %s\n' "$task_output_dir"
else
    printf 'BAGEL joint replica failed; inspect per-card logs in %s\n' "$task_output_dir" >&2
fi
exit "$task_result"
