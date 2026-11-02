# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encode each parent once and share its CPU conditions across video leaves."""

from dataclasses import replace

from .requests import group_recaption_requests


def prepare_recaption_tree(requests, batch_size, prepare_text):
    """Yield per-video cache entries while batching only unique recaptions."""
    groups = group_recaption_requests(requests)
    siblings = {id(group[0]): group for group in groups}
    leaders = [group[0] for group in groups]
    for start in range(0, len(leaders), batch_size):
        for leader, record, embeddings, mask, negative in prepare_text(leaders[start : start + batch_size]):
            for request in siblings[id(leader)]:
                extra = request.sampling_params.extra_args
                child = replace(
                    record,
                    seed=int(request.sampling_params.seed),
                    recaption_group_id=extra.get("recaption_group_id"),
                    recaption_seed=int(leader.sampling_params.seed),
                    recaption_video_index=extra.get("recaption_video_index", 0),
                    videos_per_recaption=len(siblings[id(leader)]),
                )
                yield extra["prepared_request_id"], (request, child, embeddings, mask, negative)
