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

"""Tree rollout shape and sibling request grouping (no inference dependencies)."""

from collections import defaultdict


def tree_width(settings, total=None):
    mode = settings.get("rollout_mode", "flat")
    width = settings.get("videos_per_recaption", 1)
    if mode not in {"flat", "tree"}:
        raise ValueError("rollout_mode must be flat or tree")
    if type(width) is not int or width < 1:
        raise ValueError("videos_per_recaption must be a positive integer")
    if mode == "flat" and width != 1:
        raise ValueError("videos_per_recaption > 1 requires rollout_mode=tree")
    if total is not None and (total < 1 or total % width):
        raise ValueError("rollout.n must be a positive multiple of videos_per_recaption (N*k)")
    return width


def group_recaption_requests(requests):
    """Return complete sibling groups, ordered by video index, before encoding."""
    groups = defaultdict(list)
    for request in requests:
        extra = request.sampling_params.extra_args
        key = extra.get("recaption_group_id", extra["prepared_request_id"])
        groups[key].append(request)
    for group in groups.values():
        extra = group[0].sampling_params.extra_args
        width = tree_width(extra)
        if len(group) != width:
            raise ValueError("Incomplete recaption sibling group")
        if extra.get("rollout_mode", "flat") == "tree":
            group.sort(key=lambda request: request.sampling_params.extra_args["recaption_video_index"])
            indices = [request.sampling_params.extra_args["recaption_video_index"] for request in group]
            if indices != list(range(width)):
                raise ValueError("Duplicate or missing recaption video index")
            if len({request.sampling_params.seed for request in group}) != width:
                raise ValueError("Sibling videos require distinct seeds")
    return list(groups.values())


def tree_sampling_params(settings, *, uid, session_id, total, prompt_index, base_seed):
    """Attach parent/child identity and a distinct AV seed to a leaf request."""
    from verl_omni.agent_loop.utils import derive_rollout_seed

    width = tree_width(settings, total)
    seed = base_seed if base_seed is not None else settings.get("seed", 42)
    return {
        "recaption_group_id": f"{uid}:{session_id // width}",
        "recaption_video_index": session_id % width,
        "seed": derive_rollout_seed(seed, prompt_index * total + session_id),
    }
