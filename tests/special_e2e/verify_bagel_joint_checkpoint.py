# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read-only CPU audit of real joint-training shards and durable frozen policies."""

import argparse
import json
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor

try:
    import torch_npu  # noqa: F401 (register Ascend checkpoint storage when installed)
except ModuleNotFoundError as exc:
    if exc.name != "torch_npu":
        raise


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def check_finite(values):
    count = 0
    for value in values:
        if isinstance(value, torch.Tensor):
            value = local(value)
            assert value.device.type == "cpu" and bool(torch.isfinite(value).all())
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--optimizer-updates", type=int, required=True)
    parser.add_argument("--policy-version", type=int, required=True)
    parser.add_argument("--compare-reference", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    reports = []
    expert_totals = {"understanding": 0.0, "generation": 0.0}
    for rank in range(args.world_size):
        actor = args.checkpoint / "actor"
        model = torch.load(
            actor / f"model_world_size_{args.world_size}_rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        policy = torch.load(actor / f"algorithm_state_rank_{rank}.pt", map_location="cpu", weights_only=True, mmap=True)
        assert policy["schema_version"] == 1 and policy["rank"] == rank
        assert policy["world_size"] == args.world_size and policy["policy_version"] == args.policy_version
        model_count = check_finite(model.values())
        policy_count = check_finite(list(policy["old"].values()) + list(policy["reference"].values()))
        for branch, key in (
            ("understanding", "layers.0.self_attn.q_proj.weight"),
            ("generation", "layers.0.self_attn.q_proj_moe_gen.weight"),
        ):
            current = model[key]
            assert isinstance(current, DTensor) and current.dtype == torch.float32
            delta = local(current) - policy["reference"][key]
            expert_totals[branch] += float(delta.square().sum(dtype=torch.float64))
        if args.compare_reference:
            previous = torch.load(
                args.compare_reference / "actor" / f"algorithm_state_rank_{rank}.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            assert previous["reference"].keys() == policy["reference"].keys()
            for key in previous["reference"]:
                torch.testing.assert_close(previous["reference"][key], policy["reference"][key], rtol=0, atol=0)
            del previous
        del model, policy
        optimizer = torch.load(
            actor / f"optim_world_size_{args.world_size}_rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        assert [group["lr"] for group in optimizer["param_groups"]] == [1e-6, 3e-5]
        counters = {int(state["step"]) for state in optimizer["state"].values()}
        assert counters == {args.optimizer_updates}, counters
        ids = [pid for group in optimizer["param_groups"] for pid in group["params"]]
        assert set(ids) == set(optimizer["state"])
        optimizer_count = check_finite(value for state in optimizer["state"].values() for value in state.values())
        reports.append(
            {
                "rank": rank,
                "model_tensors": model_count,
                "frozen_policy_tensors": policy_count,
                "optimizer_tensors": optimizer_count,
                "optimizer_updates": args.optimizer_updates,
            }
        )
        del optimizer
    assert all(value > 0 for value in expert_totals.values()), expert_totals
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "world_size": args.world_size,
                "all_tensors_finite": True,
                "both_experts_updated": True,
                "reference_unchanged_after_resume": bool(args.compare_reference),
                "expert_squared_deltas": expert_totals,
                "ranks": reports,
                "scope": "Execution and checkpoint integrity, not convergence or held-out improvement",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
