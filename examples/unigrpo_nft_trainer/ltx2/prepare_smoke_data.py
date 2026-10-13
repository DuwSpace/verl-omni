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

"""Reproduce the four-prompt functional experiment; not a held-out quality benchmark."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New dataset directory")
    args = parser.parse_args()
    prompts = json.loads(Path(__file__).with_name("fixed_prompts.json").read_text())
    rows = [
        {
            "data_source": "ltx2_unigrpo_nft",
            "prompt": [{"role": "user", "content": prompt}],
            "reward_model": {"ground_truth": prompt},
            "extra_info": {"index": index},
        }
        for index, prompt in enumerate(prompts)
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(rows)
    for name in ("train", "val"):
        frame.to_parquet(args.output / f"{name}.parquet", index=False)
    (args.output / "prompts.json").write_text(json.dumps(prompts, indent=2) + "\n")
    print(f"Wrote {len(rows)} fixed prompts to {args.output}; train and validation are identical smoke fixtures.")


if __name__ == "__main__":
    main()
