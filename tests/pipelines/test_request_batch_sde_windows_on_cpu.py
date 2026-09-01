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

"""CPU tests for per-sample SDE window sampling under request packing."""

import torch

from verl_omni.pipelines.request_batch import collate_prompt_mask, collate_prompt_rows, sample_per_sample_sde_windows


def _advanced_generator(seed: int, latent_numel: int = 8) -> torch.Generator:
    generator = torch.Generator().manual_seed(seed)
    _ = torch.randn(latent_numel, generator=generator)
    return generator


def test_per_sample_sde_windows_match_serial_and_ignore_pack_order():
    seeds = [11, 22, 33]
    serial = [
        sample_per_sample_sde_windows(
            sde_window_size=2,
            sde_window_range=(0, 5),
            num_timesteps=10,
            batch_size=1,
            generator=_advanced_generator(seed),
            device="cpu",
        )[0]
        for seed in seeds
    ]

    packed_order = [2, 0, 1]
    packed = sample_per_sample_sde_windows(
        sde_window_size=2,
        sde_window_range=(0, 5),
        num_timesteps=10,
        batch_size=len(packed_order),
        generator=[_advanced_generator(seeds[i]) for i in packed_order],
        device="cpu",
    )
    assert packed == [serial[i] for i in packed_order]


def test_collate_prompt_rows_left_pads_to_target_len() -> None:
    prompts = [{"prompt_token_ids": [10, 20]}, {"prompt_token_ids": [30]}]
    ids, lengths = collate_prompt_rows(
        prompts,
        ("prompt_token_ids",),
        None,
        device=torch.device("cpu"),
        field_name="prompt_token_ids",
        pad_value=0,
        pad_side="left",
        target_len=4,
    )
    assert ids.tolist() == [[0, 0, 10, 20], [0, 0, 0, 30]]
    assert lengths == [2, 1]
    mask = collate_prompt_mask(
        prompts,
        ("prompt_mask",),
        None,
        device=torch.device("cpu"),
        field_name="prompt_mask",
        token_lengths=lengths,
        target_seq_len=4,
        pad_side="left",
    )
    assert mask.tolist() == [[False, False, True, True], [False, False, False, True]]
