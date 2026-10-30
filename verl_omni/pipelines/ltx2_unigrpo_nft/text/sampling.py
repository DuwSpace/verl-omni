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

"""Transformers batched decoding with independent per-response random streams."""

import math

import torch
from transformers import GenerationConfig, LogitsProcessor

from .prompt import recaption_messages


class SeededSampler(LogitsProcessor):
    """Sample before greedy selection and retain only the selected logprobs.

    Transformers' ordinary batched multinomial shares one RNG across rows.
    Sampling each row here preserves independent request seeds. Returning one
    finite score makes greedy selection emit our sampled token; the stored
    logprob is from the original temperature-scaled distribution, not this
    forced selection. Finished rows have private RNGs and cannot affect peers.
    """

    def __init__(self, seeds, temperature, device):
        self.generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
        self.temperature = temperature
        self.logprobs = []

    def __call__(self, input_ids, scores):
        scaled = scores.float() / self.temperature
        probabilities = scaled.softmax(dim=-1)
        tokens = torch.stack(
            [torch.multinomial(row, 1, generator=rng) for row, rng in zip(probabilities, self.generators, strict=True)]
        )
        self.logprobs.append(scaled.log_softmax(dim=-1).gather(-1, tokens).squeeze(-1))
        return torch.full_like(scores, -torch.inf).scatter_(-1, tokens, 0.0)


@torch.no_grad()
def generate_recaptions(model, tokenizer, prompts, seeds, *, temperature, max_new_tokens, duration_seconds, preset):
    if not prompts or len(prompts) != len(seeds):
        raise ValueError("Batched recaption needs one seed per nonempty prompt batch")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("Recaption max_new_tokens must be a positive integer")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Recaption temperature must be finite and positive")
    prompt_ids = [
        tokenizer.apply_chat_template(
            recaption_messages(prompt, duration_seconds, preset=preset),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            return_tensors="pt",
        )[0]
        for prompt in prompts
    ]
    eos = model.generation_config.eos_token_id
    eos = (eos,) if isinstance(eos, int) else tuple(eos)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0]
    width = max(ids.numel() for ids in prompt_ids)
    ids = torch.full((len(prompts), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, prompt in enumerate(prompt_ids):
        ids[row, -prompt.numel() :] = prompt
        mask[row, -prompt.numel() :] = 1
    device = model.device
    sampler = SeededSampler(seeds, temperature, device)
    output = model.generate(
        input_ids=ids.to(device),
        attention_mask=mask.to(device),
        generation_config=GenerationConfig(
            do_sample=False,  # SeededSampler owns stochastic selection.
            max_new_tokens=max_new_tokens,
            eos_token_id=list(eos),
            pad_token_id=pad,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
        ),
        logits_processor=[sampler],
    )
    responses = output.sequences[:, width:].cpu()
    logprobs = torch.stack(sampler.logprobs, dim=1).cpu()
    if responses.shape != logprobs.shape:
        raise RuntimeError("Batched recaption lost sampled-token logprobs")
    results = []
    for row, response in enumerate(responses):
        stops = torch.isin(response, torch.tensor(eos)).nonzero()
        length = int(stops[0, 0]) + 1 if stops.numel() else response.numel()
        response = response[:length].clone()
        text = tokenizer.decode(response, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("Gemma generated an empty recaption")
        results.append((text, prompt_ids[row].cpu(), response, logprobs[row, :length].clone(), eos))
    return results
