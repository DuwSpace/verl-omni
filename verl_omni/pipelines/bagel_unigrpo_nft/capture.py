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

"""Request-local capture around native vllm-omni BAGEL, without copying its sampler."""

from contextlib import contextmanager

import torch

from .replay import BagelJointReplay


def _cpu(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=dtype).clone()


class BagelReplayCapture:
    """Collect the exact native inputs, sampled-token logprobs and final latent.

    Instance-local method interception and module hooks are restored on failure.
    Only one vocabulary vector stays live during decoding, not T×vocab logits.
    """

    def __init__(self, bagel):
        self.bagel = bagel
        self.prompt_ids = None
        self.thinking_ids = None
        self.response_ids = None
        self.log_probs = None
        self.latent = None
        self.latent_positions = None
        self.image_positions = None
        self.boundary_ids = None
        self.temperature = None
        self.eos_token_id = None

    def _prompts(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        if self.prompt_ids is None:
            generation_input = result[0]
            ids = generation_input["packed_text_ids"]
            positions = generation_input["packed_text_position_ids"]
            if not torch.equal(positions.cpu(), torch.arange(ids.numel())):
                raise ValueError("Joint T2I replay requires an initial text-only context starting at position zero")
            self.prompt_ids = _cpu(ids, torch.int64)
        return result

    def _thinking(self, original, *args, **kwargs):
        if self.thinking_ids is not None:
            raise ValueError("Only one thinking chain per request is supported")
        if not kwargs.get("do_sample", False) or float(kwargs.get("temperature", 0)) <= 0:
            raise ValueError("AR-GRPO rollout requires stochastic native text sampling with positive temperature")
        if int(kwargs["max_length"]) < 2:
            raise ValueError("At least two native thinking steps are required")
        temperature = float(kwargs["temperature"])
        self.eos_token_id = int(kwargs["end_token_id"])
        previous_logits = None
        sampled_ids, log_probs = [], []

        def record_token(module, args, inputs):
            nonlocal previous_logits
            if previous_logits is not None:
                token = inputs["packed_text_ids"].reshape(-1)
                if token.numel() != 1:
                    raise ValueError("Joint capture requires single-sequence decoding")
                sampled_ids.append(_cpu(token, torch.int64))
                log_probs.append(
                    _cpu(torch.log_softmax(previous_logits.float() / temperature, -1)[0, token], torch.float32)
                )
                previous_logits = None

        def record_logits(module, inputs, output):
            nonlocal previous_logits
            if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
                raise ValueError("Native BAGEL lm_head must expose [1, vocab] logits")
            previous_logits = output.detach()

        model = self.bagel.language_model
        input_hook = model.register_forward_pre_hook(record_token, with_kwargs=True)
        output_hook = model.lm_head.register_forward_hook(record_logits)
        try:
            tokens = original(*args, **kwargs)
            if tokens.ndim != 2 or tokens.shape[1] != 1:
                raise ValueError("Native BAGEL thinking must return [T, 1] cached token IDs")
            self.thinking_ids = _cpu(tokens[:, 0], torch.int64)
            if not torch.equal(self.thinking_ids[:1], kwargs["packed_start_tokens"].detach().cpu()):
                raise ValueError("Native thinking did not preserve its fixed start token")
            if tokens.shape[0] < int(kwargs["max_length"]):
                # Native generate_text stops on the next EOS draw without caching it.
                eos = int(kwargs["end_token_id"])
                sampled_ids.append(torch.tensor([eos], dtype=torch.int64))
                log_probs.append(
                    _cpu(torch.log_softmax(previous_logits.float() / temperature, -1)[0, eos : eos + 1], torch.float32)
                )
            self.response_ids = torch.cat(sampled_ids)
            self.log_probs = torch.cat(log_probs)
            self.temperature = temperature
        finally:
            input_hook.remove()
            output_hook.remove()
        return tokens

    def _image(self, original, *args, **kwargs):
        if self.thinking_ids is None or self.prompt_ids is None:
            raise ValueError("Native image generation did not consume a captured thinking context")
        cache_length = kwargs["past_key_values"].key_cache[0].shape[0]
        if cache_length != self.prompt_ids.numel() + self.thinking_ids.numel():
            raise ValueError("Image KV cache length differs from the captured prompt/thinking context")
        output = original(*args, **kwargs)
        latents = output[0]
        if len(latents) != 1:
            raise ValueError("Joint BAGEL capture requires one final latent per native request")
        self.latent = _cpu(latents[0], torch.float32)
        self.latent_positions = _cpu(kwargs["packed_vae_position_ids"], torch.int64)
        self.image_positions = _cpu(kwargs["packed_position_ids"], torch.int64)
        self.boundary_ids = _cpu(kwargs["packed_text_ids"], torch.int64)
        return output

    @contextmanager
    def installed(self):
        """Intercept only this BAGEL instance for one request; restore all methods."""
        if getattr(self.bagel, "_joint_capture_running", False):
            raise RuntimeError("Concurrent/reentrant BAGEL capture on one pipeline is unsupported")
        self.bagel._joint_capture_running = True
        methods = {"prepare_prompts": self._prompts, "generate_text": self._thinking, "generate_image": self._image}
        originals = {name: getattr(self.bagel, name) for name in methods}
        previous = {name: self.bagel.__dict__.get(name) for name in methods}
        try:
            for name, wrapper in methods.items():
                original = originals[name]
                setattr(
                    self.bagel,
                    name,
                    lambda *args, _wrapper=wrapper, _original=original, **kwargs: _wrapper(_original, *args, **kwargs),
                )
            yield self
        finally:
            for name in methods:
                if previous[name] is None:
                    delattr(self.bagel, name)
                else:
                    setattr(self.bagel, name, previous[name])
            self.bagel._joint_capture_running = False

    def result(self, *, policy_version: int) -> BagelJointReplay:
        """Build a validated, detached CPU record after successful native generation."""
        values = (
            self.prompt_ids,
            self.thinking_ids,
            self.response_ids,
            self.log_probs,
            self.latent,
            self.latent_positions,
            self.image_positions,
            self.boundary_ids,
            self.temperature,
        )
        if any(value is None for value in values):
            raise RuntimeError("Native BAGEL did not expose every required joint replay field")
        replay = BagelJointReplay(*values, policy_version=policy_version, eos_token_id=self.eos_token_id)
        replay.validate()
        return replay
