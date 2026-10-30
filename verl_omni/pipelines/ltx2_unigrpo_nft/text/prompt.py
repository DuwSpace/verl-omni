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

"""Official LTX audiovisual caption template and response parsing."""

from pathlib import Path


def recaption_messages(prompt, duration_seconds=None, *, preset="ltx_t2av"):
    """Use the official LTX caption template for audiovisual requests."""
    if preset != "ltx_t2av" or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Expected a nonempty prompt and the ltx_t2av template")
    system = (Path(__file__).parent / "prompts/gemma_t2v_system_prompt.txt").read_text(encoding="utf-8")
    return [{"role": "system", "content": system}, {"role": "user", "content": f"user prompt: {prompt}"}]


def direct_recaption(text, *, ended_with_eos):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Gemma returned an empty caption")
    if not ended_with_eos:
        raise ValueError("Gemma caption truncated before EOS; increase recaption_max_new_tokens")
    return text.strip()
