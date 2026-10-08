# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Map all BAGEL actor components, not just transformer blocks, to native checkpoint names."""


def native_actor_weight_name(name: str) -> str:
    """Convert the shared engine's transformer-prefixed BagelForSFT state keys."""
    if not name.startswith("transformer."):
        return name
    key = name.removeprefix("transformer.")
    if key.startswith(("layers.", "embed_tokens.", "norm.", "norm_moe_gen.")):
        return "language_model.model." + key
    if key.startswith("lm_head."):
        return "language_model." + key
    if key.startswith(
        ("time_embedder.", "latent_pos_embed.", "vae2llm.", "llm2vae.", "vit_model.", "connector.", "vit_pos_embed.")
    ):
        return key
    raise ValueError(f"Unknown BAGEL actor state key; refusing partial policy sync: {name}")
