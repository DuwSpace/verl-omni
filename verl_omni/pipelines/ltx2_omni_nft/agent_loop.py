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

"""OmniNFT agent-loop worker and LTX raw-text registry entry."""

from uuid import uuid4

import numpy as np
from verl.experimental.agent_loop.agent_loop import register
from verl.protocol import DataProto

from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopWorker
from verl_omni.pipelines.ltx2_flow_grpo.agent_loop import LTX2DiffusionSingleTurnAgentLoop

__all__ = ["LTX2OmniNFTAgentLoopWorker", "LTX2OmniNFTSingleTurnAgentLoop"]


def _ensure_omni_nft_sample_uids(batch: DataProto) -> None:
    sample_uids = batch.non_tensor_batch.get("sample_uid")
    if sample_uids is None:
        sample_uids = np.array([uuid4().hex for _ in range(len(batch))], dtype=object)
        batch.non_tensor_batch["sample_uid"] = sample_uids
    else:
        sample_uids = np.asarray(sample_uids, dtype=object)
        if sample_uids.shape != (len(batch),):
            raise ValueError(f"sample_uid must have shape ({len(batch)},), got {sample_uids.shape}.")
        batch.non_tensor_batch["sample_uid"] = sample_uids
    if len(set(map(str, sample_uids))) != len(sample_uids):
        raise ValueError("sample_uid values must be unique within an OmniNFT rollout batch.")


class LTX2OmniNFTAgentLoopWorker(DiffusionAgentLoopWorker):
    """Assign unique per-rollout identities around the shared worker."""

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Assign per-rollout identities before dispatching repeated prompt groups."""
        _ensure_omni_nft_sample_uids(batch)
        return await super().generate_sequences(batch)


@register("ltx2_omni_nft_single_turn_agent")
class LTX2OmniNFTSingleTurnAgentLoop(LTX2DiffusionSingleTurnAgentLoop):
    """Use LTX raw-text tokenization for normalized OmniNFT prompts."""
