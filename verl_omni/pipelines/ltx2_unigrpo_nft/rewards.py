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

"""Media conversion shared with training, plus a sequential scorer for offline replay.

Training dispatches through OmniRewardLoopManager. LTXJointRewardRunner is used
by score_rollout.py to reproduce the same configured scorers outside training.
"""

from copy import deepcopy
from time import perf_counter

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop.reward_manager.multi import MultiVisualRewardManager
from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor
from verl_omni.workers.config.reward import RewardModelSpec


def reward_sample(sample) -> DataProto:
    """Project generated media, always using the original task for text scorers."""
    record = sample["replay"]
    record.validate()
    video = sample["video"].detach().cpu()
    if video.ndim != 4 or video.shape[1] != 3:
        raise ValueError("Reward video requires [frames,3,height,width]")
    if video.dtype.is_floating_point:
        if not torch.isfinite(video).all() or video.min() < 0 or video.max() > 1:
            raise ValueError("Decoded float video must be finite and in [0,1]")
        video = video.float().mul(255).round().to(torch.uint8)
    elif video.dtype != torch.uint8:
        raise ValueError("Decoded video must be uint8 or floating point")
    audio = sample["audio"].detach().cpu()
    if audio.ndim != 2 or not audio.dtype.is_floating_point or not torch.isfinite(audio).all():
        raise ValueError("Reward audio requires finite floating-point [channels,time]")
    if float(sample["fps"]) != float(record.generation_params["frame_rate"]):
        raise ValueError("Reward fps must match the actual rollout geometry")
    return DataProto.from_dict(
        tensors={
            "responses": video.unsqueeze(0),
            "audio": audio.unsqueeze(0),
            "audio_sample_rate": torch.tensor([sample["audio_sample_rate"]]),
        },
        non_tensors={
            "data_source": np.array(["ltx2_unigrpo_nft"], dtype=object),
            "reward_model": np.array([{"ground_truth": record.original_prompt}], dtype=object),
            "reward_inputs": np.array(
                [{"text": {"video": record.original_prompt, "audio": record.original_prompt}}], dtype=object
            ),
            "audio_layout": np.array(["CT"], dtype=object),
            "fps": np.array([sample["fps"]]),
        },
    )


class LTXJointRewardRunner:
    """Score a rollout batch one model at a time using the shared reward manager.

    Each model is woken once per term and slept before the next model. Executors
    persist across batches, retaining CPU weights when cpu_offload is enabled.
    Scorer preprocessing, weights and routing come exclusively from config.
    The caller must release/offload rollout models before invoking this runner
    when both use the same device.
    """

    def __init__(self, config):
        self.config = config
        self._executors = {}
        if config.reward.aggregation != "preserve_components":
            raise ValueError("Joint rewards require preserve_components aggregation")
        if len(config.reward.reward_functions) != 5:
            raise ValueError("Joint LTX recipe requires all five reward functions")

    async def score(self, samples):
        if not samples:
            raise ValueError("Cannot score an empty rollout batch")
        started = perf_counter()
        data = [reward_sample(sample) for sample in samples]
        names = sorted(self.config.reward.reward_functions)
        infos = [{} for _ in samples]
        columns = []
        metrics = {}
        for name in names:
            manager, executor = self._scorer_for_term(name)
            load_start = perf_counter()
            try:
                await executor.wake_up()
                metrics[f"reward/{name}/load_time"] = perf_counter() - load_start
                score_start = perf_counter()
                scores = []
                for index, item in enumerate(data):
                    result = await manager.run_single(item)
                    info = result["reward_extra_info"]
                    scores.append(info[f"reward/{name}"])
                    infos[index].update({key: value for key, value in info.items() if key != "reward/combined"})
                metrics[f"reward/{name}/score_time"] = perf_counter() - score_start
                columns.append(scores)
            finally:
                await executor.sleep()
            metrics[f"reward/{name}/total_time"] = perf_counter() - load_start
        scores = torch.tensor(columns, dtype=torch.float32).T.contiguous()
        if not torch.isfinite(scores).all():
            raise ValueError("All five required reward components must be finite")
        metrics["reward/total_time"] = perf_counter() - started
        return scores, names, infos, metrics

    def _scorer_for_term(self, name):
        """Reuse the CPU-resident executor and route exactly one configured reward term."""
        config = deepcopy(self.config)
        term = config.reward.reward_functions[name]
        config.reward.reward_functions = {name: OmegaConf.to_container(term, resolve=True)}
        model_name = term.model
        model = config.reward.models[model_name]
        if model.backend != "native":
            raise ValueError("Initial synchronous LTX reward runner requires native models")
        if model_name not in self._executors:
            self._executors[model_name] = NativeRewardExecutor(
                RewardModelSpec(
                    name=model_name,
                    backend="native",
                    model_path=model.model_path,
                    executor_config=OmegaConf.to_container(model.executor, resolve=True),
                    cpu_offload=model.get("cpu_offload", False),
                )
            )
        executor = self._executors[model_name]
        manager = MultiVisualRewardManager(config, tokenizer=None, compute_score=None)
        manager.set_reward_executors({}, {model_name: executor})
        return manager, executor

    async def close(self):
        """Release cached executors, including weights retained in CPU memory."""
        import asyncio

        results = await asyncio.gather(
            *(executor.close() for executor in self._executors.values()), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        self._executors.clear()
