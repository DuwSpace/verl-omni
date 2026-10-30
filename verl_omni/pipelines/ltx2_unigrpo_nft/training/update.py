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

"""Composite AR/AV accumulation using the engine's existing named adapters."""

import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.diffusion_algos import OmniNFTLoss

from ..replay import LTXJointReplay
from ..text.loss import clipped_recaption_loss, recaption_reference_kl
from ..text.metrics import recaption_logprob_diff


class JointUpdate:
    def __init__(self, module, model_config, use_adapter):
        self.module = module
        self.model_config = model_config
        self.use_adapter = use_adapter

    def run(self, data, loss_function, forward_only):
        if forward_only:
            raise NotImplementedError("Joint training uses native vLLM-Omni rollout")
        config = loss_function.keywords["config"]
        loss_config = config.diffusion_loss
        mode = self.model_config.pipeline.update_mode
        update_text, update_av = mode != "av_only", mode != "ar_only"
        records = tu.get(data, "ltx_joint_replay")
        records = [record if isinstance(record, LTXJointReplay) else LTXJointReplay(**record) for record in records]
        for record in records:
            record.validate()
        if not records or len({record.policy_version for record in records}) != 1:
            raise ValueError("Joint update requires one nonempty rollout policy batch")
        micro_size = config.ppo_micro_batch_size_per_gpu
        if type(micro_size) is not int or micro_size <= 0:
            raise ValueError("Joint micro-batch size must be a positive integer")
        total, metrics = 0.0, {}
        times = data["train_timesteps"]

        def collect(values, scale):
            for name, value in values.items():
                metrics[name] = metrics.get(name, 0.0) + float(value) * scale

        for index in range(0, len(records), micro_size):
            group = records[index : index + micro_size]
            selection = slice(index, index + len(group))
            scale = len(group) / len(records)

            def av(step, group=group, selection=selection):
                return self._av_batch(
                    group,
                    data["video_latents_clean"][selection],
                    data["audio_latents_clean"][selection],
                    data["video_forward_noise"][selection, step],
                    data["audio_forward_noise"][selection, step],
                    times[selection, step],
                )

            with torch.no_grad(), self.use_adapter("old"):
                old_text = self._ar_batch(group).detach() if update_text else None
                old_av = [tuple(p.cpu() for p in av(step)[0]) for step in range(times.shape[1])] if update_av else []
            with torch.no_grad(), self.use_adapter("reference"):
                ref_text = self._ar_batch(group).detach() if update_text and loss_config.text_kl_coef else None
                ref_av = [tuple(p.cpu() for p in av(step)[0]) for step in range(times.shape[1])] if update_av else []
            if update_text:
                inputs = self._ar_batch_inputs(group)
                current = self.module(task="recaption", **inputs)
                mask = inputs["response_mask"][:, 1:]
                replay_metrics = recaption_logprob_diff(old_text, [record.rollout_logprobs for record in group], mask)
                max_key = "text/logprob_diff_abs_max"
                metrics[max_key] = max(metrics.get(max_key, 0.0), replay_metrics.pop(max_key))
                collect(replay_metrics, scale)
                loss, values = clipped_recaption_loss(
                    current,
                    old_text,
                    data["text_advantages"][selection].to(current.device),
                    mask,
                    clip_ratio=loss_config.ar_clip_ratio,
                )
                if ref_text is not None:
                    loss = loss + loss_config.text_kl_coef * recaption_reference_kl(current, ref_text, mask)
                loss = loss * loss_config.text_loss_weight
                (loss * scale).backward()
                total += loss.detach().item() * scale
                collect(values, scale)
            if update_av:
                for step in range(times.shape[1]):
                    predictions, clean, noisy, sigma = av(step)
                    output = {}
                    for column, modality in enumerate(("video", "audio")):
                        device = predictions[column].device
                        output.update(
                            {
                                f"{modality}_forward_prediction": predictions[column],
                                f"{modality}_old_prediction": old_av[step][column].to(device),
                                f"{modality}_ref_forward_prediction": ref_av[step][column].to(device),
                                f"{modality}_x0": clean[column],
                                f"{modality}_xt": noisy[column],
                                f"{modality}_t_expanded": sigma,
                            }
                        )
                    result = OmniNFTLoss()(
                        config=config,
                        model_output=output,
                        data=TensorDict(
                            {"reward_prob": data["reward_prob"][selection, step].to(predictions[0].device)},
                            batch_size=[len(group)],
                        ),
                    )
                    loss = result.loss * loss_config.av_loss_weight
                    (loss * scale / times.shape[1]).backward()
                    total += loss.detach().item() * scale / times.shape[1]
                    collect(result.metrics, scale / times.shape[1])
        # None prevents optimizer momentum/weight decay from moving an ablated branch.
        for name, parameter in self.module.named_parameters():
            if (name.startswith("gemma.") and not update_text) or (name.startswith("transformer.") and not update_av):
                parameter.grad = None
        metrics.update(
            {"joint/total_loss": total, "joint/update_text": float(update_text), "joint/update_av": float(update_av)}
        )
        return {"loss": [total], "metrics": {key: [value] for key, value in metrics.items()}, "model_output": {}}

    def _device(self):
        # Use the accelerator even when FSDP keeps local parameter shards on CPU.
        from verl.utils.device import get_device_id, get_device_name

        name = get_device_name()
        return torch.device(name, get_device_id()) if name != "cpu" else torch.device("cpu")

    def _av_inputs(self, record, video_clean, audio_clean, video_noise, audio_noise, timestep):
        device = self._device()
        t = timestep.to(device=device, dtype=torch.float32).reshape(1)
        sigma = t.reshape(1, 1, 1) / 1000.0
        clean = (video_clean.to(device).unsqueeze(0), audio_clean.to(device).unsqueeze(0))
        noise = (video_noise.to(device).unsqueeze(0), audio_noise.to(device).unsqueeze(0))
        noisy = tuple((1 - sigma) * x.float() + sigma * n.float() for x, n in zip(clean, noise, strict=True))
        params = record.generation_params
        dtype = next(self.module.transformer.parameters()).dtype
        inputs = dict(
            input_ids=record.condition_token_ids.to(device).unsqueeze(0),
            attention_mask=record.condition_attention_mask.to(device).unsqueeze(0),
            hidden_states=noisy[0].to(dtype),
            audio_hidden_states=noisy[1].to(dtype),
            timestep=t,
            sigma=t,
            audio_timestep=t,
            audio_sigma=t,
            num_frames=(params["num_frames"] - 1) // 8 + 1,
            height=params["height"] // 32,
            width=params["width"] // 32,
            fps=params["frame_rate"],
            audio_num_frames=clean[1].shape[1],
        )
        return inputs, clean, noisy, sigma

    def _ar_batch_inputs(self, records):
        from torch.nn.utils.rnn import pad_sequence

        inputs = [record.ar_inputs(self._device()) for record in records]
        temperatures = {item["temperature"] for item in inputs}
        if len(temperatures) != 1:
            raise ValueError("Joint text micro-batch requires a shared sampling temperature")
        # Right padding preserves each replay's original causal positions.
        return {
            **{
                key: pad_sequence([item[key][0] for item in inputs], batch_first=True, padding_value=0)
                for key in ("input_ids", "attention_mask", "response_mask")
            },
            "temperature": inputs[0]["temperature"],
        }

    def _ar_batch(self, records):
        return self.module(task="recaption", **self._ar_batch_inputs(records))

    def _av_batch(self, records, video_clean, audio_clean, video_noise, audio_noise, timesteps):
        if any(record.generation_params != records[0].generation_params for record in records):
            raise ValueError("Joint AV micro-batch requires uniform generation geometry")
        # Rollout encodes conditions with padding=max_length. Do not introduce
        # additional padding here: that would change the replay's positions.
        if len({record.condition_token_ids.numel() for record in records}) != 1:
            raise ValueError("Joint AV micro-batch requires equal padded condition lengths")
        items = [
            self._av_inputs(record, video_clean[i], audio_clean[i], video_noise[i], audio_noise[i], timesteps[i])
            for i, record in enumerate(records)
        ]
        inputs = {
            key: torch.cat([item[0][key] for item in items], dim=0) if torch.is_tensor(value) else value
            for key, value in items[0][0].items()
        }
        clean = tuple(torch.cat([item[1][column] for item in items]) for column in (0, 1))
        noisy = tuple(torch.cat([item[2][column] for item in items]) for column in (0, 1))
        sigma = torch.cat([item[3] for item in items])
        predictions = self.module(task="av", **inputs)
        return tuple(value.float() for value in predictions), clean, noisy, sigma
