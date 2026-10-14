# DiffusionNFT

Last updated: 10/14/2026.

DiffusionNFT ([paper](https://arxiv.org/abs/2509.16117),
[code](https://github.com/NVlabs/DiffusionNFT),
[project page](https://research.nvidia.com/labs/cosmos-lab/diffusionnft/))
is an online RL method for diffusion models that optimizes the **forward
diffusion process** instead of applying policy gradients to the reverse
sampling chain. It contrasts positive and negative generations under a reward
signal, then folds that reinforcement signal into a supervised flow-matching
objective.

This makes DiffusionNFT useful since reverse-process likelihoods are expensive
or awkward to estimate. It is solver-agnostic during rollout, only needs clean
final latents/images and rewards for actor training, and naturally supports an
off-policy split between the rollout policy and the training policy.

## Algorithm

For a prompt $c$, the old policy samples $K$ clean images
$x_0^{1:K} \sim \pi^{\text{old}}(\cdot \mid c)$. A reward model assigns raw
scores $r^{\text{raw}}(x_0, c)$, which are mapped into an optimality
probability:

$$
\begin{aligned}
r(x_0,c)
&= \frac{1}{2} + \frac{1}{2}
\mathrm{clip}\left(
\frac{
r^{\mathrm{raw}}(x_0,c)
- \mathbb{E}_{\pi^{\mathrm{old}}(\cdot \mid c)}
r^{\mathrm{raw}}(x_0,c)
}{Z_c},
-1, 1
\right).
\end{aligned}
$$

This optimality-probability transform follows the GRPO-style practice of
normalizing rewards within a prompt group before clipping them into a bounded
training signal. Here $Z_c > 0$ is the reward normalizer for prompt $c$; in
practice, it is a standard-deviation term, estimated either from the prompt's
sample group or from the global rollout batch. VeRL-Omni defaults to global
reward standard deviation normalization for DiffusionNFT
(`algorithm.global_std=True`).

DiffusionNFT then noising-samples $x_t$ from the forward process and optimizes
two implicit branches:

$$
\begin{aligned}
\mathcal{L}(\theta)
&= \mathbb{E}_{c,\,x_0 \sim \pi^{\mathrm{old}}(\cdot \mid c),\,t}
\Big[
r\left\|v_\theta^+(x_t,c,t)-v\right\|_2^2 \\
&\quad + (1-r)\left\|v_\theta^-(x_t,c,t)-v\right\|_2^2
\Big],
\end{aligned}
$$

where the implicit positive and negative velocities are

$$
v_\theta^+(x_t,c,t) =
(1-\beta)v^{\text{old}}(x_t,c,t) + \beta v_\theta(x_t,c,t),
$$

$$
v_\theta^-(x_t,c,t) =
(1+\beta)v^{\text{old}}(x_t,c,t) - \beta v_\theta(x_t,c,t).
$$

Here $\beta$ controls the reinforcement guidance strength. In VeRL-Omni this is
`actor_rollout_ref.actor.diffusion_loss.mix_beta`.

## How VeRL-Omni Implements DiffusionNFT

VeRL-Omni's DiffusionNFT path uses a direct-preference trainer loop rather than
the policy-gradient loop used by Flow-GRPO.

| Layer | What it does | Code |
|---|---|---|
| Rollout adapter | Generates images with the `old` LoRA adapter and returns clean latents plus trainable forward timesteps. | `verl_omni/pipelines/qwen_image_diffusion_nft/` |
| Actor loss | Implements the implicit positive/negative forward-process objective and optional reference prediction MSE. | `verl_omni/trainer/diffusion/diffusion_algos.py` |
| FSDP engine | Trains from clean latents by re-noising at selected forward timesteps. | `verl_omni/workers/engine/fsdp/diffusers_impl.py` |
| Trainer | Runs online rollout, reward evaluation, actor update, and old-policy adapter refresh. | `verl_omni/trainer/diffusion/ray_diffusion_trainer.py` |

The actor keeps two policy adapters:

- `default`: the trainable policy updated by actor optimization.
- `old`: the rollout policy used for data collection and the implicit branch
  definitions above.

After actor updates, the trainer refreshes the `old` adapter from `default`
using `algorithm.old_policy_decay_schedule` and
`algorithm.old_policy_update_interval`.

### Extending the NFT engine

Single-latent DiffusionNFT and joint video/audio OmniNFT use
`NFTDiffusersFSDPEngine` with `model_type=diffusion_nft_model`. The engine owns
prompt preparation, policy-adapter switching, the old/current/reference
forwards, and loss invocation. Model adapters retain their existing
`prepare_model_inputs` and `forward` methods.

Algorithm preparation lives in
`verl_omni/workers/engine/fsdp/nft_hooks.py`. The engine selects hooks using
`actor_rollout_ref.model.algorithm`: `omni_nft` selects `OmniNFTTrainingHooks`,
while existing NFT/DGPO and external adapters retain `NFTTrainingHooks` by
default. An additional specialization can extend the default hooks and add an
entry to `NFT_TRAINING_HOOKS`; it does not require another engine or a new
method on `DiffusionModelBase`.

The hooks have three data responsibilities:

- `prepare_training_inputs(micro_batch, step, scheduler)` returns
  `TrainingInputs(latents, timesteps, loss_context)`. The context belongs to
  the algorithm; the engine does not assume it contains `x0`, `xt` or `t`.
- `prepare_model_outputs(predictions, loss_context, micro_batch)` maps the
  three policy predictions and training context to the loss's output fields.
  Current-policy gradients must be preserved.
- `prepare_loss_inputs(micro_batch, step)` selects per-step reward data and
  any additional group metadata or metrics. Forward-only calls skip this hook.

Default NFT hooks retain the original single-latent noising and group fields.
OmniNFT hooks independently noise `video_latents_clean` and
`audio_latents_clean`, pack model inputs, and map joint predictions to the
video/audio loss fields. Both use the scheduler's `num_train_timesteps`;
the rollout and model adapter must use the same scale. Reward routing and
loss weighting remain in actor-batch preparation and `OmniNFTLoss`.

Algorithm hooks validate engine requirements before distributed initialization.
OmniNFT currently requires FSDP2 without Ulysses. When gradient checkpointing
is enabled, the NFT engine calls `configure_gradient_checkpointing` after
loading the model and before LoRA, compilation and FSDP wrapping. Default NFT
hooks retain the loader's checkpointing; only OmniNFT hooks install the
non-reentrant dtype wrapper. PPO/DPO engines and the public model base do
not gain a training-module configuration hook.

## Configuration

The reference Qwen-Image OCR recipe selects DiffusionNFT with:

```bash
actor_rollout_ref.model.algorithm=diffusion_nft
actor_rollout_ref.model.model_type=diffusion_nft_model
algorithm.trainer_type=direct_preference
algorithm.sample_source=online
algorithm.paired_preference=false
actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft
actor_rollout_ref.model.policy_state_adapters='["default","old"]'
actor_rollout_ref.rollout.rollout_adapter=old
actor_rollout_ref.rollout.calculate_log_probs=False
```

### Core Parameters

- `actor_rollout_ref.rollout.n`: number of images sampled per prompt. The
  example uses `16`.
- `algorithm.timestep_fraction`: fraction of rollout timesteps used for
  forward-process actor training. Use `1.0` to train on all selected rollout
  timesteps.
- `algorithm.adv_mode`: maps normalized reward advantages into
  `reward_prob`. The default recipe uses `continuous`.
- `algorithm.old_policy_decay_schedule`: old-policy update schedule. Supported
  values include `copy`, `linear_to_0_5`, and `delayed_linear_to_0_999`.
- `algorithm.old_policy_update_interval`: optimizer steps between `old`
  adapter refreshes.
- `actor_rollout_ref.actor.diffusion_loss.mix_beta`: $\beta$ in the implicit
  positive/negative velocity equations.
- `actor_rollout_ref.actor.diffusion_loss.ref_kl_coef`: coefficient for the
  prediction-space reference MSE regularizer.
- `actor_rollout_ref.actor.diffusion_loss.adv_clip_max`: clamp used when
  mapping normalized rewards into `reward_prob`.

## Reference Example

The ready-to-run OCR example post-trains `Qwen/Qwen-Image` with a visual reward
model (`Qwen/Qwen3-VL-8B-Instruct`) using `vllm_omni` rollout. It is configured
for one node with 4 GPUs, LoRA rank `64`, rollout group size `16`, 10 training
rollout steps, and 40 validation steps.

First install the OCR reward dependency after setting up the base VeRL-Omni
environment:

```bash
uv pip install -e ".[ocr]"
```

Obtain the raw OCR dataset from the original Flow-GRPO repository
([dataset/ocr](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr))
and place it under `$WORKSPACE/data/ocr`, where `WORKSPACE` defaults to
`$HOME` if unset. Then preprocess it into the parquet files consumed by the
DiffusionNFT script:

```bash
export WORKSPACE=${WORKSPACE:-$HOME}

python3 examples/flowgrpo_trainer/data_process/qwenimage_ocr.py \
  --input_dir $WORKSPACE/data/ocr \
  --output_dir $WORKSPACE/data/ocr
```

The command writes:

- `$WORKSPACE/data/ocr/train.parquet`
- `$WORKSPACE/data/ocr/test.parquet`

Launch training from the repository root:

```bash
bash examples/diffusionnft_trainer/qwen_image/run_qwen_image_ocr_lora.sh
```


## References

- K. Zheng *et al.*, *DiffusionNFT: Online Diffusion Reinforcement with
  Forward Process*, arXiv:2509.16117.
- DiffusionNFT official repository: <https://github.com/NVlabs/DiffusionNFT>.
- DiffusionNFT project page:
  <https://research.nvidia.com/labs/cosmos-lab/diffusionnft/>.

## Citation

```bibtex
@article{zheng2025diffusionnft,
  title={DiffusionNFT: Online Diffusion Reinforcement with Forward Process},
  author={Zheng, Kaiwen and Chen, Huayu and Ye, Haotian and Wang, Haoxiang and Zhang, Qinsheng and Jiang, Kai and Su, Hang and Ermon, Stefano and Zhu, Jun and Liu, Ming-Yu},
  journal={arXiv preprint arXiv:2509.16117},
  year={2025}
}
```
