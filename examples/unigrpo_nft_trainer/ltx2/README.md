# LTX-2.3 recaption GRPO + OmniNFT

For the architecture, reward equations, gradient paths, policy lifecycle and launch configuration, see the [中文实现文档](IMPLEMENTATION.md).

This recipe trains a paired Gemma/DiT LoRA policy using synchronous vLLM-Omni
rollout. Gemma enables its trainable LoRA for planning and recaption, then disables
that LoRA for AV condition encoding. GRPO updates Gemma LoRA; OmniNFT updates
only the audio/video DiT LoRA. Base weights, VAEs and connectors are frozen.
Both stages reuse the same Gemma base weights; encoding runs without gradients.

Historical results below predate frozen base encoding and do not validate the new gradient contract.

## Direct audiovisual prompt rewriting

Gemma now returns only the final audiovisual generation prompt, without tags,
headings or a separate planning section. The instruction retains the user's
requirements for intent, temporal/spatial coherence, audio and original-language
dialogue. It sets no word count. The entire response conditions AV generation
and receives GRPO credit; reward targets remain the dataset's original video/audio
prompts. Gemma LoRA is active during rewriting and disabled during encoding.

`actor_rollout_ref.rollout.ltx_joint.max_new_tokens=4096` is the response ceiling;
normal generation ends at EOS. Temperature remains 1.0. A token-budget cutoff
raises a diagnostic error instead of silently using incomplete text. The AV
encoder separately retains `max_sequence_length=1024`; an overlong final prompt
is rejected rather than silently truncated. These are different limits.

New replay uses `response_format="caption"`, `planning=null`, and retains the
complete response text, token IDs and logprobs. Historical tagged replay remains
readable. Live generation no longer invokes the two-section parser. Failure
messages retain seed, policy version, token count, EOS status and raw response.

## Historical verification before planning

**Complete under the earlier user-adjusted verification scope.** Five real update rounds, checkpoint restoration,
subsequent rollout and fixed evaluation have run successfully on eight Ascend
910B devices. Numerical alignment corrections passed the real FSDP paired forward and joint
backward in experiment `011`, which exited successfully. Native decoder freezing also passed an actual before/after audit. The user accepted the existing training duration and requested no further
training. Run 012 was stopped before another update. The user replaced human review with GPT visual/recaption spot review and waived
sound checks. Audio semantics and AV synchronization remain unverified; no
quality improvement is claimed. No additional training is scheduled. Earlier runs do not establish acceptance
of code changed afterward.

## Run the joint trainer

The default dataset uses the complete pinned
[OmniNFT VGGSound recipe](https://github.com/DuwSpace/verl-omni/tree/98dc6a06ccef1b701b2a917bf4f02a3f4d6c373b/examples/omninft_trainer):
19,487 training records and 16 validation records. Both prepared Parquet splits
are included under `data/omninft/vggsound/ltx_joint_full`; cloning this branch
is sufficient. To regenerate them from the pinned source:

```bash
python examples/unigrpo_nft_trainer/ltx2/prepare_omninft_data.py
```

The converter pins OmniNFT data revision
`fb9237f6e74edf0d0f2a683f4d975b79fde588fe`. Its default `--train_size=-1`
and `--val_size=-1` preserve every record. Use explicit positive limits and a
separate `--output_dir` for smoke datasets. Default outputs are
`data/omninft/vggsound/ltx_joint_full/train.parquet` and `test.parquet`.
The launcher uses them unless `LTX_TRAIN_FILE` / `LTX_VAL_FILE` override them.
The existing `ltx_joint_1x1` smoke files are retained.

The converter preserves `prompt_av` as the joint generation prompt and
`prompt_v` / `prompt_a` as the original video/audio reward targets, along with
source metadata and stable group IDs. The full parquet contents have been
checked against every source record; this preparation does not run training.

Defaults are `train_batch_size=32`, `rollout.n=8` (256 generated samples per
training step), actor mini-batch size 256, and four samples per micro-batch
per rank. On eight ranks, eight micro-batches accumulate into one optimizer update. Checkpoint saving and validation run every 2 steps. Total training is
capped at 100 steps, so this run uses the full data source without completing
an epoch. Validation uses all 16 records with one candidate per prompt.
The launcher does not override rollout group or actor mini-batch sizes with
the device count. `run_train.sh` is the single training entry point and uses the current recipe defaults.

The launcher leaves `LTX_ACTOR_NPU_MEMORY_FRACTION` unset by default; an
explicit value is forwarded through the Ray runtime environment.

The tested environment is eight Ascend 910B devices with 64 GB each, CPU offload,
local model/reward assets and substantial host RAM. Runtime versions are recorded
in `/tmp/ltx-joint-artifacts/training-007-runtime-environment.json`: PyTorch
2.10.0, torch-npu 2.10.0.post4, Transformers 5.14.1, Diffusers 0.40.0, PEFT 0.20.0,
vLLM 0.28.0 and vLLM-Omni 0.28.0rc2.dev68+gded893462.npu. The installed native
source has an unavailable external Git directory; package version is not proof
of a clean checkout. Other versions and GPU execution are unverified.

Initialize CANN/ATB for your installation and set the model, reward and output
paths before launching. See the [NPU installation guide](../../../docs/start/install_npu.md).
The script defaults to eight devices, CPU offload and W&B offline, and writes
`train.log` under `LTX_OUTPUT_DIR`. Additional Hydra overrides can be appended.

```bash
# Source your installation's CANN/ATB environment scripts first.
export LTX_MODEL_PATH=/path/to/LTX-2.3-Diffusers
export LTX_REWARD_ROOT=/path/to/omnift-rewards
export LTX_TRAIN_FILE="$PWD/data/omninft/vggsound/ltx_joint_full/train.parquet"
export LTX_VAL_FILE="$PWD/data/omninft/vggsound/ltx_joint_full/test.parquet"
export LTX_OUTPUT_DIR=/path/to/new-run
export LTX_NUM_DEVICES=8
bash examples/unigrpo_nft_trainer/ltx2/run_train.sh
```

Set `LTX_DISABLE_PROGRESS_BAR=1` before launching to hide the LTX denoising
progress bar. The recipe forwards it to Ray workers; `0` (default) keeps the
backend's normal display behavior. Model loading bars and training logs are unaffected.

To reproduce the historical four-prompt functional experiment instead:

```bash
python examples/unigrpo_nft_trainer/ltx2/prepare_smoke_data.py --output /path/to/new-smoke-data
export LTX_TRAIN_FILE=/path/to/new-smoke-data/train.parquet
export LTX_VAL_FILE=/path/to/new-smoke-data/val.parquet
```

The prompts are versioned in `fixed_prompts.json`. Train and validation contain
the same four prompts; these fixtures test integration and fixed-seed changes,
not held-out generalization. The reproduced parquet records were compared with
the actual experiment inputs and match exactly. Use a separate representative
validation set for broader quality conclusions.

Both parquet files follow the prompt-chat dataset contract, including
`data_source` and `reward_model.ground_truth`. The reward asset paths are explicit
in `joint_train.yaml`. Five scorers occupy distinct devices 0–4; the launcher
requires at least five devices, but real distributed verification used eight.
Extra launcher arguments are Hydra overrides; `--cfg job --resolve` checks the
resolved configuration without training.

The current geometry is 256 × 384, 121 frames at 24 fps, with 20 denoising
steps for training rollout and 40 for evaluation. A training step contains
32 prompts with eight samples each. Mini-batch size counts complete recaption/AV
samples, so 256 samples with mini-batch size 256 produce one joint optimizer
step per training step when `ppo_epochs=1`. Micro-batch size is four per rank. FP32 LoRA masters use learning
rates 1e-6 for Gemma and 1e-5 for DiT. Gemma uses original FP32 base weights;
DiT/connectors use BF16. The recipe is a short functional experiment, not a
production-quality generation setting.

To restore a paired checkpoint into a new output directory:

```bash
bash examples/unigrpo_nft_trainer/ltx2/run_train.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/path/to/previous/checkpoints/global_step_4 \
  trainer.total_training_steps=5
```

`ref` remains the training-start snapshot on resume. All mini-batches retain
the same paired EMA `old`; it is refreshed only after the full actor update.
Each export contains both adapters, a version manifest and checksums. Replicas
must acknowledge the complete pair before generation. Training rollout versions
are `2 * step - 1`; evaluation versions are `2 * step`, with zero for initial
evaluation. Immutable old/ref hashes are checked during updates and serialization.

## Start on another eight-NPU server

The host launcher uses the existing image
`crpi-gwfvpzl8eqw3jv8c.cn-shanghai.personal.cr.aliyuncs.com/verl_omni/npu:npu-a2-main-8d2a9fd`.
It mounts this checkout at `/repo`, the complete model cache at `/hub`, parquet
files at `/ltx-data`, and the chosen output directory at `/outputs`. Mounting the
whole cache preserves Hugging Face snapshot symlinks. The host provides the NPU
driver; CANN and the engine stack come from the image.

```bash
cp examples/unigrpo_nft_trainer/ltx2/server.env.example /tmp/ltx-server.env
# Edit /tmp/ltx-server.env: host model cache, parquet and output directories.
bash examples/unigrpo_nft_trainer/ltx2/start_docker.sh /tmp/ltx-server.env
bash examples/unigrpo_nft_trainer/ltx2/start_train.sh ltx23-unigrpo --name first-run
```

`start_docker.sh` installs the recipe's missing dependencies inside the new
container with `setup_npu_env.sh`. It pins the tested Transformers, Diffusers,
PEFT and reward packages, and constrains pip to retain the image's Torch/NPU,
vLLM and verl versions. It does not rebuild or replace the engine stack. For
an offline server, prepare a wheel mirror/cache before running the installer;
a local Docker image alone does not contain the missing Python packages.
Set `PIP_INDEX_URL` in the server file to select a reachable package mirror.
An existing container is never deleted or replaced automatically. To retry an
interrupted install, run `docker exec CONTAINER bash
/repo/examples/unigrpo_nft_trainer/ltx2/setup_npu_env.sh`.

Training runs in the background by default, from scratch, with Text logprob
checks enabled. HPS baseline checks and reward diagnostic captures have been
removed; all five reward scorers still run normally. No HPS reference file is
needed. Use `--no-text-check` to disable Text checks, or `--foreground` to wait
for completion and stream logs. Extra arguments after `--` are Hydra overrides.

```bash
# Two full training steps, eight samples per step, no validation/checkpoint save.
bash examples/unigrpo_nft_trainer/ltx2/start_train.sh ltx23-unigrpo --name smoke --smoke
# Inspect startup, training and the final exit code (0 means success).
docker exec ltx23-unigrpo tail -f /outputs/smoke/launcher.log
docker exec ltx23-unigrpo cat /outputs/smoke/exit_code
# Configuration-only check, without loading models or running training.
bash examples/unigrpo_nft_trainer/ltx2/start_train.sh ltx23-unigrpo --name config-check --foreground -- --cfg job --resolve
```

The container uses the checkout from which `start_docker.sh` was invoked. Use
separate checkouts/container names for the CPU-master and pretouch branches;
keep the mounted code unchanged while a training process is running. Run names
must be new, and a container permits one managed training run at a time. The
scripts do not install host drivers or change the branches' offload behavior.

## AR / OmniNFT update ablations

All three variants inherit `joint_train.yaml`, including its data, seeds,
learning rates, sample count, five rewards, validation geometry and EMA decay.
Only the two loss coefficients and the experiment label differ:

| Hydra selection | AR / Gemma LoRA | OmniNFT / DiT LoRA |
| --- | --- | --- |
| `ablation=av_only` | Fixed | Updated |
| `ablation=joint` (default) | Updated | Updated |
| `ablation=ar_only` | Updated | Fixed |

The rewriting instruction is independently selectable with `recaption_prompt`:

| Hydra selection | Instruction / chat format |
| --- | --- |
| `recaption_prompt=planner` (default) | Existing audiovisual planner, single user message |
| `recaption_prompt=ltx_t2v` | Upstream LTX-2.3 T2V system prompt, separate system and user messages |

For example, select the upstream prompt with AR-only training:

```bash
bash examples/unigrpo_nft_trainer/ltx2/start_train.sh CONTAINER --ablation ar_only -- recaption_prompt=ltx_t2v
```

Direct `run_train.sh` launches accept the same Hydra override. For the standalone
`smoke_rollout.py`, use `--recaption-prompt ltx_t2v`. Both training and validation
use the selected instruction; reward text targets remain the original dataset
prompts. Actor replay uses captured prompt token IDs, including the selected chat
format. Existing generation limits and sampling settings are unchanged.

The upstream text is bundled unmodified from
[bunter96/LTX-2.3, commit f735012](https://github.com/bunter96/LTX-2.3/blob/f73501238862129bcd5ef2fd0fb8ae2b9461010e/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_t2v_system_prompt.txt),
with its source notice and LTX license under
`verl_omni/pipelines/ltx2_unigrpo_nft/prompts/`; runtime network access is unnecessary.
Unlike the planner instruction, this preset asks for concrete quoted dialogue when
speech is mentioned without supplied words. Compare it as a separate experiment.

AR still generates recaptions in every variant, and DiT still generates audio
and video. A fixed AR policy can produce different text because sampling remains
stochastic. This ablation disables parameter updates, not recaption generation.

A zero `text_loss_weight` or `av_loss_weight` skips that branch's differentiable
forward and backward. Disabled gradients are set to `None` before the shared
optimizer step, preventing AdamW momentum and weight decay from moving the
parameters. Its rollout `old` EMA is also held fixed. Both adapters remain in
the model, checkpoint and export pair; `requires_grad` is retained because
policy snapshots and adapter export use it to discover the complete pair.
Text old-policy replay and diagnostics remain available in every variant.

Run all three from the same initial model, sequentially on one container:

```bash
for mode in av_only joint ar_only; do
  bash examples/unigrpo_nft_trainer/ltx2/start_train.sh ltx23-unigrpo \
    --ablation "$mode" --foreground \
    -- trainer.val_before_train=true || break
done
```

Run names include the variant and timestamp. Add `--smoke` before `--` for a
two-step wiring test; smoke mode disables periodic validation/checkpoint saving.
For direct container launches, pass `ablation=ar_only` (or another variant) to
`run_train.sh` / `run_text_diagnostics.sh`. Explicit Hydra overrides still take
precedence over the selected preset. Use the same variant when resuming an
experiment; switching a trained checkpoint to another variant is a different
experiment from the from-scratch comparison above.

Compare fixed-prompt validation reward curves at equal training steps, including
all five components and their combined score. Keep train and validation curves
separate. Record the initial validation score, Text logprob differences and
`joint/update_text` / `joint/update_av`; disabled branch gradient norms should
be zero. Repeat the comparison with matching seeds across variants if measuring
quality differences; a two-step smoke run only verifies execution.

## Offline logging

The recipe enables `trainer.logger=[console,wandb]`. The launcher sets
`WANDB_MODE=offline`, and the Ray runtime environment also explicitly sets offline
mode so the trainer process records locally. `WANDB_DIR` defaults to
`LTX_OUTPUT_DIR`; runs are written under `$LTX_OUTPUT_DIR/wandb/offline-run-*`.
Set `WANDB_DIR` to another writable directory before launching to relocate them.
No W&B login is required and this recipe does not upload runs. Console metrics
remain enabled. A local smoke check using the actual `Tracking` class confirmed
that an offline `.wandb` run file is created.

## Backend and numerical contract

`algorithm.trainer_type=direct_preference` and
`actor_rollout_ref.model.algorithm=ltx2_unigrpo_nft` select the joint trainer.
Each rollout replica uses one vLLM-Omni diffusion worker. Recaption invokes
Transformers `generate` on that worker's existing Gemma; it is not a separate
paged-KV autoregressive engine. The same Gemma then supplies native AV conditions.
Training rollout and validation now use separate native guidance settings from
[the reference recipe](https://github.com/DuwSpace/verl-omni/blob/split/omninft-ltx2-recipe/examples/omninft_trainer/ltx2/ltx2_omninft.yaml):

| Setting | Training rollout | Evaluation |
| --- | --- | --- |
| Height × width / frames / fps | 256 × 384 / 121 / 24 | 256 × 384 / 121 / 24 |
| Denoising steps | 20 | 40 |
| Video / audio CFG | 1.5 / 3.0 | 3.0 / 7.0 |
| Video / audio modality CFG | 1.0 / 1.0 (disabled) | 3.0 / 3.0 |
| Video / audio rescale | 0.0 / 0.0 (disabled) | 0.7 / 0.7 |

Override training fields under `actor_rollout_ref.rollout.pipeline` and evaluation
fields under `actor_rollout_ref.rollout.val_kwargs.pipeline`. Settings travel with
each batch to the cached backend; evaluation does not mutate training defaults.
Replay saves the selected geometry and guidance. To disable modality guidance
or rescale in evaluation too, set both corresponding evaluation scales to 1.0
or 0.0 respectively. Gemma allows up to 4096 generated tokens and temperature 1.0.

TP remains 1 with eight independent replicas and CPU offload; VAE tiling is
now enabled through `rollout.ltx_joint.vae_use_tiling`. VAE patch parallelism
remains 1. The reference recipe's TP, full-batch sizes, LoRA/optimizer settings,
old-adapter rollout and training schedule are not adopted. Evaluation uses independent `rollout.val_kwargs.n=1` and fixed seeds.
The FSDP collective dispatch pads requests to the actor world size, then removes
padded outputs before rewards, metrics and saving. With one prompt on eight ranks,
this currently runs eight copies with the same seed but retains only one result;
it does not yet reduce physical generation work to one active rank. With a single
candidate, group-relative probabilities are neutral (0.5); compare the five raw
reward components rather than that normalized probability. The native LTX sampler
uses its deterministic denoising path, without injecting FlowGRPO SDE noise.
The FSDP OmniNFT forward remains conditional; sampling guidance changes the
generated samples, not the actor's noise-prediction objective.

With `rollout.ltx_joint.cpu_offload=true`, the native backend delegates
residency to this repository's `JointRolloutOffload`. Gemma, connectors, DiT,
video VAE, audio VAE and vocoder run with mutually exclusive model residency.
The controller also covers `generate()` / `decode()` entry points and native
DiT LoRA tensor sidecars. Each request releases all managed components in
`finally`; consecutive DiT steps keep their weights resident. Transfers preserve
dtypes and use blocking CPU copies without pinning the entire Gemma model.
Both positive and negative CFG embeddings use the connector dtype (BF16), while
Gemma weights and encoding computation remain FP32. This addresses the observed
negative-connector OOM caused by overlap with resident Gemma and FP32 conditions.
The installed backend must support the pipeline-owned
`enable_omni_model_cpu_offload` protocol; current container initialization logs
confirm that delegation. Framework/communication allocations and live outputs
are outside this controller and can still occupy device memory after release.

All five native reward models enable `offload: true` and `cpu_offload: true`
in `rewards.yaml`. The first scoring batch loads checkpoints; afterward sleep
moves weights to CPU memory and wake moves the same model back to its assigned
device. Processors and mixed parameter dtypes are preserved, including HPSv3's
FP32 reward head and DeSync's mel transform. Sleeping executors reject inference
and wait for active requests before moving weights. Set a model's
`reward.models.<name>.cpu_offload=false` to restore destruction/reloading on
sleep; `offload=false` instead keeps it resident on the accelerator. Cached
weights consume host RAM until the executor is closed or its worker exits.
Restart training to pick up this code/config change in existing Ray workers.

The explicit offload change passed real-weight single-worker probes at
256 × 384 / 121 frames: two consecutive requests with training CFG and two
denoising steps, plus one request with the full 40-step evaluation guidance.
All three produced MP4/WAV and replay tensors. Peak PyTorch allocated memory
was 48.75 GiB (reserved 50.63–50.65 GiB); after explicit release, each worker
reported 1 KiB allocated and 2 MiB reserved. These probes use base-equivalent
zero-B adapters, not a complete training policy, and do not establish eight-rank
training memory safety or output quality. Results are under
`outputs/oom-offload-probe-train-v2/summary.json` and
`outputs/oom-offload-probe-val/summary.json`. The offload/model/planning and
trainer/GRPO CPU suites passed 90 tests; Ruff check/format and diff checks passed.
Earlier 256 × 256 / 33-frame results are historical.
Validation for this change: 84 joint CPU regression tests passed, followed by
two focused hook tests covering TensorDict sampling overrides and the fallback
path. Resolved Hydra configs instantiate the intended pipeline dataclasses;
actor defaults match training rollout, and evaluation uses its own guidance.

Replay records retain prompt context, sampled token IDs/logprobs, masks,
temperature, seeds, policy/group identifiers, clean AV latents, training noise
and times. Text rewards refer to the original user prompt. Current conditioning
is recomputed with gradients. Each policy state computes its own conditioning
using identical recaption tokens, noise and times. Both branch backwards and
micro-batch accumulation finish before one shared optimizer step.

The actor attention processors reproduce the installed native arithmetic:
BF16 connector/DiT rotary intermediates, packed video self-attention QKV and
native LoRA operand/scaling precision. FP32 LoRA master parameters retain their
autograd graph. These processors support the nonperturbed joint conditional
path; DoRA, merged adapters and LoRA dropout are rejected. Actual compatibility
must be checked again when changing the native backend.

## Verified reward source

Source: [DuwSpace/verl-omni, split/omninft-ltx2](https://github.com/DuwSpace/verl-omni/tree/66595248b0407749eb0a7cccc5107f8642519d2a),
commit `66595248b0407749eb0a7cccc5107f8642519d2a` (queried and fetched on 2026-10-12).

`rewards.yaml` preserves the complete `reward` block from
`examples/omninft_trainer/ltx2/ltx2_omninft.yaml` at that commit. It requires the
launcher to supply `actor_rollout_ref.rollout.pipeline.frame_rate`.
The five scorer implementations and shared media helpers have been ported
unchanged from this commit. The existing reward manager now forwards the complete
sample to scorers that request it, reserves routing weights for aggregation, and
supports component-preserving outputs for OmniNFT and the joint loss.

| Scorer | Video routing | Audio routing | Source scoring configuration |
| --- | ---: | ---: | --- |
| VideoAlign | 1.0 | 0.0 | Weights `[0.5, 0, 0.5]`, means `[3.6757, 0, 2.8105]`, stds `[2.2476, 1, 2.5121]` |
| HPSv3 | 1.5 | 0.0 | 5 frames, top fraction 0.3, cap 15 |
| AudioBox | 0.0 | 0.5 | CE + CU - PC + PQ, scale 0.025 |
| CLAP | 0.0 | 1.0 | Scale 0.5, offset 0.5, clamp to `[0, 1]` |
| DeSync | 1.0 | 1.0 | Source implementation and frame rate |

All five scorer weights are 1.0 and all five are required. The manager preserves
components. The source algorithm centers each component within its original
prompt group, divides by that component's population standard deviation across
the batch plus `1e-4`, then multiplies by the unnormalized routing matrix.
Continuous probability mapping is `clamp(advantage / (2 * adv_clip_max) + 0.5, 0, 1)`.

**The source produces two probabilities, ordered video then audio**, with actor
shape `[batch, timesteps, 2]`. It does not define a single joint probability.
The proposed extension keeps those two OmniNFT probabilities unchanged and uses
their arithmetic mean as the text reward, followed by within-group advantage
normalization. This interpretation was raised with the user and is provisional;
the joint actor-batch preparation now implements this explicitly as
`text_probability_reduction=mean`. The standalone text helper accepts an already
selected scalar probability and cannot change scorer aggregation.

The `OmniNFTLoss` class was ported from the source commit into
`verl_omni/trainer/diffusion/diffusion_algos.py`. `OmniNFTLossConfig` lives in the
existing worker config module and additionally rejects nonfinite coefficients.
The source modality weights are both 1.0 and reference MSE coefficients are both
`1e-4`; the launch configuration retains these values.

## Artifacts, evaluation and timing

Each saved sample has a silent MP4, stereo WAV and an `-av.mp4` with audio,
plus `samples.json` and an HTML gallery. Sample `.pt` files are disabled by
default (`trainer.joint_sample_save_tensors=false`); enable this option only
when full tensor replay is needed. JSON retains prompts, seeds, generation
parameters, response token IDs/logprobs and reward scores. Training samples are under
`$LTX_OUTPUT_DIR/samples/step-NNNNNN`; fixed evaluation is under
`samples/evaluation/step-NNNNNN`. Checkpoints contain all actor/optimizer shards
and paired old/ref state. Raw update metrics are saved atomically under
`checkpoints/update_metrics/step-NNNNNN.pt` before evaluation.

The complete step clock runs from the earliest policy synchronization start
through the end of all parameter updates. Parallel task durations must not be
summed to reconstruct it. Evaluation is reported separately. Initial loading,
diagnostic capture and ordinary updates must also be distinguished. Scorer
latencies include executor queueing but exclude model wake-up; allocator memory
peaks are process measurements, not aggregate device residency.

Run `010` step 4, with dumps disabled for that step, took 948.417 s total,
118.876 s for training and 420.051 s for rewards, or 0.01687 samples/s. Its fixed
evaluation took another 948.529 s. This is one warmed-process observation using
the earlier arithmetic, not a stable performance distribution for final code.
HPSv3 wake-up is a large cost because the source scorer initializes its backbone
before loading its checkpoint.

## Consistency diagnostics and historical evidence

Current text logprob diagnostics are configured through
`actor_rollout_ref.rollout.ltx_joint.text_diagnostics`. HPS baseline health checks
and reward diagnostic captures have been removed.
See [IMPLEMENTATION.md](IMPLEMENTATION.md#21-权重搬运与保留的检查) for the reading guide,
CPU master lifecycle, and R/R′/T comparisons.

The temporary AV/MSProbe collector, its RPC and `run_consistency_npu.sh` have
been removed. The following results describe earlier experiments; their saved
evidence is retained, but the old collection command is no longer available.

Run `010` used the same old policy and 235 valid recaption tokens: mean absolute
logprob error 4.21e-6, p99 4.28e-5, maximum ratio deviation from one 4.77e-5,
and no token crossed the 0.01 clipping interval. This establishes a baseline for
that sample/backend, not a universal threshold. Its initial AV comparison exposed
connector rotation and DiT arithmetic differences. Corrected full connector and
DiT forwards subsequently matched captured native outputs bitwise on NPU,
including FP32 LoRA masters. Run `011` then verified the integrated FSDP path:
Gemma embeddings, both connector contexts and final video/audio predictions all
matched bitwise for policy 9, seed 186, timestep 909. Its 165-token old text
replay had maximum ratio deviation 6.69e-5 and zero clipping violations. The four
automatic RoPE suspects were rejected after raw-tensor inspection: no parameters
exist in those modules, and their FP32/BF16 outputs match after native casting.

## Requirements and evidence

See [VALIDATION.md](VALIDATION.md) for the requirement-to-code/test/runtime mapping
and the user-adjusted quality-review scope.

All `/tmp` paths below are local experiment artifacts, not files shipped in the
repository. The implementation and chronological commands are in this workspace;
retain the artifacts separately before cleaning the environment.

| Requirement | Current evidence | Remaining acceptance work |
| --- | --- | --- |
| Five exact source rewards and probability routing | Pinned recipe, source-equivalence tests, real scored rollouts | Scalar text reduction remains a disclosed extension |
| vLLM-Omni recaption, shared conditioning and AV generation | Runs 007, 009, 010; audited tokens, media, versions and seeds | Run 011 also passed |
| GRPO group advantage, clipping and masks | CPU tests; actual text gradients and replay metrics | 255-test regression suite passed |
| Both loss gradients reach Gemma; AV reaches DiT | Real branch gradient metrics; all eight checkpoint shards change both LoRAs | Run 011 both gradient paths verified |
| Frozen base/connectors and connector gradient path | Runs 007, 010 step 4 and 011 step 5 all-rank checkpoint comparisons exact; actual Gemma AV gradients | Native before/after parameter hashes also passed |
| Paired default/old/ref, accumulation and resume | Five rounds, ten optimizer steps; all-rank state audits preserve original ref and previous current as old | Run 011 immutable hashes pass for every micro-batch |
| Updated policy used in later rollout | Run 010 policy 7 after step 3, evaluation policy 8 after step 4 | Further rounds waived by user; run 012 stopped |
| Current conditioning recomputed; shared noise/time/tokens | Hooks and tests; paired real forward metadata | 255-test regression suite passed |
| Same-old text, embeddings and predictions | Run 010 diagnosis and run 011 corrected actual paired dumps | Run 011 integrated paired capture and report passed |
| Step/stage/scorer/branch timing, gradients and memory | Run 010 metrics and persisted sample audits | Run 011 raw metrics verified; ordinary run-010 timings have stated limits |
| Fixed evaluation and quality review | Initial and step-4 galleries, 32 matched pairs across four prompts | Human review; no quality acceptance claimed |

Principal local evidence:

- `/tmp/ltx-joint-training-010.log` (successful exit; steps 3 and 4).
- `/tmp/ltx-joint-artifacts/training-010-step3-state-audit.json` and
  `training-010-step4-state-audit.json` (all eight ranks).
- `/tmp/ltx-joint-artifacts/training-011-frozen-weights-audit.json` and
  `/tmp/ltx-joint-artifacts/training-010-step4-frozen-weights-audit.json`
  (all frozen Gemma/DiT/connectors parameters, all eight shards, exact).
- `/tmp/ltx-joint-artifacts/training-011-resume-export-audit.json`
  (policy-9 Gemma/DiT LoRA exactly matches the step-4 checkpoint on all ranks).
- `/tmp/ltx-joint-artifacts/training-010-analysis/output_5_root_cause_report.md`
  and `dit_matched_arithmetic_npu.json` (actual paired diagnosis and correction).
- `/tmp/ltx-joint-artifacts/training-010-samples/` and
  `/tmp/ltx-joint-artifacts/baseline-001-samples/` (audited replay and AV).
- `/tmp/ltx-joint-artifacts/quality-comparison-initial-step4/index.html`
  and adjacent `human_review.csv` (unused after the user requested GPT review).
- `/tmp/ltx-joint-training-011.log`, `/tmp/ltx-joint-probe-011` and
  `/tmp/ltx-joint-training-011/checkpoints` (successful corrected joint update).
- `/tmp/ltx-joint-artifacts/training-011-analysis/output_5_root_cause_report.md`,
  `training-011-boundary-comparison.json`, `training-011-old-text-comparison.json`,
  `training-011-state-audit.json` and `training-011-metrics.json` (corrected run evidence).

Across the 32 initial/step-4 pairs, raw mean score deltas were AudioBox -0.000574,
CLAP +0.003540, DeSync 0, HPSv3 -1.04206 and VideoAlign -0.132158. These do not
establish unchanged quality. Group-normalized probabilities are unsuitable for
claiming cross-checkpoint improvement. The 1.375-second smoke clips are padded
by the source DeSync scorer to eight seconds, limiting that score's usefulness.

## Targeted tests

Run with the training dependencies installed and this worktree on `PYTHONPATH`:

```bash
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 python -m pytest -q \
  tests/pipelines/test_ltx2_joint_*_on_cpu.py \
  tests/trainer/diffusion/test_ltx_joint_trainer_on_cpu.py \
  tests/trainer/diffusion/test_omni_nft_loss_on_cpu.py \
  tests/trainer/diffusion/test_recaption_grpo_on_cpu.py \
  tests/pipelines/test_bagel_unigrpo_nft_*_on_cpu.py \
  tests/reward_loop/test_named_reward_models_on_cpu.py \
  tests/utils/reward_score/test_hpsv3_micro_batching_on_cpu.py \
  tests/utils/reward_score/test_native_av_scorers_on_cpu.py \
  tests/workers/test_diffusers_dtype_on_cpu.py
```

The latest focused checks passed 32 model tests (including differentiable native
attention), 11 trainer tests (including independent metric persistence), and
28 model tests at the snapshot-integrity revision. These runs overlap and must
not be added as distinct test counts. The combined regression suite passed 255 tests with 48 warnings in 55.94 s
(`/tmp/ltx-joint-final-regression.log`), including reward dispatch, scorer, dtype
and BAGEL regressions. CPU tests
use tiny models or explicit stand-ins and are not real LTX acceptance evidence.
See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the current implementation scope.

The opt-in native decoder audit is separate from training. After loading a
paired policy, it hashes all connectors, video/audio VAE and vocoder parameters,
runs actual generation, and compares the hashes again. It also rejects enabled
parameter gradients. Invoke the standalone probe with
`--audit-frozen-components --policy-bundle /path/to/rollout-policy-N` to check
a real exported pair. Without `--policy-bundle`, the probe creates zero-B adapters
for transport/kernel checks; these must never replace exported training adapters. Hashes are written to
`frozen_before.json` and `frozen_after.json`. The mutation-detection CPU test
passed separately (1 passed, 32 deselected, 26.58 s); the actual policy-9 native audit subsequently passed: all four parameter
hashes remained identical across two 30-step AV generations. Evidence is under
`/tmp/ltx-joint-artifacts/ltx-joint-native-frozen-001/`; the process exited zero.

An additional shared-code regression run passed 63 tests in 29.52 s with 21
warnings (`/tmp/ltx-joint-shared-regression.log`): BAGEL trainer, diffusion config,
FSDP ignored subtrees and merged-LoRA paths.

Ruff passes for the new joint modules. Across all 54 changed Python files it
reports only the pre-existing `UP042` for `DiffusionAdvantageEstimator(str, Enum)`;
that declaration is unchanged from HEAD. `git diff --check` passes.

Run 011's saved update metrics report 16/16 immutable-snapshot checks passed,
nonzero local Gemma gradients from both text (mean norm 0.95712) and AV (0.06255),
and nonzero DiT AV gradients (0.04233); text DiT gradients are zero and both
branches have zero nonfinite gradient elements. All eight ranks' 384 Gemma and
2,304 DiT LoRA tensors changed; optimizer state records ten total updates.
The diagnostic step took 1,181.669 s including tensor collection, with 316.029 s
training and 484.746 s rewards. It is not a steady performance benchmark.

For the verified local setup, the observed 16-replay worst old ratio error is
8.72612e-5. The recorded acceptance bound is 0.001 (one tenth of the actual
0.01 clipping half-width, more than 11 times the observed error), with zero
clipping violations required. `/tmp/ltx-joint-artifacts/consistency-baseline.json`
records the derivation and scope. Re-baseline changes to the model, tokenizer,
precision, sampling or backend. No further training run is required by the user; this bound is retained for
future reproductions.

Run 012 was launched to extend training through steps 6 and 7, then stopped at
the user's request during initialization, before another update or evaluation.
All recorded descendants of its verified driver have exited. The last verified
checkpoint remains run 011 step 5. Log: `/tmp/ltx-joint-training-012.log`.

## First-step diagnostics and policy reload regression

The 2026-10-13 first-step GRPO cancellation, VideoAlign raw/calibrated scores,
planning length changes, ten-step artifact cadence and second-policy LoRA
reload behavior is described in [IMPLEMENTATION.md](IMPLEMENTATION.md).
The real-worker two-policy probe used the original volleyball prompt and
produced plan/recaption word counts of 135/53 and 146/60. The probe used two
denoising steps and base-equivalent adapters; a full eight-rank training rerun
is still needed for end-to-end acceptance.


### Current frozen-encoder and reward contract (2026-10-13)

Planning uses Gemma LoRA at temperature 1 with only format/AV-goal instructions.
Positive and negative condition encoding disable that LoRA and retain frozen
base/connectors; only text GRPO updates Gemma. Historical AV-to-Gemma gradient
results above describe the previous design. Dataset video/audio reward targets
now survive native dispatch instead of both being replaced by the joint prompt.
AudioBox retains the reference `(CE+CU-PC+PQ)/40` score and reports all four axes.
See implementation sections 17–18 for branch comparison and real rescore evidence.

## Old-policy EMA and text KL

The joint recipe uses `algorithm.old_policy_decay_schedule=linear_to_0_5`
and `old_policy_update_interval=1`, matching the standalone LTX OmniNFT recipe.
After each completed actor update round (all mini-batches and PPO epochs), the
paired Gemma/DiT old snapshots are updated as
`old = decay * old + (1 - decay) * current`, using that completed training
step's `decay = min(step * 0.001, 0.5)`. This occurs before checkpoint saving
and validation, matching the standalone recipe's timing. The first rollout
copies current weights. Training and validation export the same EMA old
policy; generation never advances EMA. Old-policy replay uses that same
snapshot throughout the actor update. Current trainable weights are restored
after export. Checkpoints already contain the just-completed step's EMA, and
resume must not apply it again. The reference snapshot remains fixed.

Text optimization uses `text_loss_weight * (GRPO + text_kl_coef * KL)`.
The recipe sets `text_kl_coef=0.001`, `text_loss_weight=1.0`, and
`av_loss_weight=1.0`. KL is the sampled-token k3 penalty
`exp(logp_ref - logp_current) - (logp_ref - logp_current) - 1`, averaged over
valid response tokens and then samples. The reference is the fixed initial
Gemma policy, not EMA; reference forward passes receive no gradients.
Set `text_kl_coef=0` to disable its extra reference text forward pass.
The estimate uses rollout tokens and is not a full-vocabulary KL.
Monitor `actor/text/kl_loss`, `actor/text/weighted_kl_loss`,
`actor/text/total_loss`, `old_policy/decay`, and
`old_policy/update_applied` alongside reward and gradient metrics.

## Adapter-only model checkpoints

The joint recipe sets `actor_rollout_ref.actor.checkpoint.save_lora_only=true`.
Model shards contain both Gemma and DiT adapters, omitting frozen base weights.
Optimizer, scheduler, RNG, trainer/data-loader state and the paired old/reference
adapter snapshots remain saved for resume. Restore with the same base model,
adapter configuration and sharding layout. Existing full-model checkpoints can
still be loaded; subsequent saves use the adapter-only format. This changes new
checkpoints only and does not convert existing files.

Override `actor_rollout_ref.actor.checkpoint.save_lora_only=false` to save full
model shards again. This setting requires the installed verl checkpoint manager
to support `save_lora_only`. The saved files remain distributed training
checkpoints, not standalone PEFT `adapter_model.safetensors` exports.

## Text rollout/actor logprob diagnosis

Use this when historical rollout scores disagree with actor old-policy replay.
Diagnostics are disabled by default. With the same model/data/reward environment
variables as `run_train.sh`, launch a new diagnostic run with:

```bash
export LTX_OUTPUT_DIR=/absolute/path/to/new-diagnostic-run
bash examples/unigrpo_nft_trainer/ltx2/run_text_diagnostics.sh
```

The wrapper preserves the original launcher's model, data, parallelism, generation
length, validation schedule and reward settings, and forwards extra CLI overrides.
It does not resume the historical run automatically. It explicitly transports the
settings through actor configuration and the native policy RPC, so an existing
Ray runtime does not need to inherit extra diagnostic environment variables.

Every native worker records **R**, the original generation logprobs, then immediately
computes **Rprime** for the same original prompt/response token IDs with the actor's
teacher-forcing implementation. This happens before condition encoding, video
execution or offload. Every actor rank computes **T** under the corresponding old
policy at its original call site, before that micro-batch's backward. The text/AV
forward order is preserved, and the actor does not run an extra old-policy forward.
If a later micro-batch fails, earlier gradients may exist, but no optimizer step
is allowed to consume them.

Artifacts are written under `$LTX_OUTPUT_DIR/text_diagnostics`, or the absolute
`LTX_TEXT_DIAGNOSTIC_DIR` path. Each process has its own directory:

- `policy-<version>.json/.pt`: loaded LoRA, adapter state, all buffers including
  nonpersistent RoPE, and tensor hashes. Frozen base tensors are hashed on the
  first version. Native LoRA values and enabled/scaling state are checked against
  the exported policy after loading, beyond checking the transport file hash.
- `sample-*.json/.pt`: unique sample ID, policy version, seed, source rollout rank,
  actor rank, PID/device visibility, original tokens, actual masks, temperature,
  R/Rprime/T where available, and current LoRA/buffer hashes. Actor metadata links
  to the original native artifact. Input and score hashes detect transport changes.
- `failure-*.json/.pt`: first-failure full live Gemma parameters and **all** buffers,
  saved before restoring actor current weights or restarting native workers.
  Actor tensors retain FSDP shard coordinates/placements; they are not standalone
  full-model checkpoints. Native snapshots are unsharded. Numeric mismatch capture
  is bounded to one full snapshot per process. JSON `complete=true` is written only
  after its tensor file is saved successfully.

The default trigger is **per-sample** mean absolute logprob error above `0.001`,
or maximum absolute error above `0.05`, or a nonfinite comparison. These are
configurable diagnostic triggers, not universal numerical acceptance tolerances:

```bash
bash examples/unigrpo_nft_trainer/ltx2/run_text_diagnostics.sh \
  actor_rollout_ref.rollout.ltx_joint.text_diagnostics.mean_abs_threshold=0.001 \
  actor_rollout_ref.rollout.ltx_joint.text_diagnostics.max_abs_threshold=0.05
```

With `stop_on_mismatch=true` (the wrapper default), ranks coordinate a stop before
an optimizer update. A native R/Rprime mismatch saves the original native state
before raising; actor ranks coordinate when their rollout calls finish. An actor
mismatch captures all actor shards and requests native snapshots before exiting.
Successful samples still produce evidence files. Validation produces R/Rprime
only, since it does not execute actor training. There is one extra full text
forward per native sample, CPU transfer/hash overhead, and substantial disk/host
memory cost on a full-model failure capture (across eight replicas this can be
hundreds of GB). Use a diagnostic run with sufficient capacity.

Interpret the first divergence: R vs Rprime points to the original native
execution; Rprime vs T with R matching Rprime points to cross-worker state or
execution differences. Compare tensor hashes **by name**, and compare actor shards
by their recorded layout, not by rank-level aggregate hashes. These captures
support a subsequent paired module/MSProbe investigation; they do not themselves
prove a hardware fault or SDC, and this mode does not collect MSProbe layer dumps.

### Sharing frozen CPU weights between rollout processes

The joint recipe enables a file-backed CPU master shared by rollout processes on
one host:

```yaml
actor_rollout_ref:
  rollout:
    ltx_joint:
      cpu_offload: true
      shared_cpu_weights_dir: ${oc.env:LTX_OUTPUT_DIR}/shared_cpu_weights
```

Use a **local filesystem** for this directory. Set `shared_cpu_weights_dir: null`
to retain independent per-process masters. The dataclass default remains disabled
for other recipes. Existing training jobs are unaffected by editing the YAML.

The first initializer atomically publishes one mapped file per component, under a
hostname subdirectory. After its engine initializes successfully, it atomically
publishes `ready.json`. Other workers validate checkpoint/cache file identities
and bind empty model parameters directly to these files. Clean file-backed pages are physically shared; private
copy-on-write mappings prevent accidental CPU writes from altering other workers
or the cache. Existing integrity checks still reject altered weights. NPU copies,
LoRA adapters, and nonpersistent buffers remain private. Actor/FSDP and reward
weight management are unchanged.

Only the first initializer holds the per-host lock while loading checkpoints and
creating the cache. Once ready, followers release the lock before engine creation
and initialize concurrently. They read model configuration, tokenizer and scheduler
files, but skip checkpoint tensor loading and initial full-weight hash scans.
Gemma loads directly in FP32, without first constructing a BF16 checkpoint copy.
CPU-master and device-byte integrity checks still run at component activation.
The first run still pays the cost of writing approximately 90 GiB; this is not a
zero-cost startup. Physical sharing depends on resident pages and is best measured using
`Pss` / `Shared_Clean` / `Shared_Dirty` in `/proc/<pid>/smaps`, not summed RSS.

Allow approximately **90 GiB of local disk space per run per host**, plus room for
checkpoints and samples. This is a capacity estimate, not a measured whole-job RAM
peak. A tmpfs path such as `/dev/shm/ltx-<unique-run-name>` is also supported but
consumes host RAM and needs enough capacity alongside Ray; the old 16 GiB Docker
shared-memory mount is insufficient. Do not reuse one directory across concurrent
jobs or models. Files are retained for the run and can be deleted **after all its
workers have exited**. A partial write is never published; mismatched or corrupted
cache contents raise an error instead of silently replacing a live mapping.

This reduces replicated frozen CPU storage. It does not fix defective memory or
prove that an NPU transfer path is reliable.
