# LTX-2.3 recaption GRPO + OmniNFT

This recipe trains a paired Gemma/DiT LoRA policy using synchronous vLLM-Omni
rollout. Gemma generates recaptions and encodes them with the same adapter.
GRPO and OmniNFT both update Gemma; OmniNFT also updates the audio/video DiT.
Base weights, VAEs and connectors are frozen; connectors retain the gradient
path to the current Gemma encoder.

**Complete under the user-adjusted verification scope.** Five real update rounds, checkpoint restoration,
subsequent rollout and fixed evaluation have run successfully on eight Ascend
910B devices. Numerical alignment corrections passed the real FSDP paired forward and joint
backward in experiment `011`, which exited successfully. Native decoder freezing also passed an actual before/after audit. The user accepted the existing training duration and requested no further
training. Run 012 was stopped before another update. The user replaced human review with GPT visual/recaption spot review and waived
sound checks. Audio semantics and AV synchronization remain unverified; no
quality improvement is claimed. No additional training is scheduled. Earlier runs do not establish acceptance
of code changed afterward.

## Run the joint trainer

The tested environment is eight Ascend 910B devices with 64 GB each, CPU offload,
local model/reward assets and substantial host RAM. Runtime versions are recorded
in `/tmp/ltx-joint-artifacts/training-007-runtime-environment.json`: PyTorch
2.10.0, torch-npu 2.10.0.post4, Transformers 5.14.1, Diffusers 0.40.0, PEFT 0.20.0,
vLLM 0.28.0 and vLLM-Omni 0.28.0rc2.dev68+gded893462.npu. The installed native
source has an unavailable external Git directory; package version is not proof
of a clean checkout. Other versions and GPU execution are unverified.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LTX_MODEL_PATH=/path/to/LTX-2.3-Diffusers
export LTX_REWARD_ROOT=/path/to/omnift-rewards
export LTX_TRAIN_FILE=/path/to/train.parquet
export LTX_VAL_FILE=/path/to/fixed-eval.parquet
export LTX_OUTPUT_DIR=/path/to/new-run
export LTX_NUM_DEVICES=8
bash examples/unigrpo_nft_trainer/ltx2/run_joint_npu.sh
```

To reproduce the exact four-prompt functional experiment:

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

The default diagnostic geometry is 256 × 256, 33 frames at 24 fps, 30 denoising
steps, two prompts with eight samples each. Mini-batch size counts complete
recaption/AV samples, so 16 samples with mini-batch size eight produce two joint
optimizer steps. Micro-batch size is one per rank. FP32 LoRA masters use learning
rates 1e-6 for Gemma and 1e-5 for DiT. Gemma uses original FP32 base weights;
DiT/connectors use BF16. The recipe is a short functional experiment, not a
production-quality generation setting.

To restore a paired checkpoint into a new output directory:

```bash
bash examples/unigrpo_nft_trainer/ltx2/run_joint_npu.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/path/to/previous/checkpoints/global_step_4 \
  trainer.total_training_steps=5
```

`ref` remains the training-start snapshot on resume. Every rollout freezes a new
paired `old` from current parameters; all mini-batches retain that same `old`.
Each export contains both adapters, a version manifest and checksums. Replicas
must acknowledge the complete pair before generation. Training rollout versions
are `2 * step - 1`; evaluation versions are `2 * step`, with zero for initial
evaluation. Immutable old/ref hashes are checked during updates and serialization.

## Backend and numerical contract

`algorithm.trainer_type=direct_preference` and
`actor_rollout_ref.model.algorithm=ltx2_unigrpo_nft` select the joint trainer.
Each rollout replica uses one vLLM-Omni diffusion worker. Recaption invokes
Transformers `generate` on that worker's existing Gemma; it is not a separate
paged-KV autoregressive engine. The same Gemma then supplies native AV conditions.
The initial implementation uses conditional-only AV generation.

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

Each saved sample has replay tensors, silent MP4, stereo WAV and an `-av.mp4`
with audio, plus `samples.json` and an HTML gallery. Training samples are under
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

## Consistency collection

Use an unused shared dump directory and the desired global step:

```bash
export DUMP_PATH=/path/to/new-probe
export MSPROBE_DUMP_GLOBAL_STEP=1
bash examples/unigrpo_nft_trainer/ltx2/run_consistency_npu.sh
```

This selects a real old-policy actor AV forward and replays its exact inputs in
the native worker. It saves MSProbe L0 dumps, boundary tensors, token replay and
correlation metadata under `actor/` and `generate/`. A run can consume over
120 GB of diagnostic storage. The wrapper defaults to one training prompt and
disables evaluation; explicit trailing overrides can change these settings.
Use the documented MSProbe and root-cause-analysis workflow for dump mapping and
analysis; the current investigation is recorded in [RUN_JOURNAL.md](RUN_JOURNAL.md).

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
Historical test commands, failures and fixes are retained in
[RUN_JOURNAL.md](RUN_JOURNAL.md).

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
