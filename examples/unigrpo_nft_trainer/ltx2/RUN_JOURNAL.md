# LTX joint-training investigation journal

Historical notes recorded during implementation. Status statements below describe
the time of each experiment and can be superseded by later entries. See
[README.md](README.md) for current acceptance status and launch instructions.

## Local verification, 2026-10-12

The host Python has no training packages. Tests ran on a separate copy of this
worktree at `/tmp/ltx-joint-worktree` inside the existing
`bagel-unigrpo-nft-npu` container, using its installed dependencies:

```bash
PYTHONPATH=/tmp/ltx-joint-worktree TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 \
  python -m pytest -q \
  tests/trainer/diffusion/test_omni_nft_loss_on_cpu.py \
  tests/trainer/diffusion/test_recaption_grpo_on_cpu.py \
  tests/pipelines/test_bagel_unigrpo_nft_loss_on_cpu.py
```

Result: **27 passed**, 20 dependency warnings, 26.80 seconds. This covers the
OmniNFT hand-calculated loss and gradient ownership, probability routing,
recaption advantage/clipping/masking/accumulation, replay metric calculations,
and the existing BAGEL loss regression tests.

`scripts/generate_trainer_config.sh` completed with `All good` in that isolated
copy with all four generated config files checked against their initial staged
contents. No generated YAML changes were needed. The source worktree was not
committed. Ruff reports the existing `UP042` on
`DiffusionAdvantageEstimator(str, Enum)`; this pre-existing enum is unchanged.

The host exposes eight Ascend 910B devices (no running device processes when
queried). Container cache `/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee8edcf067e838b843f926ec4d4cc9b2be1aaf`
declares `Gemma3ForConditionalGeneration`, LTX2 DiT and connectors. Cache presence
and device visibility do not prove loadability or completed training. The
container's vLLM-Omni source is at `/workspace/src/vllm-omni`; its checkout points
to an unavailable external Git directory, so its revision still needs independent
verification against the repository pin before relying on backend compatibility.

Additional model/state tests:

```bash
PYTHONPATH=/tmp/ltx-joint-worktree TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 \
  python -m pytest -q \
  tests/pipelines/test_ltx2_joint_model_on_cpu.py \
  tests/pipelines/test_bagel_unigrpo_nft_policy_state_on_cpu.py \
  tests/pipelines/test_bagel_unigrpo_nft_hooks_on_cpu.py
```

Result: **25 passed**, 30 dependency warnings, 25.99 seconds. The Gemma tests use a real tiny random
`Gemma3ForConditionalGeneration` plus PEFT LoRA, with lightweight stand-ins for
connectors and DiT. They compare selected-token teacher-forcing logprobs with
Gemma's own full forward, check both gradient paths, and check current embeddings
change after an optimizer step. They do not validate LTX model quality or NPU
execution. The state tests cover paired old/ref snapshots, preserving the initial
reference through serialization, rejecting missing components, and restoring
both components after an exception. BAGEL retains its public policy-state class
and now reuses the same extracted trainable-parameter snapshot implementation.

The training adapter and loss are now registered under `ltx2_unigrpo_nft`.
Use of the actor requires `model_type=diffusion_model`, FSDP2, only the `default`
PEFT adapter, and `ppo_micro_batch_size_per_gpu=1` in this initial implementation.
Old/ref are hook-managed paired snapshots. The inherited `reference` adapter
convention (disable LoRA) is deliberately not used. Gemma decoder layers and DiT
blocks have explicit FSDP2 units; Gemma's tied embedding/head remain owned by the
root unit. Distributed sharding is not yet exercised.

Joint hooks were tested through the real shared `BaseEngine.train_batch` method
with the tiny CPU model. Tests assert no parameter changes during either
backward, exactly one optimizer step afterward, both branches' contributions to
Gemma, DiT gradients only from AV, frozen base/connectors, and unchanged old/ref
across another mini-batch. Invalid versions, geometry, noise and probabilities
fail before backward. Replay logprob quantiles are computed over the concatenated
valid tokens, not averaged per-sample percentiles. Hooks report reference, text,
AV and combined forward/backward wall time; the trainer still needs to supply
complete step and optimizer-inclusive timings.

The combined model/hooks/state/loss/recaption regression command on 2026-10-12
passed **61 tests** in 26.93 seconds (38 dependency warnings):

```bash
python -m pytest -q \
  tests/pipelines/test_ltx2_joint_model_on_cpu.py \
  tests/pipelines/test_bagel_unigrpo_nft_hooks_on_cpu.py \
  tests/pipelines/test_bagel_unigrpo_nft_policy_state_on_cpu.py \
  tests/pipelines/test_bagel_unigrpo_nft_loss_on_cpu.py \
  tests/trainer/diffusion/test_omni_nft_loss_on_cpu.py \
  tests/trainer/diffusion/test_recaption_grpo_on_cpu.py
```

Run in the same container/worktree copy and environment described above. Config
regeneration again checked all four generated YAML files with `All good`.
A subsequent public joint-loss test verifies the unnormalized `2 * text + 3 * AV`
value and derivative against the separate objectives. Its test file passed
**8 tests** in 27.10 seconds (including the seven previously tested OmniNFT
cases); there are now 62 distinct passing CPU tests across the listed files.

## Native rollout probe

The joint registry now has a rollout adapter and `LTXJointVllmBackend`. The
vLLM-Omni diffusion worker uses its existing Gemma instance for Transformers
`generate`, then encodes the resulting caption and runs its native AV DiT.
This is not a separate paged-KV autoregressive engine. Paired adapter bundles
include component checksums and a monotonically increasing policy version;
generation is admitted only after both components load successfully.

The standalone probe is reproducible with:

```bash
# On Ascend, initialize CANN's Python/compiler paths in the same shell first.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  python examples/unigrpo_nft_trainer/ltx2/smoke_rollout.py \
  --model /hub/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee8edcf067e838b843f926ec4d4cc9b2be1aaf \
  --output /tmp/ltx-joint-smoke-new
```

Use a new output directory. The probe injects rank-8, zero-B adapters into one
Gemma and one DiT projection, so it tests native loading and transport without
changing base predictions. It is not a substitute for actor adapter export,
reward scoring, joint updates or quality evaluation. Its short default sampling
configuration is diagnostic, not a quality baseline.

On 2026-10-12, native probe `002` loaded the real model and synchronized both
adapters on Ascend 910B, then failed because the installed tokenizer's chat
template returned `BatchEncoding` by default. The adapter now explicitly requests
tensor output. Probe `001` had failed earlier during construction because the
backend's `extras` default was a Pydantic `FieldInfo`; the backend now explicitly
passes an empty mapping. Logs are `/tmp/ltx-joint-smoke-001.log` and
`/tmp/ltx-joint-smoke-002.log` on the host. These failures do not establish a
completed rollout. The backend now propagates native worker error messages.

Probe `003` passed recaption and conditioning but failed in the first native DiT
LoRA projection: model-level CPU offload had left the LoRA tensor tuples on CPU.
The worker now uses the native layer's residency hook to keep these sidecars on
the compute device while continuing to offload base parameters. Probe `004`
subsequently completed both DiT denoising steps and reached audio VAE decoding,
where the NPU compiler failed to import `tbe`. The module is installed in the
CANN Python directory, so the command above now sources CANN's environment.
Neither LoRA component is disabled by the offload fix. Full media output still
requires verification; logs are `/tmp/ltx-joint-smoke-003.log` and
`/tmp/ltx-joint-smoke-004.log`.

The six-file CPU regression command above passed **65 tests** in 26.84 seconds
(41 dependency warnings), recorded at `/tmp/ltx-joint-cpu-regression.log`.
After the sidecar change and replacing the recaption tokenizer stub with a real
local `PreTrainedTokenizerFast`, the model test file passed **21 tests** in
27.10 seconds (40 dependency warnings), recorded at
`/tmp/ltx-joint-model-regression.log`. These CPU tests still use a tiny Gemma and
stand-in connectors/DiT, and do not verify device offload.

Probe `005` **completed real native inference and exited successfully** after
loading CANN's environment. Container artifacts are in
`/tmp/ltx-joint-smoke-005/` (`sample.pt`, `summary.json`, `video.mp4`, `audio.wav`);
the host log is `/tmp/ltx-joint-smoke-005.log`. The outputs and both clean latent
tensors are finite. Video is `[33, 3, 256, 256]` at 24 fps; native audio was
`[1, 2, 63840]` at the actual vocoder's **48000 Hz**. The backend now removes the
single audio batch dimension and retains runtime fps/sample-rate metadata for
reward scoring, instead of relying on the generic IO spec's 24000 Hz default.
Clean latent shapes are `[1, 320, 128]` (video) and `[1, 34, 128]` (audio).

Measured first-run wall times: loading 64.48 s, paired sync 1.33 s, recaption
4.86 s, and conditioning/generation/decoding 85.96 s. The last measurement
includes first-use CANN compilation and is **not** steady-state performance.
Only two denoising steps were used. Recaption reached its 32-token cap without
EOS and is visibly incomplete. These diagnostic samples therefore establish
transport and decoding, not generation quality. No scorer or joint update was
run by this probe; acceptance stage 1 is still incomplete.

## Scoring saved native rollouts

`score_rollout.py` runs all five native models sequentially through the shared
reward manager, releasing each model before loading the next. Both audio and
video text reward inputs are the original user prompt, never the recaption.
It saves resolved configuration, five scores, scorer details, per-model loading
and scoring times, source OmniNFT video/audio probabilities, and text advantages.
The text reward continues to use the provisional mean of the two probabilities.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  python examples/unigrpo_nft_trainer/ltx2/score_rollout.py \
  --sample /tmp/ltx-joint-smoke-005/sample.pt \
  --model /hub/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee8edcf067e838b843f926ec4d4cc9b2be1aaf \
  --reward-root /hub/omnift-rewards \
  --output /tmp/ltx-joint-rewards-new
```

Repeat `--sample` for independent samples in a group. A singleton group produces
probability 0.5 and zero text advantage; this diagnoses scoring, not a learning
signal. Model paths match the pinned source launch script, including
`checkpoints/clap-htsat-unfused` rather than the generic CLAP default model.
Additional native dependencies used here are `audiobox_aesthetics==0.0.4` and
`timm==1.0.27`, matching the source dependency script.

Reward regression verification on 2026-10-12 passed **184 tests** in 32.83 s
(20 dependency warnings), with **no skipped async tests**, in the same isolated
container copy. Log: `/tmp/ltx-joint-reward-tests-003.log`.

```bash
python -m pytest -q --asyncio-mode=auto \
  tests/utils/reward_score/test_native_av_scorers_on_cpu.py \
  tests/utils/reward_score/test_clap_on_cpu.py \
  tests/utils/reward_score/test_hpsv3_micro_batching_on_cpu.py \
  tests/utils/reward_score/test_reward_utils_on_cpu.py \
  tests/reward_loop/test_multi_reward_manager_on_cpu.py \
  tests/reward_loop/test_named_reward_models_on_cpu.py \
  tests/reward_loop/test_reward_loop_manager_on_cpu.py \
  tests/pipelines/test_ltx2_joint_rewards_on_cpu.py
```

The test environment uses `pytest==8.3.2` and `pytest-asyncio==0.24.0`. Tests
cover scorer arithmetic with injected model outputs, preservation of original
prompts and 48 kHz metadata, required scorer failure/cleanup, component order,
finite-score enforcement, legacy weighted sums, and both OmniNFT loss modes.
They do not validate real reward-model predictions.

Real NPU scoring run `003` exited successfully on 2026-10-12. It scored the
actual output of native rollout probe `005` with all five models and executed
`LTXJointLoss.prepare_actor_batch`, including the source probability mapping.
Artifacts are `/tmp/ltx-joint-rewards-003/` in the container and
`/tmp/ltx-joint-artifacts/rewards-003/` on the host; log:
`/tmp/ltx-joint-rewards-003.log`. All scorer/helper source files were checked
byte-for-byte against the pinned source commit.

| Component | Actual score | Model load (s) | Score (s) |
| --- | ---: | ---: | ---: |
| AudioBox | 0.222890 | 3.86 | 1.04 |
| CLAP | 0.511783 | 0.96 | 0.19 |
| DeSync | 0.833333 | 24.05 | 0.81 |
| HPSv3 | -10.224693 | 380.65 | 0.89 |
| VideoAlign | -0.559111 | 2.98 | 0.75 |

Total measured reward wall time was 419.38 s, including loading and release.
HPSv3's model construction dominates this first scoring run. The single-sample
group produced video/audio probabilities `[0.5, 0.5]`, text probability `0.5`,
text advantage `0.0`, and constant-group fraction `1.0`. This is expected for a
singleton and proves neither a useful GRPO group nor improved quality. The
input remains the two-step video with a truncated recaption described above.
Multiple independent samples per prompt, same-old consistency, real joint
updates, resume and human quality review are still outstanding.

## Consecutive grouped rollout and scoring

On 2026-10-12, group probe `002` completed two consecutive requests in one native
worker, using `--num-samples 2 --steps 30 --recaption-tokens 256`, the same prompt,
seeds 42/43, and paired policy version 0. Both recaptions ended with EOS (159 and
223 tokens). This exposed and fixed a real CPU-offload bug: after the first AV
request Gemma was on CPU, while `generate()` consulted its device before its
forward hooks ran. The recaption path now enters the native sequential-offload
residency context before calling `generate()`.

Host artifacts are `/tmp/ltx-joint-artifacts/group-002/`, including both saved
replay records, MP4/WAV files and `summary.json`. The log is
`/tmp/ltx-joint-group-002.log`. Loading took 71.21 s, paired sync 0.69 s, and both
rollouts 179.73 s. Recaption times were 21.47/31.98 s and AV generation times
98.87/25.93 s. The first request includes compilation; one later request is not
a stable throughput benchmark. The process exited successfully, although native
worker shutdown logged a timeout/termination warning.

All five real scorers then processed this group with the pinned source recipe.
Host artifacts: `/tmp/ltx-joint-artifacts/group-rewards-001/`; log:
`/tmp/ltx-joint-group-rewards-001.log`.

| Seed | AudioBox | CLAP | DeSync | HPSv3 | VideoAlign | Video/audio probability | Text advantage |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 42 | 0.235215 | 0.569674 | 0.833333 | -10.271109 | -0.598077 | 0.450849 / 0.350469 | -0.999990 |
| 43 | 0.337456 | 0.623355 | 0.833333 | -10.236874 | -1.499020 | 0.549151 / 0.649531 | +0.999990 |

The provisional mean reduction gives text probabilities 0.400659/0.599341 and
constant-group fraction 0.0. Thus the actual samples provide a nonzero relative
text signal. Reward wall time was 400.71 s, including 374.39 s loading HPSv3.
This process used `OMP_NUM_THREADS=8`; it did not establish a material speedup.

These samples still have a known quality defect: their recaptions specify
3 seconds and 5–7 seconds, while the generated interval is about 1.33 seconds.
The recaption instruction now includes the actual available duration. That
instruction change has passed the 22 model regression tests (26.86 s,
`/tmp/ltx-joint-duration-tests.log`), but has not yet been checked in a new real
generation. Neither media quality nor the effectiveness of the hint is accepted
on the basis of these scores.

The consistency collection/analysis skills were fetched to
`/tmp/ltx-joint-msagent/` from `Ascend/msagent` commit
`1ea8fff4e2de563db1dec444c9875a4cb6589b14`. They require paired MSProbe dumps and
sample correlation. No paired dumps or root-cause report have been produced
yet. The installed `mindstudio-probe==26.1.0.post1` initially failed to import
because `matplotlib` was absent.
Its missing declared dependencies have now been installed, and importing the
public `PrecisionDebugger` API succeeds with the CANN environment loaded. This
checks tool availability only, not successful tensor collection.

The subsequent sections record consistency screening and trainer integration.
Joint-update/resume/quality acceptance still requires actual training evidence.

## Recaption precision screening

`screen_recaption.py` checks saved token IDs against the actual actor's
`LTXJointModel.recaption_logprobs` method using a verified paired policy bundle.
It saves per-token tensors and metrics; `--cached-comparison` also teacher-forces
the same saved tokens through Gemma's cached forward, without sampling. This is
a diagnostic, not a replacement for paired worker/actor MSProbe collection.

The BF16 experiment `/tmp/ltx-joint-screen-002` (host copy
`/tmp/ltx-joint-artifacts/screen-002`) found that cached replay reproduced all
382 saved native rollout logprobs exactly. Full-sequence actor replay differed:

| Seed | Mean absolute logprob error | p99 | Maximum | Ratio outside 1% clip |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.016765 | 0.157249 | 0.208957 | 28.93% |
| 43 | 0.016204 | 0.187361 | 0.211951 | 23.32% |

Therefore the BF16 configuration fails the intended consistency requirement.
A separate FP32 experiment on seed 42, `/tmp/ltx-joint-screen-fp32-001`, reduced
the full-sequence versus cached maximum logprob error to 0.00002396 and the
fraction outside 1% to zero. FP32 versus the historical BF16 rollout still
differs; that comparison cannot qualify the new rollout configuration.

The shared Gemma now loads in FP32 in both actor and native worker. The actor
declares its FP32 subtree so FSDP preserves it; the custom engine loader now
honors the same precision-preservation contract as its standard loader. Hidden
states are cast to connector dtype at the conditioning boundary, preserving
the gradient. DiT/connectors retain their configured dtype. This increases
Gemma weight memory and still needs real distributed training validation.

The model and precision helper regressions passed 27 tests in 27.91 s
(`/tmp/ltx-joint-fp32-tests.log`), including gradients through a frozen BF16
connector from FP32 Gemma. The custom registry-loader regression also passed
all 5 dtype tests (`/tmp/ltx-joint-custom-dtype-tests.log`).

Native FP32 probe `001` completed two 30-step AV generations, with 168/202
recaption tokens and EOS on both. Both descriptions now explicitly fit the
1.333-second duration. Host samples are in
`/tmp/ltx-joint-artifacts/group-fp32-001/`. Same-old actor screening saved to
`/tmp/ltx-joint-artifacts/screen-fp32-native-001/` found maximum ratio errors
0.001901/0.003342 and no tokens outside the 1% clipping interval. This improves
BF16 behavior but remains substantially above the same-process FP32 control.

The first native FP32 probe upcast an already-BF16-loaded Gemma. The worker now
reloads original Gemma checkpoint weights directly in FP32 before offload hooks
or LoRA injection, avoiding irreversible rounding from that intermediate load.
The native backend currently has no per-component dtype option, so this adds
initialization work.

Probe `002` completed successfully with direct FP32 checkpoint loading. Its
168/202-token outputs were replayed by the actual actor logprob implementation:

| Seed | Mean absolute logprob error | p99 | Maximum ratio error | Ratio outside 1% clip |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.00000330 | 0.00002888 | 0.00004697 | 0% |
| 43 | 0.00000396 | 0.00003728 | 0.00006986 | 0% |

The largest observed ratio error is below 1/140 of the configured clipping
scale on these fixed samples. Host outputs are
`/tmp/ltx-joint-artifacts/group-fp32-002/` and
`/tmp/ltx-joint-artifacts/screen-fp32-native-002/`; logs are
`/tmp/ltx-joint-group-fp32-002.log` and
`/tmp/ltx-joint-screen-fp32-native-002.log`. This is a real same-old token-level
screen on one prompt and two seeds, not a universal tolerance or evidence of
post-update adapter consistency. Paired MSProbe tensor analysis, Gemma/DiT
conditioning consistency and distributed joint training remain open.

## Actor LoRA export

`export_joint_actor_lora` now gathers only trainable Gemma/DiT LoRA matrices from
FSDP2 DTensors, preserving FP32 adapter values. All ranks verify the same version,
layout, destination and host before gathering. Rank zero publishes the existing
checksummed paired bundle and broadcasts publication failures. The native
backend's `load_actor_policy` commits that bundle on every replica before
releasing the previous version; an incomplete replica commit disables admission.

The real two-process Gloo/CPU DTensor test passed in 58.48 s
(`/tmp/ltx-joint-export-tests.log`). It checks exact reconstructed values, frozen
base exclusion, paired output, mismatched rank versions and root publication
failure. A separate transport test passed in 26.60 s
(`/tmp/ltx-joint-export-backend-tests.log`), covering retention of previous files
until commit and failure admission/cleanup with a fake native RPC. These tests
do not qualify real FSDP/HCCL export or post-update native loading. The joint
generate hooks, trainer selection and recipe are now connected as described below;
their full runtime behavior is not yet qualified.

## Joint training entry point

`joint_train.yaml` composes the existing diffusion trainer and the pinned five-
reward recipe. `algorithm.trainer_type=direct_preference` with model algorithm
`ltx2_unigrpo_nft` selects `LTXJointRayTrainer`. Each actor rank owns a separate
TP=1 native worker. The hook freezes old, exports both LoRA components, commits
the pair and returns original-prompt reward inputs, AV tensors, clean latents
and exact replay records. The initial generation uses conditional-only guidance.

`ppo_mini_batch_size` is the number of complete recaption/AV samples in a joint
optimizer step; this trainer does not multiply it by rollout `n`. Gemma and DiT
have separate optimizer learning rates (1e-6 and 1e-5). The short diagnostic
recipe uses 256-pixel, 33-frame outputs and 30 generation steps; it is not a
production-quality recipe or an accepted short-training result.

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

Both parquet files use the normal prompt-chat dataset contract, including
`data_source` and `reward_model.ground_truth`. The launcher forwards every extra
argument, including `--cfg job --resolve` for a configuration-only check.
Restore uses `trainer.resume_mode=auto` and the same output directory, but actual
joint checkpoint restoration remains to be tested. Periodic fixed-prompt
evaluation is enabled at every step in the recipe; its implementation is
described below and still requires real runtime validation.

Integration tests passed 42 tests in 28.06 s (joint model/hook and BAGEL trainer
regressions) and 7 LTX trainer tests in 27.08 s, recorded in
`/tmp/ltx-joint-hooks-tests.log` and `/tmp/ltx-joint-trainer-tests.log`. Tests cover
paired export before generation, original-text AV reward inputs, replay transport,
group/seed order and actual-sample mini-batch size. Native calls are mocked in
these integration tests; earlier standalone native evidence remains separate.
All four generated configs were regenerated and verified (`All good`,
`/tmp/ltx-joint-config-verification.log`). The actual launcher resolved its full
Hydra configuration successfully (`/tmp/ltx-joint-launch-config.log`).

The first real two-NPU joint-training diagnostic used four fixed prompts in
`/tmp/ltx-joint-train-data/`, one training step and checkpoint saving disabled.
Run `001` failed before training: the explicit FSDP2 initialization helper moved
the whole Gemma/DiT module to NPU before sharding and exhausted 61 GiB of usable
memory. Log: `/tmp/ltx-joint-training-001.log`.

The helper now moves each declared unit and shards it immediately, leaving
already-sharded descendant DTensors under FSDP device ownership. Frozen buffers
and leaves are still moved for direct use. Retry `002` passed real two-NPU FSDP
initialization (zero allocated device memory after CPU offload), then failed on
missing native reward placement. Run `003` established that the existing reward
manager rejects overlapping native-model device lists. The recipe now assigns
the five scorers to distinct parent-pool indices 0–4, preserving all scoring
and aggregation settings; the launcher requires at least five devices and
defaults to eight. Actor and rollout reuse those devices in the synchronous
schedule. Run `004` uses all eight devices; log
`/tmp/ltx-joint-training-004.log`. No successful optimizer update is claimed yet.

Each saved training step now includes MP4, WAV and tensor replay artifacts,
plus an index linking original prompts, recaptions, versions, five scores and
probabilities. Real MP4/WAV serialization and trainer wall-clock tests passed
9 tests in 27.55 s (`/tmp/ltx-joint-artifact-metric-tests.log`); the disjoint
recipe placement check passed separately in 27.43 s
(`/tmp/ltx-joint-reward-placement-test.log`). These are component tests.
The full step uses the earliest replica sync start through the end of actor
updates; per-stage timings preserve parallel replica execution instead of
summing it. Initial native model loading is reported separately.

Run `004` was deliberately stopped during initial native DiT adapter loading.
A nonblocking process-stack inspection found repeated regex compilation in
`_match_target_modules`; the exported adapter had 1,152 exact module names,
exceeding the regex cache. DiT bundles now use one anchored, escaped alternation
covering exactly those names. Gemma export remains unchanged. Bundle loading
and exact matching regressions passed 2 tests in 26.30 s
(`/tmp/ltx-joint-bundle-regex-tests.log`). Real retry `005` uses this change;
its log is `/tmp/ltx-joint-training-005.log`. This is not yet successful update
evidence.

Run `005` passed collective paired-policy export/loading and entered native
30-step AV generation on the eight replicas. The full reward/update round is
still running. Peak allocated/reserved device memory reporting has also been
added for subsequent runs: native rollout resets per request; actor resets
before synchronization and reports after each mini-batch, with max reduction
across ranks/mini-batches. These are process allocator peaks, not a sum across
colocated processes. The live `005` workers were already loaded before this
monitoring change. Model/trainer regressions passed 33 tests in 28.98 s and
Ruff passed (`/tmp/ltx-joint-memory-metrics-tests.log`).

## Fixed evaluation and memory handoff

`LTXJointRayTrainer._validate` now generates the entire fixed validation set
through the same paired native backend, scores it with all five rewards and
saves the original/recaption/media/score index under
`samples/evaluation/step-NNNNNN/`. Validation order and seeds remain fixed.
It performs no optimizer update. Version 0 is reserved for optional initial
evaluation; training round s uses 2s-1 and its post-update evaluation uses 2s.
Evaluation may refresh old only after the round's complete mini-batch sequence;
the initial ref snapshot is untouched. Checkpoints saved before evaluation can
resume at the next training version. Lifecycle tests, including failure cleanup,
fixed seeds, ordering and BAGEL dispatch regressions, passed 28 tests in 27.69 s
(`/tmp/ltx-joint-evaluation-tests.log`). This is not real evaluation evidence.

Run `005` completed all 16 native AV samples and reached reward initialization.
At that boundary, native processes still occupied about 39 GiB each. The backend's
sequential offloader swaps components only when the next component runs; it does
not release the last-used model on batch completion. A new native RPC offloads
the managed Gemma/DiT modules and empties the allocator after all requests, while
retaining the existing hooks for subsequent requests. Actor generation likewise
releases cached device memory before native rollout. These changes passed Ruff;
the still-running `005` predates them and cannot validate the new handoff.

Run `005` finished generation and five-model scoring, then failed during joint
actor preparation because the driver-side Hydra mapping did not materialize
`LTXJointLossConfig` defaults (`text_probability_reduction` and
`text_advantage_epsilon`). The recipe now supplies both explicitly. A regression
composes the real Hydra recipe and calls joint batch preparation without worker
config conversion; all 11 trainer tests passed in 28.02 s
(`/tmp/ltx-joint-preparation-tests.log`). No actor backward occurred in `005`.

Retry `006` runs one diagnostic training round with checkpointing and periodic
evaluation disabled via explicit overrides. Log: `/tmp/ltx-joint-training-006.log`.
It includes the memory handoff and memory metrics. Completed rollout/reward
samples are now persisted before actor updates, retaining evidence if training
fails. Its exact paired rollout bundle was copied to
`/tmp/ltx-joint-training-006/rollout-policy-1/` before ephemeral cleanup.

Branch gradient attribution was added after `006` started. It snapshots local
accumulated FSDP gradients before/after each branch and reports the norm of the
difference for Gemma and DiT, plus nonfinite counts. These are explicitly local
shard norms, not global gradient norms. Seven branch/hook regressions passed in
28.30 s (`/tmp/ltx-joint-gradient-diagnostics-tests.log`), including isolated loss
coefficients and accumulation. Real branch-attribution evidence remains pending;
`006` loaded the previous hooks and cannot supply it.

During `006`, the first five replicas to finish their complete local batch
released native device residency from approximately 39,690 MiB to 2,314 MiB per
process, while slower replicas were still finishing generation. This confirms
the new end-of-batch handoff on real workers. A device-process snapshot is saved
at `/tmp/ltx-joint-training-006-memory-handoff.txt`; subsequent training and
next-round reactivation are still unverified.

## Run 006 artifacts and actor compatibility fixes

Run `006` scored and persisted all 16 native samples (two original-prompt groups,
eight samples each) under joint policy version 1. All recaptions ended with EOS,
with 83–227 generated tokens. Text probability ranged from 0.215327 to 0.646898.
Host samples: `/tmp/ltx-joint-artifacts/training-006-samples/step-000001/`;
container samples: `/tmp/ltx-joint-training-006/samples/step-000001/`.
The original tasks were a hand tapping wood and a drop falling into a glass.
The artifact audit decoded all 16 MP4/WAV pairs and checked finite five-score
vectors, exact replay validity, latent/noise shapes and matching paired-bundle
version (`/tmp/ltx-joint-training-006-artifact-audit.log`). All videos have 33
frames and stereo audio at 48 kHz. This verifies the inference/reward closure,
not human-reviewed quality or training/inference prediction equivalence.

Actual actor execution then failed during old DiT forward: Diffusers routed an
RMSNorm with no affine weight to `torch_npu.npu_rms_norm`, whose gamma argument
cannot be None. The joint model replaces only these parameter-free norms with
Diffusers' standard FP32-variance formula. Affine norms and checkpoint parameter
keys remain unchanged. All 26 model regressions passed in 29.73 s
(`/tmp/ltx-joint-norm-tests.log`). A real NPU check against the original Diffusers
CPU reference formula passed for FP32 and BF16. FP32 maximum output/gradient
errors were 2.38e-7/4.77e-7; BF16 matched exactly on this input
(`/tmp/ltx-nonaffine-norm-npu-check-003.log`). This is a norm-level check only.

The checkpoint path also required serializable joint model metadata. The model
now uses `LTXJointConfig`, retaining all three component configs and an explicit
base-model path, with PEFT nested-conversion scope rooted at the composite module.
A config save/load regression passed in 26.20 s before the full model regression
above (`/tmp/ltx-joint-checkpoint-config-tests-004.log`). Full FSDP checkpoint
save/resume remains unverified.

Artifacts now retain training noise and rollout timings, plus explicit teacher-
forcing input IDs/attention/response masks for future runs. Actual MP4/WAV and
exact mask serialization passed (`/tmp/ltx-joint-explicit-mask-artifacts-tests.log`,
1 test, 27.98 s). Run `007` includes the norm/config fixes and branch-gradient
metrics, enables checkpoint saving, and keeps periodic evaluation disabled for
this first optimizer diagnostic. Log: `/tmp/ltx-joint-training-007.log`.
No successful optimizer step is claimed yet.

The run `006` review gallery is available on the host at
`/tmp/ltx-joint-artifacts/training-006-samples/step-000001/index.html`.
Each card includes the original task, recaption, joint version and seed, five
scores, modality/text probabilities and an MP4 containing both video and audio.
The original MP4, lossless WAV and tensor artifacts remain available. The media
test checks audio-stream decoding, all video frames and HTML escaping
(`/tmp/ltx-joint-review-gallery-tests.log`, 1 passed in 30.01 s). No human quality
review has been recorded.

Runtime provenance for run `007` is stored in the container at
`/tmp/ltx-joint-training-007/runtime-environment.json`: package versions, model
snapshot, reward source revision, and SHA-256 inventories of the actual Python
source trees. The installed vLLM-Omni version reports
`0.28.0rc2.dev68+gded893462.npu`; its broken external Git-directory reference
prevents verification of a clean checkout, so the source inventory records the
actual files instead. This capture does not prove backend numerical consistency.

## First completed joint training round: run 007

Run `007` exited successfully (code 0). It generated 16 native samples, applied
all five rewards, completed actor updates and wrote all eight model, optimizer,
algorithm-state and extra-state checkpoint shards under
`/tmp/ltx-joint-training-007/checkpoints/global_step_1/actor/` in the container.
The checkpoint occupies about 88 GiB, including frozen model weights. All eight
optimizer shards report step 2 for every optimizer state, with DiT learning
rate `1e-5` and Gemma learning rate `1e-6`. Every rank has changed finite values
in all 384 Gemma and 2304 DiT LoRA tensors relative to old. The final old and ref
snapshots are exactly equal, as expected for this first round. This does not
independently trace their contents during each intermediate mini-batch or compare
frozen base tensors with the original model. Audit report:
`/tmp/ltx-joint-artifacts/training-007-checkpoint-audit.json` on the host.

Reported mean local branch-gradient norms were 0.865525 for text→Gemma,
0 for text→DiT, 0.291142 for AV→Gemma and 0.103678 for AV→DiT. Both branch
nonfinite-element counts were zero. These local norms establish real gradient
paths; they are not global gradient norms. Reported global gradient norm was
2.612637. Same-old text replay had zero ratios outside the 1% clipping interval;
the logged absolute logprob mean was 3.70e-6. The percentile/max diagnostics are
framework-reduced per-sample statistics and must not be read as global token
quantiles without retaining and combining the underlying token differences.

The full sync-to-update wall time was 968.265 s; actor update took 120.936 s and
rewards took 395.407 s. This first round includes initialization/warmup costs and
is not a steady-state throughput estimate. Native model loading was separately
reported as 73.380 s. Checkpoint writing took 38.018 s. Extracted logged metrics
are on the host at `/tmp/ltx-joint-artifacts/training-007-metrics.json`.

The artifact audit passed for all 16 samples, including decoded media, finite
latents/noise, paired version and explicitly serialized teacher-forcing inputs,
attention masks and response masks. The review gallery is at
`/tmp/ltx-joint-artifacts/training-007-samples/step-000001/index.html` on the host.
The audit log is `/tmp/ltx-joint-training-007-artifact-audit.log`.

Run `008` was launched with the same recipe, explicit resume from run `007`'s
`global_step_1`, `trainer.total_training_steps=3` and `trainer.test_freq=2`.
It is intended to check recovery, subsequent paired-policy rollouts and fixed
validation. Its log is `/tmp/ltx-joint-training-008.log`; none of those results
are claimed until the actual run and checkpoint comparison complete.


## Frozen-weight audit and scorer timing

The complete run `007` frozen-actor audit passed on all eight ranks. Per rank,
1,066 Gemma state entries (including the tied LM-head alias), 4,186 DiT entries
and 262 connector entries exactly match the original model snapshot after
casting to the actor dtype and taking the recorded FSDP shard. No source keys
were missing and no mismatches were found. The audit handles Gemma's checkpoint
key conversion and excludes only LoRA matrices. Report on the host:
`/tmp/ltx-joint-artifacts/training-007-frozen-weights-audit.json`; execution log:
`/tmp/ltx-joint-training-007-frozen-audit.log`; script:
`/tmp/audit-ltx-frozen-weights.py`. VAE weights are not part of the actor checkpoint
and are outside this comparison's scope.

The reward manager now records `reward/<scorer>/score_seconds` for each sample.
This is awaited wall time, including executor queueing, excluding the separate
model wake-up phase. Concurrent scorer times are not summed into reward total
time. The multi-manager and named-model regression suites passed 92 tests in
33.41 s (`/tmp/ltx-joint-scorer-timing-tests.log`), and Ruff passed. Run `008` was
already initialized before this change, so it cannot validate the new timers.

Run `008` loaded all eight model/optimizer/RNG/scheduler shards from step 1 and
completed its next native generation batch. Its paired rollout version 3 was
preserved at `/tmp/ltx-joint-training-008/rollout-policy-3/`. Every exported Gemma
and DiT LoRA tensor exactly matches the corresponding restored checkpoint shard
on all eight ranks (384 Gemma and 2,304 DiT entries per rank). Host report:
`/tmp/ltx-joint-artifacts/training-008-resume-export-audit.json`; script:
`/tmp/audit-ltx-resume-export.py`. This verifies that the next rollout receives
the trained paired weights; it does not verify native activation arithmetic or
preservation of the initial reference after the next update. The reward phase
is still running, and multi-round completion remains unverified.

## Resume OOM and opt-in AV tensor collection

Run `008` completed resumed generation and rewards but failed in the first old
AV forward on rank 3. CANN FlashAttention reported runtime memory allocation
failure (`207001` / `EL0004`); the device snapshot showed 65,519 MiB of 65,536 MiB
occupied, including 59,727 MiB attributed to that actor. After preserving the
log, paired bundle and samples, the specific driver was terminated; its session
ended with code 134 and all eight devices were confirmed free. Snapshot:
`/tmp/ltx-joint-training-008-oom-memory.txt`. This is not a successful resumed
update or fixed evaluation.

The NPU launcher now explicitly passes `LTX_ACTOR_NPU_MEMORY_FRACTION=0.8` to
Ray workers. Joint actor initialization applies the public NPU caching-allocator
limit, leaving room for CANN allocations and colocated worker contexts. This is
a configurable mitigation, not yet a verified fix. Run `009` retries the same
resume and multi-round/evaluation configuration; log:
`/tmp/ltx-joint-training-009.log`. It is running with diagnostics disabled.

`run_consistency_npu.sh` enables a single requested global-step window, minimum
legal eight-sample group, one-sample micro-batches, and no periodic validation.
It forwards all diagnostic environment variables explicitly through Ray, and
preserves caller overrides. Native requests already execute serially with
conditional-only guidance. The actor sends one rank-0 old sample's exact AV
inputs to the committed native worker before its ordinary FSDP forward. The
worker recomputes Gemma encoding and connectors, checks exact condition token
identity, predicts with its native DiT, and releases device residency afterward.
MSProbe captures native and actual old-actor module tensors, supplemented by
connector/transformer input-output tensors and correlation metadata containing
sample identity, version, seed, token hash, timestep and global step.

The adapter refactor passed all 26 joint-model CPU regressions in 28.62 s
(`/tmp/ltx-joint-probe-hooks-tests.log`). Shell syntax, Ruff and command-capture
checks passed; `/tmp/ltx-joint-probe-wrapper-argv.json` records the final wrapper
arguments. The installed MSProbe API produced nonempty connector/transformer
module tensor dumps on a small CPU model (`/tmp/ltx-joint-probe-api-check-003.log`).
Its first/last-layer exclusion in a verl process made the first tiny fixture
empty; the fixture now includes surrounding modules and collection rejects empty
dumps. This validates the API, not LTX/NPU tensor collection. Actual paired dumps,
module mapping and the required root-cause report still remain to be produced.

The diagnostic boundary comparator is available as:

```bash
python -m verl_omni.pipelines.ltx2_unigrpo_nft.consistency_probe \
  --actor /path/to/actor/host-pid \
  --generate /path/to/generate/host-pid \
  --output /path/to/boundary_comparison.json
```

Use only trusted collector artifacts. It requires completed opposite-side
collections with identical sample, policy, step, token hash, seed, timestep and
single conditional layout, then checks exact noisy latent/time/geometry inputs
and condition masks. It reports Gemma stacked embeddings, connector contexts
and video/audio predictions with mean/p99/max absolute difference and relative
L2; it does not assign an invented acceptance threshold. Seven CPU regressions
passed in 29.56 s, including rejection of changed policy/seed/branch, incomplete
captures and changed noisy latents with otherwise identical metadata
(`/tmp/ltx-joint-pairing-tests.log`). Ruff passed. Real paired collection remains
pending and this comparator does not replace the required MSProbe module mapping
and root-cause report.

Actor diagnostic artifacts additionally retain exact old teacher-forced token
logprobs, native rollout logprobs, masks, temperature and clipping configuration
in `old_text_replay.pt`, permitting token-level statistics without averaging
per-sample percentiles. This extra persistence has not yet run on real LTX.
The current `009` run uses diagnostics disabled; its rollout pair was preserved
at `/tmp/ltx-joint-training-009/rollout-policy-3/` while active.

The native paired loader now rejects a bundle whose declared base-model path
differs from the live worker's model before modifying either component. Its
existing real-tiny-Gemma pair-loading test now exercises this rejection and then
checks successful loading and version admission; it passed in 26.16 s
(`/tmp/ltx-joint-live-base-identity-tests.log`). Ruff passed. This path check does
not replace source-weight inventories for detecting changes at the same path.

The time conventions were rechecked against the installed native
`ltx2_guidance.py:timestep_kwargs` and Diffusers LTX2 forward documentation:
`timestep`, `audio_timestep`, `sigma` and `audio_sigma` all receive scaled model
time, while forward-process noise interpolation uses that value divided by 1000.
The same-input diagnostic preserves both conventions.

The resumed run `008`'s 16 version-3 samples passed the full artifact audit at
step 2: all MP4/WAV pairs decoded, and replay tokens/masks, finite latent/noise
arrays, five scores and paired version matched. Host audit:
`/tmp/ltx-joint-artifacts/training-008-samples/step-000002/artifact_audit.json`;
log: `/tmp/ltx-joint-training-008-artifact-audit.log`. This validates generated
artifacts despite that run's later actor OOM; it does not change its failure status.

Run `009` passed the previous old-AV-forward failure point and reached subsequent
joint training computation. Its actor memory snapshot shows roughly 40–47 GiB
per process at that moment (`/tmp/ltx-joint-training-009-actor-memory.txt`), rather
than the roughly 59 GiB at the previous failure. The specific actor stack was
saved to `/tmp/ltx-joint-training-009-actor-stack.txt`. Full update/checkpoint and
later evaluation results are still pending.

## Resumed update verified; repeated-sync fix and run 010

Run `009` completed its resumed actor update and saved `global_step_2`. The
all-rank checkpoint audit confirms optimizer step 4, exact preservation of the
initial reference, old exactly matching the restored step-1 current parameters,
and finite changes in all 384 Gemma and 2,304 DiT LoRA tensors on each rank.
Host report: `/tmp/ltx-joint-artifacts/training-009-resume-state-audit.json`;
script: `/tmp/audit-ltx-resumed-checkpoint.py`; log:
`/tmp/ltx-joint-training-009-resume-state-audit.log`. This provides real recovery
and another joint-update round, with the limitation that checkpoint comparison
does not trace every intermediate mini-batch. The 80% actor allocator limit
allowed this previously failing workload to complete its update.

The run then failed during synchronization for fixed evaluation, exiting with
code 1. PEFT mutates large target-module configurations during injection, and the
worker had cached that mutated configuration. The next unchanged transport
configuration was therefore rejected as a structural change. The worker now
caches the pre-injection config. A two-layer real Gemma/PEFT regression forces
the actual target-module compression, loads two versions with different weights,
and verifies the second version's logits against the actor. It passed in 26.82 s
(`/tmp/ltx-joint-repeated-sync-tests-002.log`); the initial one-layer fixture could
not exercise cardinality reduction and was replaced. Ruff passed.

Run `010` resumes from run `009`'s step-2 checkpoint, targets total step 4 and
fixed evaluation every two steps, with the original two-prompt/eight-sample-per-
prompt batch. It uses `run_consistency_npu.sh` with
`MSPROBE_DUMP_GLOBAL_STEP=3` and `DUMP_PATH=/tmp/ltx-joint-probe-010` to collect one
post-update paired AV replay. Native replay installs the same forward context as
the production model runner. Log: `/tmp/ltx-joint-training-010.log`. This run is
active; real MSProbe artifacts, repeated native synchronization and final fixed
evaluation are not yet claimed as successful.

Run `010` has completed its first 16-sample generation and entered reward
computation. Its version-5 paired bundle was preserved at
`/tmp/ltx-joint-training-010/rollout-policy-5/` before a subsequent synchronization
can remove the transport directory. Diagnostic collection remains pending.

Completed actor updates now independently save raw metrics under
`trainer.default_local_dir/update_metrics/step-NNNNNN.pt`, with global step,
rollout policy version and `phase=actor_update_complete`. An atomic rename makes
the final file visible only after serialization completes. This preserves update
timing and gradient evidence if later checkpointing or evaluation fails, as in
run `009`; it is not a resumable checkpoint. The 11 trainer CPU regressions
passed in 28.04 s (`/tmp/ltx-joint-update-metrics-tests.log`), including checking
the persisted metrics against the update result. Ruff passed. This change was
made after run `010` imported its trainer and applies to subsequent launches.

Frozen old/reference CPU snapshots now retain SHA-256 digests. Each actor
forward/backward batch checks both snapshots before and after accumulation;
checkpoint serialization and the next rollout also check them, covering the
interval containing optimizer updates. A mismatch raises before the altered
snapshot can be exported or refreshed. Successful batches emit
`actor/joint/frozen_snapshots_verified=1`. These are rank-local snapshot checks,
not a comparison of active current parameters against old/ref. The 28 model
CPU regressions passed in 28.45 s, including deliberate corruption of either
snapshot (`/tmp/ltx-joint-frozen-digest-tests.log`); Ruff passed. These checks
were added after run `010` loaded its actor and still need real-run evidence.

## Run 010: real paired capture and continued rollout

Step 3 completed its joint update and checkpoint. The same live workers then
accepted policy 7 and generated the next rollout, exercising the repeated-sync
fix. Both policy 5 and policy 7 bundles are preserved under the run directory.
All eight rank shards of policy 5 exactly match restored step-2 current LoRA;
report: `/tmp/ltx-joint-artifacts/training-010-resume-export-audit.json`.

Actual paired MSProbe collections completed at
`/tmp/ltx-joint-probe-010/{actor/localhost.localdomain-2135417,generate/localhost.localdomain-2137214}`
inside the container, totaling about 122 GiB. They share sample identity, policy
5, seed 122, timestep 909 and 1,024 condition tokens. The actor also retained 235
old recaption token logprobs: mean absolute difference 4.2105e-6, p99 4.2787e-5,
maximum 4.7684e-5; no ratio crosses the actual ±0.01 clip interval. Host evidence:
`/tmp/ltx-joint-artifacts/training-010-old-text-comparison.json`.

Boundary comparison verifies exact noise/time/geometry/masks and bitwise equal
stacked Gemma embeddings. Connector video/audio context relative L2 differences
are 0.0030084/0.0038802; DiT video/audio prediction differences are
0.0085187/0.0075694. These are measured discrepancies, not acceptance claims.
The comparator now uses NumPy's linear quantile because the real 192-million-
element stacked embeddings exceed `torch.quantile`'s input limit. The corrected
comparison succeeded; report:
`/tmp/ltx-joint-artifacts/training-010-boundary-comparison.json`.

The fixed MSAgent analysis ran against 3,723 mapped actual module boundaries.
Its outputs and agent-written interim report are in
`/tmp/ltx-joint-artifacts/training-010-analysis/`. The report explicitly leaves
root cause open: raw connector tensors must distinguish actual parameter or
kernel differences from FSDP/fused-module representation and propagated errors.
This diagnostic step includes substantial tensor-write overhead and cannot be
reported as steady-state performance. Step 4 and fixed evaluation remain pending.

Raw-tensor investigation now identifies the first connector discrepancy as
split-RoPE precision. Both first-block video/audio projections and Q/K norms
are exact; all 36 inspected parameter fields are exact, including previously
flagged block-5 norms. Native RoPE replay on actual captured normalized Q/K
reproduces all four native attention inputs bitwise. Diffusers replay differs
on roughly one third of elements, max absolute 0.0625. Native casts frequencies
to BF16 and retains BF16 split arithmetic; installed Diffusers explicitly
upcasts rotation to FP32. Evidence and the updated agent-written report:
`/tmp/ltx-joint-artifacts/training-010-analysis/{connector_raw_comparison.json,rope_replay.json,output_5_root_cause_report.md}`.
This isolates the first boundary, not all downstream DiT error. Train-side
alignment preserving autograd and a new real paired comparison remain required.

## Connector rotation alignment

The joint actor now installs `NativeConnectorAttnProcessor` on video/audio
connectors. It retains differentiable Diffusers attention while materializing
split-RoPE frequencies and intermediate arithmetic in the projection dtype,
matching the native LTX-2.3 worker. Functional addcmul updates avoid modifying
autograd views in place. Learned-register attention drops its obsolete padding
mask, as native LTX-2.3 does. DiT attention is not changed by this installation.

Replaying the actual run-010 first-block normalized Q/K through the new helper
matches all 12,582,912 captured native rotated elements exactly. Evidence:
`/tmp/ltx-joint-artifacts/training-010-analysis/aligned_rope_replay.json`;
script `/tmp/replay-ltx-aligned-rope.py`; log
`/tmp/ltx-joint-aligned-rope-replay.log`. This is CPU replay against real NPU
captures, not a new complete-model NPU forward. Thirty model CPU regressions
passed in 30.15 s (`/tmp/ltx-joint-connector-alignment-tests.log`), including
FP64 finite-difference gradient checking and nonzero finite input gradients
through a frozen real LTX attention module. Ruff passed. Run `010` loaded its
model before this change; a subsequent paired run must verify the full aligned
connector and DiT outputs and training gradients.

## Consecutive updates verified through step 4

Run `010` completed step 4 and all eight checkpoint shards. Consecutive-step
all-rank audits passed for steps 2→3 and 3→4: initial reference is exact, old
matches preceding current, all 384 Gemma and 2,304 DiT LoRA tensors per rank
changed finitely, and optimizer steps are respectively 6 and 8. Reports:
`/tmp/ltx-joint-artifacts/training-010-step3-state-audit.json` and
`/tmp/ltx-joint-artifacts/training-010-step4-state-audit.json`. The generalized
script is `/tmp/audit-ltx-consecutive-checkpoint.py`; these comparisons do not
trace intermediate mini-batch snapshots. Evaluation policy 8 was synchronized
and preserved at `/tmp/ltx-joint-training-010/rollout-policy-8/`; its 32 fixed
samples are still being generated.

Raw console metrics for diagnostic step 3 were extracted to
`/tmp/ltx-joint-artifacts/training-010-step3-metrics.json`. Wall time was
1,254.409 s, actor update 460.830 s, reward 407.157 s. The actor update includes
paired MSProbe writes; these are not steady-state figures. Mean per-sample
scorer call times (including executor queueing, excluding model wake-up) were
video_align 7.194 s, HPSv3 5.895 s, audiobox 1.672 s, CLAP 1.767 s and desync
6.259 s. They overlap and must not be summed as stage wall time. Text→Gemma,
AV→Gemma and AV→DiT local gradient norms remained nonzero; text→DiT was zero.

`/tmp/check-ltx-aligned-connectors.py` is prepared for a targeted actual NPU
forward/backward using the captured full connector input and original frozen
connector weights. It has not yet run; it will run after the ongoing evaluation
releases the devices, followed by a complete joint paired recheck as needed.

Both run-010 training sample batches passed artifact audits: 16 samples each
at policies 5 and 7, all video/audio decoded, valid token/mask/version replay,
finite clean latents and saved forward noise. Host galleries and audits are in
`/tmp/ltx-joint-artifacts/training-010-samples/step-{000003,000004}/`.
`training_sample_summary.json` in that root records five-score ranges/stds.
Each batch has 15/16 EOS-terminated recaptions. Step 3 DeSync is constant;
step 4 DeSync varies (0.47619–0.83333). The source scorer pads short media to
its fixed eight-second input; these 33-frame smoke samples do not establish
longer-clip synchrony quality. Human media review remains pending.

Native pipeline initialization now explicitly freezes base models, connectors,
VAEs and vocoder parameters. Gemma LoRA remains frozen again after native
adapter injection, as before. This makes inference-only parameter intent
explicit; it does not substitute for a before/after VAE weight audit.

Evaluation has entered reward computation on devices 0–4. After checking device
7 at zero utilization and 10% HBM occupancy, the targeted aligned-connector NPU
check was launched there. Its log is `/tmp/ltx-joint-aligned-connectors-npu.log`;
results are pending. The fixed evaluation continues independently.

The targeted full-connector NPU check exited 0. With actual captured inputs and
original frozen checkpoint weights, corrected video and audio outputs and masks
are bitwise equal to native outputs. Backward produces a finite nonzero input
gradient (norm 5.54627e-7); every connector parameter remains frozen with no grad.
Report: `/tmp/ltx-joint-artifacts/training-010-analysis/aligned_connector_npu.json`.
This confirms the full connector forward/backward on NPU for the sampled input,
not yet a new joint Gemma/DiT update. DiT must next be compared with these exact
native connector contexts to isolate its remaining arithmetic discrepancies.

## Run 010 completed; isolated DiT investigation

Run `010` exited 0 after steps 3/4 and fixed evaluation. All 32 policy-8 evaluation
samples passed media/replay audits; the gallery and audit are at
`/tmp/ltx-joint-artifacts/training-010-samples/evaluation/step-000004/`.
All evaluation recaptions ended with EOS. There is still no matched initial-policy
evaluation or human quality review, so no quality improvement/non-regression
claim is made.

Step-4 metrics: `/tmp/ltx-joint-artifacts/training-010-step4-metrics.json`.
The repeated-worker, diagnostics-inactive step took 948.417 s wall time;
actor update 118.876 s and reward 420.051 s. Fixed evaluation took 948.529 s
separately. This is one warmed-process observation before the connector fix,
not a repeated performance distribution.

An isolated actor DiT NPU forward using exactly native connector contexts and
policy-5 LoRA still differs: relative L2 video 0.0081814, audio 0.0083443.
Evidence: `training-010-analysis/dit_native_inputs_npu.json` under the host
artifact root. The first block has tiny projection discrepancies before Q/K
normalization and larger differences after attention. Native DiT RoPE inputs are
BF16 versus actor FP32; all sampled self/cross-modal frequencies become exactly
equal after casting. Reports: `dit_block0_comparison.json`, `dit_rope_inputs.json`.

A temporary diagnostic applies aligned rotation to DiT without changing the
production DiT processor. It still produces relative L2 differences about
0.00863/0.00817, so rotation alignment alone does not resolve DiT consistency.
A further NPU diagnostic with BF16 LoRA (matching native sidecar dtype) is active:
`/tmp/check-ltx-dit-bf16-lora.py`, log `/tmp/ltx-joint-dit-bf16-lora-npu.log`.
This is an inference experiment, not a change to FP32 trainable master weights.

The BF16-LoRA plus aligned-rotation DiT diagnostic exited 0 but still differs
from native (relative L2 video 0.0079267, audio 0.0075436). Both variant reports
are preserved under `training-010-analysis/` in the host artifact root. Neither
variant justifies an acceptance claim; compare their first-block traces next.

Fresh initial-policy fixed evaluation is now launched at
`/tmp/ltx-joint-baseline-001`, log `/tmp/ltx-joint-baseline-001.log`, with
`trainer.val_before_train=true trainer.val_only=true trainer.resume_mode=disable`.
It uses the same four fixed prompts, eight samples each and fixed evaluation
seeds as run `010`. No optimizer update is requested. The initial version-0
transport bundle must be retained while the run is active; baseline results
remain pending.

The BF16-LoRA/aligned-RoPE first-block trace now confirms that the entire audio
self-attention path matches native exactly, including its Q/K/V projections,
normalization and output projection. Video fused-QKV differs sparsely before
normalization; output-projection input mean absolute difference decreases from
9.396e-4 to 2.922e-6, but later layers still accumulate error. Evidence:
`/tmp/ltx-joint-artifacts/training-010-analysis/dit_bf16_lora_block0_comparison.json`.
A targeted NPU fused-versus-separate QKV experiment is prepared at
`/tmp/check-ltx-qkv-rounding.py`; it has not yet run because baseline generation
occupies the devices. No production DiT arithmetic change is made from these
partial results. Baseline version 0 is preserved at
`/tmp/ltx-joint-baseline-001/rollout-policy-0/`.

The targeted NPU QKV experiment exited 0 and establishes the video projection
boundary: concatenated Q/K/V weights in one projection, plus the same BF16 LoRA
updates, match all 3,932,160 captured native output elements exactly. Three
separate projections differ at 646 elements (mean absolute 2.4542e-7, max
0.0625). Report: `training-010-analysis/qkv_rounding_npu.json` in the host
artifact root; script `/tmp/check-ltx-qkv-rounding.py`; log
`/tmp/ltx-joint-qkv-rounding-npu.log`. The result identifies shape-dependent
projection rounding; it does not yet certify later DiT layers. A differentiable
matching projection with FP32 trainable master LoRA weights remains to implement
and validate before the next complete joint run.

The fixed-seed comparison builder is prepared at
`/tmp/build-ltx-quality-comparison.py`. It requires exactly 32 matching
(prompt, seed) pairs across four prompts, writes side-by-side audiovisual media,
recaptions, five raw scores and a blank human-review worksheet. It has not yet
been run against the pending initial-policy evaluation; it does not manufacture
human quality judgments or compare group-normalized probabilities as absolute
quality scores.

## Matched DiT arithmetic verified and integrated

`dit_attention.py` implements differentiable packed video QKV, native LoRA
operand precision and native RoPE via the shared attention processor. LoRA
master parameters and optimizer remain FP32; casting operands is part of the
autograd graph. The complete isolated DiT NPU forward now matches captured
native video and audio predictions bitwise with policy-5 weights and exact
native contexts. Report:
`/tmp/ltx-joint-artifacts/training-010-analysis/dit_matched_arithmetic_npu.json`.
The joint model now installs this processor. Thirty-two model CPU tests passed
in 29.32 s (`/tmp/ltx-joint-dit-integrated-tests.log`), covering all attention
LoRA/input gradients, frozen-base gradients and FP32 master optimizer updates.
Ruff passed. Full FSDP backward still requires the real joint run below.

Initial evaluation `baseline-001` exited 0 and all 32 samples passed the media
and replay audit. The matched comparison is available at
`/tmp/ltx-joint-artifacts/quality-comparison-initial-step4/index.html`, with
five-score summaries and a blank human-review CSV. Across 32 pairs, mean raw
score deltas are audiobox -0.000574, CLAP +0.003540, DeSync 0, HPSv3 -1.04206,
video_align -0.132158. No aggregate probability or reward-based quality claim
is made. Human review of eight cases covering all four prompts has been
requested and is pending.

Run `011` is active, resuming step 4 and targeting step 5 with two prompts ×
eight samples, two mini-batch updates, diagnostics at global step 5, no extra
evaluation. It includes the complete arithmetic corrections, immutable snapshot
checks and independent update-metrics persistence. Log:
`/tmp/ltx-joint-training-011.log`; dump root `/tmp/ltx-joint-probe-011`;
checkpoints `/tmp/ltx-joint-training-011/checkpoints`. Its policy-9 bundle should
be preserved once exported. No successful joint-update result is claimed yet.

## Consolidated regression and pending native freeze audit

The combined 255-test suite passed in 55.94 s with 48 dependency warnings
(`/tmp/ltx-joint-final-regression.log`). This includes all joint pipeline/trainer,
OmniNFT, recaption GRPO, BAGEL pipeline, named rewards, native scorer, HPSv3 and
diffusers dtype tests. The subsequent optional native frozen-component snapshot
RPC passed its mutation/unfreezing regression (1 passed, 32 deselected, 26.58 s;
`/tmp/ltx-joint-frozen-native-test.log`). Actual before/after native hashing remains
pending. The diagnostic does not run during ordinary rollout.

Run 011 policy 9 has been preserved at
`/tmp/ltx-joint-training-011/rollout-policy-9`. Its generation completed and reward
initialization is in progress; corrected integrated training is not yet verified.
A resumed-export audit and an all-rank step-4 frozen-base audit are running.
The README now separates current requirements/evidence from this historical log.

The step-4 frozen-weight audit completed: every rank's 1,066 Gemma, 4,186 DiT
and 262 connector tensors exactly match the original checkpoint cast to actor
dtype, with no missing mappings or mismatches. Report:
`/tmp/ltx-joint-artifacts/training-010-step4-frozen-weights-audit.json`.
Run 011's policy-9 export also exactly matches the restored step-4 LoRA pair
(384 Gemma and 2,304 DiT tensors on every rank):
`/tmp/ltx-joint-artifacts/training-011-resume-export-audit.json`.

The shared-code regression command covered BAGEL trainer, diffusion configuration,
FSDP ignored subtrees and merged-LoRA behavior: 63 passed, 21 warnings, 29.52 s
(`/tmp/ltx-joint-shared-regression.log`). The standalone rollout probe now accepts
`--policy-bundle` so native frozen-component hashing can use a complete trained
pair. Its real execution is waiting for the active distributed job to release
the devices. Ruff passes for the changed probe.

## Run 011 succeeded: integrated correction and paired diagnosis

The driver exited zero. Both mini-batches updated the paired LoRAs and produced
step-5 checkpoint shards. All eight rank audits report optimizer step ten,
384 Gemma and 2,304 DiT LoRA tensors changed and finite, old equal to restored
step-4 current, and initial ref preserved. Raw update metrics verify all 16
snapshot checks, both Gemma gradient sources, DiT AV gradients and zero
nonfinite gradient elements.

Actual paired captures use actor PID 2395584 and native PID 2397412, policy 9,
seed 186 and timestep 909. Gemma stacked embeddings, both connector conditions
and both DiT predictions are bitwise equal. The fixed analysis mapped 2,715
module paths. Its four RoPE parameter-sync suspects have empty parameter maps;
all actual outputs are equal after native BF16 materialization. The checked
agent report is `/tmp/ltx-joint-artifacts/training-011-analysis/output_5_root_cause_report.md`.
Text replay has 165 tokens, mean absolute error 3.570285e-6, p99 4.377842e-5,
maximum ratio error 6.687418e-5, and no clipping violation at 0.01.

All 16 media/replay samples passed the artifact audit. The diagnostic wall time
is 1,181.669 s (training 316.029 s, reward 484.746 s), with actual raw metrics
saved before checkpointing. This includes dumps and is not ordinary throughput.

The updated standalone native decoder audit is running on device 7 using the
complete policy-9 pair, two 30-step generations and 256-token recaptions:
`/tmp/ltx-joint-native-frozen-001.log`. An all-rank step-5 frozen-base audit is
also running. A subsequent corrected-policy rollout/update and final fixed
evaluation remain necessary; human review is still pending.

## Native decoder audit passed; run 012 launched

The native freeze probe exited zero. With the full policy-9 adapter pair, all
connectors, video VAE, audio VAE and vocoder parameter hashes remain identical
across two complete 30-step generations. Parameters have gradients disabled and
no accumulated gradients. Artifacts: `/tmp/ltx-joint-artifacts/ltx-joint-native-frozen-001/`.
The all-rank step-5 frozen Gemma/DiT/connectors audit also passed exactly against
original weights: `/tmp/ltx-joint-artifacts/training-011-frozen-weights-audit.json`.

The reproducible four-prompt dataset builder produces records exactly matching
the experiment's train/val parquet files. These sets intentionally overlap and
do not measure generalization. Run 012 resumes step 5 for two ordinary rounds
(steps 6 and 7), with test frequency 7 and no MSProbe environment enabled:
`/tmp/ltx-joint-training-012.log`. The same local source files are used.

## User accepted the existing training duration; run 012 stopped

On 2026-10-13 the user stated that the existing training progress was sufficient
for verification and further training rounds were unnecessary. Run 012 was
stopped during native-worker initialization before a new optimizer update or
evaluation. SIGINT was sent only to the driver verified by its command and
LTX_OUTPUT_DIR; its 86 owned descendants were recorded before stopping. All
recorded processes are gone or terminal zombies, with PID creation times checked
to avoid unrelated processes. There are no new checkpoint/sample indexes in
run 012. Five completed rounds and ten optimizer updates remain the evidence
base; no additional training is scheduled. Human quality review remains separate.

## 2026-10-13：按最终用户范围收尾

用户接受已有训练长度，要求 GPT 模型检查，并最终免去声音检验。已完成 8 对、96 张抽帧与 recaption 的 GPT 抽查，报告 `/tmp/ltx-joint-artifacts/gpt-review/REPORT.md`。未见普遍画面崩坏，但有已有任务偏差；音频语义、同步未验证，不宣称质量提升。当前交付按调整后的范围完成，未追加训练、未提交 commit/PR。宿主机最新轮次的 16 段带音轨视频位于 `/tmp/ltx-joint-artifacts/training-011-samples/step-000005/`；该轮 rollout 使用更新前的 old 策略，不能误称为 step-5 更新后固定评估。固定对照成片仍为初始与 step 4。
