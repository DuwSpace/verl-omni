# LTX joint-training acceptance record

The goal is complete under the user-adjusted scope: actual joint-training
verification plus GPT visual/recaption spot review. On 2026-10-13 the user
replaced human review with GPT review and then explicitly waived sound checks.
Audio semantics and AV synchronization remain unverified, not passed. The user accepted the existing five training rounds and
ten optimizer updates on 2026-10-13 and requested no further training. Run 012
was stopped before another update; there are no active training jobs from this
experiment. No commit or PR was created.

## Requirement-to-evidence map

Paths in the code column are relative to the repository root. Tests named below
are in `tests/pipelines/` unless a different directory is given. Runtime evidence
is local to `/tmp/ltx-joint-artifacts/`; preserve it before cleaning the machine.

| Requirement | Implementation | Targeted verification | Actual evidence / limit |
| --- | --- | --- | --- |
| LTX-2.3 and its Gemma; shared recaption/conditioning LoRA | `verl_omni/pipelines/ltx2_unigrpo_nft/model.py`, `vllm_omni_rollout_adapter.py` | `test_ltx2_joint_model_on_cpu.py`: teacher forcing, worker sampling, conditional gradients | Native runs 007/010/011; exact policy-9 paired embeddings |
| Only Gemma and DiT LoRA train | `model.py`, `policy_state.py`, `diffusers_training_adapter.py` in the joint pipeline | Gradient ownership and FP32 master update tests | `training-011-state-audit.json`: both components changed on every rank; `training-011-frozen-weights-audit.json`: all frozen actor tensors exact |
| Frozen connectors retain Gemma gradient path | `LTXJointModel.encode_condition`, `connector_attention.py` | `test_av_gradients_cross_frozen_connector_into_same_gemma_lora`, native-processor gradient tests | Run 011 AV Gemma gradient norm nonzero; frozen connector checkpoint exact |
| Frozen native VAE/audio VAE/vocoder | Joint native pipeline initialization; worker `snapshot_joint_frozen_components` | Native hash mutation/unfreezing test | `ltx-joint-native-frozen-001/frozen_before.json` equals `frozen_after.json` across two actual 30-step generations |
| Complete paired default/old/ref and fixed initial ref on resume | `policy_state.py`, `verl_omni/pipelines/trainable_policy_state.py` | Paired restore, exception, corruption and incomplete-checkpoint tests | Run 011 all eight ranks: old equals step-4 current, ref retained, optimizer step ten; all 16 runtime hash checks pass |
| Atomic paired synchronization and version admission | `weight_sync.py`, `verl_omni/workers/rollout/ltx_joint_backend.py`, `ltx_joint_worker_extension.py` | Sharded export, collective failure, tampering and repeated-load tests | `training-011-resume-export-audit.json`: policy-9 export exactly equals restored step-4 pair |
| All rollout through vLLM-Omni | Native worker `LTXJointPipeline`, backend and joint hooks | Native worker contract and grouped dispatch tests | Real native worker recaption, conditioning, AV generation; recaption uses Transformers generate inside the diffusion worker, not a separate paged-KV AR engine |
| Multiple complete results grouped by original prompt | `verl_omni/trainer/diffusion/ltx_joint_ray_trainer.py`, joint hooks/replay | Group/seed/version dispatch tests | Runs use two prompts × eight samples; audited UIDs, seeds and policy versions |
| Exact replay tokens/logprobs/masks, clean AV latents, seeds and geometry retained | `replay.py`, `artifacts.py`, native pipeline | Replay corruption rejection and real media serialization tests | `training-011-samples/step-000005/artifact_audit.json`: all 16 media and replay records valid |
| Source five scorers and unchanged aggregation/probability map | `verl_omni/utils/reward_score/{videoalign,hpsv3_reward,audiobox,clap,desync,reward_utils}.py`, `OmniNFTLoss`, `rewards.yaml` | OmniNFT numerical tests and final source comparison | `final-source-equivalence.json`: all six files byte-identical, complete reward YAML block identical, OmniNFTLoss AST identical to commit `66595248b0407749eb0a7cccc5107f8642519d2a` |
| Original prompt remains reward task | `rewards.py:reward_sample`, joint hooks | Original-task/rate preservation tests | Persisted samples contain both original prompt and recaption; scorer input uses original prompt |
| Probability is text reward, then normalized within groups | `loss.py:prepare_actor_batch`, `verl_omni/trainer/diffusion/recaption_grpo.py` | Constant/interleaved group and advantage tests | Source has separate video/audio probabilities; AV keeps both unchanged. Their mean is the disclosed scalar text extension, not a source-defined joint probability |
| Clipped GRPO over valid recaption tokens | `recaption_grpo.py`, `LTXJointModel.recaption_logprobs` | Padding, frozen denominator/advantage and teacher-forcing tests | Run 011 text loss/gradient and same-old replay; no old-ratio clipping violations |
| Current/old/ref use same tokens, noise and time, each with own conditioning | `hooks.py:_av_inputs`, `_av`, `forward_backward_batch` | Replay/geometry/probability corruption tests and joint hooks integration | Actual paired metadata verifies policy, token hash, noise, time and geometry; current encoding recomputed in each forward |
| Both backwards and micro-batch accumulation precede one optimizer step | Joint hooks through shared `BaseEngine.train_batch` | `test_joint_hooks_accumulate_both_losses_before_engine_step_and_keep_old_ref`; micro-batch weighting test | Two optimizer steps per 16-sample round; all-rank checkpoint step count ten after five rounds |
| Gemma gets both losses; DiT gets AV loss | `hooks.py`, `diagnostics.py` | Separate loss weights and gradient attribution tests | `training-011-metrics.json`: text Gemma 0.95712, AV Gemma 0.06255, AV DiT 0.04233 mean local norms; text DiT zero; no nonfinite gradients |
| Real wall time, stage/scorer/branch timing, throughput and memory | Joint trainer, hooks, reward loop, training worker metrics | Parallel-stage wall-clock test; metrics persistence test | Run 010 ordinary step and run 011 diagnostic metrics; clocks exclude final evaluation and are not sums of overlapping tasks |
| Same-old text consistency against observed baseline | `recaption_grpo.py:old_policy_logprob_metrics`, replay metadata | Masked logprob/ratio diagnostics and worker sampling tests | `consistency-baseline.json`; worst ratio error across run-011 samples 8.72612e-5, accepted local bound 0.001 versus actual clip 0.01; zero clipping violations |
| Same-weight Gemma/connector/DiT consistency | `consistency_probe.py`, native RPC, aligned attention processors | Pairing rejection and differentiable attention tests | `training-011-boundary-comparison.json`: all five boundaries bitwise equal for actual FSDP/native pair; report in `training-011-analysis/output_5_root_cause_report.md` |
| Updated policy enters later rollout and evaluation | Joint trainer version lifecycle | Fixed evaluation updated-pair/stable-seed/no-optimizer test | Run 010 training policy 7 after step 3, eval policy 8 after step 4; policy 9 in run 011 matches updated step 4. Extra rounds stopped at user request |
| Fixed prompt galleries and raw reward comparison | `artifacts.py`, trainer validation | Decodable media and fixed-seed validation tests | 32 initial/step-4 matched pairs, four prompts; `quality-comparison-initial-step4/index.html` |
| Quality review under the user-adjusted scope | GPT review of 8 initial/step-4 pairs, 96 frames and recaptions | `gpt-review/REPORT.md`; not human review or a held-out benchmark | No widespread visible collapse in the spot check; existing task deviations remain. Audio semantics and AV synchronization waived by the user and unverified. No quality-improvement claim |
| Reproducible configuration, data and known limitations | `joint_train.yaml`, launchers, `prepare_smoke_data.py`, README | Resolved launcher/config checks; exact reproduced parquet comparison | Eight Ascend devices, local model and reward assets. Four-prompt train/validation sets overlap and do not establish generalization |

## Verification commands and results

The main regression command is documented in the README and
`/tmp/ltx-joint-final-regression.log`: 255 passed, 48 warnings, 55.94 seconds.
The subsequent native hash mutation test passed separately: 1 passed,
32 deselected, 26.58 seconds, `/tmp/ltx-joint-frozen-native-test.log`.
The shared-code suite passed 63 tests, 21 warnings, 29.52 seconds:

```bash
python -m pytest -q \
  tests/trainer/diffusion/test_bagel_joint_trainer_on_cpu.py \
  tests/workers/config/test_diffusion_config_on_cpu.py \
  tests/workers/test_diffusers_fsdp_ignored_params_on_cpu.py \
  tests/workers/test_diffusers_fsdp_merged_lora_on_cpu.py
```

The three runs cover 319 distinct cases; earlier overlapping focused runs are
not added again. CPU tests use small models/stand-ins where stated and are not
substitutes for the actual NPU evidence. Source equivalence was rechecked after
the final implementation. Ruff's only finding over the reviewed changed Python
files was the unchanged pre-existing `UP042` enum declaration. `git diff --check`
passes. New data/probe utilities also passed their focused Ruff checks.

## Performance interpretation

Run 010 step 4: 948.417 s full step, 118.876 s training, 420.051 s rewards,
0.01687 samples/s, with an additional 948.529 s evaluation. This is one warmed
observation from before the actor arithmetic alignment, not a final-code stable
performance distribution. Run 011: 1,181.669 s full step, 316.029 s training,
484.746 s rewards, including large diagnostic dumps. Loading, diagnostic overhead
and evaluation are identified separately. The user accepted this training extent;
no new benchmark rounds will be started.

## Handoff and scoped completion

Latest valid checkpoint (container):
`/tmp/ltx-joint-training-011/checkpoints/global_step_5/actor`.
It contains all eight actor/optimizer/state shards. Latest training gallery:
`/tmp/ltx-joint-artifacts/training-011-samples/step-000005/index.html`.
The paired quality comparison is
`/tmp/ltx-joint-artifacts/quality-comparison-initial-step4/index.html`.

The user replaced human review with GPT visual/recaption review and waived
sound checks. The original human review CSV remains blank by design.
The comparison page's cases 1, 2, 9, 10, 17, 18, 25 and 26 cover all four prompts.
Do not fill the reviewer field with an automated judgment. The gallery is from
step 4; corrected step-5 engineering checks do not upgrade that gallery into a
new quality evaluation. No further training is required or scheduled.

### 2026-10-13：按用户要求改用 GPT 模型抽查

用户要求停止增加训练并用 GPT 检查现有样本，替代此前等待人工抽查的执行方式。已由当前 GPT 会话检查初始策略与 step 4 的 8 对样本、96 张抽帧和对应 recaption，报告位于 `/tmp/ltx-joint-artifacts/gpt-review/REPORT.md`。这是非盲定性抽查，未调用独立 GPT API，不是人工验收。抽查未见普遍画面退化，但发现绿色物体替代水滴、暗部不清晰、recaption 时间记法歧义和声画时序矛盾；不能宣称质量提升或完整质量验收通过。

宿主和容器未配置检查到的 OpenAI/Azure API 凭据，GPT 音频语义和音画同步评估仍未完成。最新训练仍为 step 5 / 10 次 optimizer 更新，不再追加训练；本次对照成片为 step 4。原人工检查 CSV 不填写模型意见。

### 最终范围确认

用户随后明确表示“可以不用检验声音了”。本次 Goal 按调整后的范围完成：联合训练机制、冻结与版本语义、一致性、短程运行和 GPT 画面/描述抽查已有证据；音频语义与音画同步标记为用户免验、未验证。64 段音频的 PCM 诊断已经完成，但不替代声音语义评估；没有执行 GPT 音频 API 调用。生成质量存在报告中列出的偏差，不宣称提升、全面无退化或泛化能力。训练仍止于 5 轮、10 次联合更新，没有重启训练。
