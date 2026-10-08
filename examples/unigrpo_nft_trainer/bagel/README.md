# BAGEL UniGRPO thinking + image OmniNFT

Last updated: 10/08/2026

This online, full-weight recipe trains both BAGEL MoT branches with one shared
optimizer: AR-GRPO on sampled thinking tokens and the shared DiffusionNFT
prediction-space objective on final image latents. It is a new hybrid algorithm,
not another run of the standalone UniGRPO image-trajectory reproduction.

## Runtime and data

Run from the repository root in the approved Ascend environment. The launch
script uses `/hub` for the already downloaded BAGEL/PickScore snapshots and
`/accepted-data/pickscore/{train,test}.parquet` for the prepared data. Override
`BAGEL_MODEL_PATH`, `HF_HUB_CACHE`, or the trailing Hydra arguments for other
local paths. The script uses offline caches and clears proxy variables.

```bash
bash examples/unigrpo_nft_trainer/bagel/run_bagel_unigrpo_nft_npu.sh
```

The default is one node with eight actor FSDP2 ranks, eight independent TP=1
native vllm-omni MP workers, and eight accelerator-bound PickScore workers.
The joint trainer interleaves prompt groups across replicas and restores original
sample/seed order before scoring, so long-thinking prompts are not confined to
one card.
`reward.accelerator_workers.enabled=True` is essential: CPU-scheduled reward
actors otherwise all select the same default NPU. Generation always runs in the
native workers; the actor's standalone UniGRPO sampler is not a fallback.

## Key configuration

- `algorithm.trainer_type=unigrpo_nft`, `model.algorithm=unigrpo_nft`, and
  `actor.diffusion_loss.loss_mode=unigrpo_nft` select the hybrid path.
- Eight prompts and `rollout.n=8` produce 64 images per step. Mini-batch size 4
  gives two optimizer updates; per-rank micro-batch size is 1.
- Resolution is 512×512. Native BAGEL's 26 schedule points perform 25 ODE
  updates. Thinking is sampled at temperature 1, with a 1024-token native limit.
  Guidance scales are 1, matching the conditional actor velocity used by NFT.
- `actor.fsdp_config.offload_policy=True` uses FSDP2 CPUOffloadPolicy. Trainable
  master parameters and frozen old/reference shards remain FP32; rollout weights
  are BF16. Offload does not guarantee that every colocated memory layout fits.
- Base/understanding learning rate is `1e-6`; generation experts use `3e-5`.
  The text/image loss weights default to 1/1 and are normalized. AR clip is 0.01;
  image reference penalty is `1.5e-5`.
- `rollout.bagel_joint.*` controls thinking, guidance and bounded weight-sync
  transport. The launch script uses 1024 MB sync buckets by default; override
  `BAGEL_SYNC_BUCKET_MB`. Each tensor is loaded through the native stacked loader,
  and generation is admitted only after the complete manifest commits.
- `BAGEL_CPU_THREADS` defaults to 4. No compile, reduced sampling limit, or
  resolution reduction is used to inflate throughput.

## Replay and resume

Rollout records contain exact primary prompt IDs, cached thinking IDs, sampled
response IDs/log probabilities, clean patchified latents and image positions.
Fixed BOS is excluded from AR credit; terminal EOS is scored without inserting
it into image conditioning. Images and decoded text are not re-encoded to
reconstruct these training inputs.

Each checkpoint includes rank-local initial reference and old-policy state,
alongside model, optimizer, RNG and dataloader state. Missing or incompatible
algorithm state fails closed. Resume a saved run by keeping its output directory:

```bash
bash examples/unigrpo_nft_trainer/bagel/run_bagel_unigrpo_nft_npu.sh \
    trainer.default_local_dir=/repo/reproduction/joint-full/checkpoints \
    trainer.resume_mode=auto trainer.total_training_steps=3
```

`tests/special_e2e/verify_bagel_joint_checkpoint.py` audits every rank's model,
Adam and frozen-policy tensors, update counters and representative changes in
both expert branches. Its `--compare-reference` option verifies that the initial
reference is unchanged across resume. Numerical actor/native equality, convergence
and held-out improvement are separate claims; finite smoke tests do not prove them.

The portable tiny trainer smoke builds local random fixtures and verifies a real
checkpoint resume. It is registered as GPU smoke test 10; on Ascend run it with
`NUM_GPUS=8 BAGEL_JOINT_DEVICE=npu bash tests/special_e2e/run_unigrpo_nft_bagel.sh`.
This small-fixture test is execution evidence, not a performance benchmark.

## Verified full-model run

The approved `verl-omni:npu-a2-main-8d2a9fd` Ascend image completed three
full-model steps on September 30, 2026. The first process saved step 1; a second
process resumed that checkpoint and completed steps 2 and 3. Each step generated
64 images at the full settings above and performed two joint Adam updates.
The full run used real cached PickScore, not the tiny reward fixture.

| Step | Mean PickScore reward | Joint loss | Gradient norm | Total seconds, including checkpoint |
| --- | --- | --- | --- | --- |
| 1 | 0.748976 | 2.326682 | 1.334447 | 1225.29 |
| 2, after resume | 0.727033 | 2.403064 | 1.909936 | 1223.27 |
| 3, warm workers | 0.733676 | 2.758642 | 1.107952 | 1177.03 |

Step 3 spent 943.74 seconds generating and synchronizing, 0.41 seconds scoring,
138.13 seconds updating, and 94.63 seconds saving. Across all eight replicas,
this is 4.07 images/minute for generation including weight synchronization,
or 3.26 images/minute for the complete step including checkpointing. The logged
`perf/throughput` is normalized per accelerator; it is not the global rate.
Different steps contain different prompts, so these timings do not establish a
controlled speedup from interleaving. The pinned native pipeline advertises
`supports_request_batch=False`; increasing `max_num_seqs` is not a supported
request-batching optimization here.

Logs are `reproduction/joint-full-v2.log` and
`reproduction/joint-full-v2-resume.log`; checkpoints are under
`reproduction/joint-full-v2/checkpoints/global_step_{1,2,3}`. Native policy
versions 0, 1 and 2 each synchronized all 1223 actor tensors, approximately
29.2 GB, in 28 buckets. The saved sample metadata and
`joint-full-v2-resume-samples-summary.json` verify that every replica processed
all eight prompt ordinals in each resumed step, with the original seed mapping.
The portable tiny training/resume smoke also passed on all eight NPUs;
CUDA execution has not been tested in this environment.

Audit the final checkpoint and its unchanged initial reference with:

```bash
python tests/special_e2e/verify_bagel_joint_checkpoint.py \
    --checkpoint reproduction/joint-full-v2/checkpoints/global_step_3 \
    --optimizer-updates 6 --policy-version 2 \
    --compare-reference reproduction/joint-full-v2/checkpoints/global_step_1
```

This final audit completed on October 8: every model, Adam, old-policy and
reference-policy tensor on all eight ranks was finite, all Adam counters were 6,
the initial reference matched step 1 at zero tolerance, and representative
understanding and generation expert matrices had both changed. The report is
`reproduction/joint-full-v2-step3-audit.json`. The 114 CPU contract/config tests
also passed again (`reproduction/hybrid-cpu-v7.{log,xml}`).

These are execution and recovery results, not convergence evidence. Rewards
are not monotonically increasing, and no held-out improvement was measured.
Native-rollout versus frozen-actor logprob MAE was 0.1347, 0.1904 and 0.2212;
the AR ratio denominator is recomputed using the frozen actor policy, and
numerical equivalence to the native sampling distribution is not established.
Do not attribute the discrepancy to a specific kernel without paired tensor
evidence, or treat this smoke run as validation of long-run training quality.
