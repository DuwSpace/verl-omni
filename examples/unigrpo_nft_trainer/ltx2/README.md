# Tree rollout

Implementation walkthrough: [tree_rollout/README.md](../../../verl_omni/pipelines/ltx2_unigrpo_nft/tree_rollout/README.md).

The staged rollout supports `actor_rollout_ref.rollout.pipeline.rollout_mode=tree`.
`rollout.n` is still the **total video count per original prompt**, so set it to
`N*k`; `pipeline.videos_per_recaption=k` determines the number of independent
videos for each of the `N` sampled recaptions. For example:

```yaml
actor_rollout_ref:
  rollout:
    n: 16  # 8 recaptions * 2 videos
    pipeline:
      rollout_mode: tree
      videos_per_recaption: 2
      recaption_batch_size: 16  # counts unique recaptions, not videos
```

Stage one generates/encodes each recaption once. Sibling videos are routed to
that replica, share the exact AR tokens, rollout logprobs and CPU condition
embeddings, and use distinct AV seeds. Stage two produces all `N*k` leaf videos.
New recaptions use the replica with the fewest in-flight recaption leases;
flat requests use the fewest in-flight requests. This routing remains load-balanced
when `rollout.full_determinism=true`; computation determinism and sample seeds
are unchanged. Routing and batch composition may differ between runs.
Each video's reward remains separate for AV training. Its AV baseline is the
mean of the k videos generated from the same recaption. `algorithm.global_std`
continues to select full-batch video std or within-recaption std.
Text credit first averages the k raw reward vectors per recaption, then computes
`component_advantage_mean` over those N parent reward vectors per prompt.
`text_global_std` selects std over all parent rewards in the batch or within
one original prompt; both the mean and std use parent rewards, not leaf rewards.
Each sibling replay row receives its parent's text advantage; sample-mean
GRPO/KL retains equal weight per recaption. Original prompt UIDs are preserved.
Actor AR forward/backward currently still processes the repeated replay rows.

The default remains `rollout_mode=flat, videos_per_recaption=1`. Validation has
independent `val_kwargs.n` and `val_kwargs.pipeline` settings (flat by default).
In tree mode the total count must be divisible by `k`. To retain N=8 while
increasing k from 1 to 2, increase `rollout.n` from 8 to 16; this doubles AV and
reward work and the actor trajectory count, while retaining 8 AR generations.

## Joint random streams

Flat and tree joint rollout use the legacy Gemma joint seed schedule. For training step `s` (starting
at 1), a generation batch with `M = prompts * rollout.n` videos assigns expanded
row `i` the seed `rollout.seed + (2*s - 1)*M + i`. Validation uses
`rollout.seed + global_prompt_index * val_kwargs.n + session_id` independently of
the training step. These seeds are attached before worker splitting, so dispatch
order cannot change them. They do not control dataset splits or DataLoader order.
Resuming retains the schedule through the restored training step. In tree mode,
each recaption uses its first child's seed; all video leaves have distinct seeds.
AR sampling uses a private generator per recaption, so dispatch and batch order
do not change its random stream.

Training forward noise matches the legacy implementation: video uses
`(replay.seed + 0x564944) % 2**63`, audio uses
`(replay.seed + 0x415544) % 2**63`. Both generate FP32 noise on the clean latent's
device. Model initialization and training timestep selection each use seed 42.
The initialization seed also applies when `full_determinism=false`.

Data files define the train/validation split. The recipe retains `data.seed=null`
and `shuffle=true`, matching the legacy fresh `torch.Generator()` default seed
(67280421310721 in the current container). Epochs advance the sampler state;
validation keeps dataset order. Changing `data.seed` to 42 changes that order.
Resume also needs the saved DataLoader state, not only the restored step.

Matching seeds does not guarantee bitwise-identical model outputs across different
precision, batch shapes, kernels, or LoRA layouts. The old reward workers did not
set an explicit RNG seed; the current deterministic reward workers retain seed 0.
The five native scorers run in evaluation mode, with no explicit random sampling
in their local scoring paths. Historical reward-worker RNG states cannot be
reconstructed from the old config.

## Joint tree ablation

```bash
bash examples/unigrpo_nft_trainer/ltx2/start_train.sh ablation=joint_tree
```

This preset trains both AR and AV with 4 original prompts per batch, 8
recaptions per prompt and 8 videos per recaption (`rollout.n=64`). Each batch
contains 32 recaptions and 256 videos. The actor mini-batch is 4 prompts;
per-device micro-batching and validation settings inherit `joint_train.yaml`.
