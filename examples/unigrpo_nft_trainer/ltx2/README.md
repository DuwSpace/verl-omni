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

## Joint tree ablation

```bash
bash examples/unigrpo_nft_trainer/ltx2/start_train.sh ablation=joint_tree
```

This preset trains both AR and AV with 4 original prompts per batch, 8
recaptions per prompt and 8 videos per recaption (`rollout.n=64`). Each batch
contains 32 recaptions and 256 videos. The actor mini-batch is 4 prompts;
per-device micro-batching and validation settings inherit `joint_train.yaml`.
