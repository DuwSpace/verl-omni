# Tree rollout

This folder contains the N-recaption × k-video feature. `rollout.n` remains
N*k, while `pipeline.videos_per_recaption` selects k. Flat mode is k=1.

Read the modules in execution order:

1. **requests.py** — validate the shape; assign parent IDs, child indices and
   independent AV seeds; group and order siblings before text generation.
2. **routing.py** — acquire one replica per parent, share it across concurrent
   sibling requests and release it once all children complete (or on failure).
3. **preparation.py** — batch unique parents through Gemma and condition encoding,
   then create k replay records sharing CPU embeddings and AR tokens/logprobs.
4. **advantages.py** — average the k raw reward vectors of each recaption,
   compute text component advantages over those parent means, then broadcast
   to the leaf rows. AV rewards are centered within each recaption's k videos.
   Text and AV keep their respective global/per-group std configuration.
5. **metrics.py** — record reward mean/std across recaptions and the mean video
   reward std within recaptions, before actor sharding.

Integration points:

- `agent_loop/diffusion_agent_loop_tq.py`: calls `tree_sampling_params` when
  dispatching leaf requests; TQ still writes N*k rows per original prompt UID.
- `rollout/stages.py`: retains the existing two-stage barrier and delegates
  replica ownership to `RecaptionLeases`.
- `rollout/adapter.py`: passes `_prepare_text` into `prepare_recaption_tree`,
  stores the returned cache entries, then generates each leaf with native DiT.
- `training/loss.py`: calls `compute_text_advantages` before splitting the actor
  batch, and uses `av_reward_group_ids` for OmniNFT reward preparation. Original
  prompt UIDs are restored afterward for replay and downstream consumers.
- `config.py` / `replay.py`: retain the configuration and replay data definitions.

No model loading, native DiT forward or general TQ transport is duplicated here.
Actor AR forward/backward still processes the repeated replay rows.

## Reward statistics (tree mode only)

For reward component m, first compute each recaption's reward:
`parent_reward[b,j,m] = mean_i reward[b,j,i,m]` over its k videos.

Text: center these parent rewards over the N recaptions of the original prompt.
Normalize with population std computed from parent rewards (all parents in the
batch when `text_global_std=true`, otherwise the N parents of that prompt).
Average the normalized components to get one text advantage per recaption,
then broadcast it to its k replay rows.

AV: center each video's component reward by its own recaption's parent reward.
The existing `algorithm.global_std` chooses full-batch video std or within-
recaption video std. Reward routing and the OmniNFT objective are unchanged.

Flat replay takes the original text and AV preparation paths, with no parent
reward aggregation or recaption grouping.

## W&B training metrics

For each of the five `reward_names`, tree mode additionally logs:

- `critic/tree/recaption/<reward>/mean`: mean of the B*N recaption reward means.
- `critic/tree/recaption/<reward>/std`: population std of those recaption means.
- `critic/tree/within_recaption/<reward>/std_mean`: compute population std over
  each recaption's k video scores, then average over all B*N recaptions.

`metrics.py` computes these before actor micro-batching. The trainer merges
first/second moments across local updates to recover the full-step parent std;
it does not average local parent stds. Internal moments are removed before
logging. Existing video-level reward metrics are unchanged. These metrics use
the reward components consumed by training, before advantage normalization;
they are not necessarily the raw outputs of each reward model. Flat mode emits
none of these tree metrics. Validation retains its existing metric path.
