# OmniNFT training

This directory contains the staged LTX-2.3 OmniNFT recipe.

## Current boundary

The command is guarded until the LTX adapters and OmniNFT loss are registered.
It reports the missing stage instead of starting model downloads or Ray. Once
those stages are complete, provide a separate validation JSONL with
`VAL_FILE=/path/to/validation.jsonl`.

## Frozen recipe contract

- `actor_rollout_ref.model.algorithm=omni_nft`
- `actor_rollout_ref.model.model_type=omni_nft_model`
- `actor_rollout_ref.actor.diffusion_loss.loss_mode=omni_nft`
- `algorithm.trainer_type=direct_preference`
- `algorithm.sample_source=online`
- `algorithm.paired_preference=false`
- `actor_rollout_ref.rollout.n=8`
- `actor_rollout_ref.rollout.calculate_log_probs=False`
- `actor_rollout_ref.model.policy_state_adapters=[default,old]`
- `actor_rollout_ref.rollout.rollout_adapter=old`

The recipe uses `OmniNFTPromptDataset` for prompt-group records.
