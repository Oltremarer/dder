# Pro Verdict For R2V-Replay Round 1 Revision 4

Date: 2026-06-09

Source: Chrome ChatGPT Pro conversation `https://chatgpt.com/c/6a282a75-8ad0-83ea-a4ed-2205e8fffe90`

## Verdict

`go_to_full_5k`

Rationale:

> v3a is the first compact, real-dynamics variant passing H2 across seeds while retaining enough zero-reward precursors for a meaningful H3 selector test. Do not do more density-only search before the full 5k diagnostic.

## Decision

- Run the full 5k diagnostic only on `level0_sparse_grid_hybrid_v3a.yaml`.
- Use seeds `0`, `1`, and `2`.
- Keep the main rarity scorer as `kNN / obs_action_next`.
- Keep diffusion frozen as a failed lightweight baseline.
- Do not run 20k yet.
- Do not enter Level 1 RL yet.
- Do not full-diagnose v3b/v3c/v3d/v3e/v3f unless Pro asks later.

## Required Ubuntu Commands

For each seed:

```bash
conda run -n c2t python experiments/r2v_replay/scripts/build_sparse_grid_replay.py \
  --config experiments/r2v_replay/configs/level0_sparse_grid_hybrid_v3a.yaml \
  --seed ${seed} \
  --n-transitions 5000 \
  --output experiments/r2v_replay/outputs/revision5_v3a_seed${seed}_full/replay.npz

conda run -n c2t python experiments/r2v_replay/scripts/run_level0_diagnostic.py \
  --dataset experiments/r2v_replay/outputs/revision5_v3a_seed${seed}_full/replay.npz \
  --config experiments/r2v_replay/configs/level0_sparse_grid_hybrid_v3a.yaml \
  --rarity-input obs_action_next \
  --rare-topk-ratio 0.10 \
  --selected-budget-ratio 0.05 \
  --seed ${seed} \
  --output-dir experiments/r2v_replay/outputs/revision5_v3a_seed${seed}_full/diagnostic_obs_action_next
```

## Required Output Files

- `metrics.csv`
- `metric_consistency_audit.csv`
- `candidate_funnel_summary.csv`
- `selection_funnel_by_baseline.csv`
- `risk_rejection_summary.csv`
- `top_selected_examples.csv`
- `selected_rare_useless.csv`
- `rejected_valuable_precursors.csv`

## Pass Criteria

H2:

- Rare-useless enrichment >= 2 for all seeds.
- kNN Top10 rare-useless count >= 20 for all seeds.
- kNN Top10 rare-useless fraction >= 0.04 for all seeds.

H3:

- R2V full selected rare-useless fraction < kNN Top10 rare-useless fraction.
- R2V selected rare-useless should preferably be near 0.
- R2V zero-precursor recall > reward-only.
- R2V positive-reward fraction <= 0.50.
- R2V selects nonzero zero-reward precursors for all seeds.

H4:

- Combined invalid-risk AUROC >= 0.85.
- False rejection <= 0.30 if implemented.

H1:

- Weak-pass is acceptable if H2/H3 are strong.
- Preferred: Recall@Top10 rare_valuable >= 0.50.
- Preferred: Recall@Top20 rare_valuable >= 0.70.

## Contingency Rules

- If v3a fails only because zero-precursor counts are low, next minimal revision is `v3a_plus` with `near_success=0.12`, while keeping grid 21, pockets 12, probe 0.10, and noisy_goal 0.20.
- If v3a passes H2 but R2V cannot remove rare-useless while retaining precursors, treat it as a method/selector problem.
- If H2 disappears in full diagnostic, treat the density micro-check as non-predictive and reconsider the H2 route.

## Forbidden

- Do not run 20k.
- Do not enter Level 1 RL.
- Do not implement RL frameworks, PER, generation, or diffusion changes.
- Do not use `full_transition` as main evidence.
- Do not use reward, done, or labels in the main rarity scorer.
- Do not mix optional_invalid into H1-H3.
- Do not sample transitions by label or use impossible dynamics.

## Next Checkpoint

Use `R2V-Replay Round 1 Revision 5 v3a Full-5k Diagnostic Report`.
