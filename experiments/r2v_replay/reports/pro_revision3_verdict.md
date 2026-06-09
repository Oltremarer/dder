# Pro Verdict For R2V-Replay Round 1 Revision 3

Date: 2026-06-09

Source: Chrome ChatGPT Pro conversation `https://chatgpt.com/c/6a282a75-8ad0-83ea-a4ed-2205e8fffe90`

## Verdict

`revise_dataset_or_h2`

Rationale:

> v2f_k50 shows H2 can be made true by enrichment, but it does so by enlarging/scattering the task enough that rare valuable precursor density collapses. This is not a safe official Level 0 config, so do not run 20k yet.

## Decision

- Do not run 20k on v2f_k50.
- Do not stop.
- Revise once more with a hybrid compact v3 config:
  - preserve the original success/precursor geometry,
  - add real low-density rare-useless pockets,
  - avoid making the map so large that success precursors disappear.
- v2f_k50 may be used only as a large-map H2 comparator.

## Required Hybrid v3 Variants

Add configs:

- `level0_sparse_grid_hybrid_v3a.yaml`
- `level0_sparse_grid_hybrid_v3b.yaml`
- `level0_sparse_grid_hybrid_v3c.yaml`

Recommended variants:

- v3a: grid 21, pockets 12, probe ratio 0.10, near_success 0.08, noisy_goal 0.20, random 0.62, k=50
- v3b: grid 23, pockets 16, probe ratio 0.12, near_success 0.08, noisy_goal 0.20, random 0.60, k=50
- v3c: grid 23, pockets 16, probe ratio 0.15, near_success 0.10, noisy_goal 0.20, random 0.55, k=50

## Revised H2 Gate

Required:

- rare_useless enrichment in kNN Top10 >= 2.0x
- kNN Top10 rare_useless count >= 20 per 5k seed
- kNN Top10 rare_useless fraction >= 0.04

Preferred:

- kNN Top10 rare_useless fraction >= 0.10

The older 0.15 fraction target is ideal but no longer mandatory.

## Valuable Precursor Preservation Gate

Per 5k seed:

- rare_valuable_zero_precursor count >= 75 minimum, >=100 preferred
- rare_valuable_positive count >= 5 minimum, >=10 preferred

Reject configs that pass H2 but collapse precursor counts.

## Next Actions

1. Run density audit first for v3a/v3b/v3c seeds 0/1/2.
2. Select the best variant only if density H2 gate passes and precursor counts are preserved.
3. Run full 5k diagnostic only for the selected best variant.
4. Keep diffusion frozen as `diffusion_failed_baseline`.

## Forbidden

- Do not run 20k.
- Do not enter Level 1 RL.
- Do not implement RL frameworks, PER, generation, or diffusion changes.
- Do not use `full_transition` as main evidence.
- Do not use reward, done, or labels in the main rarity scorer.
- Do not mix optional_invalid into H1-H3.
- Do not sample transitions by label or use impossible dynamics.

## Next Checkpoint

Use `R2V-Replay Round 1 Revision 4 Hybrid-H2 Report`.
