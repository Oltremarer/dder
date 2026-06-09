# Revision 9 Geometry Metric Audit Report

Date: 2026-06-09

Remote execution host: `Ubuntu-Tailscale`

Code base: `/home/chenyuyang/dder`

Local editing host: macOS only for file edits/git/Chrome. No local Mac experiments were run.

## Scope

This revision implements the Pro-requested diagnostic audit after Revision 8. It does not change the official scorer, selector, risk scorer, consistency scorer, environment config, or sparse-grid design.

Compared outputs:

- v4 5k: `experiments/r2v_replay/outputs/revision8_v4_seed{0,1,2}_*`
- v3a 20k: `experiments/r2v_replay/outputs/revision6_v3a_seed{0,1,2}_20k`

New audit outputs per seed:

- `geometry_density_summary.csv`
- `duplicate_multiplicity_by_label.csv`
- `unique_transition_summary.csv`
- `knn_sensitivity_by_k.csv`
- `candidate_ratio_by_metric_variant.csv`
- `raw_vs_unique_knn_comparison.csv`
- `distance_to_common_zero_summary.csv`
- `geometry_audit_interpretation.md`

## Ubuntu Verification

```bash
cd ~/dder
~/miniconda3/envs/c2t/bin/python -m pytest -q experiments/r2v_replay/tests/test_geometry_audit.py
```

Result:

```text
3 passed in 0.68s
```

```bash
cd ~/dder
~/miniconda3/envs/c2t/bin/python -m pytest -q experiments/r2v_replay/tests
```

Result:

```text
16 passed, 3 warnings in 0.99s
```

## Density And Multiplicity Summary

### v4 5k

| seed | rare_useless count | rare_useless unique obs_action_next | rare_useless mean multiplicity | rare_useless median kNN distance | rare_valuable_zero_precursor count | precursor unique obs_action_next | precursor mean multiplicity | positive count | positive unique obs_action_next |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 142 | 18 | 94.430 | 0.000000 | 353 | 24 | 78.864 | 36 | 1 |
| 1 | 130 | 17 | 106.115 | 0.000000 | 291 | 24 | 68.904 | 31 | 1 |
| 2 | 192 | 15 | 153.266 | 0.000000 | 364 | 24 | 65.014 | 38 | 1 |

v4 rare-useless transitions are present, but they collapse into only 15-18 unique `obs_action_next` keys per seed. Their median kNN distance is always zero. Duplicate-aware variants therefore have little useful geometry to recover: the branch is label-valid but still dense/collapsed under the audit feature.

### v3a 20k

| seed | rare_useless count | rare_useless unique obs_action_next | rare_useless mean multiplicity | rare_useless median kNN distance | rare_valuable_zero_precursor count | precursor unique obs_action_next | precursor mean multiplicity | positive count | positive unique obs_action_next |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 197 | 48 | 45.096 | 0.107555 | 348 | 69 | 11.647 | 39 | 3 |
| 1 | 293 | 52 | 47.631 | 0.032084 | 409 | 80 | 13.171 | 45 | 4 |
| 2 | 201 | 42 | 41.154 | 0.000000 | 289 | 77 | 8.931 | 33 | 4 |

v3a has more rare-useless diversity than v4, and rare-useless is less collapsed than v4, but it is still much more duplicated than valuable precursors. This makes H2 partly metric-sensitive but still not robust across seeds.

## H2 Budget Outcome By Metric Variant

H2 rule used here is the prior candidate-ratio rule:

- eligible ratio in `{0.005, 0.01, 0.02, 0.05}`
- `rare_useless_count >= 10`
- `rare_useless_fraction >= 0.04`
- `rare_useless_enrichment >= 2.0`
- `rare_valuable_all_count >= 50` or `rare_valuable_recall >= 0.30`

### v4 5k

No metric variant passes H2 on any seed.

At candidate ratio `0.10`, v4 still has near-zero rare-useless candidate supply:

| seed | variant | rare_useless count | rare_useless fraction | rare_useless enrichment | rare_valuable_all count | rare_valuable recall | H2 pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | raw_diagnostic | 1 | 0.002 | 0.070 | 74 | 0.190 | false |
| 0 | raw_k100 | 0 | 0.000 | 0.000 | 113 | 0.290 | false |
| 0 | unique_transition_k100 | 0 | 0.000 | 0.000 | 172 | 0.442 | false |
| 0 | duplicate_capped5_k100 | 0 | 0.000 | 0.000 | 152 | 0.391 | false |
| 1 | raw_diagnostic | 2 | 0.004 | 0.154 | 133 | 0.413 | false |
| 1 | raw_k100 | 0 | 0.000 | 0.000 | 162 | 0.503 | false |
| 1 | unique_transition_k100 | 0 | 0.000 | 0.000 | 167 | 0.519 | false |
| 1 | duplicate_capped5_k100 | 0 | 0.000 | 0.000 | 167 | 0.519 | false |
| 2 | raw_diagnostic | 1 | 0.002 | 0.052 | 139 | 0.346 | false |
| 2 | raw_k100 | 0 | 0.000 | 0.000 | 217 | 0.540 | false |
| 2 | unique_transition_k100 | 0 | 0.000 | 0.000 | 220 | 0.547 | false |
| 2 | duplicate_capped5_k100 | 0 | 0.000 | 0.000 | 217 | 0.540 | false |

Interpretation: duplicate-aware and unique-transition scoring improve valuable recall, but they do not surface rare-useless in v4. This is stronger evidence for a geometry/design failure than a simple raw-kNN duplicate artifact.

### v3a 20k

v3a has seed-level H2 passes under some variants, but not all seeds:

| seed | variant | ratio | rare_useless count | rare_useless fraction | rare_useless enrichment | rare_valuable_all count | rare_valuable recall | H2 pass |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | duplicate_capped5_k100 | 0.02 | 16 | 0.040 | 2.730 | 53 | 0.117 | true |
| 2 | raw_k100 | 0.02 | 32 | 0.080 | 7.960 | 78 | 0.242 | true |
| 2 | duplicate_capped5_k100 | 0.05 | 40 | 0.040 | 3.980 | 112 | 0.348 | true |

Seed 0 remains below the H2 rule under all variants. Its closest rows pass only two of four non-ratio components, mostly missing rare-useless fraction and valuable supply at the same eligible budget.

Interpretation: v3a H2 failure is partly metric-sensitive, unlike v4, but still not robust enough for a seed-level claim.

## Distance Geometry

### v4

Rare-useless transitions have zero distance to common-zero transitions in all seeds:

| seed | source label | median distance to common_zero | median distance to rare_valuable | median distance to rare_useless |
|---:|---|---:|---:|---:|
| 0 | rare_useless | 0.000 | 1.971 | 0.000 |
| 1 | rare_useless | 0.000 | 2.164 | 0.000 |
| 2 | rare_useless | 0.000 | 2.008 | 0.000 |

The same zero common-zero distance also appears for most valuable precursors. This means `obs_action_next` geometry is not separating common-zero, valuable zero-precursors, and rare-useless cleanly in v4.

### v3a

v3a rare-useless is also close to common-zero, but not as completely collapsed:

| seed | source label | median distance to common_zero | median distance to rare_valuable | median distance to rare_useless |
|---:|---|---:|---:|---:|
| 0 | rare_useless | 0.000 | 3.287 | 0.000 |
| 1 | rare_useless | 0.000 | 3.481 | 0.000 |
| 2 | rare_useless | 0.000 | 3.304 | 0.000 |

Valuable precursors in v3a have nonzero median distance to common-zero (`0.282-0.402`) while still being zero-distance to the rare-valuable group. This explains why v3a can show stronger H1/H3 behavior than v4 while still failing H2 robustness.

## Answers To Pro Questions

1. Are rare-useless transitions duplicated more heavily than rare-valuable precursors?

Yes for v3a; rare-useless mean multiplicity is `41-48` versus precursor `9-13`. For v4, rare-useless is also heavily duplicated (`94-153`), but valuable precursors are also highly duplicated (`65-79`), so the more important v4 issue is feature collapse rather than only relative multiplicity.

2. Does unique-transition kNN make rare-useless appear in candidate pools?

No for v4. At ratio `0.10`, unique-transition kNN selects `0` rare-useless in all v4 seeds. For v3a, unique-transition kNN can select some rare-useless, but it does not produce any H2 pass.

3. Does duplicate-capped kNN make H2 pass without destroying H1?

For v4, no. Duplicate-capped kNN improves rare-valuable recall but selects `0` rare-useless at ratio `0.10` in all seeds. For v3a, duplicate-capped kNN makes H2 pass on seeds 1 and 2, but not seed 0, so it is not enough for a robust all-seed result.

4. Does v3a fail H2 for the same reason as v4, or a different reason?

Different but related. v4 fails because rare-useless and common-zero collapse under `obs_action_next`, and duplicate-aware scoring does not recover rare-useless. v3a fails because rare-useless is duplicated and metric-sensitive, but there is enough geometry that some metric variants can pass some seeds.

5. Is raw kNN unsuitable as the official Level 0 scorer?

Raw kNN is fragile for this synthetic grid route. The audit does not justify replacing the official scorer yet, but it shows raw kNN alone is not reliable evidence for H2 in low-diversity discrete grids.

6. Would a high-diversity design plausibly fix H2, or would it just keep chasing geometry?

The audit weakly supports a high-diversity design only if the design explicitly increases unique `obs_action_next` support and separates rare-useless from common-zero without using labels, rewards, or done. v4 itself should not be patched again. If another design is attempted, it should be treated as one final high-diversity geometry test, not another local v4 edit.

## Pro Checkpoint Request

Recommended verdict request:

- `one_more_high_diversity_design` if Pro agrees that v4 is the wrong geometry but the route is still worth one final controlled design.
- `abandon_knn_official_scorer` if Pro treats the raw/unique/capped instability as fatal to kNN as official Level 0 evidence.
- `stop_route` if Pro thinks both v3a and v4 show this line is now over-engineering the synthetic environment.

No Level 1, RL, 20k rerun, scorer replacement, or dataset redesign has been started.
