# R2V-Replay Round 1 Level 0 Synthetic-Grid Route Closure Report

Date: 2026-06-09

Final GPT Pro verdict: `stop_route`

Repo: `https://github.com/Oltremarer/dder`

Branch: `main`

Ubuntu host: `Ubuntu-Tailscale`

Ubuntu path: `~/dder`

Mac experiment status: no experiments were run for this closure report.

## Final Status

Synthetic-grid Level 0 route: stopped.

R2V-Replay overall idea: not rejected.

Current evidence: useful but not sufficient for a positive Level 0 proof.

Next scientific move: GPT Pro must choose a new empirical route or story-framing route later.

Codex action after this report: documentation and route closure only.

## Evidence Timeline

### Revision 6: v3a 20k Diagnostic

Report:

```text
experiments/r2v_replay/reports/revision6_v3a_20k_report.md
```

Summary:

- H1 was strong: kNN found rare valuable precursors.
- H3 was conditionally useful: R2V could remove selected rare-useless while retaining zero-reward precursors.
- H2 failed at 20k: rare-useless enrichment was not stable across seeds.

Interpretation:

v3a gave partial positive evidence for valuable precursor mining, but the route could not robustly demonstrate the required rare-useless candidate enrichment.

### Revision 7: Candidate-Ratio Sweep

Report:

```text
experiments/r2v_replay/reports/revision7_candidate_ratio_sweep_report.md
```

Summary:

- No all-seed H2-budget pass was found across candidate ratios.
- GPT Pro stopped the v3a patch route and requested a new two-branch design instead of more threshold relaxation.

Interpretation:

H2 was not merely a bad single-budget choice. The seed-level failure persisted across candidate budgets.

### Revision 8: Two-Branch v4

Report:

```text
experiments/r2v_replay/reports/revision8_two_branch_v4_report.md
```

Summary:

- v4 created real reachable rare-useless transitions.
- v4 failed H1/H2/H3.
- rare-useless transitions were present but highly repetitive.
- rare-useless unique transition counts were only 15-18 per seed.

Interpretation:

v4 changed the failure mode from "rare-useless diluted at large budget" to "rare-useless not kNN-rare at any useful budget."

### Revision 9: Geometry Metric Audit

Report:

```text
experiments/r2v_replay/reports/revision9_geometry_metric_audit_report.md
```

Summary:

- v4 rare-useless had zero median distance under `obs_action_next` and collapsed into low unique support.
- duplicate-aware and unique-transition kNN did not rescue v4 H2.
- v3a was partly metric-sensitive but still not robust across seeds.

Interpretation:

Low-dimensional synthetic-grid density rarity is brittle. Raw kNN is fragile, but simply switching to duplicate-aware scoring was not enough to make the route succeed.

### Revision 10: High-Diversity v5 Density Gate

Report:

```text
experiments/r2v_replay/reports/revision10_high_diversity_v5_report.md
```

Summary:

- v5 was the final allowed high-diversity synthetic-grid design.
- Tests passed on Ubuntu: `17 passed, 3 warnings`.
- v5 improved rare-useless/common-zero separation.
- v5 failed density gate on unique support.

Density gate evidence:

| seed | rare_useless count | rare_useless unique obs_action_next | precursor count | precursor unique obs_action_next | rare_useless median distance to common-zero |
|---:|---:|---:|---:|---:|---:|
| 0 | 39 | 11 | 41 | 23 | 0.773281 |
| 1 | 58 | 15 | 59 | 14 | 0.869840 |
| 2 | 39 | 12 | 44 | 24 | 0.966960 |

Pro-required thresholds:

- rare_useless unique `obs_action_next >= 50` per seed: failed
- rare_valuable_zero_precursor unique `obs_action_next >= 50` per seed: failed
- rare_useless median distance to common-zero > 0 for at least 2/3 seeds: passed

Interpretation:

v5 fixed the common-zero collapse piece, but it did not produce enough diverse rare-useless or rare-valuable precursor support under a 5k replay budget.

## Final GPT Pro Decision

`stop_route`

GPT Pro stated:

```text
Synthetic-grid Level 0 route: stopped.
R2V-Replay overall idea: not rejected.
Current evidence: useful but not sufficient for a positive Level 0 proof.
Next scientific move: GPT Pro must choose a new empirical route or story-framing route later.
Codex action now: documentation and route closure only.
```

## Preserved Lesson

Rarity-based candidate discovery in low-dimensional synthetic grids is highly sensitive to representation geometry and transition multiplicity.

The current route could not robustly demonstrate `rare != valuable` at scale without over-engineering the environment.

This does not falsify R2V-Replay as a research idea. It stops the current synthetic-grid route as an official validation path.

## Forbidden Follow-Up Under Current Verdict

- no v6
- no v5 patch
- no v4 patch
- no v3a patch
- no v5 full 5k diagnostic
- no v5 candidate sweep
- no 20k
- no Level 1
- no RL training
- no CleanRL/CORL/D4RL/SynthER/HER/PER/generation
- no selector/risk/consistency/kNN/diffusion tuning
- no claim that the synthetic-grid route succeeded
- no claim that R2V-Replay overall is invalid

## Existing Artifact Paths

Revision 6 v3a 20k:

```text
experiments/r2v_replay/outputs/revision6_v3a_seed{seed}_20k/
```

Revision 7 candidate sweep:

```text
experiments/r2v_replay/outputs/revision7_v3a_seed{seed}_candidate_sweep/
```

Revision 8 v4:

```text
experiments/r2v_replay/outputs/revision8_v4_seed{seed}_density/
experiments/r2v_replay/outputs/revision8_v4_seed{seed}_full5k/
experiments/r2v_replay/outputs/revision8_v4_seed{seed}_candidate_sweep/
```

Revision 9 geometry audit:

```text
experiments/r2v_replay/outputs/revision9_v4_seed{seed}_geometry_audit/
experiments/r2v_replay/outputs/revision9_v3a20k_seed{seed}_geometry_audit/
```

Revision 10 v5 density gate:

```text
experiments/r2v_replay/outputs/revision10_v5_seed{seed}_density/
```

## Codex Recommendation

Honor GPT Pro's `stop_route` verdict.

Do not run further experiments in this route.

Wait for the user and GPT Pro to choose either a new empirical route or a story-framing route.
