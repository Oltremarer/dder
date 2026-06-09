# R2V-Replay Round 1 Revision 8 Two-Branch-v4 Report

## Context

- Workspace: `/Users/azure/Documents/adg改`
- Repo: `https://github.com/Oltremarer/dder`
- Branch: `main`
- Ubuntu env: `Ubuntu-Tailscale:~/dder`, `~/miniconda3/envs/c2t/bin/python`
- Base pushed commit before v4: `7550843 Add candidate ratio H2 sweep`
- Config: `experiments/r2v_replay/configs/level0_sparse_grid_two_branch_v4.yaml`
- Mac local experiments: not run
- Level 1 RL: not run
- 20k: not run

## Code Changes

Implemented a new `two_branch_v4` Level 0 design:

- Added `SparseGridConfig.two_branch_v4()`.
- Added `valuable_branch_states` and `useless_branch_states` metadata.
- Added `rare_valuable_branch_probe` behavior policy.
- Reused `rare_distractor_probe` for a real reachable useless branch.
- Added `level0_sparse_grid_two_branch_v4.yaml`.
- Added `--k` to `run_density_audit.py` to match Pro's command.
- Added tests for v4 branch reachability and eval-only labels.
- Saved Pro Revision 7 verdict.

No selector, risk scorer, consistency scorer, diffusion scorer, or Level 1 RL code was changed.

## v4 Environment Design

The grid has two explicit real branches from a shared start/split state:

- Rare valuable branch: a real path from the split state to the goal.
- Rare useless branch: a real path from the split state to a dead-end pocket away from the goal.

Behavior policies:

- `rare_valuable_branch_probe`: low-frequency goal-reaching branch probe.
- `rare_distractor_probe`: low-frequency dead-end branch probe.
- `near_success`, `noisy_goal`, and `random_wander` remain ordinary behavior sources.

Why this is not label hacking:

- Branch transitions are produced by the environment step function.
- There is no manual transition sampling by label.
- Rare useless transitions are real, reachable, reward-zero, return-zero, dynamics-valid, and non-progress.
- Optional invalid transitions are still separate stress samples and excluded from H1-H3.

## Ubuntu Verification

Sparse-grid tests:

- `~/miniconda3/envs/c2t/bin/python -m pytest -q ~/dder/experiments/r2v_replay/tests/test_sparse_grid.py`
- Result: `5 passed in 0.37s`

Full r2v tests:

- `~/miniconda3/envs/c2t/bin/python -m pytest -q ~/dder/experiments/r2v_replay/tests`
- Result: `13 passed, 3 warnings in 1.19s`

Warnings were the same pandas deprecation and sklearn MLP convergence warnings seen in prior rounds.

## Dataset Composition

5k real transitions plus 150 optional invalid stress transitions per seed.

| Seed | Common zero | Rare valuable zero precursor | Rare useless | Rare valuable positive | Optional invalid |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 4469 | 353 | 142 | 36 | 150 |
| 1 | 4548 | 291 | 130 | 31 | 150 |
| 2 | 4406 | 364 | 192 | 38 | 150 |

Counts are adequate at the dataset level. The failure is in kNN rarity ranking, not in raw label availability.

## Density Audit

| Seed | Label | Count | Unique transition | Mean multiplicity | Median kNN distance | Mean kNN score | Top10 count | Top10 fraction | Top10 enrichment |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | common_zero | 4469 | 87 | 87.954 | 0.000 | 0.091 | 425 | 0.850 | 0.951 |
| 0 | rare_useless | 142 | 18 | 94.430 | 0.000 | 0.012 | 1 | 0.002 | 0.070 |
| 0 | rare_valuable_positive | 36 | 1 | 36.000 | 0.528 | 0.528 | 36 | 0.072 | 10.000 |
| 0 | rare_valuable_zero_precursor | 353 | 24 | 78.864 | 0.000 | 0.140 | 38 | 0.076 | 1.076 |
| 1 | common_zero | 4548 | 87 | 100.538 | 0.000 | 0.097 | 363 | 0.726 | 0.798 |
| 1 | rare_useless | 130 | 17 | 106.115 | 0.000 | 0.010 | 2 | 0.004 | 0.154 |
| 1 | rare_valuable_positive | 31 | 1 | 31.000 | 0.959 | 0.959 | 31 | 0.062 | 10.000 |
| 1 | rare_valuable_zero_precursor | 291 | 24 | 68.904 | 0.000 | 0.266 | 104 | 0.208 | 3.574 |
| 2 | common_zero | 4406 | 85 | 105.171 | 0.000 | 0.093 | 360 | 0.720 | 0.817 |
| 2 | rare_useless | 192 | 15 | 153.266 | 0.000 | 0.005 | 1 | 0.002 | 0.052 |
| 2 | rare_valuable_positive | 38 | 1 | 38.000 | 0.551 | 0.551 | 38 | 0.076 | 10.000 |
| 2 | rare_valuable_zero_precursor | 364 | 24 | 65.014 | 0.000 | 0.281 | 101 | 0.202 | 2.775 |

Density interpretation:

- v4 rare-useless transitions are too repetitive.
- Median kNN distance is `0.0` for rare useless in all seeds.
- Mean transition multiplicity for rare useless is high: `94.4`, `106.1`, `153.3`.
- kNN therefore treats rare-useless branch transitions as dense/common, not rare.

## Minimal Full5k Diagnostic

Run because density audit does not emit a score table for candidate-ratio sweep and Pro allowed a minimal diagnostic if needed.

| Seed | H1 Top10 | H1 Top20 | H2 rare-useless count | H2 rare-useless fraction | H2 enrichment | H2 valuable precision | H3 zero full | H3 rare-useless full | R2V zero recall | Reward-only zero recall | R2V positive fraction | H4 AUROC | H4 false rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.190 | 0.406 | 1 | 0.002 | 0.073 | 0.148 | 10 | 0 | 0.028 | 0.000 | 0.000 | 1.000 | 0.579 |
| 1 | 0.413 | 0.472 | 2 | 0.004 | 0.158 | 0.266 | 0 | 0 | 0.000 | 0.000 | 0.800 | 0.999 | 1.000 |
| 2 | 0.346 | 0.480 | 1 | 0.002 | 0.054 | 0.278 | 3 | 0 | 0.008 | 0.000 | 0.560 | 1.000 | 0.864 |

Full diagnostic status:

- H1 fails: Top10 and Top20 rare valuable recall fail all seeds.
- H2 fails: rare-useless count/fraction/enrichment fail all seeds.
- H3 fails: R2V full selects too few zero precursors and positive reward fraction exceeds `0.50` in seeds 1/2.
- H4 invalid AUROC passes, but false rejection is too high.

## Candidate-Ratio Sweep

| Ratio | Min rare-useless count | Min rare-useless fraction | Min enrichment | Min rare-value all | Min rare-value recall | All H2 pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.005 | 0 | 0.000 | 0.000 | 0 | 0.000 | false |
| 0.010 | 0 | 0.000 | 0.000 | 7 | 0.021 | false |
| 0.020 | 0 | 0.000 | 0.000 | 13 | 0.040 | false |
| 0.050 | 0 | 0.000 | 0.000 | 24 | 0.062 | false |
| 0.100 | 1 | 0.002 | 0.052 | 74 | 0.190 | false |
| 0.200 | 6 | 0.006 | 0.156 | 152 | 0.406 | false |

No candidate ratio passes H2. In fact, small and medium budgets often contain zero rare-useless transitions.

## Comparison To v3a Failure

v3a failure mode:

- kNN finds rare valuable precursors well.
- rare-useless density is not stable enough at 20k.

v4 failure mode:

- dataset contains rare-useless transitions;
- but rare-useless branch transitions are highly repetitive and kNN-dense;
- kNN ranking mostly surfaces common zero and rare valuable positive/precursor transitions;
- H1 also weakens because branch geometry creates repeated low-diversity valuable structures.

So v4 did not solve the v3a H2 problem. It changed the failure mode from "rare-useless diluted at large budget" to "rare-useless not kNN-rare at any budget."

## Codex Recommendation

`stop_level0_or_redesign_after_pro`

Do not go to 20k. Do not patch v4 without Pro approval.

The likely technical cause is clear: a narrow two-branch corridor produces too few unique rare-useless transition types, making the dead-end branch dense under kNN. If Pro wants another revision, it should require a geometry where rare valuable and rare useless are both low-frequency and high-diversity in `obs_action_next` feature space.

## Artifacts On Ubuntu

Density:

- `~/dder/experiments/r2v_replay/outputs/revision8_v4_seed{seed}_density/replay.npz`
- `~/dder/experiments/r2v_replay/outputs/revision8_v4_seed{seed}_density/density_obs_action_next/density_audit_by_label.csv`
- `~/dder/experiments/r2v_replay/outputs/revision8_v4_seed{seed}_density/density_obs_action_next/rare_useless_density_audit.csv`

Full5k diagnostic:

- `~/dder/experiments/r2v_replay/outputs/revision8_v4_seed{seed}_full5k/diagnostic_obs_action_next/metrics.csv`
- `~/dder/experiments/r2v_replay/outputs/revision8_v4_seed{seed}_full5k/diagnostic_obs_action_next/score_table.csv`
- `~/dder/experiments/r2v_replay/outputs/revision8_v4_seed{seed}_full5k/diagnostic_obs_action_next/metric_consistency_audit.csv`

Candidate sweep:

- `~/dder/experiments/r2v_replay/outputs/revision8_v4_seed{seed}_candidate_sweep/candidate_ratio_sweep.csv`
- `~/dder/experiments/r2v_replay/outputs/revision8_v4_seed{seed}_candidate_sweep/candidate_ratio_sweep_summary.csv`
- `~/dder/experiments/r2v_replay/outputs/revision8_v4_seed{seed}_candidate_sweep/h2_budget_curve.csv`

## Next Pro Question

The requested two-branch-v4 design was implemented and verified as real dynamics, but it failed H1/H2/H3. Should Codex:

1. stop Level 0 for this route,
2. design one more high-diversity two-branch environment,
3. run a targeted geometry/metric audit before any new design,
4. abandon kNN rarity as the Level 0 official scorer, or
5. return to the literature/story framing instead of more synthetic environment engineering?
