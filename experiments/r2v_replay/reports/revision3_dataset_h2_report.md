# R2V-Replay Round 1 Revision 3 Dataset-H2 Report

## Context

- Workspace: `/Users/azure/Documents/adg改`
- Repo: `https://github.com/Oltremarer/dder`
- Branch: `main`
- Ubuntu env: `Ubuntu-Tailscale:~/dder`, `~/miniconda3/envs/c2t/bin/python`
- Mac local experiments: not run
- 20k run: not run

## Code Changes

- Added distractor-v2 grid generation with scattered off-route pockets.
- Added real `rare_distractor_probe` behavior policy.
- Added strict geometry/outcome-based `rare_useless` rule for distractor pockets.
- Added density audit utilities and `scripts/run_density_audit.py`.
- Added v2 config variants:
  - `level0_sparse_grid_distractor_v2a.yaml`
  - `level0_sparse_grid_distractor_v2b.yaml`
  - `level0_sparse_grid_distractor_v2c.yaml`
  - `level0_sparse_grid_distractor_v2d_probe_rich.yaml`
  - `level0_sparse_grid_distractor_v2e_many_pockets.yaml`
  - `level0_sparse_grid_distractor_v2f_many_pockets.yaml`
  - `level0_sparse_grid_distractor_v2f_k50.yaml`
- Added tests for real distractor-policy rare_useless generation and density audit calculations.

Ubuntu tests:

- `conda run -n c2t python -m pytest -q experiments/r2v_replay/tests`
- Latest relevant test run after distractor changes: sparse/density tests passed.
- Earlier full suite after metric audit: `10 passed, 3 warnings`.

## Dataset Variants

All runs are 5k real transitions plus optional invalid stress samples. No 20k run was executed.

| Variant | Grid | Pockets | Probe ratio | k | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| v2a | 21 | 6 | 0.03 | 10 | Pro baseline low probe |
| v2b | 21 | 8 | 0.04 | 10 | Pro baseline medium probe |
| v2c | 23 | 8 | 0.05 | 10 | Pro baseline high probe |
| v2d | 23 | 8 | 0.20 | 10 | probe-rich micro-check |
| v2e | 31 | 16 | 0.26 | 10 | many-pockets micro-check |
| v2f | 41 | 32 | 0.26 | 10 | more scattered pockets |
| v2f_k50 | 41 | 32 | 0.26 | 50 | radius-density check |

## Density Audit Summary

Mean/std over seeds 0/1/2.

| Variant | rare_useless ratio mean | Top10 rare_useless fraction mean | Top10 rare_useless enrichment mean | rare_useless count mean | zero-precursor count mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2a_k10 | 0.004 | 0.023 | 6.892 | 20.3 | 83.3 |
| v2b_k10 | 0.004 | 0.020 | 5.271 | 20.0 | 118.7 |
| v2c_k10 | 0.006 | 0.032 | 5.517 | 30.0 | 136.0 |
| v2d_k10 | 0.019 | 0.040 | 2.331 | 95.7 | 101.7 |
| v2e_k10 | 0.028 | 0.047 | 1.709 | 140.7 | 57.7 |
| v2f_k10 | 0.024 | 0.049 | 2.004 | 120.0 | 37.7 |
| v2f_k50 | 0.024 | 0.064 | 2.676 | 120.0 | 37.7 |

Best current variant:

- `v2f_k50`
- rare_useless enrichment is stable and passes the `>= 2.0x` criterion.
- Top10 rare_useless fraction remains below the `>= 15%` target.
- rare_useless ratio remains below the 3%-6% target.
- rare valuable zero-precursor count is much lower than the old 15x15 setup, especially in seed0.

## By-Seed v2f_k50

| Seed | rare_useless count | rare_useless ratio | unique transitions | mean multiplicity | Top10 count | Top10 fraction | Top10 enrichment | zero precursor count | positive count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 135 | 0.027 | 51 | 16.50 | 35 | 0.070 | 2.593 | 8 | 1 |
| 1 | 110 | 0.022 | 48 | 6.55 | 33 | 0.066 | 3.000 | 59 | 6 |
| 2 | 115 | 0.023 | 44 | 12.23 | 28 | 0.056 | 2.435 | 46 | 5 |

## Interpretation

- The original Pro-specified v2a/v2b/v2c variants produced real rare_useless distractors, but the rare_useless real ratio was far below 3%-6%.
- Increasing probe ratio and pocket count created more rare_useless transitions without label sampling or teleporting.
- Larger scattered-pocket layouts plus k=50 can satisfy the enrichment branch of H2, but not the Top10 fraction branch.
- Increasing map size and pockets diluted rare valuable precursor counts, which may damage H1/H3 if full diagnostic is run.
- Long pocket dwell time increased rare_useless count but made clusters dense, so it was rejected.

## Codex Recommendation

`revise_dataset_or_h2`

Do not run 20k yet. The best density-only result (`v2f_k50`) partially satisfies H2 via enrichment, but it misses the Top10 fraction target and likely weakens H1/H3 because valuable precursor counts drop. Recommended Pro decision:

1. Accept `v2f_k50` as enough for H2 enrichment and request full diagnostic on v2f_k50, or
2. Revise dataset again to preserve the original valuable precursor density while adding scattered rare-useless pockets, or
3. Stop the H2 route if the required rare_useless fraction cannot be created without harming H1/H3.

## Artifacts On Ubuntu

Density-only outputs:

- `~/dder/experiments/r2v_replay/outputs/revision3b_v2{a,b,c}_seed{0,1,2}_density/density_obs_action_next/`
- `~/dder/experiments/r2v_replay/outputs/revision3d_v2d_seed{0,1,2}_density/density_obs_action_next/`
- `~/dder/experiments/r2v_replay/outputs/revision3e_v2e_seed{0,1,2}_density/density_obs_action_next/`
- `~/dder/experiments/r2v_replay/outputs/revision3f_v2f_seed{0,1,2}_density/density_obs_action_next/`
- `~/dder/experiments/r2v_replay/outputs/revision3f_k50_seed{0,1,2}_density/density_obs_action_next/`

Key files per output:

- `rare_useless_density_audit.csv`
- `density_audit_by_label.csv`
- `transition_multiplicity_by_label.csv`
- `knn_distance_by_label.csv`
- `density_summary.csv`
