# R2V-Replay Round 1 Revision 4 Hybrid-H2 Report

## Context

- Workspace: `/Users/azure/Documents/adg改`
- Repo: `https://github.com/Oltremarer/dder`
- Branch: `main`
- Ubuntu env: `Ubuntu-Tailscale:~/dder`, `~/miniconda3/envs/c2t/bin/python`
- Mac local experiments: not run
- 20k run: not run

## Code Changes

- Added `hybrid_v3` layout alias for compact distractor-pocket configs.
- Added hybrid configs:
  - `level0_sparse_grid_hybrid_v3a.yaml`
  - `level0_sparse_grid_hybrid_v3b.yaml`
  - `level0_sparse_grid_hybrid_v3c.yaml`
  - `level0_sparse_grid_hybrid_v3d.yaml`
  - `level0_sparse_grid_hybrid_v3e.yaml`
  - `level0_sparse_grid_hybrid_v3f.yaml`
- Kept `diffusion_failed_baseline` frozen.
- No 20k and no Level 1 RL code was run.

Ubuntu tests:

- `conda run -n c2t python -m pytest -q experiments/r2v_replay/tests`
- Result: `10 passed, 3 warnings in 0.92s`

## Dataset Variants

All variants are 5k real transitions plus optional invalid stress samples.

| Variant | Grid | Pockets | Probe | Near success | k |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3a | 21 | 12 | 0.10 | 0.08 | 50 |
| v3b | 23 | 16 | 0.12 | 0.08 | 50 |
| v3c | 23 | 16 | 0.15 | 0.10 | 50 |
| v3d | 21 | 12 | 0.10 | 0.15 | 50 |
| v3e | 21 | 12 | 0.15 | 0.15 | 50 |
| v3f | 21 | 12 | 0.10 | 0.12 | 50 |

## Density Gate Results

Mean over seeds 0/1/2.

| Variant | rare_useless ratio | Top10 rare_useless count | Top10 rare_useless fraction | rare_useless enrichment | zero precursor mean | zero precursor min | positive mean | positive min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3a | 0.0117 | 30.0 | 0.060 | 5.850 | 80.3 | 58 | 9.0 | 7 |
| v3b | 0.0151 | 26.7 | 0.053 | 4.298 | 35.0 | 17 | 4.0 | 2 |
| v3c | 0.0275 | 32.3 | 0.065 | 2.458 | 48.7 | 32 | 5.7 | 4 |
| v3d | 0.0064 | 6.3 | 0.013 | 1.471 | 108.0 | 79 | 12.0 | 9 |
| v3e | 0.0097 | 15.0 | 0.030 | 3.180 | 106.7 | 45 | 12.0 | 5 |
| v3f | 0.0085 | 7.0 | 0.014 | 1.296 | 109.0 | 62 | 12.0 | 7 |

Pro's v3 gate:

- H2 required: enrichment >= 2.0x, Top10 rare_useless count >= 20 per seed, Top10 rare_useless fraction >= 0.04.
- Precursor required: zero precursor count >= 75 minimum, positive count >= 5 minimum.

## Best And Failure Modes

Best H2:

- `v3a`
- H2 passes by mean and by all-seed Top10 fraction/count/enrichment.
- But precursor preservation fails in seeds 1 and 2: zero precursor counts are 65 and 58.

Best precursor preservation:

- `v3d` / `v3e` / `v3f`
- These preserve or improve precursor counts in most seeds.
- But H2 fails, especially Top10 rare_useless count/fraction.

No variant currently satisfies both gates.

## Codex Recommendation

`revise_dataset_or_h2` leaning `stop_if_next_revision_fails`

Do not run full diagnostic, 20k, or Level 1. The compact hybrid search shows a tradeoff:

- configs that pass H2 weaken rare valuable precursor density;
- configs that preserve precursor density fail H2.

The next Pro decision should be either:

1. Allow one more targeted revision that combines v3a H2 with v3d/e precursor preservation, or
2. Stop the current H2 route because the Level 0 task may not simultaneously support both signals without more artificial environment design.

## Artifacts On Ubuntu

Density outputs:

- `~/dder/experiments/r2v_replay/outputs/revision4_v3{a,b,c,d,e,f}_seed{0,1,2}_density/density_obs_action_next/`

Key files:

- `rare_useless_density_audit.csv`
- `density_audit_by_label.csv`
- `transition_multiplicity_by_label.csv`
- `knn_distance_by_label.csv`
- `density_summary.csv`
