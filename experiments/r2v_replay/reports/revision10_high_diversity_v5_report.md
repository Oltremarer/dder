# R2V-Replay Round 1 Revision 10 High-Diversity-v5 Report

Date: 2026-06-09

Repo: `https://github.com/Oltremarer/dder`

Branch: `main`

Ubuntu host: `Ubuntu-Tailscale`

Ubuntu path: `~/dder`

Python: `~/miniconda3/envs/c2t/bin/python`

Mac experiment status: no Mac experiments were run. Mac was used only for editing, Git, and Chrome.

## Summary

GPT Pro verdict after Revision 9 was `one_more_high_diversity_design`: implement exactly one final v5 Level 0 synthetic-grid design. If v5 failed density/H1/H2/H3 after one clean implementation, stop the synthetic-grid Level 0 route.

This revision implemented `high_diversity_v5`, updated `AGENTS.md`, saved `pro_revision9_verdict.md`, and ran the Pro-approved 5k density gate on Ubuntu.

Result: **density gate failed**. Full 5k diagnostic and candidate sweep were not run.

## Files Changed

- `AGENTS.md`
- `experiments/r2v_replay/configs/level0_sparse_grid_high_diversity_v5.yaml`
- `experiments/r2v_replay/r2v_replay/sparse_grid.py`
- `experiments/r2v_replay/scripts/run_density_audit.py`
- `experiments/r2v_replay/tests/test_sparse_grid.py`
- `experiments/r2v_replay/reports/pro_revision9_verdict.md`
- `experiments/r2v_replay/reports/revision10_high_diversity_v5_report.md`

No v3a/v4 config was edited. No selector, risk, consistency, diffusion, or RL code was changed.

## v5 Design

New config:

```text
experiments/r2v_replay/configs/level0_sparse_grid_high_diversity_v5.yaml
```

Key design choices:

- layout: `high_diversity_v5`
- grid: `45 x 45`
- distractor pockets: `48`
- rare-distractor episodes start at the selected pocket entry, then enter via real `environment.step`
- distractor pocket visit length: 4-8 steps
- near-success starts sample from 20 candidate states near the goal
- official rarity input remains `obs_action_next`
- no reward, done, full transition, or eval label is used by the official rarity scorer
- optional-invalid samples remain excluded from H1/H2/H3

This was intended to increase unique `obs_action_next` support and separate rare-useless from common-zero.

## Ubuntu Verification

Full tests:

```bash
cd ~/dder
~/miniconda3/envs/c2t/bin/python -m pytest -q experiments/r2v_replay/tests
```

Result:

```text
17 passed, 3 warnings in 0.82s
```

Warnings were the existing pandas deprecation and sklearn convergence warnings.

## Commands Run

Density gate only:

```bash
for seed in 0 1 2; do
  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/build_sparse_grid_replay.py \
    --config experiments/r2v_replay/configs/level0_sparse_grid_high_diversity_v5.yaml \
    --seed ${seed} \
    --n-transitions 5000 \
    --output experiments/r2v_replay/outputs/revision10_v5_seed${seed}_density/replay.npz

  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/run_density_audit.py \
    --dataset experiments/r2v_replay/outputs/revision10_v5_seed${seed}_density/replay.npz \
    --config experiments/r2v_replay/configs/level0_sparse_grid_high_diversity_v5.yaml \
    --rarity-input obs_action_next \
    --k 50 \
    --output-dir experiments/r2v_replay/outputs/revision10_v5_seed${seed}_density/density_obs_action_next
done
```

Not run:

- full 5k diagnostic
- candidate-ratio sweep
- 20k
- Level 1
- RL training

## Dataset Counts

| seed | common_zero | rare_valuable_zero_precursor | rare_useless | rare_valuable_positive | optional_invalid |
|---:|---:|---:|---:|---:|---:|
| 0 | 4913 | 41 | 39 | 7 | 150 |
| 1 | 4876 | 59 | 58 | 7 | 150 |
| 2 | 4912 | 44 | 39 | 5 | 150 |

## Density Gate Table

Required by Pro:

- `rare_useless` unique `obs_action_next >= 50` per seed
- `rare_valuable_zero_precursor` unique `obs_action_next >= 50` per seed
- `rare_useless` mean multiplicity <= 3x precursor mean multiplicity
- `rare_useless` median distance to common-zero > 0 for at least 2/3 seeds

Observed:

| seed | label | count | unique obs_action_next | mean multiplicity | median kNN distance | mean kNN score |
|---:|---|---:|---:|---:|---:|---:|
| 0 | rare_useless | 39 | 11 | 8.872 | 1.019117 | 1.059453 |
| 0 | rare_valuable_zero_precursor | 41 | 23 | 2.122 | 1.295790 | 1.328908 |
| 0 | rare_valuable_positive | 7 | 2 | 4.143 | 1.316163 | 1.443692 |
| 1 | rare_useless | 58 | 15 | 5.345 | 0.942791 | 1.278796 |
| 1 | rare_valuable_zero_precursor | 59 | 14 | 5.271 | 0.526304 | 0.541196 |
| 1 | rare_valuable_positive | 7 | 2 | 5.286 | 0.732501 | 0.899967 |
| 2 | rare_useless | 39 | 12 | 4.128 | 1.671976 | 1.530203 |
| 2 | rare_valuable_zero_precursor | 44 | 24 | 2.386 | 0.976637 | 1.200967 |
| 2 | rare_valuable_positive | 5 | 3 | 2.200 | 1.308139 | 1.599658 |

Density gate status:

| seed | rare_useless unique >= 50 | precursor unique >= 50 | rare_useless mean <= 3x precursor mean | rare_useless median distance to common-zero > 0 |
|---:|---|---|---|---|
| 0 | fail | fail | fail | pass |
| 1 | fail | fail | pass | pass |
| 2 | fail | fail | pass | pass |

Overall density gate: **fail**.

## Common-Zero Distance

v5 did improve the geometry separation piece:

| seed | source label | median distance to common_zero | mean distance to common_zero | p90 distance to common_zero |
|---:|---|---:|---:|---:|
| 0 | rare_useless | 0.773281 | 0.611576 | 1.008446 |
| 1 | rare_useless | 0.869840 | 0.736224 | 0.992405 |
| 2 | rare_useless | 0.966960 | 1.144951 | 1.750482 |

This passes the "distance to common-zero > 0 for at least 2/3 seeds" condition. The failure is unique support and supply, not common-zero collapse.

## Interpretation

v5 fixed the zero-distance collapse seen in v4, but it did not provide enough rare-useless or rare-valuable precursor support at 5k.

The main reason is structural: even with many distractor pockets and short visits, the 5k replay budget plus episode-level policy mixture produced only 39-58 rare-useless transitions and 41-59 valuable zero precursors per seed. Unique `obs_action_next` support stayed far below Pro's density threshold:

- rare-useless unique: 11/15/12
- valuable zero-precursor unique: 23/14/24

Continuing to adjust v5 policy ratios or reset behavior after seeing these results would violate the Pro instruction that v5 is the final clean implementation attempt.

## Codex Recommendation

`stop_route`

Do not design v6. Do not run full 5k diagnostic, candidate sweep, 20k, Level 1, or RL training from this v5 result.

Recommended Pro decision options:

- `stop_route`
- `return_to_story_framing`
- `stop_level0`

## Artifact Paths On Ubuntu

Per seed:

```text
experiments/r2v_replay/outputs/revision10_v5_seed{seed}_density/replay.npz
experiments/r2v_replay/outputs/revision10_v5_seed{seed}_density/density_obs_action_next/density_audit_by_label.csv
experiments/r2v_replay/outputs/revision10_v5_seed{seed}_density/density_obs_action_next/density_summary.csv
experiments/r2v_replay/outputs/revision10_v5_seed{seed}_density/density_obs_action_next/geometry_density_summary.csv
experiments/r2v_replay/outputs/revision10_v5_seed{seed}_density/density_obs_action_next/duplicate_multiplicity_by_label.csv
experiments/r2v_replay/outputs/revision10_v5_seed{seed}_density/density_obs_action_next/distance_to_common_zero_summary.csv
experiments/r2v_replay/outputs/revision10_v5_seed{seed}_density/density_obs_action_next/rare_useless_density_audit.csv
```

## Checkpoint Question For GPT Pro

v5 was the final allowed high-diversity synthetic-grid design. It passed tests and improved rare-useless/common-zero separation, but failed the density gate badly on unique support for both rare-useless and valuable precursors.

Should Codex now:

1. `stop_route`,
2. `return_to_story_framing`, or
3. `stop_level0`?
