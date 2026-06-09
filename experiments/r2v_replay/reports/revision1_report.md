# R2V-Replay Round 1 Revision 1 Report

## Code Changes

- Added feature modes for `state_only`, `obs_action_next`, `obs_action_next_done`, and `full_transition`.
- Reworked the lightweight diffusion-like scorer into K-fold cross-fit denoising residual scoring over noise levels `[0.03, 0.05, 0.10, 0.20, 0.35]`.
- Added rank-normalized diffusion scores and `diffusion_score_components.csv`.
- Added full kNN H1/H2 reporting, R2V no-risk/full-risk variants, funnel CSVs, and risk-rejection CSVs.
- Tightened `rare_useless` to the deep-decoy transition only; seed0/1/2 real rare_useless rates are now 5.68%, 3.28%, and 2.56%.
- Updated utility and selector weights to the Pro revision request.

## Ubuntu Validation

Environment:

- Host: `Ubuntu-Tailscale`
- Repo path: `~/dder`
- Python env: `~/miniconda3/envs/c2t/bin/python`
- Tests: `conda run -n c2t python -m pytest -q experiments/r2v_replay/tests`

Test result:

- `6 passed, 2 warnings in 1.17s`
- Warnings are expected low-epoch `MLPRegressor` convergence warnings in the tiny scorer unit test.

Main diagnostic command shape:

```bash
python experiments/r2v_replay/scripts/build_sparse_grid_replay.py \
  --config experiments/r2v_replay/configs/level0_sparse_grid.yaml \
  --seed <seed> \
  --n-transitions 5000 \
  --output experiments/r2v_replay/outputs/revision1_seed<seed>_main/replay.npz

python experiments/r2v_replay/scripts/run_level0_diagnostic.py \
  --dataset experiments/r2v_replay/outputs/revision1_seed<seed>_main/replay.npz \
  --config experiments/r2v_replay/configs/level0_sparse_grid.yaml \
  --rarity-input obs_action_next \
  --rare-topk-ratio 0.10 \
  --selected-budget-ratio 0.05 \
  --seed <seed> \
  --output-dir experiments/r2v_replay/outputs/revision1_seed<seed>_main/diagnostic_obs_action_next
```

## Main Results

Three seeds, 5k real transitions plus optional invalid stress samples. Main feature mode is `obs_action_next`, which excludes reward, done, and labels.

| Metric | Mean | Std |
| --- | ---: | ---: |
| Diffusion AUROC common-vs-rare_any | 0.512 | 0.011 |
| Diffusion Top10 valuable recall | 0.077 | 0.025 |
| Diffusion Top10 valuable precision | 0.024 | 0.002 |
| Diffusion R2V-full valuable precision | 0.107 | 0.046 |
| kNN AUROC common-vs-rare_any | 0.734 | 0.068 |
| kNN Top10 valuable recall | 0.921 | 0.074 |
| kNN Top10 valuable precision | 0.303 | 0.070 |
| kNN R2V-full valuable precision | 0.853 | 0.023 |
| kNN R2V-full zero-precursor recall | 0.145 | 0.037 |
| kNN R2V-full rare_useless fraction | 0.000 | 0.000 |
| H4 combined invalid-risk AUROC | 1.000 | 0.000 |

Real label counts:

| Seed | common_zero | rare_useless | rare_valuable_zero_precursor | rare_valuable_positive |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 4492 | 284 | 201 | 23 |
| 1 | 4697 | 164 | 125 | 14 |
| 2 | 4736 | 128 | 122 | 14 |

## Seed0 Embedding Ablation

`full_transition` includes reward and is only a leakage check.

| Mode | Diffusion AUROC | Diffusion R2V valuable precision | kNN AUROC | kNN R2V valuable precision | kNN R2V zero recall | kNN R2V rare_useless fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| state_only | 0.517 | 0.16 | 0.667 | 0.64 | 0.080 | 0.00 |
| obs_action_next | 0.501 | 0.08 | 0.663 | 0.84 | 0.104 | 0.00 |
| obs_action_next_done | 0.497 | 0.52 | 0.658 | 0.84 | 0.055 | 0.00 |
| full_transition | 0.493 | 0.72 | 0.658 | 0.84 | 0.055 | 0.00 |

## Interpretation

- Cross-fit diffusion remains weak after the requested repair. It is near-random on H1 and contributes little to H3 unless leakage-like done/reward features are allowed.
- kNN density rarity is stable across seeds and non-leakage embeddings. It identifies most valuable zero-reward precursors in the rare candidate set and R2V-kNN keeps rare_useless at zero in selected subsets.
- Risk scoring works for H4 stress. Combined invalid-risk AUROC is essentially 1.0, and invalid stress samples are not mixed into real replay metrics.
- The current result supports the broader R2V selector frame more than the diffusion-specific scorer frame.

## Codex Recommendation

Do not enter Level 1 RL yet. Recommended next Pro decision:

1. Decide whether to reframe the experiment around density rarity plus utility/risk gating, with diffusion treated as a failed scorer baseline.
2. If diffusion remains required by the idea, request a different diffusion implementation or representation before RL.
3. If kNN/density rarity is acceptable, run 20k seeds for `obs_action_next` only, then consider a minimal Level 1 RL handoff.

## Artifacts On Ubuntu

- Main metrics: `~/dder/experiments/r2v_replay/outputs/revision1_seed{0,1,2}_main/diagnostic_obs_action_next/metrics.csv`
- Score tables: `score_table.csv`
- Diffusion components: `diffusion_score_components.csv`
- Funnel summaries: `candidate_funnel_summary.csv`, `selection_funnel_by_baseline.csv`
- Risk summary: `risk_rejection_summary.csv`
- Failure tables: `top_failure_cases.csv`, `selected_rare_useless.csv`, `rejected_valuable_precursors.csv`
