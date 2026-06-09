# R2V-Replay Round 1 Revision 2 Report

## Context

- Workspace: `/Users/azure/Documents/adg改`
- Repo: `https://github.com/Oltremarer/dder`
- Branch: `main`
- Base commit before this report: `572d29c1fc027d0042a2cc01a4019ee1971b0782`
- Ubuntu env: `Ubuntu-Tailscale:~/dder`, `~/miniconda3/envs/c2t/bin/python`
- Mac local experiments: not run
- 20k run: not run

## Code / Metric Audit Changes

Files changed:

- `experiments/r2v_replay/r2v_replay/metric_audit.py`
- `experiments/r2v_replay/r2v_replay/diagnostics.py`
- `experiments/r2v_replay/tests/test_metrics.py`
- `experiments/r2v_replay/reports/pro_revision1_verdict.md`

New tests:

- `tests/test_metrics.py` validates selected vs candidate denominators, optional_invalid exclusion, zero-precursor recall, positive reward fraction, rare_useless fraction, enrichment vs uniform, and episode diversity.

Ubuntu tests:

- `conda run -n c2t python -m pytest -q experiments/r2v_replay/tests`
- Result: `8 passed, 2 warnings in 1.20s`
- Warnings: expected tiny-fixture `MLPRegressor` convergence warnings.

## Metric Consistency Audit

Each seed writes `metric_consistency_audit.csv`.

| Seed | audit rows | all assertions pass | optional_invalid excluded |
| ---: | ---: | --- | --- |
| 0 | 18 | yes | yes |
| 1 | 18 | yes | yes |
| 2 | 18 | yes | yes |

The audit now reports:

- `candidate_count`
- `selected_count`
- real transition count
- total/candidate/selected counts for valuable-positive, zero-precursor, rare_useless, common_zero, optional_invalid
- valuable precision denominator
- zero-precursor recall denominator
- positive-reward fraction denominator
- selected episode diversity
- assertion pass flag

## H1 kNN And Diffusion

Main feature: `obs_action_next`.

| Metric | Mean | Std |
| --- | ---: | ---: |
| kNN AUROC common vs rare_any | 0.734 | 0.068 |
| kNN Recall@Top10 rare_valuable | 0.921 | 0.074 |
| kNN Top10 valuable precision | 0.303 | 0.070 |
| diffusion AUROC common vs rare_any | 0.512 | 0.011 |
| diffusion Top10 valuable precision | 0.024 | 0.002 |

Interpretation:

- kNN remains a strong high-recall candidate finder for rare valuable transitions.
- diffusion remains a failed lightweight baseline.

## H2 Rare != Valuable Candidate Evidence

Pro requested candidate-level evidence that the rarity Top10 contains rare_useless distractors. The new audit shows kNN does not satisfy this.

| Seed | kNN Top10 candidate count | rare_useless count | rare_useless fraction | rare_useless enrichment | valuable precision |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 500 | 0 | 0.000 | 0.000 | 0.382 |
| 1 | 500 | 4 | 0.008 | 0.251 | 0.278 |
| 2 | 500 | 0 | 0.000 | 0.000 | 0.248 |

Mean:

- rare_useless fraction: 0.0027
- rare_useless enrichment: 0.0837x

Pro threshold from previous verdict:

- rare_useless fraction >= 15%, or
- rare_useless enrichment >= 2.0x

H2 result:

- kNN H2 fails.
- kNN candidate pool is too clean; it does not demonstrate the intended rare-useless distractor problem.

## H3 Selection Funnel

| Seed | zero precursors in kNN Top10 | zero precursors selected full | rare_useless in kNN Top10 | rare_useless selected full | positive rewards selected full |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 191 | 21 | 0 | 0 | 0 |
| 1 | 125 | 22 | 4 | 0 | 0 |
| 2 | 116 | 19 | 0 | 0 | 2 |

Reward-only collapse audit:

| Metric | Mean | Std |
| --- | ---: | ---: |
| reward-only zero-precursor recall | 0.000 | 0.000 |
| R2V-kNN zero-precursor recall | 0.145 | 0.037 |
| reward-only positive reward fraction | 0.680 | 0.208 |
| R2V-kNN positive reward fraction | 0.027 | 0.046 |
| R2V-kNN selected episode diversity | 12.000 | 2.000 |

Interpretation:

- H3 is promising after denominator audit.
- R2V-kNN does not collapse to reward-only.
- R2V-kNN selects many zero-reward precursors and almost no positive-reward terminals.
- But because H2 fails, this is not yet a full R2V proof.

## H4 Risk / Consistency

| Metric | Mean | Std |
| --- | ---: | ---: |
| combined invalid-risk AUROC | 1.000 | 0.000 |

H4 remains pass.

## Artifacts

Per seed:

- `~/dder/experiments/r2v_replay/outputs/revision2_seed{0,1,2}_audit/diagnostic_obs_action_next/metrics.csv`
- `metric_consistency_audit.csv`
- `candidate_funnel_summary.csv`
- `selection_funnel_by_baseline.csv`
- `risk_rejection_summary.csv`
- `top_selected_examples.csv`
- `selected_rare_useless.csv`
- `rejected_valuable_precursors.csv`

## Codex Recommendation

`revise`

Do not run 20k yet. The denominator audit and H3 checks passed, but H2 fails under Pro's explicit candidate-level standard. The likely next step is to revise the Level 0 dataset or rarity setup so rare_useless transitions are density-rare enough to enter the kNN candidate pool without artificial label hacking.

Questions for Pro:

1. Should Codex revise the behavior-policy/data-generation side to create true rare-useless density distractors, or should H2 be reframed away from kNN candidate rarity?
2. If revising data generation, should rare_useless be generated by a separate off-route rare policy, a decoy chamber with a narrow bottleneck, or a different low-density transition family?
