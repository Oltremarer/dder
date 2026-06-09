# R2V-Replay Round 1 Revision 5 v3a Full-5k Diagnostic Report

## Context

- Workspace: `/Users/azure/Documents/adg改`
- Repo: `https://github.com/Oltremarer/dder`
- Branch: `main`
- Ubuntu env: `Ubuntu-Tailscale:~/dder`, `~/miniconda3/envs/c2t/bin/python`
- Git commit on Ubuntu: `728eac944f217c0d8ebd6cf677c60c2c89bb3177`
- Config: `experiments/r2v_replay/configs/level0_sparse_grid_hybrid_v3a.yaml`
- Rarity input: `obs_action_next`
- Seeds: `0`, `1`, `2`
- Real transitions per seed: `5000`
- Optional invalid stress transitions per seed: `150`
- Mac local experiments: not run
- 20k run: not run
- Level 1 RL: not run

## Pro Revision 4 Instruction Followed

Pro verdict: `go_to_full_5k`

Executed only:

- `level0_sparse_grid_hybrid_v3a.yaml`
- seeds `0/1/2`
- full 5k diagnostic
- main scorer: `kNN / obs_action_next`
- diffusion kept as failed lightweight baseline

No extra density search, 20k run, Level 1 RL, PER, generation, RL framework integration, or main-scorer reward/done/label leakage was added.

## Ubuntu Commands

For each seed:

```bash
~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/build_sparse_grid_replay.py \
  --config experiments/r2v_replay/configs/level0_sparse_grid_hybrid_v3a.yaml \
  --seed ${seed} \
  --n-transitions 5000 \
  --output experiments/r2v_replay/outputs/revision5_v3a_seed${seed}_full/replay.npz

~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/run_level0_diagnostic.py \
  --dataset experiments/r2v_replay/outputs/revision5_v3a_seed${seed}_full/replay.npz \
  --config experiments/r2v_replay/configs/level0_sparse_grid_hybrid_v3a.yaml \
  --rarity-input obs_action_next \
  --rare-topk-ratio 0.10 \
  --selected-budget-ratio 0.05 \
  --seed ${seed} \
  --output-dir experiments/r2v_replay/outputs/revision5_v3a_seed${seed}_full/diagnostic_obs_action_next
```

The diffusion MLP emitted sklearn convergence warnings, as in prior rounds. Diffusion remains a failed baseline; the official evidence below uses kNN.

## Dataset Composition

Counts exclude optional invalid stress transitions except the explicit optional-invalid column.

| Seed | Common zero | Rare useless | Rare valuable zero precursor | Rare valuable positive | Optional invalid |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 4831 | 38 | 118 | 13 | 150 |
| 1 | 4856 | 72 | 65 | 7 | 150 |
| 2 | 4869 | 66 | 58 | 7 | 150 |

Observation: v3a has enough rare-useless density for H2 in all seeds, but seeds 1 and 2 remain lower on zero-reward precursor supply than seed 0.

## H1 And H2: kNN Candidate Evidence

| Seed | kNN AUROC common-vs-rare | Recall@Top10 rare valuable | Recall@Top20 rare valuable | Top10 rare-useless count | Top10 rare-useless fraction | Rare-useless enrichment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.909 | 0.427 | 0.802 | 38 | 0.076 | 10.300 |
| 1 | 0.805 | 0.819 | 1.000 | 26 | 0.052 | 3.719 |
| 2 | 0.855 | 1.000 | 1.000 | 26 | 0.052 | 4.058 |

H2 status: pass on all seeds.

- Enrichment >= 2: pass all seeds.
- Top10 rare-useless count >= 20: pass all seeds.
- Top10 rare-useless fraction >= 0.04: pass all seeds.

H1 status: weak-pass/mostly pass.

- Recall@Top20 rare valuable >= 0.70: pass all seeds.
- Recall@Top10 rare valuable >= 0.50: pass seeds 1 and 2, miss seed 0 at 0.427.
- Pro allowed weak-pass H1 if H2/H3 are strong.

## H3: R2V Selection From kNN Candidates

| Seed | Zero precursors in kNN Top10 | Zero precursors selected by R2V full | Rare useless in kNN Top10 | Rare useless selected by R2V full | Positive rewards selected by R2V full | R2V full valuable precision | R2V zero-precursor recall | Reward-only zero-precursor recall | R2V positive-reward fraction | R2V rare-useless fraction | Random-from-kNN rare-useless fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 54 | 6 | 38 | 0 | 2 | 0.320 | 0.051 | 0.000 | 0.080 | 0.000 | 0.120 |
| 1 | 52 | 12 | 26 | 0 | 7 | 0.760 | 0.185 | 0.000 | 0.280 | 0.000 | 0.040 |
| 2 | 58 | 4 | 26 | 0 | 7 | 0.440 | 0.069 | 0.000 | 0.280 | 0.000 | 0.000 |

H3 status: pass on the main requested criteria.

- R2V full selected rare-useless fraction < kNN Top10 rare-useless fraction: pass all seeds.
- R2V full selected rare-useless is near 0: pass all seeds, exactly 0.
- R2V zero-precursor recall > reward-only: pass all seeds.
- R2V positive-reward fraction <= 0.50: pass all seeds.
- R2V selects nonzero zero-reward precursors: pass all seeds.

Important nuance: seed 2 has random-from-kNN rare-useless fraction 0 by chance at budget 25, so the cleanest H3 contrast is against the full kNN Top10 candidate pool, not against one random draw.

## H4 And Audit

| Seed | Combined invalid-risk AUROC | kNN precursor false rejection full-vs-no-risk | Metric audit pass | selected_rare_useless rows | rejected_valuable_precursors rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.143 | yes | 0 | 50 |
| 1 | 1.000 | 0.231 | yes | 0 | 50 |
| 2 | 1.000 | 0.333 | yes | 0 | 50 |

H4 status: AUROC pass, false-rejection borderline.

- Combined invalid-risk AUROC >= 0.85: pass all seeds.
- False rejection <= 0.30 if implemented: pass seeds 0 and 1, miss seed 2 at 0.333.
- Metric consistency audit passed for all rows in all seeds.
- Optional invalid stress transitions were excluded from H1-H3 denominators in the audit.

## Diffusion Failed Baseline

| Seed | Diffusion Recall@Top10 rare valuable | Diffusion Recall@Top20 rare valuable | Diffusion Top10 rare-useless count | Diffusion Top10 rare-useless fraction | Diffusion rare-useless enrichment | R2V diffusion zero-precursor recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.046 | 0.115 | 15 | 0.030 | 4.066 | 0.008 |
| 1 | 0.097 | 0.153 | 29 | 0.058 | 4.149 | 0.046 |
| 2 | 0.031 | 0.138 | 20 | 0.040 | 3.121 | 0.017 |

Diffusion is still not the official rarity evidence. It finds some rare-useless density but has poor rare-valuable recall and poor zero-precursor recall after selection.

## Mean Summary

| Metric | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| kNN AUROC common-vs-rare | 0.857 | 0.805 | 0.909 |
| kNN Recall@Top10 rare valuable | 0.749 | 0.427 | 1.000 |
| kNN Recall@Top20 rare valuable | 0.934 | 0.802 | 1.000 |
| kNN Top10 rare-useless count | 30.000 | 26.000 | 38.000 |
| kNN Top10 rare-useless fraction | 0.060 | 0.052 | 0.076 |
| kNN rare-useless enrichment | 6.026 | 3.719 | 10.300 |
| R2V full zero-precursor selected count | 7.333 | 4.000 | 12.000 |
| R2V full rare-useless selected count | 0.000 | 0.000 | 0.000 |
| R2V full valuable precision | 0.507 | 0.320 | 0.760 |
| R2V full zero-precursor recall | 0.101 | 0.051 | 0.185 |
| R2V full positive-reward fraction | 0.213 | 0.080 | 0.280 |
| Combined invalid-risk AUROC | 1.000 | 1.000 | 1.000 |
| kNN precursor false rejection | 0.236 | 0.143 | 0.333 |

## Codex Recommendation

`go_to_pro_checkpoint`

Reason:

- H2 now passes all seeds in the full diagnostic.
- H3 passes the main rare-to-valuable claim: kNN surfaces a mixed rare set, and R2V removes rare-useless while retaining nonzero zero-reward precursors better than reward-only.
- H4 AUROC is very strong, but seed 2 has a borderline precursor false-rejection value of 0.333.
- H1 is acceptable under Pro's weak-pass rule because Recall@Top20 passes all seeds and H2/H3 are strong.

Do not run 20k or Level 1 until Pro decides whether the seed-2 false rejection and seed-0 Recall@Top10 miss are acceptable for Round 1.

## Artifacts On Ubuntu

Per seed:

- `~/dder/experiments/r2v_replay/outputs/revision5_v3a_seed{seed}_full/replay.npz`
- `~/dder/experiments/r2v_replay/outputs/revision5_v3a_seed{seed}_full/diagnostic_obs_action_next/metrics.csv`
- `~/dder/experiments/r2v_replay/outputs/revision5_v3a_seed{seed}_full/diagnostic_obs_action_next/metric_consistency_audit.csv`
- `~/dder/experiments/r2v_replay/outputs/revision5_v3a_seed{seed}_full/diagnostic_obs_action_next/candidate_funnel_summary.csv`
- `~/dder/experiments/r2v_replay/outputs/revision5_v3a_seed{seed}_full/diagnostic_obs_action_next/selection_funnel_by_baseline.csv`
- `~/dder/experiments/r2v_replay/outputs/revision5_v3a_seed{seed}_full/diagnostic_obs_action_next/risk_rejection_summary.csv`
- `~/dder/experiments/r2v_replay/outputs/revision5_v3a_seed{seed}_full/diagnostic_obs_action_next/top_selected_examples.csv`
- `~/dder/experiments/r2v_replay/outputs/revision5_v3a_seed{seed}_full/diagnostic_obs_action_next/selected_rare_useless.csv`
- `~/dder/experiments/r2v_replay/outputs/revision5_v3a_seed{seed}_full/diagnostic_obs_action_next/rejected_valuable_precursors.csv`

## Next Pro Question

Should Round 1 now:

1. proceed to a larger 20k v3a diagnostic,
2. do a minimal selector-risk threshold revision to reduce seed-2 false rejection,
3. do a minimal v3a_plus dataset revision for precursor supply, or
4. stop Level 0 here and move to a more realistic next-level experiment?
