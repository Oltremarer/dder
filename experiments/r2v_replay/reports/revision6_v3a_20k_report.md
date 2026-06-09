# R2V-Replay Round 1 Revision 6 v3a 20k Diagnostic Report

## Context

- Workspace: `/Users/azure/Documents/adg改`
- Repo: `https://github.com/Oltremarer/dder`
- Branch: `main`
- Ubuntu env: `Ubuntu-Tailscale:~/dder`, `~/miniconda3/envs/c2t/bin/python`
- Git commit on Ubuntu: `9b6c6bb44af1e45e0899539e5b6e033dae2e0ed3`
- Config: `experiments/r2v_replay/configs/level0_sparse_grid_hybrid_v3a.yaml`
- Rarity input: `obs_action_next`
- Seeds: `0`, `1`, `2`
- Real transitions per seed: `20000`
- Optional invalid stress transitions per seed: `600`
- Mac local experiments: not run
- Level 1 RL: not run

Ubuntu tests before 20k:

- `~/miniconda3/envs/c2t/bin/python -m pytest -q ~/dder/experiments/r2v_replay/tests`
- Result: `10 passed, 3 warnings in 1.16s`

## Pro Revision 5 Instruction Followed

Pro verdict: `go_to_20k`

Executed only:

- v3a 20k Level 0 diagnostic;
- seeds `0/1/2`;
- main scorer `kNN / obs_action_next`;
- diffusion kept as failed lightweight baseline.

No dataset revision, selector/risk tuning, diffusion tuning, 20k threshold lowering, Level 1 RL, PER, generation, RL framework integration, or main-scorer reward/done/label leakage was added.

## Dataset Composition

Counts exclude optional invalid stress transitions except the explicit optional-invalid column.

| Seed | Real transitions | Common zero | Rare useless | Rare valuable zero precursor | Rare valuable positive | Optional invalid | Success episode ratio | Behavior policy ratios |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 20000 | 19416 | 197 | 348 | 39 | 600 | 0.167 | near_success:0.038, noisy_goal:0.194, random_wander:0.702, rare_distractor_probe:0.067 |
| 1 | 20000 | 19253 | 293 | 409 | 45 | 600 | 0.187 | near_success:0.043, noisy_goal:0.162, random_wander:0.715, rare_distractor_probe:0.080 |
| 2 | 20000 | 19477 | 201 | 289 | 33 | 600 | 0.145 | near_success:0.035, noisy_goal:0.137, random_wander:0.757, rare_distractor_probe:0.072 |

The 20k datasets have many more rare valuable zero precursors than the 5k runs, so precursor supply is not the bottleneck in this round.

## H1: kNN Candidate Discovery

| Seed | kNN AUROC common-vs-rare | Recall@Top10 rare valuable | Recall@Top20 rare valuable | Enrichment@Top10 rare valuable |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.822 | 0.607 | 0.819 | 6.255 |
| 1 | 0.772 | 0.522 | 0.742 | 5.377 |
| 2 | 0.832 | 0.671 | 1.000 | 6.909 |

Mean/std/min/max:

- kNN AUROC: mean `0.809`, std `0.026`, min `0.772`, max `0.832`
- Recall@Top10 rare valuable: mean `0.600`, std `0.061`, min `0.522`, max `0.671`
- Recall@Top20 rare valuable: mean `0.854`, std `0.108`, min `0.742`, max `1.000`

H1 status: pass.

- Recall@Top10 rare valuable >= 0.50 passes all seeds.
- Recall@Top20 rare valuable >= 0.70 passes all seeds.

## H2: Rare != Valuable Candidate Evidence

| Seed | kNN Top10 rare-useless count | kNN Top10 rare-useless fraction | Rare-useless enrichment | kNN Top10 valuable precision | kNN Top10 common-zero fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 21 | 0.011 | 1.098 | 0.117 | 0.872 |
| 1 | 26 | 0.013 | 0.914 | 0.118 | 0.869 |
| 2 | 64 | 0.032 | 3.280 | 0.108 | 0.860 |

Mean/std/min/max:

- rare-useless count: mean `37.000`, std `19.201`, min `21`, max `64`
- rare-useless fraction: mean `0.0185`, std `0.0096`, min `0.0105`, max `0.0320`
- rare-useless enrichment: mean `1.764`, std `1.074`, min `0.914`, max `3.280`

H2 status: fail.

- Count >= 20 passes all seeds.
- Fraction >= 0.04 fails all seeds.
- Enrichment >= 2 fails seeds 0 and 1.

Interpretation: v3a's 5k density gate did not scale to 20k. At 20k, kNN still finds rare valuable precursors, but the official kNN Top10 candidate pool is no longer sufficiently enriched with rare-useless transitions. This weakens the `rare != valuable` candidate-pool evidence.

## H3: R2V Selection Funnel

| Seed | Zero precursors in kNN Top10 | Zero precursors selected no-risk | Zero precursors selected full-risk | Rare useless in kNN Top10 | Rare useless selected no-risk | Rare useless selected full-risk | Positive rewards selected full-risk | R2V full valuable precision | R2V zero-precursor recall | Reward-only zero-precursor recall | R2V positive-reward fraction | R2V rare-useless fraction | Random-from-kNN rare-useless fraction | Selected episode diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 223 | 34 | 23 | 21 | 0 | 0 | 12 | 0.350 | 0.066 | 0.000 | 0.120 | 0.000 | 0.000 | 29 |
| 1 | 220 | 43 | 43 | 26 | 0 | 0 | 4 | 0.470 | 0.105 | 0.000 | 0.040 | 0.000 | 0.000 | 30 |
| 2 | 204 | 40 | 29 | 64 | 0 | 0 | 12 | 0.410 | 0.100 | 0.000 | 0.120 | 0.000 | 0.050 | 31 |

Mean/std/min/max:

- full-risk zero precursors selected: mean `31.667`, std `8.380`, min `23`, max `43`
- full-risk rare useless selected: mean `0.000`, std `0.000`, min `0`, max `0`
- R2V full valuable precision: mean `0.410`, std `0.049`, min `0.350`, max `0.470`
- R2V zero-precursor recall: mean `0.091`, std `0.017`, min `0.066`, max `0.105`
- R2V positive-reward fraction: mean `0.093`, std `0.038`, min `0.040`, max `0.120`
- selected episode diversity: mean `30.000`, std `0.816`, min `29`, max `31`

H3 status: pass conditional on the available mixed candidate pool.

- R2V full rare-useless fraction is `0` for all seeds.
- R2V zero-precursor recall > reward-only for all seeds.
- R2V positive-reward fraction <= 0.50 for all seeds.
- R2V selects nonzero zero-reward precursors for all seeds.

Important caveat: because H2 fails at 20k, H3 no longer proves that R2V can filter a strongly rare-useless-enriched kNN pool at scale. It does show that, when rare-useless candidates are present, the selector removes them while keeping many zero-reward precursors.

## H4: Risk / Consistency

| Seed | Combined invalid-risk AUROC | kNN precursor false rejection full-vs-no-risk | Metric audit pass | selected_rare_useless rows | rejected_valuable_precursors rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.324 | yes | 0 | 50 |
| 1 | 1.000 | 0.209 | yes | 0 | 50 |
| 2 | 1.000 | 0.325 | yes | 0 | 50 |

Mean/std/min/max:

- Combined invalid-risk AUROC: mean `0.999988`, std `0.000004`, min `0.999982`, max `0.999992`
- kNN precursor false rejection: mean `0.286`, std `0.054`, min `0.209`, max `0.325`

H4 status: AUROC pass, false-rejection warning.

- Combined invalid-risk AUROC >= 0.85 passes all seeds.
- False rejection <= 0.30 passes seed 1 but fails seeds 0 and 2.
- Metric consistency audit passed all rows.
- Optional invalid stress transitions were excluded from H1-H3 denominators.

This matches Pro's contingency: false rejection `> 0.30` in multiple seeds is evidence for a later selector-risk revision, but the H2 scale failure is the primary blocker for moving to the next level.

## Diffusion Failed Baseline

| Seed | Diffusion Recall@Top10 rare valuable | Diffusion Recall@Top20 rare valuable | Diffusion Top10 rare-useless fraction | R2V diffusion zero-precursor recall |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.052 | 0.129 | 0.027 | 0.014 |
| 1 | 0.064 | 0.156 | 0.054 | 0.012 |
| 2 | 0.081 | 0.165 | 0.040 | 0.010 |

Diffusion remains a failed lightweight baseline:

- it has poor rare-valuable recall;
- it sometimes has rare-useless density, but not enough valuable precursor recall;
- it should not replace kNN as official evidence.

## Interpretation

20k confirms:

- kNN can find rare valuable candidates more consistently than in 5k;
- R2V can remove selected rare useless to zero;
- R2V preserves zero-reward precursors better than reward-only;
- invalid-risk AUROC remains very strong.

20k does not confirm:

- stable rare-useless enrichment in the official kNN Top10 candidate pool;
- all-seed false-rejection <= 0.30.

The main surprise is that increasing from 5k to 20k improved rare-valuable discovery but diluted rare-useless density in the Top10 candidate pool. That means the Level 0 mechanism chain is no longer complete at 20k: H1 and H3 are stronger, but H2 fails.

## Codex Recommendation

`revise_dataset_or_h2`

Primary reason:

- H2 failed at 20k: rare-useless fraction is below `0.04` for all seeds and enrichment is below `2.0` for seeds 0 and 1.

Secondary warning:

- H4 false rejection is above `0.30` in two seeds, which suggests a future selector-risk revision if Pro still wants to keep the v3a family.

Do not move to Level 1 yet. The next Pro decision should choose one of:

1. revise H2/dataset density while preserving the now-strong H1/H3 behavior;
2. revise the candidate definition or H2 threshold if Pro judges the 20k Top10 fraction criterion too strict for a 2000-candidate pool;
3. stop the current synthetic Level 0 route because 5k H2 did not scale to 20k.

## Artifacts On Ubuntu

Per seed:

- `~/dder/experiments/r2v_replay/outputs/revision6_v3a_seed{seed}_20k/replay.npz`
- `~/dder/experiments/r2v_replay/outputs/revision6_v3a_seed{seed}_20k/diagnostic_obs_action_next/metrics.csv`
- `~/dder/experiments/r2v_replay/outputs/revision6_v3a_seed{seed}_20k/diagnostic_obs_action_next/metric_consistency_audit.csv`
- `~/dder/experiments/r2v_replay/outputs/revision6_v3a_seed{seed}_20k/diagnostic_obs_action_next/candidate_funnel_summary.csv`
- `~/dder/experiments/r2v_replay/outputs/revision6_v3a_seed{seed}_20k/diagnostic_obs_action_next/selection_funnel_by_baseline.csv`
- `~/dder/experiments/r2v_replay/outputs/revision6_v3a_seed{seed}_20k/diagnostic_obs_action_next/risk_rejection_summary.csv`
- `~/dder/experiments/r2v_replay/outputs/revision6_v3a_seed{seed}_20k/diagnostic_obs_action_next/top_selected_examples.csv`
- `~/dder/experiments/r2v_replay/outputs/revision6_v3a_seed{seed}_20k/diagnostic_obs_action_next/selected_rare_useless.csv`
- `~/dder/experiments/r2v_replay/outputs/revision6_v3a_seed{seed}_20k/diagnostic_obs_action_next/rejected_valuable_precursors.csv`

## Next Pro Question

The 20k run strengthened H1/H3 but failed H2. Should Codex next:

1. revise dataset/H2 density and keep selector fixed,
2. revise selector-risk first because false rejection failed in two seeds,
3. redefine H2 for 20k candidate-pool scale,
4. stop Level 0 and move to a different synthetic design, or
5. stop this route entirely?
