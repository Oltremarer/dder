# R2V-Replay Round 1 Revision 7 Candidate-Ratio Sweep Report

## Context

- Workspace: `/Users/azure/Documents/adg改`
- Repo: `https://github.com/Oltremarer/dder`
- Branch: `main`
- Ubuntu env: `Ubuntu-Tailscale:~/dder`, `~/miniconda3/envs/c2t/bin/python`
- Base 20k commit on Ubuntu: `9b6c6bb44af1e45e0899539e5b6e033dae2e0ed3`
- Local code change under test: candidate-ratio sweep utility
- Source outputs: `revision6_v3a_seed{0,1,2}_20k/diagnostic_obs_action_next/score_table.csv`
- Sweep ratios: `0.005`, `0.01`, `0.02`, `0.05`, `0.10`, `0.20`
- Scorer: `knn`
- Dataset regeneration: not run
- 20k diagnostic rerun: not run
- Selector/risk tuning: not run
- Level 1 RL: not run
- Mac local experiments: not run

## Pro Revision 6 Instruction Followed

Pro verdict: `redefine_h2`

Implemented only:

- `candidate_ratio_sweep_rows` in `metric_audit.py`
- `analyze_candidate_ratio_sweep.py`
- `test_candidate_ratio_sweep.py`
- `pro_revision6_verdict.md`

No dataset config, sparse-grid generation, selector, risk scorer, diffusion, or diagnostic behavior was changed.

## Ubuntu Verification

New test:

- `~/miniconda3/envs/c2t/bin/python -m pytest -q ~/dder/experiments/r2v_replay/tests/test_candidate_ratio_sweep.py`
- Result: `1 passed in 0.09s`

Full r2v tests:

- `~/miniconda3/envs/c2t/bin/python -m pytest -q ~/dder/experiments/r2v_replay/tests`
- Result: `11 passed, 3 warnings in 1.13s`

Warnings were the same pandas deprecation and sklearn MLP convergence warnings seen in prior tests.

## Ubuntu Sweep Command

```bash
cd ~/dder
for seed in 0 1 2; do
  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/analyze_candidate_ratio_sweep.py \
    --diagnostic-dir experiments/r2v_replay/outputs/revision6_v3a_seed${seed}_20k/diagnostic_obs_action_next \
    --scorer knn \
    --ratios 0.005,0.01,0.02,0.05,0.10,0.20 \
    --output experiments/r2v_replay/outputs/revision7_v3a_seed${seed}_20k_candidate_sweep
done
```

Generated per seed:

- `candidate_ratio_sweep.csv`
- `candidate_ratio_sweep_summary.csv`
- `h2_budget_curve.csv`

## Decision Rule From Pro

H2-budget pass requires an eligible ratio in `{0.005, 0.01, 0.02, 0.05}` such that all seeds satisfy:

- `rare_useless_count >= 10`
- `rare_useless_enrichment >= 2.0`
- `rare_useless_fraction >= 0.04`
- `rare_valuable_all_count >= 50` or `rare_valuable_recall >= 0.30`

## All-Seed Summary

| Ratio | Candidate count | Min rare-useless count | Min rare-useless fraction | Min enrichment | Min rare-value all | Min rare-value recall | All count | All fraction | All enrichment | All supply | All H2 pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.005 | 100 | 0 | 0.000 | 0.000 | 6 | 0.016 | false | false | false | false | false |
| 0.010 | 200 | 0 | 0.000 | 0.000 | 19 | 0.044 | false | false | false | false | false |
| 0.020 | 400 | 6 | 0.015 | 1.024 | 24 | 0.062 | false | false | false | false | false |
| 0.050 | 1000 | 10 | 0.010 | 1.015 | 115 | 0.253 | true | false | false | true | false |
| 0.100 | 2000 | 21 | 0.011 | 0.887 | 216 | 0.522 | true | false | false | true | false |
| 0.200 | 4000 | 65 | 0.016 | 1.109 | 317 | 0.742 | true | false | false | true | false |

No ratio passes the all-seed H2-budget rule.

## Per-Seed Compact Results

| Seed | Ratio | Candidate count | Rare-useless count | Rare-useless fraction | Enrichment | Rare-value all | Rare-value recall | H2-budget pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.005 | 100 | 2 | 0.020 | 2.030 | 6 | 0.016 | false |
| 0 | 0.010 | 200 | 7 | 0.035 | 3.553 | 19 | 0.049 | false |
| 0 | 0.020 | 400 | 10 | 0.025 | 2.538 | 24 | 0.062 | false |
| 0 | 0.050 | 1000 | 10 | 0.010 | 1.015 | 133 | 0.344 | false |
| 0 | 0.100 | 2000 | 21 | 0.011 | 1.066 | 235 | 0.607 | false |
| 0 | 0.200 | 4000 | 68 | 0.017 | 1.726 | 317 | 0.819 | false |
| 1 | 0.005 | 100 | 0 | 0.000 | 0.000 | 14 | 0.031 | false |
| 1 | 0.010 | 200 | 0 | 0.000 | 0.000 | 20 | 0.044 | false |
| 1 | 0.020 | 400 | 6 | 0.015 | 1.024 | 36 | 0.079 | false |
| 1 | 0.050 | 1000 | 16 | 0.016 | 1.092 | 115 | 0.253 | false |
| 1 | 0.100 | 2000 | 26 | 0.013 | 0.887 | 237 | 0.522 | false |
| 1 | 0.200 | 4000 | 65 | 0.016 | 1.109 | 337 | 0.742 | false |
| 2 | 0.005 | 100 | 0 | 0.000 | 0.000 | 19 | 0.059 | false |
| 2 | 0.010 | 200 | 11 | 0.055 | 5.473 | 25 | 0.078 | false |
| 2 | 0.020 | 400 | 32 | 0.080 | 7.960 | 47 | 0.146 | false |
| 2 | 0.050 | 1000 | 32 | 0.032 | 3.184 | 150 | 0.466 | false |
| 2 | 0.100 | 2000 | 64 | 0.032 | 3.184 | 216 | 0.671 | false |
| 2 | 0.200 | 4000 | 100 | 0.025 | 2.488 | 322 | 1.000 | false |

## Interpretation

The sweep does not rescue H2.

At very small budgets:

- seed 1 has zero rare-useless transitions at `0.005` and `0.010`;
- seed 2 has zero rare-useless transitions at `0.005`;
- valuable supply is also too low.

At medium budgets:

- seed 0/1 rare-useless fraction remains below `0.04`;
- seed 1 enrichment remains below `2.0`;
- seed 2 can satisfy rare-useless fraction/enrichment at `0.010` and `0.020`, but valuable supply is too low at those budgets.

At large budgets:

- valuable supply becomes adequate;
- rare-useless fraction remains too low across all seeds;
- enrichment remains below `2.0` for seed 0/1 or below fraction threshold.

Therefore the 20k H2 failure is not only an artifact of using fixed Top10%. Under the pre-specified budget curve, kNN does not stably expose a candidate pool that is both rare-useless-enriched and rare-valuable-supplied across seeds.

## Codex Recommendation

`ask_pro_after_failed_h2_budget_sweep`

Per Pro's instruction, no follow-up full diagnostic was run because no candidate ratio passed the H2-budget rule.

The next Pro decision should choose one of:

1. revise dataset/H2 density with selector fixed;
2. stop the current v3a synthetic route;
3. relax/redefine H2 further with an explicit theoretical justification;
4. move to a different synthetic design where rare-useless density is less seed-sensitive.

## Artifacts On Ubuntu

Per seed:

- `~/dder/experiments/r2v_replay/outputs/revision7_v3a_seed{seed}_20k_candidate_sweep/candidate_ratio_sweep.csv`
- `~/dder/experiments/r2v_replay/outputs/revision7_v3a_seed{seed}_20k_candidate_sweep/candidate_ratio_sweep_summary.csv`
- `~/dder/experiments/r2v_replay/outputs/revision7_v3a_seed{seed}_20k_candidate_sweep/h2_budget_curve.csv`

## Next Pro Question

The pre-specified candidate-budget curve did not find any all-seed H2-budget pass ratio. Should Codex:

1. revise dataset/H2 density while keeping selector/risk fixed,
2. stop the v3a route and design a new Level 0 environment,
3. stop Level 0 entirely,
4. relax H2 with a new justified criterion, or
5. perform a targeted metric-bug audit before any design change?
