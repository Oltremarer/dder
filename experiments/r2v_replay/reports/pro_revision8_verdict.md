# Pro Verdict For R2V-Replay Round 1 Revision 8

Date: 2026-06-09

Source: Chrome ChatGPT Pro conversation `https://chatgpt.com/c/6a282a75-8ad0-83ea-a4ed-2205e8fffe90`

## Verdict

`metric_geometry_audit`

## Short Rationale

v4 is not ready for another design patch. The failure is technically clear, but Pro wants one targeted audit to separate bad geometry from bad official kNN metric. The rare-useless branch is real and label-valid, but under `obs_action_next` kNN it is highly repetitive and therefore dense, so H2 fails at every candidate budget. This resembles the earlier v3a problem where H2 failed at 20k while H1/H3 were stronger.

## Failure Classification

Not an implementation bug.

The v4 rollout and labels appear to work:

- rare-useless transitions are present in all seeds;
- optional invalid remains separate;
- tests pass;
- the branch is reachable through real dynamics.

The failure is mainly a design failure under the current kNN metric, with a secondary metric artifact from discrete transition duplication.

Evidence:

- rare_useless counts: seed0 `142`, seed1 `130`, seed2 `192`
- rare_useless unique transitions: seed0 `18`, seed1 `17`, seed2 `15`
- rare_useless mean multiplicity: seed0 `94.430`, seed1 `106.115`, seed2 `153.266`
- rare_useless median kNN distance: `0.000` for all seeds
- kNN Top10 rare-useless: seed0 `1`, seed1 `2`, seed2 `1`

Conclusion:

`v4 failed because the two-branch geometry creates low-diversity rare-useless transitions.`

This is not evidence that rare-useless cannot exist. It is also not evidence that R2V works. It is evidence that low-dimensional grid plus raw kNN over `obs_action_next` is fragile for H2.

## Next Action

Do not run one more design yet. First audit whether the failure would persist under duplicate-aware and geometry-aware density diagnostics.

Allowed edits only for audit code:

- `experiments/r2v_replay/scripts/analyze_geometry_metric_failure.py`
- `experiments/r2v_replay/r2v_replay/geometry_audit.py`
- `experiments/r2v_replay/tests/test_geometry_audit.py`
- `experiments/r2v_replay/reports/pro_revision8_verdict.md`

Allowed small helper edits if necessary:

- `experiments/r2v_replay/r2v_replay/replay_dataset.py`
- `experiments/r2v_replay/r2v_replay/metric_audit.py`
- `experiments/r2v_replay/r2v_replay/utils.py`

Do not edit yet:

- `experiments/r2v_replay/configs/level0_sparse_grid_two_branch_v4.yaml`
- `experiments/r2v_replay/configs/level0_sparse_grid_hybrid_v3a.yaml`
- `experiments/r2v_replay/r2v_replay/sparse_grid.py`
- `experiments/r2v_replay/r2v_replay/selector.py`
- `experiments/r2v_replay/r2v_replay/risk_scorers.py`
- `experiments/r2v_replay/r2v_replay/consistency_scorers.py`
- `experiments/r2v_replay/r2v_replay/rarity_scorers.py`

## Required Audit Outputs

The audit script should compute, per label and per seed:

- count
- unique obs_action_next count
- unique state count
- unique state_action count
- unique state_action_next count
- mean multiplicity
- median multiplicity
- max multiplicity
- median kNN distance
- mean kNN score
- TopK composition under raw kNN
- TopK composition under unique-transition kNN
- TopK composition under duplicate-capped kNN
- TopK composition under k = 5, 10, 50, 100
- distance-to-common-zero distribution
- distance-to-rare-valuable distribution
- distance-to-rare-useless distribution

Compare at least:

- v3a 20k
- v4 5k

Goal: diagnose whether raw kNN fails because of duplicate transition multiplicity, geometry, or feature collapse. Do not choose a new official scorer yet.

## Ubuntu Commands

Verify repo/tests:

```bash
cd ~/dder
git status --short
git rev-parse HEAD
~/miniconda3/envs/c2t/bin/python -m pytest -q experiments/r2v_replay/tests
```

After implementing the geometry audit:

```bash
~/miniconda3/envs/c2t/bin/python -m pytest -q experiments/r2v_replay/tests/test_geometry_audit.py
```

Run audit on v4 seeds:

```bash
for seed in 0 1 2; do
  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/analyze_geometry_metric_failure.py \
    --dataset experiments/r2v_replay/outputs/revision8_v4_seed${seed}_density/replay.npz \
    --diagnostic-dir experiments/r2v_replay/outputs/revision8_v4_seed${seed}_full5k/diagnostic_obs_action_next \
    --config experiments/r2v_replay/configs/level0_sparse_grid_two_branch_v4.yaml \
    --rarity-input obs_action_next \
    --k-values 5,10,50,100 \
    --candidate-ratios 0.005,0.01,0.02,0.05,0.10,0.20 \
    --output-dir experiments/r2v_replay/outputs/revision9_v4_seed${seed}_geometry_audit
done
```

Run the same audit on v3a 20k:

```bash
for seed in 0 1 2; do
  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/analyze_geometry_metric_failure.py \
    --dataset experiments/r2v_replay/outputs/revision6_v3a_seed${seed}_20k/replay.npz \
    --diagnostic-dir experiments/r2v_replay/outputs/revision6_v3a_seed${seed}_20k/diagnostic_obs_action_next \
    --config experiments/r2v_replay/configs/level0_sparse_grid_hybrid_v3a.yaml \
    --rarity-input obs_action_next \
    --k-values 5,10,50,100 \
    --candidate-ratios 0.005,0.01,0.02,0.05,0.10,0.20 \
    --output-dir experiments/r2v_replay/outputs/revision9_v3a20k_seed${seed}_geometry_audit
done
```

Expected artifacts:

- `geometry_density_summary.csv`
- `duplicate_multiplicity_by_label.csv`
- `unique_transition_summary.csv`
- `knn_sensitivity_by_k.csv`
- `candidate_ratio_by_metric_variant.csv`
- `raw_vs_unique_knn_comparison.csv`
- `distance_to_common_zero_summary.csv`
- `geometry_audit_interpretation.md`

## Next Report Questions

The next report should answer:

1. Are rare-useless transitions duplicated more heavily than rare-valuable precursors?
2. Does unique-transition kNN make rare-useless appear in candidate pools?
3. Does duplicate-capped kNN make H2 pass without destroying H1?
4. Does v3a fail H2 for the same reason as v4, or a different reason?
5. Is raw kNN unsuitable as the official Level 0 scorer?
6. Would a high-diversity design plausibly fix H2, or would it just keep chasing geometry?

## Decision Rules For Next Checkpoint

`abandon_knn_official_scorer` if raw kNN fails due duplicate artifacts and duplicate-aware kNN or unique-transition kNN changes H2 materially.

`one_more_high_diversity_design` if audit shows geometry is the main issue and a high-diversity design can plausibly make rare valuable and rare useless both low-frequency in `obs_action_next`.

`stop_route` if both raw and duplicate-aware metrics fail H2 on v3a/v4 and fixing H2 would require increasingly artificial environment construction.

`return_to_story_framing` if the diagnostic evidence remains useful as a negative result: density rarity finds valuable precursors, but synthetic grid H2 is too brittle for a clean main experiment.

`stop_level0` only if audit reveals label leakage, invalid samples in H1-H3, or core metric bugs that invalidate prior results.

## Forbidden Actions

- Do not run another dataset design yet.
- Do not run 20k.
- Do not enter Level 1.
- Do not implement RL training.
- Do not add CleanRL, CORL, D4RL, SynthER, PER, HER, or generation.
- Do not tune selector, risk, consistency, or diffusion.
- Do not change v3a or v4 configs.
- Do not use `full_transition` as official evidence.
- Do not use reward, done, or labels inside the official rarity scorer.
- Do not mix optional_invalid into H1-H3.
- Do not relabel rare_useless after seeing kNN rankings.
- Do not call zero-reward transitions corrupted, dirty, or bad data.
- Do not claim diffusion works.
- Do not hide seed-level failures behind averages.
- Do not relax H2 again without a written justification tied to the audit.
- Do not choose one more high-diversity design until the audit shows that geometry, not metric definition, is the main blocker.
