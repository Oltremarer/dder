# Pro Verdict For R2V-Replay Round 1 Revision 6

Date: 2026-06-09

Source: Chrome ChatGPT Pro conversation `https://chatgpt.com/c/6a282a75-8ad0-83ea-a4ed-2205e8fffe90`

## Verdict

`redefine_h2`

## Short Rationale

The 20k result must not move to the next level. H1 and H3 became stronger, but the old fixed Top10% H2 failed in seeds 0 and 1. Pro judged that the next step should not be another dataset revision and should not be selector tuning. Instead, H2 must first be reframed as a pre-specified candidate-budget curve because Top10% at 20k equals 2000 candidates and is no longer the same rarity budget as the smaller 5k check.

## H1-H4 Judgment

| Hypothesis | Decision | Evidence | Audit judgment |
| --- | --- | --- | --- |
| H1: kNN rarity high-recall finds rare valuable | pass | Top10 rare valuable recall passes all seeds; Top20 passes all seeds; AUROC all seeds >= 0.772 | Pass. 20k is more stable than 5k for H1. |
| H2: rare candidate pool contains rare-useless distractors | fail under old fixed-Top10 criterion | rare-useless count all seeds >= 20, but fraction all seeds < 0.04; enrichment seeds 0/1 < 2.0 | Cannot ignore. Current fixed Top10% evidence is insufficient. |
| H3: R2V filters candidate pool into valuable subset | conditional pass | R2V selected rare-useless = 0; zero-precursor recall > reward-only; positive-reward fraction <= 0.50 | Conditional pass. Because H2 is weak, H3 cannot be treated as a complete mechanism proof yet. |
| H4: risk / consistency | weak pass | invalid-risk AUROC ~= 1.0; false rejection seeds 0/2 are 0.324 / 0.325, above 0.30 | Not the primary blocker, but risk softness should be revisited later. |

## Primary And Secondary Blockers

Primary blocker:

`H2 fails at 20k under fixed Top10% candidate definition.`

Rare-useless transitions exist in the candidate pool:

- seed 0: 21
- seed 1: 26
- seed 2: 64

But the Top10% candidate pool at 20k has 2000 transitions, so rare-useless transitions are diluted by common-zero low-density candidates. The old H2 thresholds fail in seed 0 and seed 1.

Secondary blocker:

`H4 false rejection > 0.30 in seed 0 and seed 2.`

This suggests the selector/risk penalty may be too strong, but it is not the next priority because fixing risk would not make H2 true.

## Next Action

Do not rebuild the dataset, rerun 20k, or tune selector/risk. On the existing 20k outputs, add and run a candidate-ratio sweep to test whether H2 failure is caused by the fixed Top10% budget or by kNN failing to stably find rare-useless distractors.

Pre-specified ratios:

- `0.005`
- `0.01`
- `0.02`
- `0.05`
- `0.10`
- `0.20`

Required command shape:

```bash
for seed in 0 1 2; do
  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/analyze_candidate_ratio_sweep.py \
    --diagnostic-dir experiments/r2v_replay/outputs/revision6_v3a_seed${seed}_20k/diagnostic_obs_action_next \
    --scorer knn \
    --ratios 0.005,0.01,0.02,0.05,0.10,0.20 \
    --output experiments/r2v_replay/outputs/revision7_v3a_seed${seed}_20k_candidate_sweep
done
```

Alternative explicit input form:

```bash
for seed in 0 1 2; do
  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/analyze_candidate_ratio_sweep.py \
    --score-table experiments/r2v_replay/outputs/revision6_v3a_seed${seed}_20k/diagnostic_obs_action_next/score_table.csv \
    --metric-audit experiments/r2v_replay/outputs/revision6_v3a_seed${seed}_20k/diagnostic_obs_action_next/metric_consistency_audit.csv \
    --scorer knn \
    --ratios 0.005,0.01,0.02,0.05,0.10,0.20 \
    --output experiments/r2v_replay/outputs/revision7_v3a_seed${seed}_20k_candidate_sweep
done
```

Required outputs:

- `candidate_ratio_sweep.csv`
- `candidate_ratio_sweep_summary.csv`
- `h2_budget_curve.csv`

Required fields per ratio and seed:

- `ratio`
- `candidate_count`
- `rare_valuable_positive_count`
- `rare_valuable_zero_precursor_count`
- `rare_valuable_all_count`
- `rare_useless_count`
- `common_zero_count`
- `valuable_precision`
- `rare_valuable_recall`
- `zero_precursor_recall`
- `rare_useless_fraction`
- `rare_useless_enrichment`
- `common_zero_fraction`
- `nonvaluable_fraction`

## Sweep Decision Rule

H2-budget pass if there exists a ratio in `{0.005, 0.01, 0.02, 0.05}` such that all seeds satisfy:

- `rare_useless_count >= 10`
- `rare_useless_enrichment >= 2.0`
- `rare_useless_fraction >= 0.04`
- `rare_valuable_all_count >= 50` or `rare_valuable_recall >= 0.30`

The old fixed Top10% result must remain reported unchanged. Do not delete or overwrite Revision 6 outputs.

If the sweep finds a valid ratio, run one follow-up full diagnostic at that ratio only. Otherwise stop after the sweep report and ask Pro.

## Allowed Edits

- `experiments/r2v_replay/scripts/analyze_candidate_ratio_sweep.py`
- `experiments/r2v_replay/r2v_replay/metric_audit.py`
- `experiments/r2v_replay/tests/test_candidate_ratio_sweep.py`
- `experiments/r2v_replay/reports/pro_revision6_verdict.md`

Optional helper edit if needed:

- `experiments/r2v_replay/r2v_replay/diagnostics.py`

Do not edit dataset configs or selector/risk/dynamics code unless the sweep uncovers a metric bug. If a bug is found, report it before rerunning anything.

## Forbidden Actions

- Do not move to Level 1.
- Do not implement RL training.
- Do not add CleanRL, CORL, D4RL, SynthER, HER, PER, or generation.
- Do not revise dataset generation before the candidate-ratio sweep.
- Do not tune selector or risk weights before resolving H2.
- Do not rerun 20k with a changed dataset.
- Do not claim fixed Top10% H2 passed; it failed at 20k.
- Do not hide seed 0/1 H2 failures behind the mean.
- Do not redefine H2 by simply lowering the threshold after seeing results. The only acceptable redefinition is the pre-specified candidate-budget curve above.
- Do not treat common zero as corrupted data.
- Do not call zero-reward transitions dirty or bad data.
- Do not treat diffusion as working evidence. It remains a failed lightweight baseline.
- Do not use `full_transition`, reward, done, or labels for the main rarity scorer.
- Do not mix optional_invalid into H1-H3.
- Do not manually relabel rare_useless after inspecting candidate pools.
- Do not delete or overwrite the Revision 6 fixed-Top10 outputs.
