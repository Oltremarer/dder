# Pro Verdict For R2V-Replay Round 1 Revision 1

Date: 2026-06-09

Source: Chrome ChatGPT Pro conversation `https://chatgpt.com/c/6a282a75-8ad0-83ea-a4ed-2205e8fffe90`

## Verdict

`revise`

One-sentence rationale from Pro:

> Level 0 has proven the code path and a kNN-density R2V signal, but the diffusion mainline fails, H2 evidence is incomplete, and H3 may have denominator or precision/recall accounting issues; do not enter Level 1 RL.

## Hypothesis Judgement

| Hypothesis | Decision | Reason |
| --- | --- | --- |
| H1 rarity high-recall candidate discovery | diffusion fail; kNN weak/partial pass | diffusion AUROC 0.512 and Top10 valuable recall 0.077; kNN Top10 valuable recall 0.921 is strong, but AUROC 0.734 is below the earlier 0.75 threshold. |
| H2 rare-only mixes rare useless | fail / incomplete | kNN selected rare_useless = 0 only shows selector exclusion; Pro wants kNN candidate-level rare_useless count/fraction/enrichment. |
| H3 utility filtering improves valuable subset | kNN promising but not pass; diffusion fail | kNN R2V-full precision 0.853 is strong, but selected counts, denominators, and label-count consistency must be audited first. |
| H4 risk / consistency admission | pass | optional_invalid is excluded from real metrics; combined invalid-risk AUROC is essentially 1.0. |

## Diffusion Decision

- Do not repair the current sklearn cross-fit diffusion scorer again in this phase.
- Reframe it as `failed_lightweight_diffusion_baseline`.
- Do not use diffusion as main evidence.
- Do not claim diffusion works.
- Do not permanently abandon diffusion at the paper level; a proper PyTorch DDPM or segment-latent diffusion scorer can be revisited later only if density-R2V passes Level 0/1.

Suggested framing:

> R2V separates rare candidate discovery from task utility and risk-controlled admission. In the first diagnostic, simple density rarity works while the lightweight diffusion scorer fails.

## kNN / Density-Rarity Decision

kNN-density rarity supports a targeted 20k Level 0 run only after a metric audit revision. It does not yet support Level 1 RL planning.

Allowed 20k preconditions:

- Fix or audit H3 metric denominators.
- Report kNN H2 candidate-level rare_useless enrichment.
- Confirm no label leakage.
- Confirm R2V-kNN does not collapse to reward-only.

If the 5k revised audit passes, run:

- 20k transitions
- seeds 0, 1, 2
- `obs_action_next` only
- kNN as main
- diffusion retained as failed baseline

## Required Next Codex Actions

1. Do not run 20k yet.
2. Add metric audit fields:
   - `candidate_count`
   - `selected_count`
   - `real_transition_count`
   - `total_rare_valuable_positive`
   - `total_rare_valuable_zero_precursor`
   - `total_rare_valuable_all`
   - `candidate_rare_valuable_positive_count`
   - `candidate_rare_valuable_zero_precursor_count`
   - `candidate_rare_useless_count`
   - `selected_rare_valuable_positive_count`
   - `selected_rare_valuable_zero_precursor_count`
   - `selected_rare_useless_count`
   - `selected_common_zero_count`
   - `valuable_precision_denominator`
   - `zero_precursor_recall_denominator`
   - `positive_reward_fraction_denominator`
3. Add assertion checks:
   - selected valuable counts do not exceed `selected_count`
   - selected per-label counts do not exceed total per-label counts
   - optional_invalid is excluded from H1-H3
4. Add `tests/test_metrics.py` with hand-built label arrays for:
   - precision@budget
   - recall@budget
   - zero precursor recall
   - positive reward fraction
   - rare_useless fraction
   - enrichment vs uniform
   - selected_count != candidate_count
   - optional_invalid exclusion
5. Report kNN H2 candidate evidence per seed:
   - candidate count
   - rare_useless count/fraction
   - rare_useless enrichment vs uniform
   - valuable precision
   - common_zero fraction
6. Split H3 into candidate pool and selector effect:
   - zero precursors available in kNN Top10
   - zero precursors selected by no-risk/full-risk
   - rare_useless available in kNN Top10
   - rare_useless selected by no-risk/full-risk
   - positive rewards available/selected
7. Add reward-only collapse audit:
   - reward-only selected count
   - reward-only positive reward fraction
   - reward-only zero precursor recall
   - R2V-kNN positive reward fraction
   - R2V-kNN zero precursor recall
   - R2V-kNN selected episode diversity
8. Freeze diffusion for this phase.
9. Only after steps 1-7 pass on 5k, run 20k Level 0.

## Forbidden Actions

- Do not enter Level 1 RL.
- Do not implement DQN, CleanRL, replay wrappers, PER, generation, or diffusion-as-generator.
- Do not clone SynthER, CORL, or D4RL.
- Do not treat `full_transition` as a main result.
- Do not use reward/done rescued diffusion as main evidence.
- Do not mix optional_invalid into H1-H3.
- Do not modify label rules just to improve metrics.
- Do not report only kNN R2V precision; report candidate composition and selector funnel.
- Do not run 50k before metric denominator checks.
- Do not enter Level 1 planning before 20k passes.

## Next Checkpoint Format

Use `R2V-Replay Round 1 Revision 2 Report` with:

1. Code / metric audit changes
2. Metric consistency audit for seeds 0/1/2
3. H1 kNN and diffusion comparison
4. H2 rare != valuable candidate evidence
5. H3 selection funnel
6. H4 risk / consistency
7. 20k section only if run
8. Artifacts
9. Interpretation
10. Codex recommendation: `go | revise | stop`
11. Questions for Pro

Pro next decision standards:

- Go to 20k Level 0 if metric audit passes, kNN H1 passes or weak-passes with strong TopK recall, kNN H2 passes, kNN H3 passes, and no label leakage is found.
- Go to Level 1 planning only if 20k seeds 0/1/2 confirm H1-H4 for kNN-density R2V, H2 clearly shows rare != valuable in the candidate pool, R2V removes rare_useless while retaining zero-reward precursors, R2V does not collapse to reward-only, and artifacts include funnel tables and failure cases.
- Revise if kNN H1/H3 is promising but H2 is unclear, metric audit reveals fixable denominator bugs, or rare_useless generation needs revision.
- Stop if the metric fix removes kNN support, H2 cannot be established without artificial labeling, R2V advantage disappears after denominator correction, or label leakage is found.
