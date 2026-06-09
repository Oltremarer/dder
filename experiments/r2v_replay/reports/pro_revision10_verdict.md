# Pro Verdict For R2V-Replay Round 1 Revision 10

Date: 2026-06-09

Source: Chrome ChatGPT Pro conversation `https://chatgpt.com/c/6a282a75-8ad0-83ea-a4ed-2205e8fffe90`

## Verdict

`stop_route`

## Short Rationale

v5 was the final allowed high-diversity synthetic-grid attempt. It improved rare-useless/common-zero geometry distance, but it failed the density gate's minimum unique-support requirements.

GPT Pro ruled that the synthetic-grid Level 0 route should stop. Codex must not continue with v6, v5 patches, full v5 diagnostics, 20k, Level 1, or RL training.

This is not `stop_level0`. The code framework, audit tools, and v3a/v4/v5 failure analyses remain useful as negative diagnostic and method-risk evidence.

This is not `return_to_story_framing` as the primary verdict. The immediate decision is to stop this synthetic-grid route. Story framing is allowed only as later documentation or a future GPT Pro-directed route.

## Decision Basis

v5 density gate failed:

| gate item | required | observed | decision |
|---|---|---|---|
| rare_useless unique obs_action_next | >= 50 per seed | 11 / 15 / 12 | fail |
| rare_valuable_zero_precursor unique obs_action_next | >= 50 per seed | 23 / 14 / 24 | fail |
| rare_useless mean multiplicity <= 3x precursor | all seeds preferred | seed 0 fail | weak/fail |
| rare_useless median distance to common-zero > 0 | >= 2/3 seeds | 3/3 pass | pass |

v5 fixed one part of the previous failure: common-zero separation. It did not generate enough diverse rare-useless or rare-valuable precursor support.

This is consistent with the broader pattern:

- v3a 20k: H1/H3 could look good, but H2 failed because rare-useless enrichment was not stable across seeds.
- v4: rare-useless collapsed into too few unique `obs_action_next` keys.
- v5: final high-diversity repair still failed unique-support density gate.

## Next Allowed Actions

Codex may do only cleanup, reporting, and route-closure work.

Allowed actions:

1. Update `AGENTS.md` to mark:
   - current verdict: `stop_route`
   - v5 failed density gate
   - no v6
   - no more synthetic-grid Level 0 design without explicit new GPT Pro approval
2. Add route-closure report:
   - `experiments/r2v_replay/reports/round1_level0_synthetic_grid_stop_route.md`
3. Summarize evidence from:
   - Revision 6 v3a 20k
   - Revision 7 candidate-ratio sweep
   - Revision 8 v4
   - Revision 9 geometry audit
   - Revision 10 v5 density gate
4. Package existing artifact paths only.
5. Commit documentation-only changes.
6. Report final route status.

## Allowed Files

```text
AGENTS.md
experiments/r2v_replay/reports/pro_revision10_verdict.md
experiments/r2v_replay/reports/round1_level0_synthetic_grid_stop_route.md
```

Optional if needed:

```text
experiments/r2v_replay/README.md
experiments/r2v_replay/reports/README.md
```

Do not edit code unless fixing a typo in a report path.

## Forbidden Actions

- Do not create v6.
- Do not patch v5.
- Do not patch v4.
- Do not patch v3a.
- Do not run full 5k diagnostic on v5.
- Do not run candidate sweep on v5.
- Do not run 20k.
- Do not enter Level 1.
- Do not implement RL training.
- Do not add DQN, CleanRL, CORL, D4RL, SynthER, HER, PER, or generation.
- Do not tune selector, risk, consistency, kNN, or diffusion.
- Do not replace kNN as official scorer in this route.
- Do not revive diffusion as evidence.
- Do not use `full_transition`, reward, done, or labels for rarity scoring.
- Do not mix optional-invalid into H1/H2/H3.
- Do not manually sample or relabel rare-useless.
- Do not call zero-reward replay corrupted, dirty, polluted, or bad data.
- Do not hide seed-level failures behind means.
- Do not claim the synthetic-grid Level 0 route succeeded.
- Do not claim R2V-Replay is invalid overall.

## Final Project State

```text
Synthetic-grid Level 0 route: stopped.
R2V-Replay overall idea: not rejected.
Current evidence: useful but not sufficient for a positive Level 0 proof.
Next scientific move: GPT Pro must choose a new empirical route or story-framing route later.
Codex action now: documentation and route closure only.
```

Main lesson to preserve:

```text
Rarity-based candidate discovery in low-dimensional synthetic grids is highly sensitive to representation geometry and transition multiplicity. The current route could not robustly demonstrate rare != valuable at scale without over-engineering the environment.
```
