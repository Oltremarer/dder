# AGENTS.md - R2V-Replay Codex Workspace Rules

This file is the project-level operating manual for Codex in the `dder` / R2V-Replay workspace.

Any Codex session must read this file before editing code, running experiments, using Git, or touching remote machines.

Current project date: 2026-06-09.

Current GPT Pro verdict: `stop_route`.

Synthetic-grid Level 0 route status: stopped.

R2V-Replay overall idea status: not rejected.

Codex action now: documentation and route-closure only.

## 1. Project Background

R2V-Replay means:

```text
Rare-to-Valuable Experience Mining for Sparse-Reward Replay
```

The project studies sparse-reward replay buffers where useful learning signal is buried under many real but low-information zero-reward transitions.

Core framing:

- `rare != valuable`
- Rarity is only a candidate-discovery signal.
- R2V-Replay is not primarily a generation method.
- Diffusion is not the final value judge.
- Real zero-reward replay is not corrupted data.

Use this wording:

```text
real but mostly low-information zero-reward transitions
```

Do not call zero-reward replay corrupted, dirty, polluted, bad data, or broken data.

## 2. Current Route State

Completed Level 0 synthetic-grid artifacts:

- Revision 6: v3a 20k diagnostic
- Revision 7: candidate-ratio sweep
- Revision 8: two-branch v4 diagnostic
- Revision 9: geometry metric audit
- Revision 10: high-diversity v5 density gate

Final GPT Pro verdict:

```text
stop_route
```

Meaning:

- Synthetic-grid Level 0 route is stopped.
- Do not design v6.
- Do not patch v5.
- Do not patch v4.
- Do not patch v3a.
- Do not run v5 full 5k diagnostic.
- Do not run v5 candidate sweep.
- Do not run 20k.
- Do not enter Level 1.
- Do not run RL training.

The synthetic-grid route produced useful engineering infrastructure and negative diagnostic evidence, but it did not produce a positive Level 0 proof.

Main lesson to preserve:

```text
Rarity-based candidate discovery in low-dimensional synthetic grids is highly sensitive to representation geometry and transition multiplicity.
```

The current route could not robustly demonstrate `rare != valuable` at scale without over-engineering the synthetic environment.

## 3. Role Split

GPT Pro is the research PI and review gate.

GPT Pro decides:

- new empirical route
- story-framing route
- whether any stopped route can be reopened
- whether Level 1 or 20k can ever be run

Codex is the code and experiment executor.

Codex may now do only:

- documentation cleanup
- route-closure reports
- artifact path summaries
- Git hygiene
- final status reporting

Codex must not independently expand scope into RL, generation, PER, HER, CleanRL, CORL, D4RL, SynthER, or another synthetic-grid design.

## 4. Workspace And Remote Rules

Canonical GitHub repo:

```text
https://github.com/Oltremarer/dder
```

Ubuntu execution host:

```text
Ubuntu-Tailscale
```

Ubuntu code path:

```text
~/dder
```

Mac rule:

- Mac may be used for editing, Git, and browser.
- Mac must not run experiments unless the user explicitly overrides.

Ubuntu rule:

- No new experiments are allowed under the current verdict.
- If a future user/GPT Pro instruction explicitly permits commands, run tests and experiments only on `Ubuntu-Tailscale`.

Before any Ubuntu command:

```bash
cd ~/dder
git status --short
git rev-parse HEAD
```

Use the existing Python env unless GPT Pro or the user says otherwise:

```bash
~/miniconda3/envs/c2t/bin/python
```

Protected local literature directories:

```text
literature_story_mining/
flow_literature_mining/
```

Do not write to, move, rename, delete, format, stage, or commit these directories unless the user explicitly asks.

## 5. Allowed Files Under Current Verdict

Documentation-only closure files:

```text
AGENTS.md
experiments/r2v_replay/reports/pro_revision10_verdict.md
experiments/r2v_replay/reports/round1_level0_synthetic_grid_stop_route.md
```

Optional documentation files if needed:

```text
experiments/r2v_replay/README.md
experiments/r2v_replay/reports/README.md
```

Do not edit code unless fixing a typo in a report path.

## 6. Forbidden Actions

Do not create v6.

Do not patch v5.

Do not patch v4.

Do not patch v3a.

Do not run full 5k diagnostic on v5.

Do not run candidate sweep on v5.

Do not run 20k.

Do not enter Level 1.

Do not implement RL training.

Do not add DQN, CleanRL, CORL, D4RL, SynthER, HER, PER, or generation.

Do not tune selector, risk, consistency, kNN, or diffusion.

Do not replace kNN as official scorer in this route.

Do not revive diffusion as evidence.

Do not use `full_transition`, reward, done, or labels for rarity scoring.

Do not mix optional-invalid into H1/H2/H3.

Do not manually sample or relabel rare-useless.

Do not call zero-reward replay corrupted, dirty, polluted, or bad data.

Do not hide seed-level failures behind means.

Do not claim the synthetic-grid Level 0 route succeeded.

Do not claim R2V-Replay is invalid overall.

## 7. Reporting Requirements

Every closure or future checkpoint must include:

- repo
- branch
- commit
- Ubuntu host
- Ubuntu path
- Python executable if any command was run
- Mac experiment status
- exact commands run, or explicit "no experiments run"
- files changed
- seed-level evidence when discussing results
- artifact paths
- Codex recommendation
- questions for GPT Pro

Do not report only averages.

Do not omit failed seeds.

Do not omit optional-invalid exclusion status when discussing Level 0 diagnostics.

Do not omit whether reward/done/labels were excluded from official rarity scoring.

## 8. Final Route State

Correct state after Pro Revision 10 verdict:

```text
Synthetic-grid Level 0 route: stopped.
R2V-Replay overall idea: not rejected.
Current evidence: useful but not sufficient for a positive Level 0 proof.
Next scientific move: GPT Pro must choose a new empirical route or story-framing route later.
Codex action now: documentation and route closure only.
```

No automatic next stage.

No automatic Level 1.

No automatic 20k.

No automatic new dataset.
