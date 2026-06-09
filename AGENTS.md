# AGENTS.md - R2V-Replay Codex Workspace Rules

This file is the project-level operating manual for Codex in the `dder` / R2V-Replay workspace.

Any Codex session must read this file before editing code, running experiments, using Git, or touching remote machines.

Current project date: 2026-06-09.

Current GPT Pro verdict: `one_more_high_diversity_design`.

This means exactly one final high-diversity Level 0 synthetic-grid design is allowed. No Level 1 RL, no 20k rerun, no RL-framework integration, and no generation work are allowed unless GPT Pro gives a later explicit go.

## 1. Project Background

R2V-Replay means:

```text
Rare-to-Valuable Experience Mining for Sparse-Reward Replay
```

The project studies sparse-reward replay buffers where useful learning signal is buried under many real but low-information zero-reward transitions.

The core idea is:

```text
Stage 1: rare / low-density candidate discovery
Stage 2: task utility filtering
Stage 3: support / OOD / dynamics / reward-consistency admission
Stage 4: replay use
```

Important framing:

- `rare != valuable`
- Rarity is only a candidate-discovery signal. It is not a value judgment.
- R2V-Replay is not primarily a generation method.
- Synthetic generation may become a later baseline or extension, but it is not the current method.

## 2. Non-Negotiable Research Boundaries

Never call real zero-reward replay corrupted, dirty, polluted, bad data, or broken data.

Use this wording:

```text
real but mostly low-information zero-reward transitions
```

Never assume rare equals valuable. The whole point is to separate:

- `rare_valuable`
- `rare_useless`
- `common_zero`

Diffusion is not the final value judge. In the current Level 0 scaffold, the lightweight diffusion scorer is a failed baseline and must not be reported as working evidence.

Do not import ADG clean/dirty semantics into experience replay.

Do not describe R2V as `SynthER + filter`. SynthER-style generation is not the current method.

Do not use reward, done, `full_transition`, or eval labels in the official rarity scorer.

Do not mix optional invalid stress samples into H1/H2/H3.

## 3. Role Split

GPT Pro is the research PI and review gate.

GPT Pro decides:

- go
- revise
- stop

GPT Pro owns:

- research framing
- novelty judgment
- experiment route
- reviewer-risk decisions
- stage transitions

Codex is the code and experiment executor.

Codex owns:

- implementation
- tests
- diagnostics
- Ubuntu execution
- artifact packaging
- Git hygiene
- checkpoint reports

Codex must not independently expand scope into RL, repo survey, generation, PER, or high-dimensional experiments.

## 4. Current Status

The project is in Round 1 Level 0 diagnostic.

Completed:

- v3a 20k diagnostic
- v4 two-branch diagnostic
- Revision 9 geometry metric audit

Key results:

v3a 20k:

- kNN finds rare valuable precursors well.
- H1 passes.
- R2V removes selected rare useless and keeps zero-reward precursors.
- H3 is conditionally useful.
- H2 fails because rare-useless enrichment is not stable across seeds.

v4 5k:

- rare-useless transitions are real and label-valid.
- rare-useless collapses into too few unique `obs_action_next` keys.
- kNN treats them as dense.
- H1/H2/H3 fail.

Revision 9 audit:

- raw kNN is fragile on low-diversity discrete grid data.
- duplicate-aware and unique-transition variants do not rescue v4 H2.
- v3a is partly metric-sensitive but not robust across seeds.

Current GPT Pro decision:

```text
one_more_high_diversity_design
```

Only one final Level 0 synthetic design is allowed:

```text
level0_sparse_grid_high_diversity_v5.yaml
```

If v5 fails, stop the synthetic-grid Level 0 route. Do not design v6.

## 5. Workspace And Remote Rules

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

- All tests and experiments must run on `Ubuntu-Tailscale`.

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

Always run relevant tests before experiments:

```bash
~/miniconda3/envs/c2t/bin/python -m pytest -q experiments/r2v_replay/tests
```

Protected local literature directories:

```text
literature_story_mining/
flow_literature_mining/
```

Do not write to, move, rename, delete, format, stage, or commit these directories unless the user explicitly asks.

## 6. Allowed Scope For The Next Codex Round

Allowed:

- Create high-diversity v5 dataset config.
- Add high-diversity observation support if needed.
- Run 5k density audit.
- Run 5k diagnostic only if density audit passes.
- Run candidate-ratio sweep on v5 only.
- Update `AGENTS.md`.
- Update tests for v5.
- Write checkpoint report.

Allowed files:

```text
AGENTS.md
experiments/r2v_replay/configs/level0_sparse_grid_high_diversity_v5.yaml
experiments/r2v_replay/r2v_replay/sparse_grid.py
experiments/r2v_replay/r2v_replay/replay_dataset.py
experiments/r2v_replay/scripts/build_sparse_grid_replay.py
experiments/r2v_replay/scripts/run_density_audit.py
experiments/r2v_replay/tests/test_sparse_grid.py
experiments/r2v_replay/tests/test_replay_dataset.py
experiments/r2v_replay/tests/test_geometry_audit.py
experiments/r2v_replay/reports/pro_revision9_verdict.md
```

Allowed helper files only if necessary:

```text
experiments/r2v_replay/r2v_replay/metric_audit.py
experiments/r2v_replay/r2v_replay/geometry_audit.py
experiments/r2v_replay/scripts/analyze_geometry_metric_failure.py
experiments/r2v_replay/scripts/analyze_candidate_ratio_sweep.py
```

## 7. Forbidden Scope

Do not patch v3a again.

Do not patch v4 again.

Do not create v6 unless GPT Pro explicitly approves after v5.

Do not run 20k.

Do not enter Level 1.

Do not implement RL training.

Do not add or inspect CleanRL, CORL, D4RL, SynthER, PER, HER, or generation.

Do not tune selector, risk, consistency, or diffusion before v5 H1/H2 is established.

Do not replace kNN as official scorer during v5.

Do not use `full_transition` as official evidence.

Do not use reward, done, or labels in main rarity scoring.

Do not mix optional invalid samples into H1/H2/H3.

Do not directly sample transitions by eval label.

Do not use teleport or impossible dynamics as rare-useless.

Do not call zero-reward transitions corrupted, dirty, polluted, or bad data.

Do not hide seed-level failures behind averages.

Do not relax H2 thresholds without GPT Pro approval.

## 8. v5 Design Contract

The v5 design must fix the failure found in Revision 9:

```text
low-diversity rare-useless transitions collapse under obs_action_next kNN
```

v5 must explicitly increase unique `obs_action_next` support for both:

- `rare_valuable_zero_precursor`
- `rare_useless`

v5 must keep rare-useless separate from common-zero in official observation geometry.

Required properties:

- `rare_useless` unique `obs_action_next >= 50` per seed
- `rare_valuable_zero_precursor` unique `obs_action_next >= 50` per seed
- `rare_useless` mean multiplicity <= 3x precursor mean multiplicity
- `rare_useless` median distance to common-zero > 0 for at least 2/3 seeds

Possible design tools:

- continuous jittered observations
- region-specific nuisance features generated by environment observation
- multiple diverse valuable corridors
- multiple diverse useless distractor rooms
- short distractor visits, not long repeated dwell loops

Not allowed:

- label-conditioned features
- reward-conditioned features
- done-conditioned features
- post-hoc relabeling
- teleport transitions
- invalid dynamics
- manual sampling by label

## 9. v5 Command Plan

Run tests:

```bash
cd ~/dder
~/miniconda3/envs/c2t/bin/python -m pytest -q experiments/r2v_replay/tests
```

Run v5 density audit:

```bash
for seed in 0 1 2; do
  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/build_sparse_grid_replay.py \
    --config experiments/r2v_replay/configs/level0_sparse_grid_high_diversity_v5.yaml \
    --seed ${seed} \
    --n-transitions 5000 \
    --output experiments/r2v_replay/outputs/revision10_v5_seed${seed}_density/replay.npz

  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/run_density_audit.py \
    --dataset experiments/r2v_replay/outputs/revision10_v5_seed${seed}_density/replay.npz \
    --config experiments/r2v_replay/configs/level0_sparse_grid_high_diversity_v5.yaml \
    --rarity-input obs_action_next \
    --k 50 \
    --output-dir experiments/r2v_replay/outputs/revision10_v5_seed${seed}_density/density_obs_action_next
done
```

Only if density gate passes, run full 5k diagnostic:

```bash
for seed in 0 1 2; do
  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/run_level0_diagnostic.py \
    --dataset experiments/r2v_replay/outputs/revision10_v5_seed${seed}_density/replay.npz \
    --config experiments/r2v_replay/configs/level0_sparse_grid_high_diversity_v5.yaml \
    --rarity-input obs_action_next \
    --rare-topk-ratio 0.10 \
    --selected-budget-ratio 0.05 \
    --seed ${seed} \
    --output-dir experiments/r2v_replay/outputs/revision10_v5_seed${seed}_full5k/diagnostic_obs_action_next
done
```

Only if full 5k diagnostic runs, run candidate sweep:

```bash
for seed in 0 1 2; do
  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/analyze_candidate_ratio_sweep.py \
    --diagnostic-dir experiments/r2v_replay/outputs/revision10_v5_seed${seed}_full5k/diagnostic_obs_action_next \
    --scorer knn \
    --ratios 0.005,0.01,0.02,0.05,0.10,0.20 \
    --output experiments/r2v_replay/outputs/revision10_v5_seed${seed}_candidate_sweep
done
```

## 10. v5 Pass/Fail Rules

Density gate passes only if all seeds satisfy:

- `rare_useless` unique `obs_action_next >= 50`
- `rare_valuable_zero_precursor` unique `obs_action_next >= 50`

And at least 2/3 seeds satisfy:

- `rare_useless` median distance to common-zero > 0

H1 passes if all seeds satisfy either:

- kNN Recall@Top10 rare valuable >= 0.50
- kNN Recall@Top20 rare valuable >= 0.70

H2 passes if there exists one ratio in `{0.005, 0.01, 0.02, 0.05, 0.10}` such that all seeds satisfy:

- `rare_useless_count >= 10`
- `rare_useless_fraction >= 0.04`
- `rare_useless_enrichment >= 2.0`
- `rare_valuable_all_count >= 50` or `rare_valuable_recall >= 0.30`

H3 passes if all seeds satisfy:

- R2V selected rare-useless fraction < candidate rare-useless fraction
- R2V selects nonzero zero-reward precursors
- R2V positive reward fraction <= 0.50
- R2V zero-precursor recall > reward-only zero-precursor recall

H4 passes if all seeds satisfy:

- optional-invalid excluded from H1-H3
- invalid-risk AUROC >= 0.85

If v5 fails any of density/H1/H2/H3 after one clean implementation, stop the synthetic-grid Level 0 route.

## 11. Reporting Requirements

Every checkpoint must include:

- repo
- branch
- commit
- Ubuntu host
- Ubuntu path
- Python executable
- test command and result
- Mac experiment status
- exact commands run
- files changed
- configs changed
- seed-level results
- mean/std only after seed-level table
- artifact paths
- Codex recommendation
- questions for GPT Pro

Do not report only averages.

Do not omit failed seeds.

Do not omit optional-invalid exclusion status.

Do not omit whether reward/done/labels were excluded from official rarity scoring.

Required v5 checkpoint title:

```text
R2V-Replay Round 1 Revision 10 High-Diversity-v5 Report
```

Required final Codex recommendation options:

- `go_to_20k`
- `stop_route`
- `return_to_story_framing`
- `stop_level0`

## 12. Secret And Safety Rules

Never commit or print:

- tokens
- passwords
- SSH private keys
- server credentials
- `.env` files
- private host details beyond approved alias names

Do not assume unknown server paths or credentials.

Use placeholders when needed:

```text
<REMOTE_WORKDIR>
<UBUNTU_USER>
<UBUNTU_HOST>
<GITHUB_DDER_URL>
```

Do not run network calls, package installs, or external downloads unless the user or GPT Pro explicitly asks.

Do not expose sensitive local files.

Do not share unrelated local assets.

## 13. Git Hygiene

Prefer small, reviewable diffs.

Do not use `git add .`.

Before staging, run:

```bash
git status --short
```

Stage exact files only.

Never stage or commit:

- `literature_story_mining/`
- `flow_literature_mining/`
- experiment `outputs/`
- caches
- credentials

Commit reports and small source/config/test files only unless GPT Pro explicitly requests otherwise.

## 14. Checkpoint Rule

After v5 results, Codex must stop and ask GPT Pro.

No automatic next stage.

No automatic Level 1.

No automatic 20k.

No automatic new dataset.

If v5 succeeds, GPT Pro decides whether to run 20k.

If v5 fails, GPT Pro decides between:

- `stop_route`
- `return_to_story_framing`
- `stop_level0`
