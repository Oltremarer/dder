# R2V-Replay Codebase Search - Round A Protocol Lock

Date: 2026-06-10

Pro thread: `https://chatgpt.com/c/6a28e139-8c08-83ea-8dd4-e38c6d2631f4`

Repo: `https://github.com/Oltremarer/dder`

Branch: `main`

Commit at protocol lock: `5f366737a20e0e7c716e9763705e1ec13a6476c5`

Ubuntu host: `Ubuntu-Tailscale`

Ubuntu path: `~/dder`

Python executable for future Ubuntu commands: `~/miniconda3/envs/c2t/bin/python`

Mac experiment status: no experiments run on Mac.

Commands run for this protocol lock: no experiments run.

## Pro Round A Verdict

Pro did not choose a final codebase in Round A. Pro locked a search-and-audit
workflow. Codex must now run a high-recall codebase search, collect evidence,
score candidates, and return a Round B package to Pro.

## Objective

Find an executable public codebase / benchmark / algorithm base that can fairly
test the R2V-Replay hypothesis:

```text
In sparse-reward replay or offline datasets, many zero-reward transitions are
real but mostly low-information zero-reward transitions. R2V-Replay should use
rarity only to discover candidates, then use learning utility / value evidence
to decide which rare transitions are valuable enough to replay.
```

The desired codebase must support an insertion point like:

```text
uniform sampler -> rarity candidate miner -> value filter / scorer -> replay sampling weight
```

The insertion point should be in replay, dataset sampling, batch construction,
or loss weighting. It should not require changing the reward function or
claiming zero-reward data is corrupted.

## Hard Non-Goals

- Do not revive the stopped low-dimensional synthetic-grid route.
- Do not create v6, patch v5/v4/v3a, rerun 20k, or enter the old Level 1.
- Do not treat diffusion or generative replay as final evidence.
- Do not call zero-reward replay corrupted, dirty, polluted, bad data, or broken data.
- Do not manufacture sparsity by reward hacking.
- Do not choose a repository only because installation is convenient.
- Do not claim final route selection before Pro Round C/E approval.

## Candidate Families To Search

Pro requested high recall across these families:

- Benchmark-first: OGBench, D4RL / Minari, AntMaze, Maze2D, Kitchen, Adroit,
  Meta-World, offline goal-conditioned RL benchmarks.
- Offline RL implementation bases: CORL, d3rlpy, OfflineRL-Kit, rlkit-style
  codebases, IQL, CQL, TD3+BC, BCQ, BRAC.
- Sparse-reward / replay / credit-assignment methods: HER, PER, RUDDER, EBU,
  OPER or neighboring replay-prioritization families.
- Boundary / contrast methods: SynthER, PGR, flow or diffusion replay code.
- Existing project repo `dder` only as a project container, not as an external
  benchmark evidence source.

## Selection Rubric

Each dimension is scored 0-5. Do not assign a score without evidence.

| Dimension | 0 | 1-2 | 3 | 4-5 |
|---|---|---|---|---|
| Scientific fit | unrelated to sparse/offline replay | only generic RL or dense reward | has offline data or replay but weak sparsity | directly supports sparse/delayed reward replay questions |
| Implementation health | no runnable code | stale or undocumented | install path exists with risk | maintained, documented, standard entrypoints |
| Dataset / benchmark fit | toy-only | nonstandard tasks | standard tasks but weak sparse evidence | accepted benchmark with sparse or delayed reward tasks |
| Modification surface | no clear hook | invasive algorithm rewrite needed | possible hook but unclear | clear replay/dataloader/sampler hook |
| Smoke-test feasibility | cannot import | heavy or fragile install | likely importable with small smoke | documented minimal run/data-load path |
| Baseline fairness | no baselines | unfair or mismatched baselines | some fair comparisons | multiple standard baselines under same data/tasks |
| Reviewer risk | likely unacceptable | novelty confounded | defensible with caveats | clean story and credible benchmark |
| License/reproducibility | missing license | unclear dependencies | workable with notes | license, scripts, seeds/logging clear |

Hard veto overrides total score.

## Round Protocol

### Round A - Flow Lock

Codex saves this protocol, creates artifact templates, and does not run
experiments.

Gate to Round B: protocol and audit prompts are saved.

### Round B - High-Recall Code / Literature Search

Codex searches candidate repos and papers, then produces:

- `search_log.md`
- `codebase_candidates.csv`
- `repo_cards.md`
- `exclusion_log.md`

Gate to Round C: enough evidence exists for Pro to audit candidate coverage,
scoring evidence, and whether any candidate offers a real replay/sampler hook.

### Round C - Pro Short-List Audit

Pro checks:

- candidate pool coverage
- evidence quality behind scores
- whether R2V is insertable as rarity candidate + value filter
- whether zero-reward data is treated correctly
- whether benchmarks are reviewer-credible

Gate to Round D: Pro approves a short list for Ubuntu smoke tests.

### Round D - Ubuntu Smoke Test

Codex tests only approved candidates on `Ubuntu-Tailscale`.

Smoke test is engineering feasibility only, not scientific success. It should
verify import, dependency setup, dataset listing or loading, and a minimal batch
or algorithm entrypoint where possible.

### Round E - Final Route Recommendation

Codex returns smoke-test evidence. Pro issues go / revise / reject for the
empirical route and names the preferred codebase if ready.

## Stop Conditions

- Round B finds fewer than four candidates with scientific fit >= 4.
- Round C finds no candidate with a clear replay, sampler, dataset, or
  batch-construction hook.
- Round D finds no approved candidate can complete import + dataset access +
  minimal sample / batch construction.
- The only promising routes require reward hacking, toy-grid revival,
  zero-reward-as-corruption framing, or generative replay as evidence.

