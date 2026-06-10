# Round B Summary For Pro

Date: 2026-06-10

Status: ready for Pro Round C audit after Codex review.

No experiments run. No Ubuntu smoke tests run yet.

Mac activity was metadata-only: official web search and read-only shallow sparse
clones under `/tmp/r2v_codebase_repos`.

## Top Candidates By Current Score

| rank | candidate | score | role |
|---|---:|---:|---|
| 1 | `C001 OGBench` | 36 | strongest benchmark-first route |
| 2 | `C003 CORL` | 33 | clean single-file offline RL implementation base |
| 2 | `C006 TorchRL` | 33 | modern replay/sampler engineering substrate |
| 4 | `C002 horizon-reduction` | 32 | OGBench algorithm route with large-data risk |
| 4 | `C005 d3rlpy` | 32 | mature offline RL library substrate |
| 6 | `C004 OfflineRL-Kit` | 30 | explicit ReplayBuffer PyTorch library |
| 6 | `C009 OPER` | 30 | closest prioritized replay baseline / contrast |
| 8 | `C007 Minari` | 29 | dataset layer, not standalone algorithm base |
| 8 | `C013 stable-baselines3` | 29 | replay/HER mechanics, weaker offline fit |
| 10 | `C011 implicit_q_learning` | 28 | simple official IQL baseline, older stack |

Boundary-only:

- `C010 SynthER`
- `C012 PG`

## Proposed Round C Shortlist

Codex proposes that Pro audit these as smoke-test candidates:

1. `C001 OGBench`
2. `C003 CORL`
3. `C006 TorchRL + C007 Minari`
4. `C005 d3rlpy`
5. `C004 OfflineRL-Kit`

`C009 OPER` should be audited as a nearest baseline or contrast, not necessarily
as the main substrate.

## Top Uncertainties For Pro

1. Should the next empirical route be benchmark-first (`OGBench`) or
   implementation-first (`CORL` / `OfflineRL-Kit`)?
2. Is a modern substrate route (`TorchRL + Minari`) scientifically acceptable,
   or does it look too engineering-infrastructure-heavy for a paper route?
3. Should D4RL AntMaze remain acceptable despite Farama's migration warning, or
   should we avoid D4RL legacy dependency and prioritize OGBench/Minari?

## Codex Recommendation Before Pro Audit

Recommended primary audit direction:

```text
OGBench as benchmark-first route, with either OGBench impls or TorchRL/d3rlpy
as the sampler implementation layer.
```

Recommended implementation fallback:

```text
CORL or OfflineRL-Kit if Pro prefers a visibly patchable ReplayBuffer.sample
path over a newer benchmark.
```

Do not proceed to Ubuntu smoke tests until Pro approves the shortlist.

