# R2V-Replay Codebase Search - Exclusion Log

Status: in progress

| candidate_or_repo | source_url | reason | evidence | date_checked |
|---|---|---|---|---|
| `aviralkumar2907/CQL` | https://github.com/aviralkumar2907/CQL | Not selected as primary codebase; missing root license in shallow clone and stale commit. Use CORL/d3rlpy/OfflineRL-Kit for CQL-style baselines instead. | Local shallow clone latest commit 2020-09-14; root checkout exposed README only and no LICENSE file. | 2026-06-10 |
| `ml-jku/rudder` | https://github.com/ml-jku/rudder | Not suitable as executable base for current route. Old concept/code, no license in shallow clone, no clear replay/dataset sampling substrate. | Local shallow clone latest commit 2020-09-17; root files only README and poster PDF. | 2026-06-10 |
| `suyoung-lee/Episodic-Backward-Update` | https://github.com/suyoung-lee/Episodic-Backward-Update | Not suitable as modern executable base. Old Atari-focused implementation, no license in shallow clone, ROM files create extra reproducibility risk. | Local shallow clone latest commit 2019-09-24; root checkout exposed README and `roms/*` in tree. | 2026-06-10 |
| `jannerm/diffuser` | https://github.com/jannerm/diffuser | Boundary-only, not primary route. Diffusion trajectory planning is not replay sampler selection and would shift mechanism away from R2V. | README and tree show diffusion planner with D4RL dataset buffer; local shallow clone latest commit 2022-12-20. | 2026-06-10 |
| `Planning with Generated Rewards` ambiguous route | https://github.com/ku-dmlab/PG | Search found Prior-Guided Diffusion Planning, not a separate replay codebase for Planning with Generated Rewards. Keep PG as boundary method only. | Web search returned `ku-dmlab/PG`, NeurIPS 2025 diffusion planning repository. | 2026-06-10 |
| GitHub API metadata route | https://api.github.com/repos/... | Metadata route blocked by unauthenticated API rate limit; not a candidate exclusion but logged as failed evidence channel. | HTTP 403 rate limit exceeded for most repo metadata requests. Fallback used official pages and shallow clones. | 2026-06-10 |
