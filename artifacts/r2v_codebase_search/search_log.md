# R2V-Replay Codebase Search Log

Date started: 2026-06-10

Round: B high-recall code / literature search

Status: in progress

## Search Rules

- Prefer official repository, paper, benchmark documentation, and maintained package documentation.
- Record source URL, access date, and what evidence was extracted.
- Do not treat GPT Pro output as evidence.
- Do not score a candidate without evidence for the score.
- Keep excluded repos in `exclusion_log.md` instead of silently dropping them.

## Seed Query Families From Pro

```text
offline RL sparse reward replay buffer GitHub IQL CQL TD3+BC AntMaze
offline reinforcement learning AntMaze IQL CQL TD3+BC code
OGBench offline goal-conditioned RL GitHub replay sampler
D4RL Minari AntMaze offline RL GitHub IQL CQL
d3rlpy offline RL dataset sampler GitHub
CORL offline RL GitHub IQL CQL TD3+BC
OfflineRL-Kit GitHub offline reinforcement learning
HER sparse reward replay buffer GitHub
RUDDER reinforcement learning delayed rewards GitHub
episodic backward update replay GitHub
SynthER synthetic experience replay GitHub
Planning with Generated Rewards PGR GitHub offline RL
diffusion replay buffer offline RL GitHub
```

## Queries Run

### Web search batch 1 - benchmark and implementation bases

Date checked: 2026-06-10

Queries:

- `OGBench offline goal-conditioned reinforcement learning GitHub official`
- `CORL clean offline reinforcement learning GitHub IQL CQL TD3+BC`
- `d3rlpy offline reinforcement learning GitHub official dataset sampler`
- `OfflineRL-Kit GitHub offline reinforcement learning official`
- `D4RL GitHub Farama Foundation Minari official AntMaze offline RL`
- `implicit Q-learning GitHub official IQL AntMaze offline RL`
- `Conservative Q-Learning GitHub official CQL offline RL D4RL`
- `RLkit GitHub offline RL HER replay buffer official`

Evidence recorded:

- OGBench official GitHub README: 85 datasets, 410 standard offline RL tasks,
  reward is 1 on goal success and 0 otherwise, `ogbench.make_env_and_datasets`,
  reference algorithms in `impls`.
- CORL official GitHub README: Clean Offline RL single-file implementations,
  Apache-2.0, D4RL benchmark tables including Maze2D and AntMaze, CQL/IQL/TD3+BC.
- d3rlpy official GitHub README: offline RL library, PyPI install, D4RL and
  Atari dataset helpers, replay buffer API.
- OfflineRL-Kit official GitHub README: PyTorch offline RL library, MIT,
  CQL/TD3+BC/IQL/EDAC/MOPO/COMBO/RAMBO/MOBILE, explicit `ReplayBuffer`.

### Web search batch 2 - dataset layers and replay methods

Date checked: 2026-06-10

Queries:

- `Minari Farama GitHub offline RL dataset official replay buffer`
- `TorchRL offline RL replay buffer Minari IQL tutorial official`
- `Stable Baselines3 HER replay buffer sparse reward official docs`
- `OGBench reward success 1 otherwise 0 offline goal-conditioned RL reward GitHub`
- `RUDDER reinforcement learning delayed rewards GitHub official`
- `Episodic Backward Update EBU replay GitHub reinforcement learning official`
- `SynthER synthetic experience replay GitHub offline reinforcement learning official`
- `Planning with Generated Rewards PGR GitHub offline reinforcement learning`

Evidence recorded:

- Minari official README: offline RL dataset library, `pip install minari`,
  CLI for remote datasets, `minari.load_dataset`, integrations with TorchRL and
  d3rlpy.
- TorchRL official README and docs: modular replay buffers, samplers, storage,
  prioritized replay, HER, offline dataset helpers, `MinariExperienceReplay`,
  and Minari IQL tutorial.
- Stable-Baselines3 official README/docs: maintained PyTorch RL library,
  `HerReplayBuffer`, replay buffer save/load, strong online sparse-reward
  substrate but not an offline RL benchmark base.
- OPER official README: offline prioritized experience replay with
  resampling/reweighting on D4RL, close baseline family for R2V.
- SynthER official README: diffusion-based synthetic experience replay,
  integrated with SAC, TD3+BC, IQL, EDAC, and CQL. Marked boundary-only for
  this project because generation cannot be final evidence.

### Web search batch 3 - generation boundary and recent planning repos

Date checked: 2026-06-10

Queries:

- `"Planning with Generated Rewards" "ku-dmlab" GitHub`
- `"PG: Planning with Generated Rewards" offline reinforcement learning GitHub`
- `"Planning with Generated Rewards" "D4RL"`
- `"ku-dmlab/PG"`

Evidence recorded:

- `ku-dmlab/PG` is Prior-Guided Diffusion Planning for Offline Reinforcement
  Learning, NeurIPS 2025, with D4RL AntMaze, Maze2D, Kitchen, and MuJoCo
  dataset classes. It is a useful boundary or contrast but not a replay-sampler
  base.
- No unambiguous official executable repository for a separate "Planning with
  Generated Rewards" route was selected in this round.

### GitHub metadata route

Date checked: 2026-06-10

Attempted public GitHub API metadata lookup for candidate repos. Result:
rate-limited with HTTP 403 for most repos. Fallback used official GitHub pages
plus read-only shallow sparse clones under `/tmp/r2v_codebase_repos`.

### Read-only shallow clone metadata

Date checked: 2026-06-10

Local metadata path: `/tmp/r2v_codebase_repos`

Commands were metadata-only. No experiments or training were run on Mac.

Repos shallow-cloned:

- `seohongpark/ogbench`
- `tinkoff-ai/CORL`
- `takuseno/d3rlpy`
- `yihaosun1124/OfflineRL-Kit`
- `Farama-Foundation/D4RL`
- `Farama-Foundation/Minari`
- `DLR-RM/stable-baselines3`
- `sail-sg/OPER`
- `ikostrikov/implicit_q_learning`
- `aviralkumar2907/CQL`
- `conglu1997/synther`
- `ml-jku/rudder`
- `suyoung-lee/Episodic-Backward-Update`
- `jannerm/diffuser`
- `ku-dmlab/PG`
- `seohongpark/horizon-reduction`
- `pytorch/rl`

Key local evidence extracted:

- `ogbench`: last cloned commit date 2026-01-14; MIT; `impls/utils/datasets.py`
  has `Dataset.sample` and `GCDataset.sample`; README documents 1/0 success
  rewards and `make_env_and_datasets`.
- `horizon-reduction`: last cloned commit date 2025-08-01; MIT; OGBench-based
  offline goal-conditioned algorithms; very large dataset pathway is a smoke
  risk.
- `CORL`: last cloned commit date 2023-08-01; Apache-2.0; each algorithm has a
  local `ReplayBuffer`, `load_d4rl_dataset`, and `sample`.
- `OfflineRL-Kit`: last cloned commit date 2026-05-01; MIT; `offlinerlkit/buffer/buffer.py`
  has `ReplayBuffer.load_dataset` and `ReplayBuffer.sample`.
- `d3rlpy`: last cloned commit date 2025-09-11; MIT; `d3rlpy/dataset/replay_buffer.py`
  and algorithm base call `sample_transition_batch`.
- `TorchRL`: last cloned commit date 2026-06-09; MIT; README documents modular
  replay buffers, samplers, storage, prioritized replay, HER, and offline data.
- `OPER`: last cloned commit date 2023-06-13; Apache-2.0; `utils.py` has
  `ReplayBuffer`, `RandSampler`, `PrefetchBalancedSampler`, `sample`, and weight
  replacement.
- `SynthER`: last cloned commit date 2026-04-16; MIT; boundary-only generative
  replay code with CORL-derived algorithms.
- `implicit_q_learning`: last cloned commit date 2022-01-23; MIT; simple JAX
  IQL baseline with AntMaze configs and `Dataset.sample`.
- `PG`: last cloned commit date 2025-10-24; MIT; diffusion-planning boundary
  with D4RL AntMaze, Maze2D, Kitchen dataset `sample` methods.
