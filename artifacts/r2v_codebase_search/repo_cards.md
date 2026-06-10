# R2V-Replay Codebase Search - Repo Cards

Status: in progress

## Current Round B Shortlist View

This is not a final recommendation. It is the first evidence-backed candidate
pool to send back to Pro for Round C audit.

Top candidate groups:

- Benchmark-first: `C001 OGBench`, `C002 horizon-reduction`
- Clean offline RL implementation: `C003 CORL`, `C004 OfflineRL-Kit`, `C005 d3rlpy`
- Modern library substrate: `C006 TorchRL` with `C007 Minari`
- Neighbor baseline: `C009 OPER`
- Boundary-only generation/planning: `C010 SynthER`, `C012 PG`

### `C001` OGBench

- Repo: https://github.com/seohongpark/ogbench
- Paper / docs: https://seohong.me/projects/ogbench/
- Candidate family: benchmark-first route for offline goal-conditioned RL.
- Benchmark tasks: humanoidmaze, antmaze, cube, scene, puzzle, powderworld;
  both goal-conditioned and standard single-task variants.
- Why it may fit R2V-Replay: official README states success reward is `1` and
  otherwise `0`, which directly gives real but mostly low-information
  zero-reward transitions.
- Sparse or delayed reward evidence: README documents `reward` as 1 on goal
  reach and 0 otherwise.
- Replay / sampler / dataloader hook: `impls/utils/datasets.py` has
  `Dataset.sample` and `GCDataset.sample`; `impls/main.py` calls
  `train_dataset.sample(config['batch_size'])`.
- Baselines available: GCBC, GCIVL, GCIQL, QRL, CRL, HIQL.
- Install and smoke-test path: `pip install ogbench`; import; call
  `ogbench.make_env_and_datasets(..., env_only=True)` first, then a small
  dataset load only if Pro approves.
- Risks: MuJoCo/dm_control dependencies and dataset download size.
- Current score: 36 / 40.
- Hard veto: false.

### `C002` horizon-reduction

- Repo: https://github.com/seohongpark/horizon-reduction
- Candidate family: OGBench algorithm implementation route.
- Why it may fit R2V-Replay: built on OGBench and offers several offline
  goal-conditioned algorithms including IQL and hierarchical variants.
- Replay / sampler hook: `utils/datasets.py` and `main.py` sampling path.
- Baselines available: Flow BC, IQL, CRL, SAC+BC, FQL, HIQL, SHARSA.
- Smoke-test concern: README emphasizes 100M/1B datasets for main paper; smoke
  must avoid those and use a small OGBench task if Pro approves.
- Current score: 32 / 40.
- Hard veto: false.

### `C003` CORL

- Repo: https://github.com/tinkoff-ai/CORL
- Paper / docs: https://openreview.net/forum?id=SyAS49bBcv
- Candidate family: CleanRL-style D4RL implementation base.
- Why it may fit R2V-Replay: single-file algorithms make the sampler/batch path
  visible and patchable; AntMaze and Maze2D benchmark tables are documented.
- Sparse evidence: AntMaze and Maze2D D4RL tasks in README tables and configs.
- Replay hook: each algorithm file has local `ReplayBuffer`,
  `load_d4rl_dataset`, and `sample`.
- Baselines available: CQL, IQL, TD3+BC, AWAC, BC, EDAC, ReBRAC, DT, Cal-QL.
- Risks: D4RL and MuJoCo dependency fragility; repo last cloned commit 2023.
- Current score: 33 / 40.
- Hard veto: false.

### `C004` OfflineRL-Kit

- Repo: https://github.com/yihaosun1124/OfflineRL-Kit
- Candidate family: PyTorch offline RL library.
- Why it may fit R2V-Replay: very explicit reusable `ReplayBuffer` component and
  trainer path; recent commit in shallow clone.
- Replay hook: `offlinerlkit/buffer/buffer.py` has `ReplayBuffer.load_dataset`
  and `ReplayBuffer.sample`; trainers call `buffer.sample`.
- Baselines available: CQL, TD3+BC, IQL, EDAC, MCQ, MOPO, COMBO, RAMBO, MOBILE.
- Weakness: README results focus on MuJoCo medium/replay/expert rather than
  sparse success tasks.
- Current score: 30 / 40.
- Hard veto: false.

### `C005` d3rlpy

- Repo: https://github.com/takuseno/d3rlpy
- Paper: https://arxiv.org/abs/2111.03788
- Candidate family: mature offline RL library.
- Why it may fit R2V-Replay: strong package health, many baselines, and a
  dataset/replay API that can be smoke-tested without full MuJoCo.
- Replay hook: `d3rlpy/dataset/replay_buffer.py` and q-learning base call
  `sample_transition_batch`.
- Weakness: sparse benchmark fit is indirect unless paired with D4RL AntMaze,
  Minari, or OGBench conversion.
- Current score: 32 / 40.
- Hard veto: false.

### `C006` TorchRL

- Repo: https://github.com/pytorch/rl
- Paper: https://arxiv.org/abs/2306.00577
- Candidate family: modern PyTorch substrate for custom replay/sampler design.
- Why it may fit R2V-Replay: replay buffers are modular by storage, sampler,
  writer, transforms, priority updates, and offline data; Minari IQL tutorial
  exists.
- Replay hook: `torchrl.data` replay buffers and samplers; docs show
  `TensorDictPrioritizedReplayBuffer`, `PrioritizedSampler`, and
  `MinariExperienceReplay`.
- Weakness: not a specific benchmark or paper repo; Pro must decide whether a
  library substrate is acceptable as the main route.
- Current score: 33 / 40.
- Hard veto: false.

### `C007` Minari

- Repo: https://github.com/Farama-Foundation/Minari
- Candidate family: offline dataset layer.
- Why it may fit R2V-Replay: standard offline dataset interface, remote dataset
  CLI, and integration with TorchRL and d3rlpy.
- Hook: dataset iteration/loading only; needs TorchRL/d3rlpy or custom sampler.
- Current score: 29 / 40.
- Hard veto: false, but not standalone.

### `C009` OPER

- Repo: https://github.com/sail-sg/OPER
- Candidate family: nearest replay-prioritization baseline.
- Why it may fit R2V-Replay: explicitly resamples/reweights offline data using
  priority weights. This is close enough that reviewers may compare against it.
- Replay hook: `utils.py` includes `ReplayBuffer`, samplers, `sample`, and
  `replace_weights`.
- Weakness: it may be a better baseline/contrast than a base repo, because R2V
  needs rarity discovery plus value filtering rather than only priority weights.
- Current score: 30 / 40.
- Hard veto: false.

### Boundary-Only Cards

`C010 SynthER` and `C012 PG` are useful to keep in the literature and reviewer
contrast layer, but they should not become the primary route unless Pro
explicitly reopens generation/planning as a different idea. They violate the
current project boundary if used as final evidence because generation or
diffusion planning would become the main mechanism.

## Card Template

### `[candidate_id]` repo_name

- Repo:
- Paper / docs:
- Candidate family:
- Benchmark tasks:
- Why it may fit R2V-Replay:
- Sparse or delayed reward evidence:
- Zero-reward transition evidence:
- Replay / sampler / dataloader hook:
- Baselines available:
- Expected modification files:
- Install and smoke-test path:
- License / reproducibility:
- Risks:
- Evidence URLs:
- Current score:
- Hard veto:
