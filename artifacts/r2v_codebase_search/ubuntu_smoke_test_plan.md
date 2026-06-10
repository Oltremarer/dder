# Ubuntu Smoke Test Plan

Date: 2026-06-10

Status: draft for Pro review. Do not execute before Pro approves the dry-run
command list.

Ubuntu host: `Ubuntu-Tailscale`

Ubuntu path: `~/dder`

Required preflight before any Ubuntu command:

```bash
cd ~/dder
git status --short
git rev-parse HEAD
```

Python policy:

- Use `~/miniconda3/envs/c2t/bin/python` as the trusted base Python.
- To avoid polluting the main project environment, create isolated venvs under
  `/tmp/r2v_smoke_envs` using the trusted base Python.
- Do not install global CUDA, change drivers, or install system packages.

Global caps:

- No dataset download above 10 GB.
- First-pass preferred per-candidate download cap: 2 GB.
- No full RL training.
- Tiny update means at most 10-50 gradient steps, only if import, dataset load,
  and sampler hook already passed.

## C001 OGBench

Repo path:

```text
/tmp/r2v_smoke_repos/ogbench
```

Environment name:

```text
env-only: pointmaze-medium-navigate-v0
real dataset preferred: antmaze-medium-navigate-v0
real dataset fallback: pointmaze-medium-navigate-v0 only if AntMaze fails or is too large
```

Install command:

```bash
~/miniconda3/envs/c2t/bin/python -m venv /tmp/r2v_smoke_envs/ogbench
. /tmp/r2v_smoke_envs/ogbench/bin/activate
python -m pip install --upgrade pip
python -m pip install ogbench
```

Import command:

```bash
python - <<'PY'
import ogbench
print("OGBENCH_IMPORT_OK", getattr(ogbench, "__version__", "unknown"))
PY
```

Dataset load command:

```bash
mkdir -p /tmp/r2v_smoke_data/ogbench
python - <<'PY'
import json
import numpy as np
import ogbench

dataset_name = "pointmaze-medium-navigate-v0"
obj = ogbench.make_env_and_datasets(dataset_name, env_only=True)
env = obj[0] if isinstance(obj, tuple) else obj
print("OGBENCH_ENV_ONLY_OK", dataset_name, type(env).__name__)

real_dataset_name = "antmaze-medium-navigate-v0"
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
    real_dataset_name,
    dataset_dir="/tmp/r2v_smoke_data/ogbench",
    compact_dataset=False,
)
rewards = np.asarray(train_dataset["rewards"])
summary = {
    "dataset_name": real_dataset_name,
    "n_transitions": int(rewards.shape[0]),
    "reward_min": float(np.min(rewards)),
    "reward_max": float(np.max(rewards)),
    "zero_reward_ratio": float(np.mean(rewards == 0)),
    "positive_reward_ratio": float(np.mean(rewards > 0)),
    "keys": sorted(list(train_dataset.keys())),
}
print("OGBENCH_DATASET_SUMMARY", json.dumps(summary, sort_keys=True))
PY
du -sh /tmp/r2v_smoke_data/ogbench
```

Batch sampling command:

```bash
python - <<'PY'
import ast
from pathlib import Path

repo = Path("/tmp/r2v_smoke_repos/ogbench")
if not repo.exists():
    raise SystemExit("clone repo first")
hook = repo / "impls/utils/datasets.py"
ast.parse(hook.read_text())
print("OGBENCH_STATIC_PARSE_OK", hook)
print("OGBENCH_HOOK_FILE", hook)
PY
```

Tiny train:

```text
No tiny train in first pass. Pro only approved import, dataset load, reward
ratio, and sampler hook confirmation before any method patch.
```

Max time: 60 minutes.

Max download size: 2 GB first-pass, hard stop at 10 GB.

Expected log files:

- `artifacts/r2v_codebase_search/logs/ubuntu_C001_ogbench.log`

## C003 CORL

Repo path:

```text
/tmp/r2v_smoke_repos/CORL
```

Environment name:

```text
antmaze-umaze-v2 for optional D4RL check; dummy replay first
```

Install command:

```bash
~/miniconda3/envs/c2t/bin/python -m venv /tmp/r2v_smoke_envs/corl
. /tmp/r2v_smoke_envs/corl/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Clone/source command:

```bash
mkdir -p /tmp/r2v_smoke_repos
git clone --depth 1 --filter=blob:none --sparse https://github.com/tinkoff-ai/CORL.git /tmp/r2v_smoke_repos/CORL
cd /tmp/r2v_smoke_repos/CORL
git sparse-checkout set algorithms/offline configs requirements
git rev-parse HEAD
```

Import command:

```text
No full algorithm import before dependency review because CORL imports d4rl,
gym, wandb, and pyrallis at module top level. First smoke executes only the
ReplayBuffer class source with torch/numpy.
```

Dataset load command:

```text
No D4RL dataset load in first dummy pass. Optional D4RL load requires Pro
approval after dummy replay succeeds.
```

Batch sampling command:

```bash
python - <<'PY'
import ast
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List

source_path = Path("/tmp/r2v_smoke_repos/CORL/algorithms/offline/iql.py")
source = source_path.read_text()
module = ast.parse(source)
cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "ReplayBuffer")
segment = ast.get_source_segment(source, cls)
ns = {"np": np, "torch": torch, "Dict": Dict, "List": List, "TensorBatch": List[torch.Tensor]}
exec(segment, ns)
ReplayBuffer = ns["ReplayBuffer"]
rb = ReplayBuffer(state_dim=3, action_dim=2, buffer_size=16, device="cpu")
data = {
    "observations": np.zeros((8, 3), dtype=np.float32),
    "actions": np.ones((8, 2), dtype=np.float32),
    "next_observations": np.ones((8, 3), dtype=np.float32),
    "rewards": np.array([0,0,0,0,0,1,0,0], dtype=np.float32),
    "terminals": np.zeros((8,), dtype=np.float32),
}
rb.load_d4rl_dataset(data)
batch = rb.sample(4)
print("CORL_DUMMY_REPLAY_OK", [tuple(x.shape) for x in batch])
print("CORL_ZERO_REWARD_RATIO_DUMMY", float((data["rewards"] == 0).mean()))
PY
```

Tiny train:

```text
No tiny train in first pass. This candidate first proves replay hook only.
```

Max time: 45 minutes.

Max download size: 0 GB first pass.

Expected log files:

- `artifacts/r2v_codebase_search/logs/ubuntu_C003_corl.log`

## C006 + C007 TorchRL + Minari

Repo path:

```text
/tmp/r2v_smoke_repos/torchrl
```

Environment / dataset name:

```text
Minari remote listing first; optional sparse dataset only after Pro approval
```

Install command:

```bash
~/miniconda3/envs/c2t/bin/python -m venv /tmp/r2v_smoke_envs/torchrl_minari
. /tmp/r2v_smoke_envs/torchrl_minari/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install torchrl tensordict minari
```

Import command:

```bash
python - <<'PY'
import torch
import torchrl
import minari
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
print("TORCHRL_MINARI_IMPORT_OK", torch.__version__, getattr(torchrl, "__version__", "unknown"))
PY
```

Dataset list/load command:

```bash
python -m minari list remote > /tmp/r2v_minari_remote.txt
sed -n '1,80p' /tmp/r2v_minari_remote.txt
```

Batch sampling command:

```bash
python - <<'PY'
import torch
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler

storage = LazyMemmapStorage(32, scratch_dir="/tmp/r2v_torchrl_storage")
rb = TensorDictReplayBuffer(storage=storage, sampler=RandomSampler())
td = TensorDict({
    "observation": torch.zeros(16, 3),
    "action": torch.ones(16, 2),
    "next": {"observation": torch.ones(16, 3), "reward": torch.tensor([[0.0]] * 15 + [[1.0]])},
    "done": torch.zeros(16, 1, dtype=torch.bool),
}, batch_size=[16])
rb.extend(td)
batch = rb.sample(4)
rewards = td["next", "reward"].reshape(-1)
print("TORCHRL_DUMMY_REPLAY_OK", batch.batch_size, float((rewards == 0).float().mean()))
PY
```

Tiny train:

```text
No tiny train in first pass. Tiny IQL only after a real Minari sparse dataset
is selected and loaded under cap.
```

Max time: 60 minutes.

Max download size: 2 GB for one real sparse dataset attempt.

Real sparse Minari dataset attempt:

```text
Try in order:
1. D4RL/pointmaze/medium-v2
2. D4RL/antmaze/medium-diverse-v1
3. D4RL/door/human-v2 as API-bridge fallback only
```

Expected log files:

- `artifacts/r2v_codebase_search/logs/ubuntu_C006_C007_torchrl_minari.log`

## C005 d3rlpy

Repo path:

```text
/tmp/r2v_smoke_repos/d3rlpy
```

Trigger:

```text
Run only if Pro approves fallback after first three candidates fail or remain
inconclusive.
```

Planned first fallback command:

```bash
~/miniconda3/envs/c2t/bin/python -m venv /tmp/r2v_smoke_envs/d3rlpy
. /tmp/r2v_smoke_envs/d3rlpy/bin/activate
python -m pip install --upgrade pip
python -m pip install d3rlpy
python - <<'PY'
import d3rlpy
print("D3RLPY_IMPORT_OK", getattr(d3rlpy, "__version__", "unknown"))
PY
```

Max time: 75 minutes if activated.

Max download size: 0 GB for import-only fallback; real sparse dataset requires
new Pro approval.

## C009 OPER

Repo path:

```text
/tmp/r2v_smoke_repos/OPER
```

Read-only audit command:

```bash
mkdir -p /tmp/r2v_smoke_repos
git clone --depth 1 --filter=blob:none --sparse https://github.com/sail-sg/OPER.git /tmp/r2v_smoke_repos/OPER
cd /tmp/r2v_smoke_repos/OPER
git rev-parse HEAD
rg -n "class ReplayBuffer|def sample|replace_weights|resample|reweight|PrefetchBalancedSampler" README.md main.py utils.py advantage.py
```

Max time: 30 minutes.

Max download size: 0 GB.

Forbidden: no full priority-weight training.

Additional audit field: classify priority as static, dynamic, return-based,
advantage-based, density-like, or other.
