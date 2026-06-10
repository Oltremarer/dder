# Dry-Run Command List For Pro Review

Date: 2026-06-10

Status: approved by Pro with modifications recorded in `pro_dry_run_approval.md`.

All Ubuntu commands must begin with:

```bash
ssh Ubuntu-Tailscale 'cd ~/dder && git status --short && git rev-parse HEAD'
```

After Pro approval, Codex will run candidate commands as separate logged blocks.

## C001 OGBench Dry Run

```bash
ssh Ubuntu-Tailscale 'set -euo pipefail
cd ~/dder
git status --short
git rev-parse HEAD
mkdir -p artifacts/r2v_codebase_search/logs /tmp/r2v_smoke_envs /tmp/r2v_smoke_data/ogbench /tmp/r2v_smoke_repos
~/miniconda3/envs/c2t/bin/python -m venv /tmp/r2v_smoke_envs/ogbench
. /tmp/r2v_smoke_envs/ogbench/bin/activate
python -m pip install --upgrade pip
python -m pip install ogbench
python - <<'"'"'PY'"'"'
import json
import numpy as np
import ogbench
dataset_name = "pointmaze-medium-navigate-v0"
obj = ogbench.make_env_and_datasets(dataset_name, env_only=True)
env = obj[0] if isinstance(obj, tuple) else obj
print("OGBENCH_ENV_ONLY_OK", dataset_name, type(env).__name__)
real_dataset_name = "antmaze-medium-navigate-v0"
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(real_dataset_name, dataset_dir="/tmp/r2v_smoke_data/ogbench", compact_dataset=False)
rewards = np.asarray(train_dataset["rewards"])
print("OGBENCH_DATASET_SUMMARY", json.dumps({"dataset_name": real_dataset_name, "n_transitions": int(rewards.shape[0]), "reward_min": float(np.min(rewards)), "reward_max": float(np.max(rewards)), "zero_reward_ratio": float(np.mean(rewards == 0)), "positive_reward_ratio": float(np.mean(rewards > 0)), "keys": sorted(list(train_dataset.keys()))}, sort_keys=True))
PY
du -sh /tmp/r2v_smoke_data/ogbench
git clone --depth 1 --filter=blob:none --sparse https://github.com/seohongpark/ogbench.git /tmp/r2v_smoke_repos/ogbench
cd /tmp/r2v_smoke_repos/ogbench
git sparse-checkout set impls
python -m py_compile impls/utils/datasets.py
rg -n "class Dataset|def sample|class GCDataset|train_dataset.sample" impls
'
```

## C003 CORL Dry Run

```bash
ssh Ubuntu-Tailscale 'set -euo pipefail
cd ~/dder
git status --short
git rev-parse HEAD
mkdir -p artifacts/r2v_codebase_search/logs /tmp/r2v_smoke_envs /tmp/r2v_smoke_repos
~/miniconda3/envs/c2t/bin/python -m venv /tmp/r2v_smoke_envs/corl
. /tmp/r2v_smoke_envs/corl/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone --depth 1 --filter=blob:none --sparse https://github.com/tinkoff-ai/CORL.git /tmp/r2v_smoke_repos/CORL
cd /tmp/r2v_smoke_repos/CORL
git sparse-checkout set algorithms/offline configs requirements
git rev-parse HEAD
python - <<'"'"'PY'"'"'
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
data = {"observations": np.zeros((8, 3), dtype=np.float32), "actions": np.ones((8, 2), dtype=np.float32), "next_observations": np.ones((8, 3), dtype=np.float32), "rewards": np.array([0,0,0,0,0,1,0,0], dtype=np.float32), "terminals": np.zeros((8,), dtype=np.float32)}
rb.load_d4rl_dataset(data)
batch = rb.sample(4)
print("CORL_DUMMY_REPLAY_OK", [tuple(x.shape) for x in batch])
print("CORL_ZERO_REWARD_RATIO_DUMMY", float((data["rewards"] == 0).mean()))
PY
'
```

## C006 + C007 TorchRL + Minari Dry Run

```bash
ssh Ubuntu-Tailscale 'set -euo pipefail
cd ~/dder
git status --short
git rev-parse HEAD
mkdir -p artifacts/r2v_codebase_search/logs /tmp/r2v_smoke_envs /tmp/r2v_torchrl_storage
~/miniconda3/envs/c2t/bin/python -m venv /tmp/r2v_smoke_envs/torchrl_minari
. /tmp/r2v_smoke_envs/torchrl_minari/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install torchrl tensordict minari
python -m minari list remote > /tmp/r2v_minari_remote.txt
sed -n '1,80p' /tmp/r2v_minari_remote.txt
python - <<'"'"'PY'"'"'
import torch
import torchrl
import minari
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
print("TORCHRL_MINARI_IMPORT_OK", torch.__version__, getattr(torchrl, "__version__", "unknown"))
storage = LazyMemmapStorage(32, scratch_dir="/tmp/r2v_torchrl_storage")
rb = TensorDictReplayBuffer(storage=storage, sampler=RandomSampler())
td = TensorDict({"observation": torch.zeros(16, 3), "action": torch.ones(16, 2), "next": {"observation": torch.ones(16, 3), "reward": torch.tensor([[0.0]] * 15 + [[1.0]])}, "done": torch.zeros(16, 1, dtype=torch.bool)}, batch_size=[16])
rb.extend(td)
batch = rb.sample(4)
rewards = td["next", "reward"].reshape(-1)
print("TORCHRL_DUMMY_REPLAY_OK", batch.batch_size, float((rewards == 0).float().mean()))
PY
python - <<'"'"'PY'"'"'
import json
import numpy as np
import minari

dataset_candidates = ["D4RL/pointmaze/medium-v2", "D4RL/antmaze/medium-diverse-v1", "D4RL/door/human-v2"]
loaded = None
errors = {}
for dataset_id in dataset_candidates:
    try:
        dataset = minari.load_dataset(dataset_id, download=True)
        loaded = (dataset_id, dataset)
        break
    except Exception as exc:
        errors[dataset_id] = repr(exc)
if loaded is None:
    print("MINARI_DATASET_LOAD_FAIL", json.dumps(errors, sort_keys=True))
else:
    dataset_id, dataset = loaded
    first = next(dataset.iterate_episodes())
    rewards = np.asarray(first.rewards)
    print("MINARI_DATASET_EPISODE_SUMMARY", json.dumps({"dataset_id": dataset_id, "episode_reward_min": float(np.min(rewards)), "episode_reward_max": float(np.max(rewards)), "episode_zero_reward_ratio": float(np.mean(rewards == 0)), "episode_positive_reward_ratio": float(np.mean(rewards > 0)), "episode_len": int(rewards.shape[0])}, sort_keys=True))
PY
'
```

## C009 OPER Read-Only Audit Dry Run

```bash
ssh Ubuntu-Tailscale 'set -euo pipefail
cd ~/dder
git status --short
git rev-parse HEAD
mkdir -p /tmp/r2v_smoke_repos
git clone --depth 1 --filter=blob:none --sparse https://github.com/sail-sg/OPER.git /tmp/r2v_smoke_repos/OPER
cd /tmp/r2v_smoke_repos/OPER
git rev-parse HEAD
rg -n "class ReplayBuffer|def sample|replace_weights|resample|reweight|PrefetchBalancedSampler" README.md main.py utils.py advantage.py
rg -n "advantage|return|priority|weight|static|dynamic|density" README.md main.py utils.py advantage.py
'
```

## C005 d3rlpy Fallback Dry Run

Run only if Pro later authorizes fallback.

```bash
ssh Ubuntu-Tailscale 'set -euo pipefail
cd ~/dder
git status --short
git rev-parse HEAD
mkdir -p /tmp/r2v_smoke_envs
~/miniconda3/envs/c2t/bin/python -m venv /tmp/r2v_smoke_envs/d3rlpy
. /tmp/r2v_smoke_envs/d3rlpy/bin/activate
python -m pip install --upgrade pip
python -m pip install d3rlpy
python - <<'"'"'PY'"'"'
import d3rlpy
print("D3RLPY_IMPORT_OK", getattr(d3rlpy, "__version__", "unknown"))
PY
'
```
