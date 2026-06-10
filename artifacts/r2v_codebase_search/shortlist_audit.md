# Round C Shortlist Audit

Date: 2026-06-10

Status: draft for Pro dry-run review.

No Ubuntu commands have been run.

## Approved First-Batch Candidates

### C001 OGBench

Smoke role: primary benchmark-first smoke.

Objective:

- Verify package import.
- Verify an environment can be created without dataset download.
- Verify a small real sparse dataset can be loaded within the download cap.
- Measure zero-reward ratio on the loaded dataset.
- Confirm exact sampler hook in `impls/utils/datasets.py`.

Pass criteria:

- `import ogbench` succeeds.
- `ogbench.make_env_and_datasets(..., env_only=True)` succeeds.
- A small state-based dataset, preferably `pointmaze-medium-navigate-v0`, loads
  under the Pro download cap.
- `rewards == 0` ratio is measured and logged.
- `Dataset.sample` or `GCDataset.sample` hook is confirmed.

Failure classification:

- Dependency failure: `INSTALL_FAIL_DEP_VERSION`, `INSTALL_FAIL_SYSTEM_LIB`, or
  `IMPORT_FAIL_CODE`.
- Dataset failure: `DATASET_FAIL_NETWORK`, `DATASET_FAIL_AUTH_OR_MISSING`, or
  `ENV_FAIL_API`.
- Partial pass: environment and dummy sample pass but real sparse dataset does
  not load under cap.

### C003 CORL

Smoke role: primary implementation-first smoke.

Objective:

- Verify clone and required source tree.
- Confirm replay-buffer source and exact `sample` hook.
- Run a dummy `ReplayBuffer` sampling test without installing D4RL first.
- Only attempt D4RL import/load after the dummy replay path is proven and Pro
  approves the command list.

Pass criteria:

- Source contains `ReplayBuffer`, `load_d4rl_dataset`, and `sample`.
- Dummy replay buffer can load synthetic transitions and return a batch.
- Any D4RL failure is classified as dependency risk, not route failure.

Failure classification:

- No source hook: `NO_REPLAY_HOOK`.
- Import dependency issues: `INSTALL_FAIL_DEP_VERSION` or `IMPORT_FAIL_CODE`.
- Dummy-only success: `PARTIAL_PASS_DUMMY_ONLY`.

### C006 + C007 TorchRL + Minari

Smoke role: primary modern-substrate smoke.

Objective:

- Verify `torchrl`, `tensordict`, and `minari` import in an isolated env.
- Instantiate a replay buffer with explicit sampler.
- Sample dummy mostly-zero reward transitions and update priority if supported.
- Verify Minari can list remote datasets.
- Optionally load one small sparse Minari dataset under cap if Pro approves.

Pass criteria:

- TorchRL replay buffer samples a batch from dummy TensorDict data.
- Sampler or priority update mechanism is accessible.
- Minari CLI or API can list remote datasets.
- Real sparse Minari dataset load is a full pass; dummy-only is partial.

Failure classification:

- Torch or dependency failure: `INSTALL_FAIL_DEP_VERSION`.
- Dataset listing/load failure: `DATASET_FAIL_NETWORK` or
  `DATASET_FAIL_AUTH_OR_MISSING`.
- Dummy-only success: `PARTIAL_PASS_DUMMY_ONLY`.

### C005 d3rlpy

Smoke role: fallback only.

Trigger:

- Use only if at least two of `OGBench`, `CORL`, and `TorchRL + Minari` fail, or
  if OGBench dataset load works but algorithm hook remains unclear.

Objective:

- Verify package import and offline RL configs.
- Verify a dataset/replay API can sample a batch.
- Do not count dense Hopper-only success as scientific sparse-reward evidence.

### C009 OPER

Smoke role: read-only baseline / contrast audit.

Allowed:

- Clone/read source.
- Record commit.
- Locate resampling/reweighting interface and priority-weight path.

Forbidden:

- Do not run full priority-weight training.
- Do not run 500k / 1M step case studies.

