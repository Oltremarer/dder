# Smoke Failure Taxonomy

Use exactly these labels in Round D logs and feasibility matrix.

| label | meaning |
|---|---|
| `INSTALL_FAIL_DEP_VERSION` | Python or package version conflict prevents install/import. |
| `INSTALL_FAIL_SYSTEM_LIB` | Missing system library, compiler, MuJoCo, OpenGL, or OS dependency blocks install/import. |
| `DATASET_FAIL_NETWORK` | Dataset listing or download fails because of network, timeout, or mirror issue. |
| `DATASET_FAIL_AUTH_OR_MISSING` | Dataset requires unavailable auth, is missing, or is no longer hosted. |
| `IMPORT_FAIL_CODE` | Candidate source imports fail because of code-level import errors not explained by dependency version alone. |
| `ENV_FAIL_API` | Environment creation fails because the documented API or env name no longer works. |
| `TRAIN_FAIL_RUNTIME` | Tiny update or minimal training step fails at runtime after import and data load. |
| `TRAIN_FAIL_NUMERICAL` | Tiny update runs but produces NaN, Inf, or numerically invalid losses. |
| `NO_REPLAY_HOOK` | No replay, sampler, dataloader, or batch-construction hook can be identified. |
| `NO_SPARSE_TASK` | Candidate only passes dense or generic tasks and cannot expose a sparse/delayed reward task. |
| `LICENSE_BLOCKER` | Missing or incompatible license blocks use as project substrate. |
| `PARTIAL_PASS_DUMMY_ONLY` | Dummy replay/buffer sample works but no real sparse dataset was loaded. |
| `DATASET_SIZE_BLOCKED` | Candidate tries to download a visual, 100M, 1B, or otherwise too-large dataset beyond the approved cap. |
