# Pro Round C Audit

Date: 2026-06-10

Pro thread: `https://chatgpt.com/c/6a28e139-8c08-83ea-8dd4-e38c6d2631f4`

Verdict:

```text
pass_to_smoke_test
```

This is a limited permission to prepare for smoke tests. It is not final route
approval and it does not permit R2V implementation patches, Level 1 experiments,
RL training, or paper-result claims.

## Fatal Objections

None.

Pro judged that the Round B candidate pool did not revive the stopped toy-grid
route and did not treat generative replay / diffusion replay as the main R2V
evidence.

## Major Objections

- `C001 OGBench` can enter smoke test, but its score is benchmark-first.
  OGBench is scientifically strong for sparse success reward, but standard
  offline RL may not have complete official reference implementations and
  benchmark tables inside OGBench itself.
- `C003 CORL` can enter smoke test only as an implementation-first route. D4RL
  dependency failure should be classified as dependency risk, not as R2V
  scientific failure.
- `C006 TorchRL + C007 Minari` can enter smoke test only as a modern-substrate
  audit. They need a concrete sparse dataset and runnable IQL/CQL/TD3+BC route
  before they can support a paper story.
- `C005 d3rlpy` is fallback only.
- `C004 OfflineRL-Kit` is second-wave only.
- `C009 OPER` is an adversarial baseline / contrast, not the main substrate.

## Approved First-Batch Smoke Scope

1. `C001 OGBench`: primary benchmark-first smoke.
2. `C003 CORL`: primary implementation-first smoke.
3. `C006 + C007 TorchRL + Minari`: primary modern-substrate smoke.
4. `C005 d3rlpy`: fallback smoke only.
5. `C009 OPER`: read-only baseline audit, no full run.

## Hard Gates Before Round D

Pro requires Codex to create and submit a dry-run command list before any Ubuntu
execution:

- `shortlist_audit.md`
- `ubuntu_smoke_test_plan.md`
- `feasibility_matrix.csv`
- `smoke_failure_taxonomy.md`
- dry-run command list for Pro approval

Do not proceed until the smoke-test plan includes exact commands, time limits,
download limits, pass criteria, and failure classification.

## Pro Limits

- Do not download more than 10 GB without returning to Pro.
- Do not install global CUDA.
- Do not change system drivers.
- Do not pollute the main project environment.
- Do not modify benchmark reward functions.
- Do not delete or relabel zero-reward transitions as corrupted or dirty data.
- Do not interpret smoke pass as final route approval.
- Do not let dense Hopper-only d3rlpy smoke substitute for sparse-reward
  evidence.
- Do not run OPER full 500k / 1M step case studies.
- Do not start R2V implementation patches until at least one candidate completes
  real sparse dataset load, batch sample, hook-file confirmation, and
  zero-reward ratio measurement.

