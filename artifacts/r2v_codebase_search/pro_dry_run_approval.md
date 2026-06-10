# Pro Dry-Run Approval

Date: 2026-06-10

Pro thread: `https://chatgpt.com/c/6a28e139-8c08-83ea-8dd4-e38c6d2631f4`

Verdict:

```text
approve_dry_run
```

Can Codex run Ubuntu now:

```text
yes
```

## Approved First-Batch Commands

Approved with modifications:

- `C001 OGBench`
- `C003 CORL`
- `C006 + C007 TorchRL + Minari`
- `C009 OPER` read-only audit

Do not run `C005 d3rlpy` until Pro reviews first-batch smoke results.

## Required Modifications Before Execution

- OGBench:
  - Env-only check with `pointmaze-medium-navigate-v0` is allowed.
  - Real dataset load should prefer `antmaze-medium-navigate-v0`.
  - Use `pointmaze-medium-navigate-v0` for real dataset only if AntMaze download
    fails or is too large.
  - Add static source health check before hook extraction.
- CORL:
  - AST/class-extraction dummy replay smoke is acceptable as first-pass hook
    proof.
  - Log whether `ReplayBuffer.sample` returns indices.
  - If it does not, record `batch_indices_available=no` and identify the
    minimal future modification. Do not patch.
- TorchRL + Minari:
  - Use a clean venv without `--system-site-packages`.
  - Install CPU torch explicitly, then install `torchrl`, `tensordict`, and
    `minari`.
  - After `minari list remote`, attempt one real sparse dataset in this order:
    `D4RL/pointmaze/medium-v2`; if absent, `D4RL/antmaze/medium-diverse-v1`;
    if absent, `D4RL/door/human-v2` only as API-bridge fallback, not scientific
    sparse pass.
- OPER:
  - Read-only audit should classify priority as static, dynamic, return-based,
    advantage-based, density-like, or other.

## Additional Limits

- C001 cap: 80 minutes.
- C003 cap: 95 minutes.
- C006 + C007 cap: 105 minutes.
- C009 cap: 30 minutes.
- First-pass download cap: 2 GB per candidate.
- Hard stop: 10 GB, do not approach without Pro approval.
- No sudo apt install.
- No CUDA driver changes.
- No global package installs.
- No main environment mutation.
- No reward-function modification.
- No dataset filtering or deletion of zero-reward transitions.
- No R2V implementation patch.
- Static hook identification is allowed; code modification is not.

## Required Log Fields

- `timestamp_start`, `timestamp_end`, `hostname`, `pwd`
- `git_status_short`, `project_commit`, `candidate_id`, `candidate_repo_commit`
- `python_path`, `python_version`, `pip_freeze`, `venv_path`
- `cuda_visible`, `torch_version`, `device_used`
- `install_command`, `install_exit_code`, `import_command`, `import_exit_code`
- `dataset_id`, `dataset_source`, `dataset_dir`, `download_size_estimate`, `download_exit_code`
- `dataset_keys`, `num_transitions`, `observation_shape`, `action_shape`
- `reward_field_present`, `reward_min`, `reward_max`, `num_reward_zero`, `num_reward_positive`
- `zero_reward_ratio`, `positive_reward_ratio`
- `terminals_present`, `terminal_sum`, `masks_present`, `mask_sum`
- `sampler_function`, `sample_batch_size`, `batch_keys`, `batch_reward_zero_ratio`
- `batch_indices_available_yes_no`, `priority_update_possible_yes_no`
- `r2v_hook_exact_file`, `r2v_hook_exact_function`
- `requires_reward_function_change_yes_no`, `requires_deleting_zero_reward_yes_no`
- `failure_label`
- `provisional_result`

Allowed `provisional_result` values:

- `PASS_REAL_SPARSE`
- `PASS_HOOK_ONLY`
- `PARTIAL_PASS_DUMMY_ONLY`
- `FAIL_ENGINEERING`
- `FAIL_ROUTE_LEVEL`

## Next Pro Checkpoint

Stop after C001, C003, C006+C007, and C009 read-only audit complete, or earlier
if two candidates hit route-level failures. Return:

- `smoke_results.md`
- all raw logs
- updated `feasibility_matrix.csv`

