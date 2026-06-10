# R2V-Replay Codebase Search - Pro Audit Prompts

## Round C Short-List Audit Prompt

You are GPT Pro acting as PI and adversarial reviewer for R2V-Replay. Codex has
completed Round B high-recall codebase search and produced:

- `search_log.md`
- `codebase_candidates.csv`
- `repo_cards.md`
- `exclusion_log.md`

Please audit the package before any Ubuntu smoke test.

Return:

```text
verdict: pass_to_smoke_test | revise_search | reject_route
fatal_objections:
major_objections:
minor_objections:
missing_evidence:
baseline_attacks:
risk_attacks:
shortlist_for_ubuntu_smoke_test:
required_revision_tasks:
do_not_proceed_until:
```

Audit questions:

1. Does this route test R2V-Replay rather than resurrecting the stopped toy-grid route?
2. Does it preserve `rare != valuable`?
3. Does it treat zero-reward replay as real but mostly low-information data?
4. Is sparse reward genuine, not manufactured by reward hacking?
5. Is the R2V hook in replay / sampler / dataloader / loss weighting rather than environment reward?
6. Are baselines fair under the same data, same budget, and same seeds?
7. Would a reviewer accept the benchmark and codebase as credible?
8. Can failure be interpreted scientifically rather than as only install noise?
9. Are license and reproducibility acceptable?
10. Is a smoke-test failure engineering-only, or route-level?

## Round D Smoke-Test Audit Prompt

You are GPT Pro acting as PI and adversarial reviewer. Codex ran Ubuntu smoke
tests only on the Round C approved short list.

Please audit:

- exact Ubuntu commands
- environment / Python executable
- install logs
- import results
- dataset access results
- minimal sampler / batch results
- failure modes
- whether smoke-test evidence is enough to select a route

Return:

```text
verdict: final_go | revise_smoke_test | reject_all_current_candidates
preferred_codebase:
backup_codebase:
fatal_objections:
major_objections:
missing_evidence:
minimum_next_experiment:
do_not_claim:
```

