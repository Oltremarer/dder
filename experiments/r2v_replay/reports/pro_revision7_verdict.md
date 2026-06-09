# Pro Verdict For R2V-Replay Round 1 Revision 7

Date: 2026-06-09

Source: Chrome ChatGPT Pro conversation `https://chatgpt.com/c/6a282a75-8ad0-83ea-a4ed-2205e8fffe90`

## Verdict

`stop_v3a_route`

## Short Rationale

The candidate-ratio sweep shows that v3a's 20k H2 failure is not just a bad fixed-Top10 budget choice. kNN reliably finds rare valuable precursors, but v3a does not produce a stable candidate pool that is both rare-useless-enriched and rare-valuable-supplied across seeds. Therefore the v3a route should stop rather than be patched again.

## Fixed-Budget Explanation

Pro judgment: yes, the sweep disproves the fixed-budget-only explanation.

The tested explanation was: Top10% may be too large at 20k, so rare-useless distractors get diluted. The sweep tested smaller budgets `{0.5%, 1%, 2%, 5%}` plus larger references. No ratio passed the all-seed H2-budget rule.

Observed pattern:

- Small budgets: rare-useless sometimes appears, but not across all seeds; rare valuable supply is too low.
- Medium budgets: rare valuable supply improves, but rare-useless fraction/enrichment fails in seed 0/1.
- Large budgets: rare valuable supply is adequate, but rare-useless remains diluted.

Conclusion: this is a data-generating structure issue. v3a does not create a stable overlap between density-rare useless distractors and density-rare valuable precursors.

## Next Dataset/H2 Revision

Do not keep patching v3a. The next revision should be a new Level 0 dataset design, not another v3a/v3a_plus variant.

Recommended config name:

`level0_sparse_grid_two_branch_v4.yaml`

Core design:

- Branch A: rare valuable branch leading toward goal / bottleneck / success precursor.
- Branch B: rare useless branch leading to a dead-end pocket with no reward and no progress.
- Both branches should be similarly low-frequency under the behavior policy.
- Only Branch A eventually connects to success.
- Branch B must be real, reachable, reward-zero, dynamics-valid, and task-useless.

Rationale:

- rare valuable: low-density, task-relevant
- rare useless: low-density, task-irrelevant

This is cleaner than scattered pockets because H2 requires feature-space competition between two low-density structures.

## Allowed Files

Allowed edits:

- `experiments/r2v_replay/configs/level0_sparse_grid_two_branch_v4.yaml`
- `experiments/r2v_replay/r2v_replay/sparse_grid.py`
- `experiments/r2v_replay/r2v_replay/replay_dataset.py`
- `experiments/r2v_replay/scripts/build_sparse_grid_replay.py`
- `experiments/r2v_replay/scripts/run_density_audit.py`
- `experiments/r2v_replay/tests/test_sparse_grid.py`
- `experiments/r2v_replay/tests/test_replay_dataset.py`
- `experiments/r2v_replay/tests/test_candidate_ratio_sweep.py`

Only edit diagnostics or metrics if the new dataset needs existing audits to expose branch labels:

- `experiments/r2v_replay/r2v_replay/metric_audit.py`
- `experiments/r2v_replay/r2v_replay/diagnostics.py`

Do not edit selector/risk/diffusion code yet:

- `experiments/r2v_replay/r2v_replay/selector.py`
- `experiments/r2v_replay/r2v_replay/risk_scorers.py`
- `experiments/r2v_replay/r2v_replay/consistency_scorers.py`
- `experiments/r2v_replay/r2v_replay/rarity_scorers.py`

## Next Commands On Ubuntu

After code/config changes:

```bash
cd ~/dder
git status --short
git rev-parse HEAD
~/miniconda3/envs/c2t/bin/python -m pytest -q experiments/r2v_replay/tests
```

Run v4 density audit, 5k, seeds `0/1/2`:

```bash
for seed in 0 1 2; do
  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/build_sparse_grid_replay.py \
    --config experiments/r2v_replay/configs/level0_sparse_grid_two_branch_v4.yaml \
    --seed ${seed} \
    --n-transitions 5000 \
    --output experiments/r2v_replay/outputs/revision8_v4_seed${seed}_density/replay.npz

  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/run_density_audit.py \
    --dataset experiments/r2v_replay/outputs/revision8_v4_seed${seed}_density/replay.npz \
    --config experiments/r2v_replay/configs/level0_sparse_grid_two_branch_v4.yaml \
    --rarity-input obs_action_next \
    --k 50 \
    --output-dir experiments/r2v_replay/outputs/revision8_v4_seed${seed}_density/density_obs_action_next
done
```

If density audit cannot support the candidate sweep directly, run one minimal full diagnostic:

```bash
for seed in 0 1 2; do
  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/run_level0_diagnostic.py \
    --dataset experiments/r2v_replay/outputs/revision8_v4_seed${seed}_density/replay.npz \
    --config experiments/r2v_replay/configs/level0_sparse_grid_two_branch_v4.yaml \
    --rarity-input obs_action_next \
    --rare-topk-ratio 0.10 \
    --selected-budget-ratio 0.05 \
    --seed ${seed} \
    --output-dir experiments/r2v_replay/outputs/revision8_v4_seed${seed}_full5k/diagnostic_obs_action_next
done
```

## Acceptance Gate Before Any 20k

H1:

- kNN Recall@Top10 rare valuable >= 0.50 all seeds, or
- kNN Recall@Top20 rare valuable >= 0.70 all seeds.

H2:

- at some pre-declared ratio in `{0.01, 0.02, 0.05, 0.10}`:
  - rare_useless_count >= 10 all seeds;
  - rare_useless_fraction >= 0.04 all seeds;
  - rare_useless_enrichment >= 2.0 all seeds;
  - rare_valuable_all_count >= 30 or rare_valuable_recall >= 0.30 all seeds.

H3:

- R2V selected rare_useless fraction < candidate rare_useless fraction all seeds;
- R2V selects nonzero zero-reward precursors all seeds;
- R2V positive reward fraction <= 0.50 all seeds;
- reward-only zero-precursor recall remains 0 or below R2V.

H4:

- optional_invalid excluded from H1-H3;
- invalid-risk AUROC >= 0.85 all seeds.

## Next Checkpoint

Use `R2V-Replay Round 1 Revision 8 Two-Branch-v4 Report`.

Required sections:

- Context
- Code changes
- v4 environment design
- Dataset composition
- Density audit
- Candidate-ratio sweep
- H1/H2/H3/H4 table
- Comparison to v3a failure
- Codex recommendation: `go_to_20k | revise_dataset_or_h2 | stop_level0`

## Forbidden Actions

- Do not relax H2 further.
- Do not claim v3a passed H2 at 20k.
- Do not keep patching v3a with more probe ratios, bigger maps, or more pockets.
- Do not move to Level 1.
- Do not implement RL training.
- Do not implement PER, HER, generation, CleanRL, CORL, SynthER, D4RL, or high-dimensional tasks.
- Do not tune selector/risk before v4 H1/H2 is established.
- Do not tune diffusion.
- Do not use `full_transition`, reward, done, or labels in the main rarity scorer.
- Do not treat diffusion as working evidence.
- Do not mix optional_invalid into H1-H3.
- Do not manually sample transitions by label.
- Do not use teleport, impossible dynamics, or synthetic invalid transitions as rare-useless.
- Do not call zero-reward transitions corrupted, dirty, or bad data.
- Do not hide seed-level failures behind means.
