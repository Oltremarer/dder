# R2V-Replay Level 0 Diagnostic

This is the isolated Level 0 No-RL diagnostic scaffold for R2V-Replay.

The first stage tests whether a sparse-reward replay buffer contains rare valuable and rare useless transitions, and whether a rare-to-valuable selector improves replay subset composition without using evaluation labels as scorer inputs.

## Local Policy

Mac is used only for editing and Git. Do not run local experiments here unless the user explicitly changes that instruction.

Run tests and diagnostics on Ubuntu.

## Ubuntu Smoke Test

```bash
cd dder
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r experiments/r2v_replay/requirements.txt
pytest -q experiments/r2v_replay/tests
python experiments/r2v_replay/scripts/build_sparse_grid_replay.py --config experiments/r2v_replay/configs/level0_sparse_grid.yaml --seed 0 --n-transitions 50000 --output experiments/r2v_replay/outputs/seed0/replay.npz
python experiments/r2v_replay/scripts/run_level0_diagnostic.py --dataset experiments/r2v_replay/outputs/seed0/replay.npz --config experiments/r2v_replay/configs/level0_sparse_grid.yaml --rarity-input obs_action_next --rare-topk-ratio 0.10 --selected-budget-ratio 0.05 --seed 0 --output-dir experiments/r2v_replay/outputs/seed0/diagnostic_obs_action_next
python experiments/r2v_replay/scripts/plot_level0.py --diagnostic-dir experiments/r2v_replay/outputs/seed0/diagnostic_obs_action_next --output-dir experiments/r2v_replay/outputs/seed0/figures
python experiments/r2v_replay/scripts/package_level0_report.py --diagnostic-dir experiments/r2v_replay/outputs/seed0/diagnostic_obs_action_next --figures-dir experiments/r2v_replay/outputs/seed0/figures --output experiments/r2v_replay/outputs/seed0/reports/level0_report.zip
```

## Boundaries

- Do not call zero-reward replay corrupted data.
- Do not assume rare equals valuable.
- Do not use `label_for_eval_only` as scorer/model/selector input.
- Do not generate replay transitions in Level 0.
- Do not run RL training in Level 0.
