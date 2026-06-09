# Pro Verdict For R2V-Replay Round 1 Revision 5

Date: 2026-06-09

Source: Chrome ChatGPT Pro conversation `https://chatgpt.com/c/6a282a75-8ad0-83ea-a4ed-2205e8fffe90`

## Verdict

`go_to_20k`

## Short Rationale

v3a full-5k is the first run that satisfies the core Level 0 mechanism chain:

- the kNN-density candidate pool contains both rare valuable and rare useless transitions;
- R2V removes rare useless from that mixed rare pool;
- R2V keeps a nonzero number of zero-reward valuable precursors.

Pro judged seed-2 false rejection `0.333` and seed-0 Top10 rare-valuable recall `0.427` as warnings, not blockers for a 20k Level 0 stability check.

## H1-H4 Judgment

| Hypothesis | Decision | Evidence | Pro judgment |
| --- | --- | --- | --- |
| H1: rarity scorer high-recall finds rare valuable candidates | weak pass | kNN Recall@Top20 rare valuable passes all seeds, min `0.802`; Top10 seed 0 is `0.427`, below `0.50` | Acceptable. v3a is a harder mixed-distractor diagnostic; Top20 all-seed pass is enough for 20k. |
| H2: rare-only candidate pool mixes rare useless | pass | Top10 rare-useless count all seeds >= 20; fraction all seeds >= 0.04; enrichment all seeds >= 2.0 | Pass. The candidate-pool-level `rare != valuable` condition now holds. |
| H3: R2V selects valuable while removing rare useless | pass | kNN Top10 has rare useless, but R2V selected rare-useless = 0; R2V zero-precursor recall > reward-only; positive-reward fraction <= 0.50; all seeds select nonzero zero-reward precursors | Pass. This is not reward-only and shows rare-to-valuable filtering. |
| H4: risk / consistency catches invalid and does not overkill useful precursors | weak pass | invalid-risk AUROC = 1.0; false rejection seeds 0/1 pass, seed 2 = 0.333 slightly above 0.30 | Acceptable for 20k, but 20k must keep reporting false rejection and cannot ignore it before Level 1. |

## Seed-2 False Rejection

Pro decision: acceptable for 20k Level 0. Do not do a selector-risk revision before 20k.

Reasons:

- threshold miss is small: `0.333` vs `0.300`;
- invalid-risk AUROC is perfect;
- seed 2 still selects nonzero zero-reward precursors;
- R2V selected rare-useless remains `0`;
- seed 2 has relatively low precursor supply, so 5k variance may be high.

20k must continue reporting:

- false rejection rate on zero precursor;
- no-risk vs full-risk selected precursor count;
- no-risk vs full-risk selected rare-useless count;
- risk-rejected valuable precursor examples.

If 20k still has false rejection `> 0.30` in multiple seeds, or full-risk removes many precursors while no-risk already removes rare-useless, then the next step is `revise_selector_risk`.

## Seed-0 Recall@Top10

Pro decision: acceptable under weak-pass H1 for 20k Level 0.

Reasons:

- seed 0 Recall@Top10 rare valuable = `0.427`;
- seed 0 Recall@Top20 rare valuable = `0.802`;
- mean Recall@Top10 rare valuable = `0.749`;
- mean Recall@Top20 rare valuable = `0.934`.

Top10 seed 0 is weak, but Top20 passes all seeds and H2/H3 already hold under the Top10 budget. If 20k loses both Top10 and Top20, then candidate discovery must be revised.

## Next Commands On Ubuntu

Before running:

```bash
cd ~/dder
git status --short
git rev-parse HEAD
```

Run tests:

```bash
~/miniconda3/envs/c2t/bin/python -m pytest -q experiments/r2v_replay/tests
```

Run 20k v3a for seeds `0/1/2`:

```bash
for seed in 0 1 2; do
  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/build_sparse_grid_replay.py \
    --config experiments/r2v_replay/configs/level0_sparse_grid_hybrid_v3a.yaml \
    --seed ${seed} \
    --n-transitions 20000 \
    --output experiments/r2v_replay/outputs/revision6_v3a_seed${seed}_20k/replay.npz

  ~/miniconda3/envs/c2t/bin/python experiments/r2v_replay/scripts/run_level0_diagnostic.py \
    --dataset experiments/r2v_replay/outputs/revision6_v3a_seed${seed}_20k/replay.npz \
    --config experiments/r2v_replay/configs/level0_sparse_grid_hybrid_v3a.yaml \
    --rarity-input obs_action_next \
    --rare-topk-ratio 0.10 \
    --selected-budget-ratio 0.05 \
    --seed ${seed} \
    --output-dir experiments/r2v_replay/outputs/revision6_v3a_seed${seed}_20k/diagnostic_obs_action_next
done
```

If runtime is too long, Pro allows running seed `0` first, then seeds `1/2`.

## Required 20k Artifact Paths

- `experiments/r2v_replay/outputs/revision6_v3a_seed{0,1,2}_20k/diagnostic_obs_action_next/metrics.csv`
- `experiments/r2v_replay/outputs/revision6_v3a_seed{0,1,2}_20k/diagnostic_obs_action_next/metric_consistency_audit.csv`
- `experiments/r2v_replay/outputs/revision6_v3a_seed{0,1,2}_20k/diagnostic_obs_action_next/candidate_funnel_summary.csv`
- `experiments/r2v_replay/outputs/revision6_v3a_seed{0,1,2}_20k/diagnostic_obs_action_next/selection_funnel_by_baseline.csv`
- `experiments/r2v_replay/outputs/revision6_v3a_seed{0,1,2}_20k/diagnostic_obs_action_next/risk_rejection_summary.csv`
- `experiments/r2v_replay/outputs/revision6_v3a_seed{0,1,2}_20k/diagnostic_obs_action_next/top_selected_examples.csv`
- `experiments/r2v_replay/outputs/revision6_v3a_seed{0,1,2}_20k/diagnostic_obs_action_next/selected_rare_useless.csv`
- `experiments/r2v_replay/outputs/revision6_v3a_seed{0,1,2}_20k/diagnostic_obs_action_next/rejected_valuable_precursors.csv`

## Next Checkpoint

Use `R2V-Replay Round 1 Revision 6 v3a 20k Diagnostic Report`.

The report must include per-seed and mean/std values for:

- dataset composition;
- H1 kNN candidate discovery;
- H2 rare-useless candidate evidence;
- H3 R2V selection funnel;
- H4 risk / consistency;
- diffusion failed baseline;
- interpretation of whether 20k confirms `rare != valuable`, rare-useless removal, zero-reward precursor preservation, and candidate discovery stability.

Codex recommendation should be one of:

- `go_to_next_level`
- `revise_selector_risk`
- `revise_dataset_or_h2`
- `stop_level0`

## Forbidden Actions For The Next Round

- Do not enter Level 1 RL.
- Do not write DQN, replay wrapper, PER, HER, generation, CleanRL integration, SynthER integration, CORL integration, D4RL integration, or high-dimensional code.
- Do not revise the dataset before the 20k v3a run.
- Do not tune selector/risk weights before the 20k v3a run.
- Do not tune diffusion.
- Do not use `full_transition` as main evidence.
- Do not use reward, done, or labels inside the main rarity scorer.
- Do not mix optional_invalid into H1-H3.
- Do not manually relabel rare_useless or precursors.
- Do not lower H2 criteria after seeing 20k.
- Do not report only mean values; report per-seed values and mean/std.
- Do not hide seed-level failures behind averages.
- Do not treat zero-reward transitions as corrupted data.
- Do not claim diffusion works; keep it as a failed lightweight baseline unless future evidence changes.
