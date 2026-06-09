#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2v_replay.diagnostics import run_level0_diagnostic
from r2v_replay.replay_dataset import ReplayDataset
from r2v_replay.utils import load_yaml, run_metadata, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rarity-input", default=None)
    parser.add_argument("--rare-topk-ratio", type=float, default=None)
    parser.add_argument("--selected-budget-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.rarity_input:
        cfg.setdefault("encoder", {})["rarity_input"] = args.rarity_input
    if args.rare_topk_ratio is not None:
        cfg.setdefault("selector", {})["rare_topk_ratio"] = args.rare_topk_ratio
    if args.selected_budget_ratio is not None:
        cfg.setdefault("selector", {})["selected_budget_ratio"] = args.selected_budget_ratio

    dataset = ReplayDataset.load_npz(args.dataset)
    result = run_level0_diagnostic(dataset, cfg, args.output_dir, seed=args.seed)
    write_json(Path(args.output_dir) / "run_metadata.json", run_metadata(" ".join(sys.argv), args.config, args.seed))
    print(result["metrics"])


if __name__ == "__main__":
    main()
