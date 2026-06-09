#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2v_replay.sparse_grid import SparseGridConfig, build_sparse_grid_replay
from r2v_replay.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-transitions", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg_raw = load_yaml(args.config)
    grid_cfg = SparseGridConfig.default()
    n_transitions = args.n_transitions or int(cfg_raw.get("dataset", {}).get("n_transitions", 50000))
    dataset_cfg = cfg_raw.get("dataset", {})
    dataset = build_sparse_grid_replay(
        grid_cfg,
        n_transitions=n_transitions,
        seed=args.seed,
        policy_mix=dataset_cfg.get("policy_mix"),
        include_optional_invalid=bool(dataset_cfg.get("include_optional_invalid", False)),
        optional_invalid_ratio=float(dataset_cfg.get("optional_invalid_ratio", 0.03)),
    )
    dataset.save_npz(args.output)
    print(f"saved {len(dataset)} transitions to {args.output}")
    print(dataset.to_frame()["label_for_eval_only"].value_counts().to_string())


if __name__ == "__main__":
    main()
