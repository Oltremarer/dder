#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2v_replay.density_audit import density_audit_tables
from r2v_replay.encoders import TransitionEncoder
from r2v_replay.geometry_audit import distance_to_label_rows, multiplicity_summary_rows
from r2v_replay.rarity_scorers import KNNRarityScorer
from r2v_replay.replay_dataset import ReplayDataset
from r2v_replay.utils import load_yaml, run_metadata, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rarity-input", default=None)
    parser.add_argument("--rare-topk-ratio", type=float, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rarity_input = args.rarity_input or cfg.get("encoder", {}).get("rarity_input", "obs_action_next")
    top_ratio = float(
        args.rare_topk_ratio
        if args.rare_topk_ratio is not None
        else cfg.get("selector", {}).get("rare_topk_ratio", 0.10)
    )
    knn_k = int(args.k if args.k is not None else cfg.get("scorers", {}).get("knn_k", 10))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ReplayDataset.load_npz(args.dataset)
    real_mask = dataset.real_mask()
    z = TransitionEncoder(input_mode=rarity_input).fit_transform(dataset, mask=real_mask)
    knn_scores = KNNRarityScorer(k=knn_k).fit(z[real_mask]).score(z)
    tables = density_audit_tables(dataset, knn_scores=knn_scores, top_ratio=top_ratio, eligible_mask=real_mask)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    summary = tables["density_audit_by_label"].copy()
    summary.insert(0, "rarity_input", rarity_input)
    summary.to_csv(output_dir / "density_summary.csv", index=False)
    geometry_density = pd.DataFrame(
        multiplicity_summary_rows(
            labels=dataset.labels,
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            scores=knn_scores,
            raw_distances=knn_scores,
        )
    )
    geometry_density.to_csv(output_dir / "geometry_density_summary.csv", index=False)
    geometry_density.to_csv(output_dir / "duplicate_multiplicity_by_label.csv", index=False)
    pd.DataFrame(
        distance_to_label_rows(
            dataset,
            rarity_input=rarity_input,
            target_labels=["common_zero", "rare_valuable", "rare_useless"],
        )
    ).to_csv(output_dir / "distance_to_common_zero_summary.csv", index=False)
    write_json(output_dir / "run_metadata.json", run_metadata(" ".join(sys.argv), args.config, seed=0))
    print(pd.read_csv(output_dir / "rare_useless_density_audit.csv").to_string(index=False))


if __name__ == "__main__":
    main()
