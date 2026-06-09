#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2v_replay.geometry_audit import (
    candidate_composition_rows,
    distance_to_label_rows,
    multiplicity_summary_rows,
    score_table_raw_scores,
    score_variants,
)
from r2v_replay.replay_dataset import ReplayDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--diagnostic-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rarity-input", default="obs_action_next")
    parser.add_argument("--k-values", required=True)
    parser.add_argument("--candidate-ratios", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset = ReplayDataset.load_npz(args.dataset)
    diagnostic_dir = Path(args.diagnostic_dir)
    score_table = pd.read_csv(diagnostic_dir / "score_table.csv")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ratios = _parse_float_list(args.candidate_ratios)
    k_values = [int(k) for k in _parse_float_list(args.k_values)]
    real_mask = dataset.real_mask()
    raw_scores, raw_distances = score_table_raw_scores(score_table)

    density_rows = multiplicity_summary_rows(
        labels=dataset.labels,
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        scores=raw_scores,
        raw_distances=raw_distances,
    )
    density = pd.DataFrame(density_rows)
    density.to_csv(output_dir / "geometry_density_summary.csv", index=False)
    density.to_csv(output_dir / "duplicate_multiplicity_by_label.csv", index=False)

    unique_summary = _unique_transition_summary(dataset, args.rarity_input)
    unique_summary.to_csv(output_dir / "unique_transition_summary.csv", index=False)

    variants = score_variants(dataset, rarity_input=args.rarity_input, k_values=k_values)
    sensitivity_rows = []
    for name, scores in variants.items():
        if not name.startswith("raw_k"):
            continue
        k_value = int(name.replace("raw_k", ""))
        sensitivity_rows.extend(
            candidate_composition_rows(dataset.labels, scores, ratios=ratios, variant=name, k_value=k_value, real_mask=real_mask)
        )
    pd.DataFrame(sensitivity_rows).to_csv(output_dir / "knn_sensitivity_by_k.csv", index=False)

    variant_rows = []
    main_k = k_values[-1]
    variant_rows.extend(
        candidate_composition_rows(
            dataset.labels,
            raw_scores,
            ratios=ratios,
            variant="raw_diagnostic",
            k_value=main_k,
            real_mask=real_mask,
        )
    )
    for name, scores in variants.items():
        if name.startswith("raw_k") and name != f"raw_k{main_k}":
            continue
        variant_rows.extend(
            candidate_composition_rows(dataset.labels, scores, ratios=ratios, variant=name, k_value=main_k, real_mask=real_mask)
        )
    variant_df = pd.DataFrame(variant_rows)
    variant_df.to_csv(output_dir / "candidate_ratio_by_metric_variant.csv", index=False)
    variant_df[variant_df["ratio"] == 0.10].to_csv(output_dir / "raw_vs_unique_knn_comparison.csv", index=False)

    distances = pd.DataFrame(
        distance_to_label_rows(
            dataset,
            rarity_input=args.rarity_input,
            target_labels=["common_zero", "rare_valuable", "rare_useless"],
        )
    )
    distances.to_csv(output_dir / "distance_to_common_zero_summary.csv", index=False)

    _write_interpretation(output_dir / "geometry_audit_interpretation.md", density, variant_df, args)
    print(density.to_string(index=False))


def _parse_float_list(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one comma-separated value")
    return parsed


def _unique_transition_summary(dataset: ReplayDataset, rarity_input: str) -> pd.DataFrame:
    frame = dataset.to_frame()
    feature_columns = list(dataset.feature_matrix(rarity_input).columns)
    grouped = (
        frame.groupby(feature_columns + ["label_for_eval_only"], dropna=False)
        .size()
        .reset_index(name="label_count")
    )
    pivot = grouped.pivot_table(
        index=feature_columns,
        columns="label_for_eval_only",
        values="label_count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    label_columns = [col for col in pivot.columns if col not in feature_columns]
    pivot["total_count"] = pivot[label_columns].sum(axis=1)
    pivot["label_collision_count"] = (pivot[label_columns] > 0).sum(axis=1)
    return pivot.sort_values("total_count", ascending=False)


def _write_interpretation(path: Path, density: pd.DataFrame, variants: pd.DataFrame, args: argparse.Namespace) -> None:
    rare_useless = density[density["label"] == "rare_useless"]
    h2_candidates = variants[
        (variants["variant"].isin(["raw_diagnostic", f"raw_k{_parse_float_list(args.k_values)[-1]:.0f}", f"unique_transition_k{_parse_float_list(args.k_values)[-1]:.0f}", f"duplicate_capped5_k{_parse_float_list(args.k_values)[-1]:.0f}"]))
        & (variants["ratio"].isin([0.01, 0.02, 0.05, 0.10]))
    ]
    any_h2 = bool(h2_candidates["h2_budget_pass"].any()) if "h2_budget_pass" in h2_candidates else False
    lines = [
        "# Geometry Audit Interpretation",
        "",
        f"Dataset: `{args.dataset}`",
        f"Diagnostic dir: `{args.diagnostic_dir}`",
        "",
        f"Rare-useless mean multiplicity: `{float(rare_useless['mean_multiplicity'].iloc[0]) if len(rare_useless) else float('nan'):.3f}`",
        f"Rare-useless median kNN distance: `{float(rare_useless['median_knn_distance'].iloc[0]) if len(rare_useless) else float('nan'):.3f}`",
        f"Any H2-budget pass among raw/unique/capped variants: `{str(any_h2).lower()}`",
        "",
        "This audit is diagnostic only. It does not redefine the official scorer.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
