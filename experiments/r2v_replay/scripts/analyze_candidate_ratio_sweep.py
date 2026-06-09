#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2v_replay.metric_audit import candidate_ratio_sweep_rows


SCORER_COLUMNS = {
    "knn": "knn_rarity",
    "diffusion": "diffusion_rarity",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-dir", default=None)
    parser.add_argument("--score-table", default=None)
    parser.add_argument("--metric-audit", default=None)
    parser.add_argument("--scorer", default="knn")
    parser.add_argument("--ratios", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    score_table_path = _resolve_score_table(args.diagnostic_dir, args.score_table)
    ratios = _parse_ratios(args.ratios)
    score_table = pd.read_csv(score_table_path)
    score_column = SCORER_COLUMNS.get(args.scorer, args.scorer)
    if score_column not in score_table.columns:
        raise ValueError(f"score column not found: {score_column}")
    labels = score_table["label_for_eval_only"].astype(str).to_numpy()
    real_mask = ~_to_bool(score_table["optional_invalid"])
    rows = candidate_ratio_sweep_rows(
        labels=labels,
        scores=score_table[score_column].to_numpy(),
        ratios=ratios,
        real_mask=real_mask.to_numpy(),
    )
    for row in rows:
        row["scorer"] = args.scorer
        row["score_column"] = score_column
        row.update(_h2_budget_flags(row))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep = pd.DataFrame(rows)
    sweep.to_csv(output_dir / "candidate_ratio_sweep.csv", index=False)
    curve_cols = [
        "scorer",
        "ratio",
        "candidate_count",
        "rare_useless_count",
        "rare_useless_fraction",
        "rare_useless_enrichment",
        "rare_valuable_all_count",
        "rare_valuable_recall",
        "zero_precursor_recall",
        "h2_budget_pass",
    ]
    sweep[curve_cols].to_csv(output_dir / "h2_budget_curve.csv", index=False)
    summary_cols = [
        "scorer",
        "ratio",
        "candidate_count",
        "rare_useless_count",
        "rare_useless_fraction",
        "rare_useless_enrichment",
        "rare_valuable_all_count",
        "rare_valuable_recall",
        "h2_count_pass",
        "h2_fraction_pass",
        "h2_enrichment_pass",
        "h2_valuable_supply_pass",
        "h2_budget_pass",
    ]
    sweep[summary_cols].to_csv(output_dir / "candidate_ratio_sweep_summary.csv", index=False)
    print(sweep[summary_cols].to_string(index=False))


def _resolve_score_table(diagnostic_dir: str | None, score_table: str | None) -> Path:
    if diagnostic_dir:
        return Path(diagnostic_dir) / "score_table.csv"
    if score_table:
        return Path(score_table)
    raise ValueError("provide --diagnostic-dir or --score-table")


def _parse_ratios(value: str) -> list[float]:
    ratios = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not ratios:
        raise ValueError("at least one ratio is required")
    return ratios


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"1", "true", "yes"})


def _h2_budget_flags(row: dict[str, float | int]) -> dict[str, bool]:
    count_pass = int(row["rare_useless_count"]) >= 10
    fraction_pass = float(row["rare_useless_fraction"]) >= 0.04
    enrichment_pass = float(row["rare_useless_enrichment"]) >= 2.0
    valuable_supply_pass = int(row["rare_valuable_all_count"]) >= 50 or float(row["rare_valuable_recall"]) >= 0.30
    eligible_ratio = float(row["ratio"]) in {0.005, 0.01, 0.02, 0.05}
    return {
        "h2_count_pass": count_pass,
        "h2_fraction_pass": fraction_pass,
        "h2_enrichment_pass": enrichment_pass,
        "h2_valuable_supply_pass": valuable_supply_pass,
        "h2_budget_rule_ratio": eligible_ratio,
        "h2_budget_pass": eligible_ratio
        and count_pass
        and fraction_pass
        and enrichment_pass
        and valuable_supply_pass,
    }


if __name__ == "__main__":
    main()
