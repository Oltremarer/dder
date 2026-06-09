from __future__ import annotations

import numpy as np
import pandas as pd

from .replay_dataset import ReplayDataset


TRANSITION_KEY_COLUMNS = ["state_x", "state_y", "action", "next_state_x", "next_state_y"]


def density_audit_tables(
    dataset: ReplayDataset,
    knn_scores: np.ndarray,
    top_ratio: float = 0.10,
    eligible_mask: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    frame = dataset.to_frame()
    scores = np.asarray(knn_scores, dtype=np.float32)
    if eligible_mask is None:
        eligible_mask = dataset.real_mask()
    eligible_mask = np.asarray(eligible_mask, dtype=bool)
    eligible_indices = np.where(eligible_mask)[0]
    top_n = max(1, int(np.ceil(len(eligible_indices) * top_ratio)))
    top_indices = eligible_indices[np.argsort(-scores[eligible_indices])[:top_n]]
    top_mask = np.zeros(len(frame), dtype=bool)
    top_mask[top_indices] = True

    multiplicity = _transition_multiplicity(frame)
    frame = frame.assign(knn_score=scores, transition_multiplicity=multiplicity, in_top10=top_mask)
    real = frame[eligible_mask].copy()

    rows = []
    for label, group in real.groupby("label_for_eval_only", sort=True):
        label_mask = real["label_for_eval_only"] == label
        count = int(len(group))
        unique_count = int(group[TRANSITION_KEY_COLUMNS].drop_duplicates().shape[0])
        top_count = int(group["in_top10"].sum())
        top_fraction = _safe_fraction(top_count, top_n)
        base_fraction = _safe_fraction(count, len(real))
        rows.append(
            {
                "label": label,
                "count": count,
                "unique_state_action_next_count": unique_count,
                "mean_transition_multiplicity": float(group["transition_multiplicity"].mean()),
                "median_knn_distance": float(group["knn_score"].median()),
                "mean_knn_score": float(group["knn_score"].mean()),
                "top10_count": top_count,
                "top10_fraction": top_fraction,
                "top10_label_recall": _safe_fraction(top_count, count),
                "top10_enrichment": _safe_enrichment(top_fraction, base_fraction),
                "base_fraction": base_fraction,
                "is_top10_label": bool(label_mask.any()),
            }
        )
    by_label = pd.DataFrame(rows)
    transition_multiplicity = by_label[
        ["label", "count", "unique_state_action_next_count", "mean_transition_multiplicity"]
    ].copy()
    knn_distance = by_label[["label", "median_knn_distance", "mean_knn_score", "top10_count"]].copy()
    rare_useless = by_label[by_label["label"] == "rare_useless"].copy()
    if rare_useless.empty:
        rare_useless = pd.DataFrame(
            [
                {
                    "label": "rare_useless",
                    "count": 0,
                    "unique_state_action_next_count": 0,
                    "mean_transition_multiplicity": float("nan"),
                    "median_knn_distance": float("nan"),
                    "mean_knn_score": float("nan"),
                    "top10_count": 0,
                    "top10_fraction": 0.0,
                    "top10_label_recall": float("nan"),
                    "top10_enrichment": float("nan"),
                    "base_fraction": 0.0,
                    "is_top10_label": False,
                }
            ]
        )
    return {
        "transition_multiplicity_by_label": transition_multiplicity.reset_index(drop=True),
        "knn_distance_by_label": knn_distance.reset_index(drop=True),
        "rare_useless_density_audit": rare_useless.reset_index(drop=True),
        "density_audit_by_label": by_label.reset_index(drop=True),
    }


def _transition_multiplicity(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby(TRANSITION_KEY_COLUMNS, dropna=False).size().rename("transition_multiplicity")
    merged = frame[TRANSITION_KEY_COLUMNS].merge(
        counts.reset_index(), on=TRANSITION_KEY_COLUMNS, how="left", sort=False
    )
    return merged["transition_multiplicity"].to_numpy(dtype=np.float32)


def _safe_fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _safe_enrichment(top_fraction: float, base_fraction: float) -> float:
    if not np.isfinite(top_fraction) or not np.isfinite(base_fraction) or base_fraction <= 0:
        return float("nan")
    return float(top_fraction / base_fraction)
