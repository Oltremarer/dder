from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .encoders import TransitionEncoder
from .metric_audit import RARE_VALUABLE, candidate_ratio_sweep_rows
from .rarity_scorers import KNNRarityScorer
from .replay_dataset import ReplayDataset


LABEL_ORDER = [
    "common_zero",
    "rare_useless",
    "rare_valuable_zero_precursor",
    "rare_valuable_positive",
    "optional_invalid",
]


def multiplicity_summary_rows(
    labels: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    scores: np.ndarray | None = None,
    raw_distances: np.ndarray | None = None,
) -> list[dict[str, float | int | str]]:
    labels = np.asarray(labels).astype(str)
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    next_states = np.asarray(next_states, dtype=np.float32)
    state_keys = states
    state_action_keys = np.column_stack([states, actions])
    state_action_next_keys = np.column_stack([states, actions, next_states])
    _, inverse = np.unique(state_action_next_keys, axis=0, return_inverse=True)
    multiplicities = np.bincount(inverse)[inverse]

    rows: list[dict[str, float | int | str]] = []
    for label in _ordered_labels(labels):
        mask = labels == label
        if not bool(mask.any()):
            continue
        row: dict[str, float | int | str] = {
            "label": label,
            "count": int(mask.sum()),
            "unique_obs_action_next_count": _unique_count(state_action_next_keys[mask]),
            "unique_state_count": _unique_count(state_keys[mask]),
            "unique_state_action_count": _unique_count(state_action_keys[mask]),
            "unique_state_action_next_count": _unique_count(state_action_next_keys[mask]),
            "mean_multiplicity": float(np.mean(multiplicities[mask])),
            "median_multiplicity": float(np.median(multiplicities[mask])),
            "max_multiplicity": int(np.max(multiplicities[mask])),
        }
        if raw_distances is not None:
            row["median_knn_distance"] = float(np.median(np.asarray(raw_distances)[mask]))
        if scores is not None:
            row["mean_knn_score"] = float(np.mean(np.asarray(scores)[mask]))
        rows.append(row)
    return rows


def candidate_composition_rows(
    labels: np.ndarray,
    scores: np.ndarray,
    ratios: Iterable[float],
    variant: str,
    k_value: int | None = None,
    real_mask: np.ndarray | None = None,
) -> list[dict[str, float | int | str]]:
    rows = candidate_ratio_sweep_rows(labels=labels, scores=scores, ratios=ratios, real_mask=real_mask)
    for row in rows:
        row["variant"] = variant
        if k_value is not None:
            row["k"] = int(k_value)
        row.update(_h2_budget_flags(row))
    return rows


def score_variants(
    dataset: ReplayDataset,
    rarity_input: str,
    k_values: Iterable[int],
    duplicate_cap: int = 5,
) -> dict[str, np.ndarray]:
    k_values = [int(k) for k in k_values]
    real_mask = dataset.real_mask()
    z = TransitionEncoder(input_mode=rarity_input).fit_transform(dataset, mask=real_mask)
    real_z = z[real_mask]
    variants: dict[str, np.ndarray] = {}
    for k in k_values:
        variants[f"raw_k{k}"] = KNNRarityScorer(k=k).fit(real_z).score(z)
    main_k = k_values[-1]
    variants[f"unique_transition_k{main_k}"] = _unique_transition_scores(z, real_mask, k=main_k)
    variants[f"duplicate_capped{duplicate_cap}_k{main_k}"] = _duplicate_capped_scores(
        z, real_mask, cap=duplicate_cap, k=main_k
    )
    return variants


def distance_to_label_rows(dataset: ReplayDataset, rarity_input: str, target_labels: Iterable[str]) -> list[dict[str, float | int | str]]:
    real_mask = dataset.real_mask()
    labels = dataset.labels.astype(str)
    z = TransitionEncoder(input_mode=rarity_input).fit_transform(dataset, mask=real_mask)
    rows: list[dict[str, float | int | str]] = []
    real_indices = np.flatnonzero(real_mask)
    for target in target_labels:
        target_mask = real_mask & _label_group_mask(labels, target)
        if not bool(target_mask.any()):
            continue
        nn = NearestNeighbors(n_neighbors=1).fit(z[target_mask])
        distances, _ = nn.kneighbors(z[real_indices])
        nearest = distances[:, 0]
        real_labels = labels[real_indices]
        for label in _ordered_labels(real_labels):
            mask = real_labels == label
            if not bool(mask.any()):
                continue
            values = nearest[mask]
            rows.append(
                {
                    "target_label_group": target,
                    "source_label": label,
                    "count": int(mask.sum()),
                    "median_distance": float(np.median(values)),
                    "mean_distance": float(np.mean(values)),
                    "p90_distance": float(np.quantile(values, 0.90)),
                }
            )
    return rows


def _unique_transition_scores(z: np.ndarray, real_mask: np.ndarray, k: int) -> np.ndarray:
    real_z = z[real_mask]
    unique_z, inverse = np.unique(real_z, axis=0, return_inverse=True)
    unique_scores = KNNRarityScorer(k=k).fit(unique_z).score(unique_z)
    real_scores = unique_scores[inverse]
    scores = np.zeros(len(z), dtype=np.float32)
    scores[real_mask] = real_scores
    return scores


def _duplicate_capped_scores(z: np.ndarray, real_mask: np.ndarray, cap: int, k: int) -> np.ndarray:
    real_indices = np.flatnonzero(real_mask)
    real_z = z[real_mask]
    _, inverse = np.unique(real_z, axis=0, return_inverse=True)
    selected: list[int] = []
    for group_id in np.unique(inverse):
        group_positions = np.flatnonzero(inverse == group_id)
        selected.extend(real_indices[group_positions[:cap]].tolist())
    fit_z = z[np.asarray(selected, dtype=np.int64)]
    return KNNRarityScorer(k=k).fit(fit_z).score(z)


def score_table_raw_scores(score_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scores = score_table["knn_rarity"].to_numpy(dtype=np.float32)
    if "knn_rarity_raw" in score_table:
        raw = score_table["knn_rarity_raw"].to_numpy(dtype=np.float32)
    else:
        raw = scores
    return scores, raw


def _label_group_mask(labels: np.ndarray, label_group: str) -> np.ndarray:
    if label_group == "rare_valuable":
        return np.isin(labels, list(RARE_VALUABLE))
    return labels == label_group


def _ordered_labels(labels: np.ndarray) -> list[str]:
    present = set(np.asarray(labels).astype(str).tolist())
    ordered = [label for label in LABEL_ORDER if label in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _unique_count(values: np.ndarray) -> int:
    if len(values) == 0:
        return 0
    return int(len(np.unique(values, axis=0)))


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
