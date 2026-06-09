from __future__ import annotations

import numpy as np


RARE_VALUABLE = {"rare_valuable_positive", "rare_valuable_zero_precursor"}


def selection_audit_row(
    labels: np.ndarray,
    rewards: np.ndarray,
    selected_indices: np.ndarray,
    candidate_indices: np.ndarray | None = None,
    real_mask: np.ndarray | None = None,
    episode_ids: np.ndarray | None = None,
) -> dict[str, bool | float | int]:
    labels = np.asarray(labels).astype(str)
    rewards = np.asarray(rewards, dtype=np.float32)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if candidate_indices is None:
        candidate_indices = selected_indices
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if real_mask is None:
        real_mask = labels != "optional_invalid"
    real_mask = np.asarray(real_mask, dtype=bool)

    selected_labels = labels[selected_indices]
    candidate_labels = labels[candidate_indices]
    real_labels = labels[real_mask]

    row: dict[str, bool | float | int] = {
        "real_transition_count": int(real_mask.sum()),
        "candidate_count": int(len(candidate_indices)),
        "selected_count": int(len(selected_indices)),
        "total_rare_valuable_positive": _count(real_labels, "rare_valuable_positive"),
        "total_rare_valuable_zero_precursor": _count(real_labels, "rare_valuable_zero_precursor"),
        "total_rare_valuable_all": _count_any(real_labels, RARE_VALUABLE),
        "candidate_rare_valuable_positive_count": _count(candidate_labels, "rare_valuable_positive"),
        "candidate_rare_valuable_zero_precursor_count": _count(candidate_labels, "rare_valuable_zero_precursor"),
        "candidate_rare_valuable_all_count": _count_any(candidate_labels, RARE_VALUABLE),
        "candidate_rare_useless_count": _count(candidate_labels, "rare_useless"),
        "candidate_common_zero_count": _count(candidate_labels, "common_zero"),
        "candidate_optional_invalid_count": _count(candidate_labels, "optional_invalid"),
        "selected_rare_valuable_positive_count": _count(selected_labels, "rare_valuable_positive"),
        "selected_rare_valuable_zero_precursor_count": _count(selected_labels, "rare_valuable_zero_precursor"),
        "selected_rare_valuable_all_count": _count_any(selected_labels, RARE_VALUABLE),
        "selected_rare_useless_count": _count(selected_labels, "rare_useless"),
        "selected_common_zero_count": _count(selected_labels, "common_zero"),
        "selected_optional_invalid_count": _count(selected_labels, "optional_invalid"),
        "valuable_precision_denominator": int(len(selected_indices)),
        "zero_precursor_recall_denominator": _count(real_labels, "rare_valuable_zero_precursor"),
        "positive_reward_fraction_denominator": int(len(selected_indices)),
    }

    row["valuable_precision"] = _safe_fraction(row["selected_rare_valuable_all_count"], row["selected_count"])
    row["zero_precursor_recall"] = _safe_fraction(
        row["selected_rare_valuable_zero_precursor_count"], row["zero_precursor_recall_denominator"]
    )
    row["positive_reward_fraction"] = _safe_mean(rewards[selected_indices] > 0)
    row["rare_useless_fraction"] = _safe_fraction(row["selected_rare_useless_count"], row["selected_count"])
    row["candidate_rare_useless_fraction"] = _safe_fraction(row["candidate_rare_useless_count"], row["candidate_count"])
    row["candidate_common_zero_fraction"] = _safe_fraction(row["candidate_common_zero_count"], row["candidate_count"])
    row["candidate_valuable_precision"] = _safe_fraction(
        row["candidate_rare_valuable_all_count"], row["candidate_count"]
    )
    row["candidate_rare_useless_enrichment_vs_uniform"] = _enrichment_vs_uniform(
        row["candidate_rare_useless_fraction"],
        _safe_fraction(_count(real_labels, "rare_useless"), row["real_transition_count"]),
    )
    row["selected_episode_diversity"] = _episode_diversity(episode_ids, selected_indices)
    row["optional_invalid_excluded"] = bool(
        row["selected_optional_invalid_count"] == 0 and row["candidate_optional_invalid_count"] == 0
    )
    row["all_assertion_checks_pass"] = bool(_assertion_checks(row))
    return row


def _count(labels: np.ndarray, label: str) -> int:
    return int(np.sum(labels == label))


def _count_any(labels: np.ndarray, label_set: set[str]) -> int:
    return int(np.isin(labels, list(label_set)).sum())


def _safe_fraction(numerator: bool | float | int, denominator: bool | float | int) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return float("nan")
    return float(numerator) / denominator


def _safe_mean(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _enrichment_vs_uniform(candidate_fraction: float, base_fraction: float) -> float:
    if not np.isfinite(candidate_fraction) or not np.isfinite(base_fraction) or base_fraction <= 0:
        return float("nan")
    return float(candidate_fraction / base_fraction)


def _episode_diversity(episode_ids: np.ndarray | None, selected_indices: np.ndarray) -> int:
    if episode_ids is None or len(selected_indices) == 0:
        return 0
    selected_episodes = np.asarray(episode_ids, dtype=np.int64)[selected_indices]
    selected_episodes = selected_episodes[selected_episodes >= 0]
    return int(len(np.unique(selected_episodes)))


def _assertion_checks(row: dict[str, bool | float | int]) -> bool:
    selected_valuable = int(row["selected_rare_valuable_all_count"])
    selected_count = int(row["selected_count"])
    return (
        selected_valuable <= selected_count
        and int(row["selected_rare_valuable_positive_count"]) <= int(row["total_rare_valuable_positive"])
        and int(row["selected_rare_valuable_zero_precursor_count"])
        <= int(row["total_rare_valuable_zero_precursor"])
        and bool(row["optional_invalid_excluded"])
    )
