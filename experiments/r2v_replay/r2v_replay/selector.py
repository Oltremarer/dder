from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class SelectionScores:
    indices: np.ndarray
    rarity: np.ndarray
    utility: np.ndarray
    ood_risk: np.ndarray
    dynamics_error: np.ndarray
    reward_error: np.ndarray


@dataclass
class SelectedSubset:
    indices: np.ndarray
    final_score: np.ndarray
    candidate_indices: np.ndarray


class R2VSelector:
    def __init__(
        self,
        rare_topk_ratio: float = 0.10,
        selected_budget_ratio: float = 0.05,
        alpha: float = 0.25,
        beta: float = 1.0,
        gamma: float = 0.35,
        eta: float = 0.25,
        rho: float = 0.15,
    ):
        self.rare_topk_ratio = rare_topk_ratio
        self.selected_budget_ratio = selected_budget_ratio
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.rho = rho

    def select(self, scores: SelectionScores) -> SelectedSubset:
        n = len(scores.indices)
        candidate_n = max(1, int(np.ceil(n * self.rare_topk_ratio)))
        candidate_order = np.argsort(-scores.rarity)[:candidate_n]
        candidate_indices = scores.indices[candidate_order]
        final = (
            self.alpha * _rank01(scores.rarity[candidate_order])
            + self.beta * _rank01(scores.utility[candidate_order])
            - self.gamma * _rank01(scores.ood_risk[candidate_order])
            - self.eta * _rank01(scores.dynamics_error[candidate_order])
            - self.rho * _rank01(scores.reward_error[candidate_order])
        )
        budget_n = max(1, int(np.ceil(candidate_n * self.selected_budget_ratio)))
        selected_order = np.argsort(-final)[:budget_n]
        return SelectedSubset(
            indices=candidate_indices[selected_order].astype(np.int64),
            final_score=final[selected_order].astype(np.float32),
            candidate_indices=candidate_indices.astype(np.int64),
        )


def composition_by_label(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    counts = Counter(labels[indices].tolist())
    return dict(sorted(counts.items()))


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 1:
        return np.ones_like(values, dtype=np.float32)
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks
