from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .replay_dataset import ReplayDataset


@dataclass
class UtilityScores:
    total: np.ndarray
    components: dict[str, np.ndarray]


class TaskUtilityScorer:
    def __init__(self, w_reward: float = 0.2, w_rtg: float = 0.4, w_progress: float = 0.4):
        self.w_reward = w_reward
        self.w_rtg = w_rtg
        self.w_progress = w_progress

    def score(self, dataset: ReplayDataset) -> UtilityScores:
        reward = (dataset.rewards > 0).astype(np.float32)
        rtg = _normalize(dataset.returns_to_go.astype(np.float32))
        progress_raw = np.maximum(0.0, dataset.distances_to_goal - dataset.next_distances_to_goal)
        progress = _normalize(progress_raw.astype(np.float32))
        total = self.w_reward * reward + self.w_rtg * rtg + self.w_progress * progress
        return UtilityScores(
            total=total.astype(np.float32),
            components={
                "reward": reward.astype(np.float32),
                "rtg": rtg.astype(np.float32),
                "progress": progress.astype(np.float32),
            },
        )


def _normalize(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)
