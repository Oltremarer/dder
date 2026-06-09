from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPRegressor

from .replay_dataset import ReplayDataset


@dataclass
class ConsistencyScores:
    dynamics_error: np.ndarray
    reward_error: np.ndarray


class DynamicsConsistencyScorer:
    def __init__(self, epochs: int = 80, hidden_dim: int = 64, learning_rate: float = 1e-3, seed: int = 0):
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.seed = seed
        self.model_: MLPRegressor | None = None

    def fit(self, dataset: ReplayDataset, mask: np.ndarray | None = None) -> "DynamicsConsistencyScorer":
        x, y = _make_xy(dataset)
        if mask is not None:
            x = x[mask]
            y = y[mask]
        self.model_ = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim, self.hidden_dim),
            activation="relu",
            learning_rate_init=self.learning_rate,
            max_iter=self.epochs,
            random_state=self.seed,
            early_stopping=False,
        )
        self.model_.fit(x, y)
        return self

    def score(self, dataset: ReplayDataset) -> ConsistencyScores:
        if self.model_ is None:
            raise RuntimeError("DynamicsConsistencyScorer must be fit before score")
        x, y = _make_xy(dataset)
        pred = self.model_.predict(x)
        dynamics_error = ((pred[:, :2] - y[:, :2]) ** 2).mean(axis=1)
        reward_error = (pred[:, 2] - y[:, 2]) ** 2
        return ConsistencyScores(
            dynamics_error=dynamics_error.astype(np.float32),
            reward_error=reward_error.astype(np.float32),
        )


def _make_xy(dataset: ReplayDataset) -> tuple[np.ndarray, np.ndarray]:
    actions = np.zeros((len(dataset), 4), dtype=np.float32)
    actions[np.arange(len(dataset)), dataset.actions.clip(0, 3)] = 1.0
    x = np.hstack([dataset.states.astype(np.float32), actions])
    y = np.hstack([dataset.next_states.astype(np.float32), dataset.rewards[:, None].astype(np.float32)])
    return x.astype(np.float32), y.astype(np.float32)
