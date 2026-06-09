from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
        self.model_ = None

    def fit(self, dataset: ReplayDataset, mask: np.ndarray | None = None) -> "DynamicsConsistencyScorer":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        x, y = _make_xy(dataset)
        if mask is not None:
            x = x[mask]
            y = y[mask]
        x_tensor = torch.as_tensor(x)
        y_tensor = torch.as_tensor(y)
        self.model_ = nn.Sequential(
            nn.Linear(x.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, y.shape[1]),
        )
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        for _ in range(self.epochs):
            pred = self.model_(x_tensor)
            loss = ((pred - y_tensor) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    def score(self, dataset: ReplayDataset) -> ConsistencyScores:
        if self.model_ is None:
            raise RuntimeError("DynamicsConsistencyScorer must be fit before score")
        import torch

        x, y = _make_xy(dataset)
        with torch.no_grad():
            pred = self.model_(torch.as_tensor(x)).numpy()
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
