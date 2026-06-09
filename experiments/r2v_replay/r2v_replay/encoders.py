from __future__ import annotations

import numpy as np

from .replay_dataset import ReplayDataset


class TransitionEncoder:
    def __init__(self, input_mode: str = "obs_action_next", eps: float = 1e-6):
        self.input_mode = input_mode
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.feature_columns_: list[str] = []

    def fit(self, dataset: ReplayDataset, mask: np.ndarray | None = None) -> "TransitionEncoder":
        frame = dataset.feature_matrix(self.input_mode)
        self.feature_columns_ = list(frame.columns)
        values = frame.to_numpy(dtype=np.float32)
        fit_values = values if mask is None else values[mask]
        self.mean_ = fit_values.mean(axis=0)
        self.std_ = fit_values.std(axis=0)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, dataset: ReplayDataset) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("encoder must be fit before transform")
        values = dataset.feature_matrix(self.input_mode).to_numpy(dtype=np.float32)
        return ((values - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, dataset: ReplayDataset, mask: np.ndarray | None = None) -> np.ndarray:
        return self.fit(dataset, mask=mask).transform(dataset)
