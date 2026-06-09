from __future__ import annotations

import numpy as np

from .rarity_scorers import KNNRarityScorer


class OODRiskScorer:
    def __init__(self, k: int = 10):
        self._knn = KNNRarityScorer(k=k)

    def fit(self, z: np.ndarray) -> "OODRiskScorer":
        self._knn.fit(z)
        return self

    def score(self, z: np.ndarray) -> np.ndarray:
        return self._knn.score(z)
