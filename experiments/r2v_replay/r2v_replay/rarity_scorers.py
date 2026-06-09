from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors


class KNNRarityScorer:
    def __init__(self, k: int = 10):
        self.k = k
        self.model_: NearestNeighbors | None = None
        self.fit_z_: np.ndarray | None = None

    def fit(self, z: np.ndarray) -> "KNNRarityScorer":
        self.fit_z_ = np.asarray(z, dtype=np.float32)
        n_neighbors = min(len(self.fit_z_), self.k + 1)
        self.model_ = NearestNeighbors(n_neighbors=n_neighbors)
        self.model_.fit(self.fit_z_)
        return self

    def score(self, z: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("KNNRarityScorer must be fit before score")
        distances, _ = self.model_.kneighbors(np.asarray(z, dtype=np.float32))
        if distances.shape[1] > 1:
            distances = distances[:, 1:]
        return distances.mean(axis=1).astype(np.float32)


@dataclass
class DiffusionConfig:
    steps: int = 32
    epochs: int = 40
    batch_size: int = 256
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    eval_repeats: int = 4
    seed: int = 0


class DiffusionRarityScorer:
    """Small denoising rarity scorer.

    This keeps the Level 0 scorer lightweight enough for Ubuntu smoke tests by
    using scikit-learn instead of a heavyweight deep-learning runtime. It still
    trains on noisy latent transitions and scores denoising error; it does not
    generate replay samples.
    """

    def __init__(self, config: DiffusionConfig | None = None):
        self.config = config or DiffusionConfig()
        self.model_: MLPRegressor | None = None
        self.dim_: int | None = None

    def fit(self, z: np.ndarray) -> "DiffusionRarityScorer":
        rng = np.random.default_rng(self.config.seed)
        z_arr = np.asarray(z, dtype=np.float32)
        self.dim_ = int(z_arr.shape[1])
        noisy_inputs: list[np.ndarray] = []
        eps_targets: list[np.ndarray] = []
        for _ in range(max(1, self.config.eval_repeats)):
            t = rng.integers(0, self.config.steps, size=len(z_arr))
            alpha = np.linspace(0.98, 0.02, self.config.steps, dtype=np.float32)[t][:, None]
            eps = rng.normal(size=z_arr.shape).astype(np.float32)
            noisy = np.sqrt(alpha) * z_arr + np.sqrt(1.0 - alpha) * eps
            noisy_inputs.append(np.hstack([noisy, _time_embedding_np(t, self.config.steps, 8)]))
            eps_targets.append(eps)
        x = np.vstack(noisy_inputs)
        y = np.vstack(eps_targets)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=(self.config.hidden_dim, self.config.hidden_dim),
            activation="relu",
            learning_rate_init=self.config.learning_rate,
            max_iter=self.config.epochs,
            random_state=self.config.seed,
            batch_size=min(self.config.batch_size, len(x)),
            early_stopping=False,
        )
        self.model_.fit(x, y)
        return self

    def score(self, z: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.dim_ is None:
            raise RuntimeError("DiffusionRarityScorer must be fit before score")
        rng = np.random.default_rng(self.config.seed + 17)
        z_arr = np.asarray(z, dtype=np.float32)
        scores = np.zeros(len(z_arr), dtype=np.float32)
        for _ in range(max(1, self.config.eval_repeats)):
            t = rng.integers(0, self.config.steps, size=len(z_arr))
            alpha = np.linspace(0.98, 0.02, self.config.steps, dtype=np.float32)[t][:, None]
            eps = rng.normal(size=z_arr.shape).astype(np.float32)
            noisy = np.sqrt(alpha) * z_arr + np.sqrt(1.0 - alpha) * eps
            pred = self.model_.predict(np.hstack([noisy, _time_embedding_np(t, self.config.steps, 8)]))
            scores += ((pred - eps) ** 2).mean(axis=1).astype(np.float32)
        return scores / max(1, self.config.eval_repeats)


def _time_embedding_np(t_idx: np.ndarray, steps: int, dim: int) -> np.ndarray:
    t = t_idx.astype(np.float32)[:, None] / max(1, steps - 1)
    freqs = np.arange(1, dim // 2 + 1, dtype=np.float32)[None, :]
    emb = np.concatenate([np.sin(t * freqs), np.cos(t * freqs)], axis=1)
    if emb.shape[1] < dim:
        emb = np.concatenate([emb, t], axis=1)
    return emb[:, :dim].astype(np.float32)
