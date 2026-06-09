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
    folds: int = 3
    noise_levels: tuple[float, ...] = (0.03, 0.05, 0.10, 0.20, 0.35)
    time_embedding_dim: int = 8
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
        z_arr = np.asarray(z, dtype=np.float32)
        if len(z_arr) == 0:
            raise ValueError("cannot fit diffusion rarity scorer on an empty array")
        self.dim_ = int(z_arr.shape[1])
        self.model_ = self._fit_model(z_arr, seed=self.config.seed)
        return self

    def score(self, z: np.ndarray) -> np.ndarray:
        components = self.score_components(z)
        return components["rank01_mean"]

    def score_components(self, z: np.ndarray) -> dict[str, np.ndarray]:
        if self.model_ is None or self.dim_ is None:
            raise RuntimeError("DiffusionRarityScorer must be fit before score")
        z_arr = np.asarray(z, dtype=np.float32)
        components = self._score_components_with_model(self.model_, z_arr, seed=self.config.seed + 17)
        return self._summarize_components(components)

    def cross_fit_score(self, z: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z_arr = np.asarray(z, dtype=np.float32)
        if len(z_arr) == 0:
            raise ValueError("cannot cross-fit diffusion rarity scorer on an empty array")
        if len(z_arr) == 1:
            self.fit(z_arr)
            components = self.score_components(z_arr)
            return components["rank01_mean"], components

        folds = min(max(2, int(self.config.folds)), len(z_arr))
        rng = np.random.default_rng(self.config.seed + 313)
        shuffled = rng.permutation(len(z_arr))
        holdout_folds = np.array_split(shuffled, folds)
        component_values = {
            _noise_key(sigma): np.zeros(len(z_arr), dtype=np.float32) for sigma in self._noise_levels()
        }

        all_indices = np.arange(len(z_arr))
        for fold_id, holdout in enumerate(holdout_folds):
            if len(holdout) == 0:
                continue
            train_mask = np.ones(len(z_arr), dtype=bool)
            train_mask[holdout] = False
            train_indices = all_indices[train_mask]
            model = self._fit_model(z_arr[train_indices], seed=self.config.seed + fold_id)
            fold_components = self._score_components_with_model(
                model, z_arr[holdout], seed=self.config.seed + 1000 + fold_id
            )
            for name, values in fold_components.items():
                component_values[name][holdout] = values

        components = self._summarize_components(component_values)
        return components["rank01_mean"], components

    def _fit_model(self, z_arr: np.ndarray, seed: int) -> MLPRegressor:
        rng = np.random.default_rng(seed)
        noise_levels = self._noise_levels()
        noisy_inputs: list[np.ndarray] = []
        eps_targets: list[np.ndarray] = []
        for _ in range(max(1, int(self.config.eval_repeats))):
            for level_idx, sigma in enumerate(noise_levels):
                eps = rng.normal(size=z_arr.shape).astype(np.float32)
                noisy = z_arr + float(sigma) * eps
                level = np.full(len(z_arr), level_idx, dtype=np.int64)
                noisy_inputs.append(
                    np.hstack([noisy, _time_embedding_np(level, len(noise_levels), self.config.time_embedding_dim)])
                )
                eps_targets.append(eps)
        x = np.vstack(noisy_inputs)
        y = np.vstack(eps_targets)
        model = MLPRegressor(
            hidden_layer_sizes=(self.config.hidden_dim, self.config.hidden_dim),
            activation="relu",
            learning_rate_init=self.config.learning_rate,
            max_iter=self.config.epochs,
            random_state=seed,
            batch_size=min(self.config.batch_size, len(x)),
            early_stopping=False,
        )
        model.fit(x, y)
        return model

    def _score_components_with_model(self, model: MLPRegressor, z_arr: np.ndarray, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        noise_levels = self._noise_levels()
        components: dict[str, np.ndarray] = {}
        for level_idx, sigma in enumerate(noise_levels):
            scores = np.zeros(len(z_arr), dtype=np.float32)
            level = np.full(len(z_arr), level_idx, dtype=np.int64)
            emb = _time_embedding_np(level, len(noise_levels), self.config.time_embedding_dim)
            for _ in range(max(1, int(self.config.eval_repeats))):
                eps = rng.normal(size=z_arr.shape).astype(np.float32)
                noisy = z_arr + float(sigma) * eps
                pred = model.predict(np.hstack([noisy, emb]))
                scores += ((pred - eps) ** 2).mean(axis=1).astype(np.float32)
            components[_noise_key(float(sigma))] = scores / max(1, int(self.config.eval_repeats))
        return components

    def _summarize_components(self, components: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        summarized = {name: values.astype(np.float32) for name, values in components.items()}
        noise_levels = self._noise_levels()
        noise_columns = [summarized[_noise_key(sigma)] for sigma in noise_levels]
        stacked = np.vstack(noise_columns)
        summarized["low_noise"] = _masked_noise_mean(stacked, noise_levels <= 0.05)
        summarized["mid_noise"] = _masked_noise_mean(stacked, (noise_levels > 0.05) & (noise_levels <= 0.20))
        summarized["high_noise"] = _masked_noise_mean(stacked, noise_levels > 0.20)
        summarized["mean_raw"] = stacked.mean(axis=0).astype(np.float32)
        summarized["rank01_mean"] = _rank01(summarized["mean_raw"])
        return summarized

    def _noise_levels(self) -> np.ndarray:
        levels = np.asarray(tuple(self.config.noise_levels), dtype=np.float32)
        if len(levels) == 0:
            raise ValueError("DiffusionConfig.noise_levels must not be empty")
        return levels


def _time_embedding_np(t_idx: np.ndarray, steps: int, dim: int) -> np.ndarray:
    t = t_idx.astype(np.float32)[:, None] / max(1, steps - 1)
    freqs = np.arange(1, dim // 2 + 1, dtype=np.float32)[None, :]
    emb = np.concatenate([np.sin(t * freqs), np.cos(t * freqs)], axis=1)
    if emb.shape[1] < dim:
        emb = np.concatenate([emb, t], axis=1)
    return emb[:, :dim].astype(np.float32)


def _noise_key(sigma: float) -> str:
    return f"noise_{sigma:.2f}"


def _masked_noise_mean(stacked: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not bool(mask.any()):
        return np.zeros(stacked.shape[1], dtype=np.float32)
    return stacked[mask].mean(axis=0).astype(np.float32)


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 1:
        return np.ones_like(values, dtype=np.float32)
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks
