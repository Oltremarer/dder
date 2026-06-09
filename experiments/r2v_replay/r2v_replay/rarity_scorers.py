from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
    def __init__(self, config: DiffusionConfig | None = None):
        self.config = config or DiffusionConfig()
        self.model_ = None
        self.dim_: int | None = None

    def fit(self, z: np.ndarray) -> "DiffusionRarityScorer":
        import torch
        from torch import nn

        torch.manual_seed(self.config.seed)
        z_tensor = torch.as_tensor(np.asarray(z, dtype=np.float32))
        self.dim_ = int(z_tensor.shape[1])
        time_dim = 8
        self.model_ = nn.Sequential(
            nn.Linear(self.dim_ + time_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.dim_),
        )
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.config.learning_rate)
        alpha_bar = torch.linspace(0.98, 0.02, self.config.steps)
        generator = torch.Generator().manual_seed(self.config.seed)
        n = len(z_tensor)
        for _ in range(self.config.epochs):
            perm = torch.randperm(n, generator=generator)
            for start in range(0, n, self.config.batch_size):
                idx = perm[start : start + self.config.batch_size]
                clean = z_tensor[idx]
                t_idx = torch.randint(0, self.config.steps, (len(idx),), generator=generator)
                eps = torch.randn(clean.shape, generator=generator)
                a = alpha_bar[t_idx].unsqueeze(1)
                noisy = torch.sqrt(a) * clean + torch.sqrt(1.0 - a) * eps
                t_embed = _time_embedding(t_idx, self.config.steps, time_dim)
                pred = self.model_(torch.cat([noisy, t_embed], dim=1))
                loss = ((pred - eps) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def score(self, z: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.dim_ is None:
            raise RuntimeError("DiffusionRarityScorer must be fit before score")
        import torch

        z_tensor = torch.as_tensor(np.asarray(z, dtype=np.float32))
        alpha_bar = torch.linspace(0.98, 0.02, self.config.steps)
        generator = torch.Generator().manual_seed(self.config.seed + 17)
        scores = torch.zeros(len(z_tensor))
        self.model_.eval()
        with torch.no_grad():
            for _ in range(self.config.eval_repeats):
                t_idx = torch.randint(0, self.config.steps, (len(z_tensor),), generator=generator)
                eps = torch.randn(z_tensor.shape, generator=generator)
                a = alpha_bar[t_idx].unsqueeze(1)
                noisy = torch.sqrt(a) * z_tensor + torch.sqrt(1.0 - a) * eps
                t_embed = _time_embedding(t_idx, self.config.steps, 8)
                pred = self.model_(torch.cat([noisy, t_embed], dim=1))
                scores += ((pred - eps) ** 2).mean(dim=1)
        return (scores / max(1, self.config.eval_repeats)).numpy().astype(np.float32)


def _time_embedding(t_idx, steps: int, dim: int):
    import torch

    t = t_idx.float().unsqueeze(1) / max(1, steps - 1)
    freqs = torch.arange(1, dim // 2 + 1).float().unsqueeze(0)
    emb = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=1)
    if emb.shape[1] < dim:
        emb = torch.cat([emb, t], dim=1)
    return emb[:, :dim]
