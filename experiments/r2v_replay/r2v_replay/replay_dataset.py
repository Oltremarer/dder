from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = {
    "obs_action_next": [
        "state_x",
        "state_y",
        "action",
        "next_state_x",
        "next_state_y",
        "done",
    ],
    "full_transition": [
        "state_x",
        "state_y",
        "action",
        "reward",
        "next_state_x",
        "next_state_y",
        "done",
    ],
}


@dataclass
class ReplayDataset:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    episode_ids: np.ndarray
    timesteps: np.ndarray
    behavior_policy_ids: np.ndarray
    distances_to_goal: np.ndarray
    next_distances_to_goal: np.ndarray
    returns_to_go: np.ndarray
    labels: np.ndarray
    optional_invalid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.states)
        arrays: Iterable[np.ndarray] = (
            self.actions,
            self.rewards,
            self.next_states,
            self.dones,
            self.episode_ids,
            self.timesteps,
            self.behavior_policy_ids,
            self.distances_to_goal,
            self.next_distances_to_goal,
            self.returns_to_go,
            self.labels,
        )
        for arr in arrays:
            if len(arr) != n:
                raise ValueError("all replay fields must have the same length")

        self.states = np.asarray(self.states, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.int64)
        self.rewards = np.asarray(self.rewards, dtype=np.float32)
        self.next_states = np.asarray(self.next_states, dtype=np.float32)
        self.dones = np.asarray(self.dones, dtype=bool)
        self.episode_ids = np.asarray(self.episode_ids, dtype=np.int64)
        self.timesteps = np.asarray(self.timesteps, dtype=np.int64)
        self.behavior_policy_ids = np.asarray(self.behavior_policy_ids).astype(str)
        self.distances_to_goal = np.asarray(self.distances_to_goal, dtype=np.float32)
        self.next_distances_to_goal = np.asarray(self.next_distances_to_goal, dtype=np.float32)
        self.returns_to_go = np.asarray(self.returns_to_go, dtype=np.float32)
        self.labels = np.asarray(self.labels).astype(str)
        if self.optional_invalid_mask is None:
            self.optional_invalid_mask = np.zeros(n, dtype=bool)
        else:
            self.optional_invalid_mask = np.asarray(self.optional_invalid_mask, dtype=bool)

    def __len__(self) -> int:
        return int(len(self.states))

    def real_mask(self) -> np.ndarray:
        return ~self.optional_invalid_mask

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "state_x": self.states[:, 0],
                "state_y": self.states[:, 1],
                "action": self.actions,
                "reward": self.rewards,
                "next_state_x": self.next_states[:, 0],
                "next_state_y": self.next_states[:, 1],
                "done": self.dones.astype(np.int64),
                "episode_id": self.episode_ids,
                "timestep": self.timesteps,
                "behavior_policy_id": self.behavior_policy_ids,
                "distance_to_goal": self.distances_to_goal,
                "next_distance_to_goal": self.next_distances_to_goal,
                "return_to_go": self.returns_to_go,
                "label_for_eval_only": self.labels,
                "optional_invalid": self.optional_invalid_mask,
            }
        )

    def feature_matrix(self, mode: str) -> pd.DataFrame:
        if mode not in FEATURE_COLUMNS:
            raise ValueError(f"unknown feature mode: {mode}")
        return self.to_frame()[FEATURE_COLUMNS[mode]].copy()

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            next_states=self.next_states,
            dones=self.dones,
            episode_ids=self.episode_ids,
            timesteps=self.timesteps,
            behavior_policy_ids=self.behavior_policy_ids,
            distances_to_goal=self.distances_to_goal,
            next_distances_to_goal=self.next_distances_to_goal,
            returns_to_go=self.returns_to_go,
            labels=self.labels,
            optional_invalid_mask=self.optional_invalid_mask,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "ReplayDataset":
        data = np.load(path, allow_pickle=False)
        optional_invalid_mask = data["optional_invalid_mask"] if "optional_invalid_mask" in data else None
        return cls(
            states=data["states"],
            actions=data["actions"],
            rewards=data["rewards"],
            next_states=data["next_states"],
            dones=data["dones"],
            episode_ids=data["episode_ids"],
            timesteps=data["timesteps"],
            behavior_policy_ids=data["behavior_policy_ids"],
            distances_to_goal=data["distances_to_goal"],
            next_distances_to_goal=data["next_distances_to_goal"],
            returns_to_go=data["returns_to_go"],
            labels=data["labels"],
            optional_invalid_mask=optional_invalid_mask,
        )
