from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .replay_dataset import ReplayDataset


Action = int
State = tuple[int, int]


ACTION_DELTAS: dict[Action, State] = {
    0: (0, 1),
    1: (0, -1),
    2: (-1, 0),
    3: (1, 0),
}


@dataclass(frozen=True)
class SparseGridConfig:
    width: int = 15
    height: int = 15
    start_states: tuple[State, ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
    goal_state: State = (14, 14)
    walls: frozenset[State] = frozenset()
    decoy_states: frozenset[State] = frozenset()
    bottleneck_states: frozenset[State] = frozenset()
    max_steps: int = 80
    precursor_window: int = 8

    @classmethod
    def default(cls) -> "SparseGridConfig":
        width = 15
        height = 15
        walls = {(7, y) for y in range(height) if y != 7}
        walls.update({(x, 7) for x in range(width) if x not in {6, 7, 8}})
        decoy_states = {(2, 6), (3, 5), (3, 6)}
        bottleneck_states = {(6, 7), (7, 7), (8, 7)}
        return cls(
            width=width,
            height=height,
            start_states=((0, 0), (0, 1), (1, 0), (1, 1)),
            goal_state=(14, 14),
            walls=frozenset(walls),
            decoy_states=frozenset(decoy_states),
            bottleneck_states=frozenset(bottleneck_states),
            max_steps=80,
            precursor_window=8,
        )


class SparseGridEnv:
    def __init__(self, config: SparseGridConfig, seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.state: State = config.start_states[0]
        self.timestep = 0

    def reset(self, start_state: State | None = None) -> State:
        if start_state is None:
            idx = int(self.rng.integers(0, len(self.config.start_states)))
            start_state = self.config.start_states[idx]
        self.state = start_state
        self.timestep = 0
        return self.state

    def step(self, action: Action) -> tuple[State, float, bool, dict[str, object]]:
        if action not in ACTION_DELTAS:
            raise ValueError(f"unknown action: {action}")
        dx, dy = ACTION_DELTAS[action]
        candidate = (self.state[0] + dx, self.state[1] + dy)
        hit_wall = not self._is_valid(candidate)
        next_state = self.state if hit_wall else candidate
        self.timestep += 1
        reward = 1.0 if next_state == self.config.goal_state else 0.0
        done = bool(reward > 0.0 or self.timestep >= self.config.max_steps)
        self.state = next_state
        return next_state, reward, done, {"hit_wall": hit_wall}

    def distance_to_goal(self, state: State) -> int:
        return abs(state[0] - self.config.goal_state[0]) + abs(state[1] - self.config.goal_state[1])

    def _is_valid(self, state: State) -> bool:
        x, y = state
        return 0 <= x < self.config.width and 0 <= y < self.config.height and state not in self.config.walls


def _greedy_action_towards(state: State, target: State, env: SparseGridEnv, rng: np.random.Generator) -> Action:
    candidates: list[tuple[int, Action]] = []
    for action, (dx, dy) in ACTION_DELTAS.items():
        nxt = (state[0] + dx, state[1] + dy)
        if env._is_valid(nxt):
            dist = abs(nxt[0] - target[0]) + abs(nxt[1] - target[1])
            candidates.append((dist, action))
    if not candidates:
        return int(rng.integers(0, 4))
    best_dist = min(dist for dist, _ in candidates)
    best_actions = [action for dist, action in candidates if dist == best_dist]
    return int(rng.choice(best_actions))


def _policy_action(policy_name: str, env: SparseGridEnv, rng: np.random.Generator) -> Action:
    state = env.state
    if policy_name == "random_wander":
        return int(rng.integers(0, 4))
    if policy_name == "noisy_goal":
        if rng.random() < 0.75:
            return _greedy_action_towards(state, env.config.goal_state, env, rng)
        return int(rng.integers(0, 4))
    if policy_name == "decoy":
        decoy_target = min(env.config.decoy_states, key=lambda s: abs(s[0] - 3) + abs(s[1] - 6))
        if rng.random() < 0.85:
            return _greedy_action_towards(state, decoy_target, env, rng)
        return int(rng.integers(0, 4))
    if policy_name == "near_success":
        if rng.random() < 0.92:
            return _greedy_action_towards(state, env.config.goal_state, env, rng)
        return int(rng.integers(0, 4))
    raise ValueError(f"unknown policy: {policy_name}")


def _start_for_policy(policy_name: str, cfg: SparseGridConfig, rng: np.random.Generator) -> State:
    if policy_name == "near_success":
        candidates = ((8, 7), (8, 8), (9, 8), (10, 9))
        valid = [s for s in candidates if 0 <= s[0] < cfg.width and 0 <= s[1] < cfg.height and s not in cfg.walls]
        return valid[int(rng.integers(0, len(valid)))]
    idx = int(rng.integers(0, len(cfg.start_states)))
    return cfg.start_states[idx]


def _label_episode(records: list[dict[str, object]], cfg: SparseGridConfig) -> None:
    episode_return = float(sum(float(r["reward"]) for r in records))
    success_indices = [i for i, r in enumerate(records) if float(r["reward"]) > 0.0]
    success_index = success_indices[0] if success_indices else None

    for i, row in enumerate(records):
        state = tuple(row["state"])  # type: ignore[arg-type]
        next_state = tuple(row["next_state"])  # type: ignore[arg-type]
        reward = float(row["reward"])
        progress = float(row["distance_to_goal"]) - float(row["next_distance_to_goal"])
        if reward > 0.0:
            label = "rare_valuable_positive"
        elif success_index is not None and episode_return > 0.0 and 0 < (success_index - i) <= cfg.precursor_window:
            in_bottleneck = state in cfg.bottleneck_states or next_state in cfg.bottleneck_states
            label = "rare_valuable_zero_precursor" if progress >= 1.0 or in_bottleneck else "common_zero"
        elif state in cfg.decoy_states or next_state in cfg.decoy_states:
            label = "rare_useless"
        else:
            label = "common_zero"
        row["label_for_eval_only"] = label
        row["return_to_go"] = episode_return


def _make_optional_invalid(dataset: ReplayDataset, ratio: float, seed: int) -> ReplayDataset:
    if ratio <= 0:
        return dataset
    rng = np.random.default_rng(seed + 991)
    n_invalid = max(1, int(round(len(dataset) * ratio)))
    source_idx = rng.choice(np.where(dataset.real_mask())[0], size=n_invalid, replace=True)
    states = dataset.states[source_idx].copy()
    actions = dataset.actions[source_idx].copy()
    rewards = 1.0 - dataset.rewards[source_idx].copy()
    next_states = dataset.next_states[source_idx].copy()
    next_states[:, 0] = rng.integers(0, int(np.max(dataset.states[:, 0])) + 1, size=n_invalid)
    next_states[:, 1] = rng.integers(0, int(np.max(dataset.states[:, 1])) + 1, size=n_invalid)

    invalid = ReplayDataset(
        states=states,
        actions=actions,
        rewards=rewards.astype(np.float32),
        next_states=next_states,
        dones=np.zeros(n_invalid, dtype=bool),
        episode_ids=np.full(n_invalid, -1, dtype=np.int64),
        timesteps=np.arange(n_invalid, dtype=np.int64),
        behavior_policy_ids=np.full(n_invalid, "optional_invalid"),
        distances_to_goal=dataset.distances_to_goal[source_idx],
        next_distances_to_goal=dataset.next_distances_to_goal[source_idx],
        returns_to_go=np.zeros(n_invalid, dtype=np.float32),
        labels=np.full(n_invalid, "optional_invalid"),
        optional_invalid_mask=np.ones(n_invalid, dtype=bool),
    )
    return ReplayDataset(
        states=np.vstack([dataset.states, invalid.states]),
        actions=np.concatenate([dataset.actions, invalid.actions]),
        rewards=np.concatenate([dataset.rewards, invalid.rewards]),
        next_states=np.vstack([dataset.next_states, invalid.next_states]),
        dones=np.concatenate([dataset.dones, invalid.dones]),
        episode_ids=np.concatenate([dataset.episode_ids, invalid.episode_ids]),
        timesteps=np.concatenate([dataset.timesteps, invalid.timesteps]),
        behavior_policy_ids=np.concatenate([dataset.behavior_policy_ids, invalid.behavior_policy_ids]),
        distances_to_goal=np.concatenate([dataset.distances_to_goal, invalid.distances_to_goal]),
        next_distances_to_goal=np.concatenate([dataset.next_distances_to_goal, invalid.next_distances_to_goal]),
        returns_to_go=np.concatenate([dataset.returns_to_go, invalid.returns_to_go]),
        labels=np.concatenate([dataset.labels, invalid.labels]),
        optional_invalid_mask=np.concatenate([dataset.optional_invalid_mask, invalid.optional_invalid_mask]),
    )


def build_sparse_grid_replay(
    config: SparseGridConfig,
    n_transitions: int,
    seed: int,
    policy_mix: Mapping[str, float] | None = None,
    include_optional_invalid: bool = False,
    optional_invalid_ratio: float = 0.03,
) -> ReplayDataset:
    rng = np.random.default_rng(seed)
    env = SparseGridEnv(config, seed=seed)
    policy_mix = dict(
        policy_mix
        or {
            "random_wander": 0.58,
            "noisy_goal": 0.23,
            "decoy": 0.04,
            "near_success": 0.15,
        }
    )
    names = np.array(list(policy_mix.keys()))
    weights = np.array(list(policy_mix.values()), dtype=np.float64)
    weights = weights / weights.sum()

    rows: list[dict[str, object]] = []
    episode_id = 0
    while len(rows) < n_transitions:
        policy = str(rng.choice(names, p=weights))
        env.reset(_start_for_policy(policy, config, rng))
        episode_rows: list[dict[str, object]] = []
        for t in range(config.max_steps):
            state = env.state
            distance = env.distance_to_goal(state)
            action = _policy_action(policy, env, rng)
            next_state, reward, done, _ = env.step(action)
            episode_rows.append(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                    "episode_id": episode_id,
                    "timestep": t,
                    "behavior_policy_id": policy,
                    "distance_to_goal": distance,
                    "next_distance_to_goal": env.distance_to_goal(next_state),
                    "return_to_go": 0.0,
                    "label_for_eval_only": "common_zero",
                }
            )
            if done:
                break
        _label_episode(episode_rows, config)
        rows.extend(episode_rows)
        episode_id += 1

    rows = rows[:n_transitions]
    dataset = ReplayDataset(
        states=np.asarray([r["state"] for r in rows], dtype=np.float32),
        actions=np.asarray([r["action"] for r in rows], dtype=np.int64),
        rewards=np.asarray([r["reward"] for r in rows], dtype=np.float32),
        next_states=np.asarray([r["next_state"] for r in rows], dtype=np.float32),
        dones=np.asarray([r["done"] for r in rows], dtype=bool),
        episode_ids=np.asarray([r["episode_id"] for r in rows], dtype=np.int64),
        timesteps=np.asarray([r["timestep"] for r in rows], dtype=np.int64),
        behavior_policy_ids=np.asarray([r["behavior_policy_id"] for r in rows]).astype(str),
        distances_to_goal=np.asarray([r["distance_to_goal"] for r in rows], dtype=np.float32),
        next_distances_to_goal=np.asarray([r["next_distance_to_goal"] for r in rows], dtype=np.float32),
        returns_to_go=np.asarray([r["return_to_go"] for r in rows], dtype=np.float32),
        labels=np.asarray([r["label_for_eval_only"] for r in rows]).astype(str),
    )
    if include_optional_invalid:
        dataset = _make_optional_invalid(dataset, optional_invalid_ratio, seed)
    return dataset
