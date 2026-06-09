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
    main_corridor_states: frozenset[State] = frozenset()
    distractor_pockets: tuple[frozenset[State], ...] = ()
    deep_distractor_states: frozenset[State] = frozenset()
    max_steps: int = 80
    precursor_window: int = 10

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
            precursor_window=10,
        )

    @classmethod
    def distractor_v2(
        cls,
        width: int = 21,
        height: int = 21,
        num_distractor_pockets: int = 6,
        max_steps: int = 90,
        precursor_window: int = 10,
    ) -> "SparseGridConfig":
        if width < 15 or height < 15:
            raise ValueError("distractor_v2 requires width and height >= 15")
        goal_state = (width - 2, height - 2)
        main_corridor = _main_corridor(width, height, goal_state)
        pockets, pocket_walls = _make_distractor_pockets(width, height, num_distractor_pockets)
        deep_states = frozenset().union(*pockets) if pockets else frozenset()
        return cls(
            width=width,
            height=height,
            start_states=((1, 1), (1, 2), (2, 1), (2, 2)),
            goal_state=goal_state,
            walls=frozenset(pocket_walls),
            decoy_states=frozenset(),
            bottleneck_states=frozenset({(width // 2, height // 2), (width // 2 + 1, height // 2)}),
            main_corridor_states=frozenset(main_corridor),
            distractor_pockets=tuple(frozenset(pocket) for pocket in pockets),
            deep_distractor_states=deep_states,
            max_steps=max_steps,
            precursor_window=precursor_window,
        )


def sparse_grid_config_from_dict(raw: Mapping[str, object] | None) -> SparseGridConfig:
    raw = dict(raw or {})
    layout = str(raw.get("layout", "default"))
    if layout in {"distractor_v2", "hybrid_v3"}:
        return SparseGridConfig.distractor_v2(
            width=int(raw.get("width", 21)),
            height=int(raw.get("height", 21)),
            num_distractor_pockets=int(raw.get("num_distractor_pockets", 6)),
            max_steps=int(raw.get("max_steps", 90)),
            precursor_window=int(raw.get("precursor_window", 10)),
        )
    cfg = SparseGridConfig.default()
    return SparseGridConfig(
        width=int(raw.get("width", cfg.width)),
        height=int(raw.get("height", cfg.height)),
        start_states=cfg.start_states,
        goal_state=cfg.goal_state,
        walls=cfg.walls,
        decoy_states=cfg.decoy_states,
        bottleneck_states=cfg.bottleneck_states,
        main_corridor_states=cfg.main_corridor_states,
        distractor_pockets=cfg.distractor_pockets,
        deep_distractor_states=cfg.deep_distractor_states,
        max_steps=int(raw.get("max_steps", cfg.max_steps)),
        precursor_window=int(raw.get("precursor_window", cfg.precursor_window)),
    )


def _main_corridor(width: int, height: int, goal_state: State) -> set[State]:
    corridor: set[State] = set()
    y_mid = max(2, height // 2)
    for x in range(1, goal_state[0] + 1):
        corridor.add((x, 1))
    for y in range(1, y_mid + 1):
        corridor.add((goal_state[0], y))
    for x in range(width // 2, goal_state[0] + 1):
        corridor.add((x, y_mid))
    for y in range(y_mid, goal_state[1] + 1):
        corridor.add((width // 2, y))
    for x in range(width // 2, goal_state[0] + 1):
        corridor.add((x, goal_state[1]))
    return corridor


def _make_distractor_pockets(width: int, height: int, count: int) -> tuple[list[set[State]], set[State]]:
    centers: list[State] = [
        (3, height - 5),
        (6, height - 8),
        (width - 7, 4),
        (width - 5, height // 2),
        (width // 2 - 3, height - 4),
        (width // 2 + 3, 5),
        (4, height // 2),
        (width - 8, height - 6),
    ]
    for x in range(3, width - 4, 5):
        for y in range(4, height - 4, 5):
            center = (x, y)
            if center not in centers and abs(x - (width - 2)) + abs(y - (height - 2)) > 8:
                centers.append(center)
    pockets: list[set[State]] = []
    walls: set[State] = set()
    used: set[State] = set()
    for cx, cy in centers[:count]:
        cx = min(max(2, cx), width - 4)
        cy = min(max(2, cy), height - 4)
        pocket = {(cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1)}
        entry = (cx - 1, cy)
        if pocket & used or entry in used:
            continue
        used.update(pocket)
        used.add(entry)
        pockets.append(pocket)
        for x in range(cx - 1, cx + 3):
            for y in range(cy - 1, cy + 3):
                state = (x, y)
                if state in pocket or state == entry:
                    continue
                if 0 <= x < width and 0 <= y < height:
                    walls.add(state)
    return pockets, walls


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


def _policy_action(
    policy_name: str,
    env: SparseGridEnv,
    rng: np.random.Generator,
    episode_context: dict[str, object] | None = None,
) -> Action:
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
    if policy_name == "rare_distractor_probe":
        if not env.config.deep_distractor_states:
            return int(rng.integers(0, 4))
        context = episode_context if episode_context is not None else {}
        target = context.get("distractor_target")
        entry = context.get("distractor_entry")
        if target is None:
            target = tuple(sorted(env.config.deep_distractor_states))[0]
        target = tuple(target)  # type: ignore[arg-type]
        if state in env.config.deep_distractor_states:
            context["entered_distractor"] = True
            context["inside_steps_left"] = max(0, int(context.get("inside_steps_left", 1)) - 1)
            return int(rng.integers(0, 4))
        if entry is not None and state != tuple(entry):  # type: ignore[arg-type]
            target = tuple(entry)  # type: ignore[arg-type]
        if rng.random() < 0.88:
            return _greedy_action_towards(state, target, env, rng)
        return int(rng.integers(0, 4))
    raise ValueError(f"unknown policy: {policy_name}")


def _start_for_policy(policy_name: str, cfg: SparseGridConfig, rng: np.random.Generator) -> State:
    if policy_name == "near_success":
        candidates = ((8, 7), (8, 8), (9, 8), (10, 9))
        valid = [s for s in candidates if 0 <= s[0] < cfg.width and 0 <= s[1] < cfg.height and s not in cfg.walls]
        return valid[int(rng.integers(0, len(valid)))]
    idx = int(rng.integers(0, len(cfg.start_states)))
    return cfg.start_states[idx]


def _episode_context_for_policy(
    policy_name: str, cfg: SparseGridConfig, rng: np.random.Generator
) -> dict[str, object]:
    if policy_name != "rare_distractor_probe" or not cfg.distractor_pockets:
        return {}
    pocket = cfg.distractor_pockets[int(rng.integers(0, len(cfg.distractor_pockets)))]
    target = sorted(pocket)[int(rng.integers(0, len(pocket)))]
    min_x = min(x for x, _ in pocket)
    min_y = min(y for _, y in pocket)
    return {
        "distractor_target": target,
        "distractor_entry": (min_x - 1, min_y),
        "inside_steps_left": int(rng.integers(12, 25)),
        "entered_distractor": False,
    }


def _should_end_episode_for_policy(policy_name: str, episode_context: dict[str, object]) -> bool:
    return (
        policy_name == "rare_distractor_probe"
        and bool(episode_context.get("entered_distractor", False))
        and int(episode_context.get("inside_steps_left", 0)) <= 0
    )


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
        elif _is_rare_useless_distractor_transition(row, state, next_state, cfg, episode_return, success_index, i):
            label = "rare_useless"
        elif not cfg.deep_distractor_states and _is_deep_decoy_transition(state, next_state, cfg):
            label = "rare_useless"
        else:
            label = "common_zero"
        row["label_for_eval_only"] = label
        row["return_to_go"] = episode_return


def _is_deep_decoy_transition(state: State, next_state: State, cfg: SparseGridConfig) -> bool:
    if not cfg.decoy_states:
        return False
    deep_decoy = max(cfg.decoy_states, key=lambda s: (s[0] + s[1], s[0], s[1]))
    return state == deep_decoy or next_state == deep_decoy


def _is_rare_useless_distractor_transition(
    row: dict[str, object],
    state: State,
    next_state: State,
    cfg: SparseGridConfig,
    episode_return: float,
    success_index: int | None,
    row_index: int,
) -> bool:
    if not cfg.deep_distractor_states:
        return False
    if str(row["behavior_policy_id"]) != "rare_distractor_probe":
        return False
    if float(row["reward"]) != 0.0 or episode_return != 0.0 or success_index is not None:
        return False
    progress = float(row["distance_to_goal"]) - float(row["next_distance_to_goal"])
    if progress > 0.0:
        return False
    if state in cfg.main_corridor_states or next_state in cfg.main_corridor_states:
        return False
    if state not in cfg.deep_distractor_states and next_state not in cfg.deep_distractor_states:
        return False
    if success_index is not None and 0 < (success_index - row_index) <= cfg.precursor_window:
        return False
    return True


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
            "random_wander": 0.60,
            "noisy_goal": 0.23,
            "decoy": 0.02,
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
        episode_context = _episode_context_for_policy(policy, config, rng)
        env.reset(_start_for_policy(policy, config, rng))
        episode_rows: list[dict[str, object]] = []
        for t in range(config.max_steps):
            state = env.state
            distance = env.distance_to_goal(state)
            action = _policy_action(policy, env, rng, episode_context)
            next_state, reward, done, _ = env.step(action)
            policy_done = _should_end_episode_for_policy(policy, episode_context)
            episode_rows.append(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done or policy_done,
                    "episode_id": episode_id,
                    "timestep": t,
                    "behavior_policy_id": policy,
                    "distance_to_goal": distance,
                    "next_distance_to_goal": env.distance_to_goal(next_state),
                    "return_to_go": 0.0,
                    "label_for_eval_only": "common_zero",
                }
            )
            if done or policy_done:
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
