from collections import Counter

import numpy as np

from r2v_replay.sparse_grid import SparseGridConfig, SparseGridEnv, build_sparse_grid_replay


def test_sparse_grid_step_is_deterministic_and_blocks_walls():
    cfg = SparseGridConfig(
        width=7,
        height=7,
        start_states=((0, 0),),
        goal_state=(6, 6),
        walls=frozenset({(1, 0)}),
        decoy_states=frozenset({(0, 6)}),
        bottleneck_states=frozenset({(3, 3)}),
        max_steps=20,
    )
    env = SparseGridEnv(cfg, seed=0)
    state = env.reset(start_state=(0, 0))

    next_state, reward, done, info = env.step(3)

    assert state == (0, 0)
    assert next_state == (0, 0)
    assert reward == 0.0
    assert done is False
    assert info["hit_wall"] is True


def test_build_sparse_grid_replay_uses_eval_only_labels_and_policy_mixture():
    cfg = SparseGridConfig.default()
    dataset = build_sparse_grid_replay(
        cfg,
        n_transitions=2500,
        seed=4,
        policy_mix={
            "random_wander": 0.55,
            "noisy_goal": 0.20,
            "decoy": 0.15,
            "near_success": 0.10,
        },
    )

    assert len(dataset) >= 2500
    assert set(dataset.labels).issuperset(
        {
            "common_zero",
            "rare_valuable_positive",
            "rare_valuable_zero_precursor",
            "rare_useless",
        }
    )
    assert "label_for_eval_only" in dataset.to_frame().columns
    assert set(np.unique(dataset.rewards)).issubset({0.0, 1.0})

    counts = Counter(dataset.labels)
    assert counts["common_zero"] > counts["rare_valuable_positive"]
    assert counts["rare_valuable_zero_precursor"] > 0
    assert counts["rare_useless"] > 0
