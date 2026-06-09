from collections import Counter

import numpy as np

from r2v_replay.sparse_grid import SparseGridConfig, SparseGridEnv, build_sparse_grid_replay, sparse_grid_config_from_dict


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


def test_distractor_v2_generates_real_rare_useless_from_probe_policy():
    cfg = SparseGridConfig.distractor_v2(width=21, height=21, num_distractor_pockets=4)
    dataset = build_sparse_grid_replay(
        cfg,
        n_transitions=1200,
        seed=11,
        policy_mix={
            "random_wander": 0.15,
            "noisy_goal": 0.10,
            "near_success": 0.05,
            "rare_distractor_probe": 0.70,
        },
    )
    frame = dataset.to_frame()
    rare = frame[frame["label_for_eval_only"] == "rare_useless"]

    assert len(rare) > 0
    assert set(rare["behavior_policy_id"]) == {"rare_distractor_probe"}
    assert (rare["reward"] == 0.0).all()
    assert (rare["return_to_go"] == 0.0).all()
    assert ((rare["distance_to_goal"] - rare["next_distance_to_goal"]) <= 0.0).all()


def test_two_branch_v4_has_reachable_valuable_and_useless_branches():
    cfg = sparse_grid_config_from_dict({"layout": "two_branch_v4"})
    env = SparseGridEnv(cfg, seed=0)
    start = env.reset()

    assert start == cfg.start_states[0]
    assert cfg.valuable_branch_states
    assert cfg.useless_branch_states
    assert cfg.valuable_branch_states.isdisjoint(cfg.useless_branch_states)

    valuable_next, _, _, valuable_info = env.step(3)
    env.reset()
    useless_next, _, _, useless_info = env.step(2)

    assert valuable_next in cfg.valuable_branch_states
    assert useless_next in cfg.useless_branch_states
    assert valuable_info["hit_wall"] is False
    assert useless_info["hit_wall"] is False


def test_two_branch_v4_generates_real_rare_value_and_rare_useless():
    cfg = sparse_grid_config_from_dict({"layout": "two_branch_v4"})
    dataset = build_sparse_grid_replay(
        cfg,
        n_transitions=1800,
        seed=21,
        policy_mix={
            "random_wander": 0.20,
            "near_success": 0.05,
            "rare_valuable_branch_probe": 0.35,
            "rare_distractor_probe": 0.40,
        },
    )
    frame = dataset.to_frame()
    rare_useless = frame[frame["label_for_eval_only"] == "rare_useless"]
    zero_precursors = frame[frame["label_for_eval_only"] == "rare_valuable_zero_precursor"]
    positives = frame[frame["label_for_eval_only"] == "rare_valuable_positive"]

    assert len(rare_useless) > 0
    assert len(zero_precursors) > 0
    assert len(positives) > 0
    assert set(rare_useless["behavior_policy_id"]) == {"rare_distractor_probe"}
    assert (rare_useless["reward"] == 0.0).all()
    assert (rare_useless["return_to_go"] == 0.0).all()
    assert ((rare_useless["distance_to_goal"] - rare_useless["next_distance_to_goal"]) <= 0.0).all()
    assert (zero_precursors["reward"] == 0.0).all()
    assert (zero_precursors["return_to_go"] > 0.0).all()
