from pathlib import Path

import numpy as np

from r2v_replay.replay_dataset import ReplayDataset


def test_replay_dataset_round_trips_npz_and_keeps_labels_eval_only(tmp_path: Path):
    dataset = ReplayDataset(
        states=np.array([[0, 0], [0, 1]], dtype=np.float32),
        actions=np.array([1, 3], dtype=np.int64),
        rewards=np.array([0.0, 1.0], dtype=np.float32),
        next_states=np.array([[0, 1], [1, 1]], dtype=np.float32),
        dones=np.array([False, True]),
        episode_ids=np.array([0, 0], dtype=np.int64),
        timesteps=np.array([0, 1], dtype=np.int64),
        behavior_policy_ids=np.array(["random_wander", "noisy_goal"]),
        distances_to_goal=np.array([4.0, 1.0], dtype=np.float32),
        next_distances_to_goal=np.array([3.0, 0.0], dtype=np.float32),
        returns_to_go=np.array([1.0, 1.0], dtype=np.float32),
        labels=np.array(["rare_valuable_zero_precursor", "rare_valuable_positive"]),
    )

    path = tmp_path / "replay.npz"
    dataset.save_npz(path)
    loaded = ReplayDataset.load_npz(path)

    assert len(loaded) == 2
    assert loaded.labels.tolist() == dataset.labels.tolist()
    assert "label_for_eval_only" in loaded.to_frame().columns
    assert "label_for_eval_only" not in loaded.feature_matrix("obs_action_next").columns
    assert "reward" in loaded.feature_matrix("full_transition").columns
    assert "reward" not in loaded.feature_matrix("obs_action_next").columns
