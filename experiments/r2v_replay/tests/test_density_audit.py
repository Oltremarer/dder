import numpy as np

from r2v_replay.density_audit import density_audit_tables
from r2v_replay.replay_dataset import ReplayDataset


def test_density_audit_reports_label_multiplicity_and_top10_stats():
    dataset = ReplayDataset(
        states=np.array([[0, 0], [0, 0], [5, 5], [9, 9]], dtype=np.float32),
        actions=np.array([0, 0, 1, 2], dtype=np.int64),
        rewards=np.array([0, 0, 0, 1], dtype=np.float32),
        next_states=np.array([[0, 1], [0, 1], [5, 4], [9, 8]], dtype=np.float32),
        dones=np.array([False, False, False, True]),
        episode_ids=np.array([0, 0, 1, 2], dtype=np.int64),
        timesteps=np.array([0, 1, 0, 0], dtype=np.int64),
        behavior_policy_ids=np.array(["random_wander", "random_wander", "rare_distractor_probe", "near_success"]),
        distances_to_goal=np.array([4, 4, 8, 1], dtype=np.float32),
        next_distances_to_goal=np.array([3, 3, 9, 0], dtype=np.float32),
        returns_to_go=np.array([0, 0, 0, 1], dtype=np.float32),
        labels=np.array(["common_zero", "common_zero", "rare_useless", "rare_valuable_positive"]),
    )
    knn_scores = np.array([0.1, 0.2, 0.9, 0.8], dtype=np.float32)

    tables = density_audit_tables(dataset, knn_scores=knn_scores, top_ratio=0.5)
    rare_row = tables["rare_useless_density_audit"].set_index("label").loc["rare_useless"]
    common_row = tables["transition_multiplicity_by_label"].set_index("label").loc["common_zero"]

    assert common_row["mean_transition_multiplicity"] == 2.0
    assert rare_row["count"] == 1
    assert rare_row["unique_state_action_next_count"] == 1
    assert rare_row["top10_fraction"] == 0.5
    assert rare_row["top10_enrichment"] == 2.0
