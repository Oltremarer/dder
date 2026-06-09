import numpy as np

from r2v_replay.encoders import TransitionEncoder
from r2v_replay.rarity_scorers import KNNRarityScorer
from r2v_replay.risk_scorers import OODRiskScorer
from r2v_replay.utility_scorers import TaskUtilityScorer
from r2v_replay.replay_dataset import ReplayDataset


def _small_dataset() -> ReplayDataset:
    states = np.array([[0, 0], [0, 1], [1, 1], [8, 8], [8, 7]], dtype=np.float32)
    next_states = np.array([[0, 1], [1, 1], [2, 1], [8, 7], [8, 6]], dtype=np.float32)
    return ReplayDataset(
        states=states,
        actions=np.array([1, 3, 3, 0, 0], dtype=np.int64),
        rewards=np.array([0, 0, 1, 0, 0], dtype=np.float32),
        next_states=next_states,
        dones=np.array([False, False, True, False, False]),
        episode_ids=np.array([0, 0, 0, 1, 1], dtype=np.int64),
        timesteps=np.array([0, 1, 2, 0, 1], dtype=np.int64),
        behavior_policy_ids=np.array(["near_success", "near_success", "near_success", "decoy", "decoy"]),
        distances_to_goal=np.array([4, 3, 1, 9, 10], dtype=np.float32),
        next_distances_to_goal=np.array([3, 1, 0, 10, 11], dtype=np.float32),
        returns_to_go=np.array([1, 1, 1, 0, 0], dtype=np.float32),
        labels=np.array(
            [
                "rare_valuable_zero_precursor",
                "rare_valuable_zero_precursor",
                "rare_valuable_positive",
                "rare_useless",
                "rare_useless",
            ]
        ),
    )


def test_encoder_and_scorers_return_one_score_per_transition_without_labels():
    dataset = _small_dataset()
    encoder = TransitionEncoder(input_mode="obs_action_next")
    z = encoder.fit_transform(dataset)

    assert z.shape[0] == len(dataset)
    assert "label_for_eval_only" not in encoder.feature_columns_
    assert "reward" not in encoder.feature_columns_

    rarity = KNNRarityScorer(k=2).fit(z).score(z)
    risk = OODRiskScorer(k=2).fit(z).score(z)
    utility = TaskUtilityScorer(w_reward=0.2, w_rtg=0.4, w_progress=0.4).score(dataset)

    assert rarity.shape == (len(dataset),)
    assert risk.shape == (len(dataset),)
    assert utility.total.shape == (len(dataset),)
    assert utility.components["progress"][0] > 0.0
    assert utility.components["reward"][2] > utility.components["reward"][0]
