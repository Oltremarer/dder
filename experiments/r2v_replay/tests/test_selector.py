import numpy as np

from r2v_replay.selector import R2VSelector, SelectionScores, composition_by_label


def test_r2v_selector_keeps_rarity_as_proposal_and_ranks_by_utility_risk():
    labels = np.array(
        [
            "common_zero",
            "rare_valuable_zero_precursor",
            "rare_useless",
            "rare_valuable_positive",
            "common_zero",
            "rare_useless",
        ]
    )
    scores = SelectionScores(
        indices=np.arange(6),
        rarity=np.array([0.1, 0.9, 0.95, 0.85, 0.2, 0.8]),
        utility=np.array([0.1, 0.8, 0.2, 0.9, 0.2, 0.1]),
        ood_risk=np.array([0.1, 0.1, 0.7, 0.0, 0.1, 0.8]),
        dynamics_error=np.array([0.0, 0.1, 0.8, 0.0, 0.0, 0.9]),
        reward_error=np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.7]),
    )

    selector = R2VSelector(rare_topk_ratio=0.66, selected_budget_ratio=0.5)
    selected = selector.select(scores)

    assert selected.indices.tolist() == [3, 1]
    assert composition_by_label(labels, selected.indices)["rare_valuable_positive"] == 1
    assert composition_by_label(labels, selected.indices)["rare_valuable_zero_precursor"] == 1
