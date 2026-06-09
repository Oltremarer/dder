import numpy as np

from r2v_replay.metric_audit import selection_audit_row


def test_selection_audit_uses_real_denominators_and_separates_candidate_from_selected():
    labels = np.array(
        [
            "common_zero",
            "rare_useless",
            "rare_valuable_zero_precursor",
            "rare_valuable_positive",
            "optional_invalid",
            "rare_useless",
        ]
    )
    rewards = np.array([0, 0, 0, 1, 1, 0], dtype=np.float32)
    episode_ids = np.array([0, 0, 1, 2, -1, 3], dtype=np.int64)
    real_mask = labels != "optional_invalid"
    candidate_indices = np.array([1, 2, 3, 5], dtype=np.int64)
    selected_indices = np.array([2, 3], dtype=np.int64)

    audit = selection_audit_row(
        labels=labels,
        rewards=rewards,
        selected_indices=selected_indices,
        candidate_indices=candidate_indices,
        real_mask=real_mask,
        episode_ids=episode_ids,
    )

    assert audit["real_transition_count"] == 5
    assert audit["candidate_count"] == 4
    assert audit["selected_count"] == 2
    assert audit["total_rare_valuable_positive"] == 1
    assert audit["total_rare_valuable_zero_precursor"] == 1
    assert audit["candidate_rare_useless_count"] == 2
    assert audit["selected_rare_valuable_positive_count"] == 1
    assert audit["selected_rare_valuable_zero_precursor_count"] == 1
    assert audit["valuable_precision_denominator"] == 2
    assert audit["zero_precursor_recall_denominator"] == 1
    assert audit["positive_reward_fraction_denominator"] == 2
    assert audit["valuable_precision"] == 1.0
    assert audit["zero_precursor_recall"] == 1.0
    assert audit["positive_reward_fraction"] == 0.5
    assert audit["rare_useless_fraction"] == 0.0
    assert audit["candidate_rare_useless_fraction"] == 0.5
    assert audit["candidate_rare_useless_enrichment_vs_uniform"] == 1.25
    assert audit["selected_episode_diversity"] == 2
    assert audit["optional_invalid_excluded"] is True
    assert audit["all_assertion_checks_pass"] is True


def test_selection_audit_flags_optional_invalid_selection():
    labels = np.array(["common_zero", "rare_valuable_zero_precursor", "optional_invalid"])
    rewards = np.array([0, 0, 1], dtype=np.float32)
    real_mask = labels != "optional_invalid"

    audit = selection_audit_row(
        labels=labels,
        rewards=rewards,
        selected_indices=np.array([1, 2], dtype=np.int64),
        candidate_indices=np.array([1, 2], dtype=np.int64),
        real_mask=real_mask,
    )

    assert audit["selected_optional_invalid_count"] == 1
    assert audit["optional_invalid_excluded"] is False
    assert audit["all_assertion_checks_pass"] is False
