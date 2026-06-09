import numpy as np

from r2v_replay.geometry_audit import candidate_composition_rows, multiplicity_summary_rows


def test_multiplicity_summary_reports_duplicate_transition_pressure():
    labels = np.array(
        [
            "rare_useless",
            "rare_useless",
            "rare_useless",
            "rare_valuable_zero_precursor",
            "rare_valuable_positive",
            "common_zero",
        ]
    )
    states = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
    actions = np.array([1, 1, 1, 2, 2, 3], dtype=np.int64)
    next_states = np.array([[0, 1], [0, 1], [0, 1], [1, 1], [2, 1], [3, 1]], dtype=np.float32)
    rows = multiplicity_summary_rows(labels, states, actions, next_states)
    by_label = {row["label"]: row for row in rows}

    assert by_label["rare_useless"]["count"] == 3
    assert by_label["rare_useless"]["unique_state_action_next_count"] == 1
    assert by_label["rare_useless"]["mean_multiplicity"] == 3.0
    assert by_label["rare_valuable_zero_precursor"]["mean_multiplicity"] == 1.0


def test_candidate_composition_rows_counts_rare_valuable_and_useless():
    labels = np.array(
        [
            "rare_useless",
            "rare_valuable_zero_precursor",
            "common_zero",
            "rare_valuable_positive",
        ]
    )
    scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)

    rows = candidate_composition_rows(labels, scores, ratios=[0.5], variant="raw", k_value=5)

    assert len(rows) == 1
    row = rows[0]
    assert row["variant"] == "raw"
    assert row["candidate_count"] == 2
    assert row["rare_useless_count"] == 1
    assert row["rare_valuable_all_count"] == 1
    assert row["rare_useless_fraction"] == 0.5
    assert row["rare_valuable_recall"] == 0.5


def test_candidate_composition_rows_adds_h2_budget_flags():
    labels = np.array(
        ["rare_useless"] * 10
        + ["rare_valuable_zero_precursor"] * 20
        + ["rare_valuable_positive"] * 10
        + ["common_zero"] * 10
        + ["common_zero"] * 950
    )
    scores = np.concatenate([np.linspace(1.0, 0.5, 50), np.linspace(0.4, 0.0, 950)]).astype(np.float32)

    rows = candidate_composition_rows(labels, scores, ratios=[0.05, 0.10], variant="raw", k_value=50)
    by_ratio = {row["ratio"]: row for row in rows}

    assert by_ratio[0.05]["h2_count_pass"]
    assert by_ratio[0.05]["h2_fraction_pass"]
    assert by_ratio[0.05]["h2_enrichment_pass"]
    assert by_ratio[0.05]["h2_valuable_supply_pass"]
    assert by_ratio[0.05]["h2_budget_rule_ratio"]
    assert by_ratio[0.05]["h2_budget_pass"]
    assert not by_ratio[0.10]["h2_budget_rule_ratio"]
    assert not by_ratio[0.10]["h2_budget_pass"]
