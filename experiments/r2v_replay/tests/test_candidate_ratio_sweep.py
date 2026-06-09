import math

import numpy as np

from r2v_replay.metric_audit import candidate_ratio_sweep_rows


def test_candidate_ratio_sweep_excludes_optional_invalid_and_counts_by_budget():
    labels = np.array(
        [
            "rare_useless",
            "common_zero",
            "rare_valuable_zero_precursor",
            "rare_useless",
            "rare_valuable_positive",
            "common_zero",
            "rare_valuable_zero_precursor",
            "common_zero",
            "rare_useless",
            "common_zero",
            "optional_invalid",
        ]
    )
    scores = np.array([0.99, 0.98, 0.97, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 1.00])
    real_mask = labels != "optional_invalid"

    rows = candidate_ratio_sweep_rows(labels=labels, scores=scores, ratios=[0.2, 0.5], real_mask=real_mask)

    first = rows[0]
    assert first["ratio"] == 0.2
    assert first["real_transition_count"] == 10
    assert first["candidate_count"] == 2
    assert first["rare_useless_count"] == 1
    assert first["common_zero_count"] == 1
    assert first["rare_valuable_all_count"] == 0
    assert first["optional_invalid_count"] == 0
    assert first["rare_useless_fraction"] == 0.5
    assert math.isclose(first["rare_useless_enrichment"], 0.5 / 0.3)
    assert first["rare_valuable_recall"] == 0.0
    assert first["zero_precursor_recall"] == 0.0
    assert first["nonvaluable_fraction"] == 1.0

    second = rows[1]
    assert second["ratio"] == 0.5
    assert second["candidate_count"] == 5
    assert second["rare_useless_count"] == 2
    assert second["common_zero_count"] == 1
    assert second["rare_valuable_zero_precursor_count"] == 1
    assert second["rare_valuable_positive_count"] == 1
    assert second["rare_valuable_all_count"] == 2
    assert second["valuable_precision"] == 0.4
    assert math.isclose(second["rare_valuable_recall"], 2 / 3)
    assert second["zero_precursor_recall"] == 0.5
    assert second["rare_useless_fraction"] == 0.4
    assert math.isclose(second["rare_useless_enrichment"], 0.4 / 0.3)
    assert second["common_zero_fraction"] == 0.2
    assert second["nonvaluable_fraction"] == 0.6
