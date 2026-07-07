import numpy as np
import pytest

from labelfree.metrics import (
    expected_anomaly_gap_score,
    normalized_pseudo_discrepancy_score,
    relative_top_median_score,
)


def test_relative_top_median_uses_top_score_tail():
    scores = [1.0, 2.0, 3.0, 10.0, 20.0]

    result = relative_top_median_score(scores, top_fraction=0.4)

    assert result == pytest.approx((15.0 - 3.0) / (3.0 + 1e-9))


def test_expected_anomaly_gap_matches_discrete_formula():
    scores = np.array([0.0, 1.0, 2.0, 10.0, 12.0])

    result = expected_anomaly_gap_score(scores, top_fraction=0.4)

    ordered = np.array([12.0, 10.0, 2.0, 1.0, 0.0])
    terms = []
    for k in (1, 2):
        high = ordered[:k]
        low = ordered[k:]
        numerator = k * len(low) * (high.mean() - low.mean()) ** 2
        denominator = len(ordered) * (
            k * high.var() + len(low) * low.var() + 1e-9
        )
        terms.append(numerator / denominator)

    assert result == pytest.approx(np.mean(terms))


def test_normalized_pseudo_discrepancy_matches_formula():
    validation = [1.0, 2.0, 3.0]
    generated = [4.0, 5.0, 6.0]

    result = normalized_pseudo_discrepancy_score(validation, generated)

    assert result == pytest.approx(9.0 / (2 * (np.var(validation) + np.var(generated)) + 1e-9))


def test_autouad_metrics_apply_score_polarity():
    high_is_normal = [-1.0, -2.0, -3.0, -10.0, -20.0]
    high_is_anomalous = [1.0, 2.0, 3.0, 10.0, 20.0]

    assert relative_top_median_score(
        high_is_normal,
        top_fraction=0.4,
        score_polarity="higher_is_normal",
    ) == pytest.approx(relative_top_median_score(high_is_anomalous, top_fraction=0.4))


def test_normalized_pseudo_discrepancy_allows_different_sample_sizes():
    result = normalized_pseudo_discrepancy_score([1.0, 2.0], [3.0, 4.0, 5.0])

    assert result > 0
