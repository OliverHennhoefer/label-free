import math

import pytest

from labelfree.metrics import asi_score, asoi_score


def test_asi_matches_paper_formula():
    X = [[0.0], [2.0], [4.0], [10.0], [12.0]]
    scores = [0.0, 0.1, 0.2, 4.0, 5.0]

    result = asi_score(X, scores, n_outliers=2)

    pooled_var = ((3 - 1) * 4.0 + (2 - 1) * 2.0) / (3 + 2 - 2)
    assert result == pytest.approx(9.0 / math.sqrt(pooled_var))


def test_asoi_matches_algorithm_components():
    X = [[0.0], [2.0], [4.0], [10.0], [12.0]]
    scores = [0.0, 0.1, 0.2, 4.0, 5.0]

    result = asoi_score(X, scores, n_outliers=2)

    separation_norm = 9.0 / 12.0
    hellinger = 1.0
    expected = 0.5314 * separation_norm + 0.4686 * hellinger
    assert result == pytest.approx(expected)


def test_asoi_score_polarity_controls_outlier_tail():
    X = [[0.0], [2.0], [4.0], [10.0], [12.0]]
    low_is_anomalous = [0.0, -0.1, -0.2, -4.0, -5.0]
    high_is_anomalous = [0.0, 0.1, 0.2, 4.0, 5.0]

    result = asoi_score(
        X,
        low_is_anomalous,
        n_outliers=2,
        score_polarity="higher_is_normal",
    )
    expected = asoi_score(X, high_is_anomalous, n_outliers=2)

    assert result == pytest.approx(expected)


def test_asoi_validates_weights():
    with pytest.raises(ValueError, match="sum to 1"):
        asoi_score([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0], n_outliers=1, beta=0.2)
