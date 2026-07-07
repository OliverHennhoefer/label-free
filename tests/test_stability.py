import pytest

from labelfree.metrics import ranking_stability_score, top_k_stability_score


def test_ranking_stability_is_one_for_identical_rankings():
    score_matrix = [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
    ]

    result = ranking_stability_score(score_matrix, contamination=0.2)

    assert result == pytest.approx(1.0)


def test_ranking_stability_drops_for_variable_rankings():
    stable = [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
    ]
    variable = [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0, 0.0],
        [0.0, 2.0, 4.0, 1.0, 3.0],
    ]

    assert ranking_stability_score(variable, contamination=0.2) < ranking_stability_score(
        stable,
        contamination=0.2,
    )


def test_top_k_stability_is_average_jaccard_overlap():
    score_matrix = [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [0.1, 1.1, 2.1, 3.1, 4.1],
        [0.0, 1.0, 3.0, 2.0, 4.0],
    ]

    result = top_k_stability_score(score_matrix, top_k=2)

    assert result == pytest.approx((1.0 + 1 / 3 + 1 / 3) / 3)


def test_stability_metrics_apply_score_polarity():
    high_is_anomalous = [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [0.1, 1.1, 2.1, 3.1, 4.1],
    ]
    high_is_normal = [[-value for value in row] for row in high_is_anomalous]

    result = top_k_stability_score(
        high_is_normal,
        top_k=2,
        score_polarity="higher_is_normal",
    )
    expected = top_k_stability_score(high_is_anomalous, top_k=2)

    assert result == pytest.approx(expected)


def test_ranking_stability_validates_contamination():
    with pytest.raises(ValueError, match="contamination"):
        ranking_stability_score([[0.0, 1.0], [1.0, 0.0]], contamination=0.5)
