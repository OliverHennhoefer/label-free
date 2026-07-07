import pytest

from labelfree.metrics import ireos_score


def test_ireos_rewards_separable_high_scored_candidates():
    X = [[0.0], [0.1], [0.2], [10.0]]

    isolated = ireos_score(X, [0.0, 0.0, 0.0, 10.0], n_outliers=1, gammas=[0.1, 1.0])
    clustered = ireos_score(X, [10.0, 0.0, 0.0, 0.0], n_outliers=1, gammas=[0.1, 1.0])

    assert isolated > clustered


def test_ireos_applies_score_polarity():
    X = [[0.0], [0.1], [0.2], [10.0]]

    result = ireos_score(
        X,
        [0.0, 0.0, 0.0, -10.0],
        n_outliers=1,
        gammas=[0.1, 1.0],
        score_polarity="higher_is_normal",
    )
    expected = ireos_score(X, [0.0, 0.0, 0.0, 10.0], n_outliers=1, gammas=[0.1, 1.0])

    assert result == pytest.approx(expected)


def test_ireos_validates_gamma_grid():
    with pytest.raises(ValueError, match="gammas"):
        ireos_score([[0.0], [1.0]], [0.0, 1.0], n_outliers=1, gammas=[0.0])
