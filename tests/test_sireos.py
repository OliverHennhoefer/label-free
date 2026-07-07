import pytest

from labelfree.metrics import sireos_score


def test_sireos_rewards_weight_on_isolated_points_with_lower_score():
    X = [[0.0], [0.1], [0.2], [10.0]]

    isolated_weighted = sireos_score(X, [0.0, 0.0, 0.0, 10.0], kernel_width=1.0)
    cluster_weighted = sireos_score(X, [10.0, 0.0, 0.0, 0.0], kernel_width=1.0)

    assert isolated_weighted < cluster_weighted


def test_sireos_applies_score_polarity():
    X = [[0.0], [0.1], [0.2], [10.0]]

    result = sireos_score(
        X,
        [0.0, 0.0, 0.0, -10.0],
        kernel_width=1.0,
        score_polarity="higher_is_normal",
    )
    expected = sireos_score(X, [0.0, 0.0, 0.0, 10.0], kernel_width=1.0)

    assert result == pytest.approx(expected)


def test_sireos_validates_kernel_width():
    with pytest.raises(ValueError, match="kernel_width"):
        sireos_score([[0.0], [1.0]], [0.0, 1.0], kernel_width=0.0)
