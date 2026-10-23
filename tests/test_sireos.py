import numpy as np
import pytest
from sklearn.metrics.pairwise import pairwise_distances

from labelfree import unify_scores
from labelfree.metrics import sireos_score


def _author_sireos(data, solution):
    """Loop transcribed from the authors' sireos.py (homarques/SIREOS)."""
    n_samples = data.shape[0]
    D = pairwise_distances(data)
    t = np.quantile(D[np.nonzero(D)], 0.01)
    X = solution[:, None]
    X = X / X.sum()
    score = 0
    for j in range(n_samples):
        score += (
            np.mean(
                np.exp(
                    -(
                        np.linalg.norm(
                            data[None, j, :] - np.delete(data[None, :, :], j, axis=1),
                            axis=-1,
                        )
                        ** 2
                    )
                    / (2 * t * t)
                )
            )
            * X[j]
        )
    return float(score[0])


def test_sireos_matches_author_implementation():
    rng = np.random.default_rng(5)
    data = np.vstack([rng.normal(size=(40, 3)), rng.normal(6, 1, size=(3, 3))])
    solution = rng.uniform(size=43)

    assert sireos_score(data, solution) == pytest.approx(_author_sireos(data, solution))


def test_sireos_rewards_weight_on_isolated_points():
    X = [[0.0], [0.1], [0.2], [10.0]]

    isolated_weighted = sireos_score(X, [0.0, 0.0, 0.0, 10.0], kernel_width=1.0)
    cluster_weighted = sireos_score(X, [10.0, 0.0, 0.0, 0.0], kernel_width=1.0)

    assert isolated_weighted < cluster_weighted


def test_sireos_is_invariant_to_score_scale_but_not_offset():
    X = np.random.default_rng(0).normal(size=(20, 2))
    scores = np.random.default_rng(1).uniform(size=20)

    assert sireos_score(X, 3 * scores) == pytest.approx(sireos_score(X, scores))
    assert sireos_score(X, scores + 1) != pytest.approx(sireos_score(X, scores))


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


def test_sireos_requires_non_negative_scores():
    X = [[0.0], [1.0], [2.0]]
    with pytest.raises(ValueError, match="unify_scores"):
        sireos_score(X, [-1.0, 0.0, 1.0])

    signed = [-1.0, 0.0, 1.0]
    assert sireos_score(X, unify_scores(signed, method="linear")) >= 0


def test_sireos_validates_kernel_width():
    with pytest.raises(ValueError, match="kernel_width"):
        sireos_score([[0.0], [1.0]], [0.0, 1.0], kernel_width=0.0)
