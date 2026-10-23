import math

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from labelfree.metrics import laplacian_score


def _he_reference(X, f, k, t=None):
    """Dense Laplacian Score from He, Cai, and Niyogi (2005), Algorithm steps 1-4."""
    distances = cdist(X, X, metric="sqeuclidean")
    order = np.argsort(distances, axis=1)
    S = np.zeros_like(distances)
    for i in range(len(X)):
        for j in order[i, 1 : k + 1]:
            weight = 1.0 if t is None else np.exp(-distances[i, j] / t)
            S[i, j] = S[j, i] = weight
    D = np.diag(S.sum(axis=1))
    L = D - S
    ones = np.ones(len(X))
    f_tilde = f - (f @ D @ ones) / (ones @ D @ ones) * ones
    return (f_tilde @ L @ f_tilde) / (f_tilde @ D @ f_tilde)


@pytest.mark.parametrize("kernel_width", [None, 0.5])
def test_laplacian_score_matches_dense_definition(kernel_width):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 3))
    scores = rng.normal(size=30)

    result = laplacian_score(X, scores, n_neighbors=4, kernel_width=kernel_width)

    assert result == pytest.approx(_he_reference(X, scores, 4, kernel_width))


def test_laplacian_score_rewards_graph_smooth_scores():
    X = [[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]]

    smooth = laplacian_score(X, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], n_neighbors=1)
    rough = laplacian_score(X, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], n_neighbors=1)

    assert smooth == 0.0
    assert rough > smooth


def test_laplacian_score_constant_scores_are_uninformative():
    result = laplacian_score([[0.0], [1.0], [2.0]], [1.0, 1.0, 1.0], n_neighbors=1)

    assert math.isinf(result)


def test_laplacian_score_is_sign_invariant_after_polarity():
    X = [[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]]

    result = laplacian_score(
        X,
        [0.0, 0.0, 0.0, -1.0, -1.0, -1.0],
        n_neighbors=1,
        score_polarity="higher_is_normal",
    )
    expected = laplacian_score(X, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], n_neighbors=1)

    assert result == pytest.approx(expected)


def test_laplacian_score_handles_duplicate_rows():
    X = [[0.0], [0.0], [5.0], [5.0]]

    assert laplacian_score(X, [0.0, 0.0, 1.0, 1.0], n_neighbors=1) == 0.0


def test_laplacian_score_validates_inputs():
    with pytest.raises(ValueError, match="n_neighbors"):
        laplacian_score([[0.0], [1.0]], [0.0, 1.0], n_neighbors=2)
    with pytest.raises(ValueError, match="kernel_width"):
        laplacian_score(
            [[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0], n_neighbors=1, kernel_width=-1.0
        )
