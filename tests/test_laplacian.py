import math

import pytest

from labelfree.metrics import laplacian_score
from labelfree.metrics.laplacian import _neighbor_edges


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
    expected = laplacian_score(
        X,
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        n_neighbors=1,
    )

    assert result == pytest.approx(expected)


def test_laplacian_score_validates_neighbor_count():
    with pytest.raises(ValueError, match="n_neighbors"):
        laplacian_score([[0.0], [1.0]], [0.0, 1.0], n_neighbors=2)


def test_neighbor_graph_uses_exact_requested_count():
    assert _neighbor_edges([[0.0], [10.0], [11.0]], 1) == {(0, 1), (1, 2)}
