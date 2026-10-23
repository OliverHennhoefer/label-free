"""Laplacian Score of an anomaly-score vector (He, Cai, and Niyogi, 2005)."""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors

from labelfree._validation import (
    as_2d_finite,
    check_integer,
    check_positive,
    check_same_length,
    orient_scores,
)


def laplacian_score(
    X,
    scores,
    *,
    n_neighbors: int = 5,
    kernel_width: float | None = None,
    score_polarity: str = "higher_is_anomalous",
) -> float:
    """Graph smoothness of scores over the data manifold. Lower is better.

    He et al. define the Laplacian Score of a feature ``f`` as
    ``L = f~' L f~ / f~' D f~`` with ``f~ = f - (f' D 1 / 1' D 1) 1``, where
    ``S`` connects ``x_i`` and ``x_j`` when either is among the other's
    ``n_neighbors`` nearest neighbors, ``D = diag(S 1)``, and ``L = D - S``.
    Marques et al. (SISAP 2022) use it as an internal outlier evaluation
    baseline by treating the score vector as the feature.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix. Scale features first.
    scores : array-like of shape (n_samples,)
        Detector scores.
    n_neighbors : int
        Neighborhood size of the k-nearest-neighbor graph.
    kernel_width : float, optional
        Heat-kernel width ``t`` for edge weights ``exp(-||x_i - x_j||^2 / t)``.
        The default uses the paper's simple-minded 0/1 weights.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores. The score is invariant to negation, so
        polarity only matters for consistency.

    Returns
    -------
    float
        Laplacian Score. ``inf`` for constant scores.
    """
    X = as_2d_finite(X, name="X")
    scores = orient_scores(scores, score_polarity=score_polarity)
    check_same_length(X, scores)
    check_integer(n_neighbors, name="n_neighbors")
    if not 1 <= n_neighbors < X.shape[0]:
        raise ValueError("n_neighbors must be between 1 and len(X) - 1")
    if kernel_width is not None:
        kernel_width = check_positive(kernel_width, name="kernel_width")

    weights = _neighbor_weights(X, n_neighbors, kernel_width)
    degree = np.asarray(weights.sum(axis=1)).ravel()
    centered = scores - float(scores @ degree) / float(degree.sum())
    denominator = float(centered @ (degree * centered))
    if denominator == 0:
        return math.inf
    numerator = denominator - float(centered @ (weights @ centered))
    return max(numerator, 0.0) / denominator


def _neighbor_weights(X: np.ndarray, n_neighbors: int, kernel_width: float | None):
    """Symmetric k-nearest-neighbor weight matrix without self-loops."""
    distances, indices = NearestNeighbors(n_neighbors=n_neighbors).fit(X).kneighbors()
    rows = np.repeat(np.arange(X.shape[0]), n_neighbors)
    if kernel_width is None:
        values = np.ones(rows.size)
    else:
        values = np.exp(-(distances.ravel() ** 2) / kernel_width)
    graph = coo_matrix((values, (rows, indices.ravel())), shape=(X.shape[0],) * 2)
    graph = graph.tocsr()
    return graph.maximum(graph.T)
