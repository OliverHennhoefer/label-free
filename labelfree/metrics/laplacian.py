"""Laplacian Score baseline for anomaly-score smoothness."""

from __future__ import annotations

import math

import numpy as np
from sklearn.neighbors import NearestNeighbors

from labelfree.utils.validation import as_2d_finite, orient_scores


def laplacian_score(
    X,
    scores,
    *,
    n_neighbors: int = 5,
    score_polarity: str = "higher_is_anomalous",
) -> float:
    """Score smoothness on a nearest-neighbor graph over rows of X. Lower is better."""
    X = as_2d_finite(X, name="X")
    scores = orient_scores(scores, score_polarity=score_polarity)
    if X.shape[0] != scores.size:
        raise ValueError("X and scores must have the same number of samples")
    if not 1 <= n_neighbors < X.shape[0]:
        raise ValueError("n_neighbors must be between 1 and len(X) - 1")

    edges = _neighbor_edges(X, n_neighbors)
    degree = np.zeros(X.shape[0], dtype=float)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    degree_sum = float(degree.sum())
    if degree_sum == 0:
        return math.inf

    centered = scores - float(np.dot(scores, degree) / degree_sum)
    numerator = sum(float((centered[i] - centered[j]) ** 2) for i, j in edges)
    denominator = float(np.dot(degree, centered * centered))
    if denominator == 0:
        return math.inf
    return numerator / denominator


def _neighbor_edges(X: np.ndarray, n_neighbors: int) -> set[tuple[int, int]]:
    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    indices = neighbors.kneighbors(X, return_distance=False)
    edges = set()
    for i, row in enumerate(indices):
        for j in row:
            if i == j:
                continue
            edges.add((min(i, int(j)), max(i, int(j))))
    return edges
