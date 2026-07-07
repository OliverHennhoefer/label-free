"""Similarity-based IREOS."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances

from labelfree.utils.validation import as_2d_finite, orient_scores


def sireos_score(
    X,
    scores,
    *,
    score_polarity: str = "higher_is_anomalous",
    kernel_width: float | None = None,
    kernel_quantile: float = 0.01,
) -> float:
    """Score-weighted average similarity to other samples. Lower is better."""
    X = as_2d_finite(X, name="X")
    scores = orient_scores(scores, score_polarity=score_polarity)
    if X.shape[0] != scores.size:
        raise ValueError("X and scores must have the same number of samples")

    distances = pairwise_distances(X)
    width = _kernel_width(distances, kernel_width, kernel_quantile)
    similarity = _leave_one_out_similarity(distances, width)
    weights = _score_weights(scores)
    return float(np.dot(weights, similarity))


def _kernel_width(
    distances: np.ndarray,
    kernel_width: float | None,
    kernel_quantile: float,
) -> float:
    if kernel_width is not None:
        if kernel_width <= 0:
            raise ValueError("kernel_width must be positive")
        return float(kernel_width)
    if not 0 < kernel_quantile < 1:
        raise ValueError("kernel_quantile must be between 0 and 1")

    nonzero = distances[distances > 0]
    if nonzero.size == 0:
        return 0.0
    return float(np.quantile(nonzero, kernel_quantile))


def _leave_one_out_similarity(distances: np.ndarray, width: float) -> np.ndarray:
    if distances.shape[0] < 2:
        raise ValueError("X must contain at least two samples")
    if width == 0:
        return np.ones(distances.shape[0])

    kernel = np.exp(-(distances * distances) / (2 * width * width))
    np.fill_diagonal(kernel, 0.0)
    return kernel.sum(axis=1) / (kernel.shape[0] - 1)


def _score_weights(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.min()
    total = float(shifted.sum())
    if total == 0:
        return np.full(scores.size, 1 / scores.size)
    return shifted / total
