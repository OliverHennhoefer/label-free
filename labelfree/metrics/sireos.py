"""Similarity-based Internal, Relative Evaluation of Outlier Solutions (SIREOS)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances

from labelfree._validation import (
    as_2d_finite,
    check_positive,
    check_same_length,
    orient_scores,
)


def sireos_score(
    X,
    scores,
    *,
    kernel_width: float | None = None,
    kernel_quantile: float = 0.01,
    score_polarity: str = "higher_is_anomalous",
) -> float:
    """SIREOS index of Marques, Zimek, Campello, and Sander (SISAP 2022).

    ``SIREOS = sum_j w_j * mean_{i != j} exp(-||x_j - x_i||^2 / (2 t^2))`` with
    weights ``w_j = s_j / sum(s)``. A good solution puts its weight on samples
    that are dissimilar to the rest of the data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix. Scale features before comparing solutions.
    scores : array-like of shape (n_samples,)
        Outlier scores. After orientation they must be non-negative with a
        positive sum, as the index normalizes them by their total. Shifted or
        signed scores, such as scikit-learn's ``decision_function``, should
        first be mapped with :func:`labelfree.unify_scores`.
    kernel_width : float, optional
        Heat-kernel width ``t``. Defaults to the ``kernel_quantile`` quantile of
        the non-zero pairwise distances, as in the authors' code.
    kernel_quantile : float
        Distance quantile used when ``kernel_width`` is omitted.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores.

    Returns
    -------
    float
        SIREOS in ``[0, 1]``. Lower is better.
    """
    X = as_2d_finite(X, name="X")
    scores = orient_scores(scores, score_polarity=score_polarity)
    check_same_length(X, scores)
    if X.shape[0] < 2:
        raise ValueError("X must contain at least two samples")
    if np.any(scores < 0) or scores.sum() == 0:
        raise ValueError(
            "SIREOS needs non-negative oriented scores with a positive sum; "
            "map signed scores with labelfree.unify_scores first"
        )

    distances = pairwise_distances(X)
    width = _kernel_width(distances, kernel_width, kernel_quantile)
    if width == 0:
        similarity = np.ones(X.shape[0])
    else:
        kernel = np.exp(-(distances**2) / (2 * width**2))
        np.fill_diagonal(kernel, 0.0)
        similarity = kernel.sum(axis=1) / (X.shape[0] - 1)
    return float(np.dot(scores / scores.sum(), similarity))


def _kernel_width(
    distances: np.ndarray,
    kernel_width: float | None,
    kernel_quantile: float,
) -> float:
    if kernel_width is not None:
        return check_positive(kernel_width, name="kernel_width")
    if not 0 < kernel_quantile < 1:
        raise ValueError("kernel_quantile must be between 0 and 1")
    nonzero = distances[distances > 0]
    if nonzero.size == 0:
        return 0.0
    return float(np.quantile(nonzero, kernel_quantile))
