"""Combine several label-free metrics into one model ranking.

Implements the Borda-based rank aggregation of Goswami, Challu, Callot,
Minorics, and Kan ("Unsupervised Model Selection for Time-series Anomaly
Detection", ICLR 2023), who treat each label-free metric as a noisy ranking of
the candidate models.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import rankdata

from labelfree._validation import check_integer

__all__ = ["borda_scores", "robust_borda_scores"]


def borda_scores(
    metric_matrix,
    *,
    higher_is_better=True,
    top_k: int | None = None,
) -> np.ndarray:
    """Borda count over metric rankings of candidate models.

    A model ranked at position ``r`` (1 is best) by a metric receives
    ``N - r`` points. With ``top_k``, only models in a metric's top ``k``
    receive points (partial Borda). Tied models share the average position.

    Parameters
    ----------
    metric_matrix : array-like of shape (n_metrics, n_models)
        One row of metric values per metric.
    higher_is_better : bool or sequence of bool
        Direction of every metric, or one direction for all rows.
    top_k : int, optional
        Number of top positions per metric that earn points.

    Returns
    -------
    numpy.ndarray of shape (n_models,)
        Borda points per model. Higher is better.
    """
    positions = _positions(metric_matrix, higher_is_better)
    return _borda(positions, _top_k(top_k, positions.shape[1]))


def robust_borda_scores(
    metric_matrix,
    *,
    higher_is_better=True,
    top_k: int | None = None,
) -> np.ndarray:
    """Borda count after removing metrics with high empirical influence.

    The empirical influence of ranking ``s`` in the set ``S`` is
    ``EI(s) = f(S) - f(S without s)`` with ``f(A)`` the mean Kendall tau
    distance between ``Borda(A)`` and the rankings in ``A``. Influences are
    split into two groups by single-linkage clustering, the group with the
    larger mean influence is discarded, and Borda (partial with ``top_k``)
    aggregates the remaining rankings. Goswami et al. call the partial version
    robust Borda and the full one trimmed Borda. With fewer than three metrics
    nothing is trimmed.

    Parameters
    ----------
    metric_matrix : array-like of shape (n_metrics, n_models)
        One row of metric values per metric.
    higher_is_better : bool or sequence of bool
        Direction of every metric, or one direction for all rows.
    top_k : int, optional
        Number of top positions per metric that earn points.

    Returns
    -------
    numpy.ndarray of shape (n_models,)
        Borda points per model from the retained metrics. Higher is better.
    """
    positions = _positions(metric_matrix, higher_is_better)
    top_k = _top_k(top_k, positions.shape[1])
    if positions.shape[0] < 3:
        return _borda(positions, top_k)

    every = np.arange(positions.shape[0])
    total = _mean_distance(positions, every)
    influence = np.array(
        [total - _mean_distance(positions, every[every != m]) for m in every]
    )
    if np.ptp(influence) == 0:
        return _borda(positions, top_k)
    groups = fcluster(linkage(influence[:, None], method="single"), 2, "maxclust")
    means = {g: influence[groups == g].mean() for g in np.unique(groups)}
    keep = groups != max(means, key=means.get)
    return _borda(positions[keep], top_k)


def _positions(metric_matrix, higher_is_better) -> np.ndarray:
    matrix = np.asarray(metric_matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("metric_matrix must have shape (n_metrics, n_models >= 2)")
    if not np.isfinite(matrix).all():
        raise ValueError("metric_matrix contains non-finite values")
    directions = np.broadcast_to(
        np.asarray(higher_is_better, dtype=bool), (matrix.shape[0],)
    )
    oriented = np.where(directions[:, None], -matrix, matrix)
    return rankdata(oriented, axis=1, method="average")


def _top_k(top_k: int | None, n_models: int) -> int:
    if top_k is None:
        return n_models
    check_integer(top_k, name="top_k")
    if not 1 <= top_k <= n_models:
        raise ValueError(f"top_k must be between 1 and {n_models}")
    return int(top_k)


def _borda(positions: np.ndarray, top_k: int) -> np.ndarray:
    n_models = positions.shape[1]
    points = np.where(positions <= top_k, n_models - positions, 0.0)
    return points.sum(axis=0)


def _mean_distance(positions: np.ndarray, rows: np.ndarray) -> float:
    consensus = rankdata(-_borda(positions[rows], positions.shape[1]))
    return float(np.mean([_kendall_distance(consensus, positions[r]) for r in rows]))


def _kendall_distance(left: np.ndarray, right: np.ndarray) -> int:
    """Number of model pairs ordered differently by two rankings."""
    left_order = np.sign(left[:, None] - left[None, :])
    right_order = np.sign(right[:, None] - right[None, :])
    return int(np.count_nonzero(np.triu(left_order * right_order < 0, k=1)))
