"""IREOS-style classifier separability score."""

from __future__ import annotations

import numpy as np

from labelfree.utils.validation import as_2d_finite, split_score_labels


def ireos_score(
    X,
    scores,
    *,
    n_outliers: int | None = None,
    contamination: float | None = None,
    score_polarity: str = "higher_is_anomalous",
    gammas=None,
    gamma_count: int = 10,
) -> float:
    """Score-weighted one-vs-rest RBF separability. Higher is better."""
    X = as_2d_finite(X, name="X")
    oriented, labels = split_score_labels(
        scores,
        n_outliers=n_outliers,
        contamination=contamination,
        score_polarity=score_polarity,
    )
    if X.shape[0] != oriented.size:
        raise ValueError("X and scores must have the same number of samples")

    candidate_indices = np.flatnonzero(labels == 1)
    gammas = _gamma_grid(X, gammas=gammas, gamma_count=gamma_count)
    weights = _candidate_weights(oriented[candidate_indices])
    separability = np.array(
        [_point_separability(X, index, gammas) for index in candidate_indices]
    )
    return float(np.dot(weights, separability))


def _point_separability(X: np.ndarray, index: int, gammas: np.ndarray) -> float:
    squared_distances = np.sum((X - X[index]) ** 2, axis=1)
    squared_distances = np.delete(squared_distances, index)
    separability = [
        1.0 - float(np.mean(np.exp(-float(gamma) * squared_distances)))
        for gamma in gammas
    ]
    return float(np.mean(separability))


def _gamma_grid(X: np.ndarray, *, gammas, gamma_count: int) -> np.ndarray:
    if gammas is not None:
        gammas = np.asarray(gammas, dtype=float)
        if gammas.ndim != 1 or gammas.size == 0 or np.any(gammas <= 0):
            raise ValueError("gammas must be a non-empty 1D array of positive values")
        return gammas
    if gamma_count < 1:
        raise ValueError("gamma_count must be positive")

    distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    nonzero = distances[distances > 0]
    if nonzero.size == 0:
        return np.ones(gamma_count)
    scale = float(np.quantile(nonzero, 0.1))
    gamma_max = 1.0 / (2.0 * scale * scale)
    return np.linspace(gamma_max / gamma_count, gamma_max, gamma_count)


def _candidate_weights(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.min()
    total = float(shifted.sum())
    if total == 0:
        return np.full(scores.size, 1 / scores.size)
    return shifted / total
