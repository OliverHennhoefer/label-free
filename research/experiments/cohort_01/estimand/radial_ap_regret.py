"""Minimize worst-case population AP regret when radial risk rises by unknown amounts across dyadic severity bands."""

from itertools import combinations

import numpy as np
from scipy.linalg import eig

from benchmarks.protocol import top_membership


def radial_profiles(X):
    """Upper-tail memberships of sample squared Mahalanobis radius."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n < 2 or X.shape[1] == 0:
        return None
    centered = X - X.mean(axis=0)
    left, singular, _ = np.linalg.svd(centered, full_matrices=False)
    rank = np.count_nonzero(singular > singular[0] * max(X.shape) * np.finfo(float).eps)
    if rank == 0 or rank >= n - 1:
        return None
    # Projection leverage is squared radius up to a common scale. Round away
    # floating-point differences between physically equal radii.
    radius = np.round(np.sum(left[:, :rank] ** 2, axis=1), 12)
    if np.ptp(radius) == 0:
        return None
    budgets = np.unique(np.ceil(n / 2 ** np.arange(7)).astype(int))
    return np.column_stack([top_membership(radius, k) for k in budgets])


def ap_matrix(scores, profiles):
    """Symmetric numerator for tie-grouped population AP on the row support."""
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order])) + 1, len(order)]
    cumulative = np.cumsum(profiles[order], axis=0)[ends - 1]
    increments = np.diff(cumulative, axis=0, prepend=np.zeros_like(cumulative[:1]))
    terms = increments.T @ (cumulative / ends[:, None]) / len(order)
    return (terms + terms.T) / 2


def maximum_gain(difference, prevalence):
    """Return the sharp largest AP gain and an attaining calibration."""
    denominator = (prevalence[:, None] + prevalence[None, :]) / 2
    # The value on any face is a weighted average of its entrywise ratios.
    upper = difference / denominator
    vertex = np.argmax(np.diag(upper))
    witness = np.eye(len(prevalence))[vertex]
    best = upper[vertex, vertex]
    if upper.max() <= best + 1e-13:
        return best, witness

    for size in range(2, len(prevalence) + 1):
        for support in combinations(range(len(prevalence)), size):
            indices = np.ix_(support, support)
            if upper[indices].max() <= best + 1e-13:
                continue
            numerator = difference[indices]
            values, vectors = eig(numerator, denominator[indices], check_finite=False)
            for value, vector in zip(values, vectors.T, strict=True):
                if not np.isfinite(value) or abs(value.imag) > 1e-9:
                    continue
                weights = vector.real.copy()
                if abs(weights.sum()) < 1e-12:
                    continue
                weights /= weights.sum()
                if weights.min() < -1e-9:
                    continue
                weights = np.maximum(weights, 0)
                weights /= weights.sum()
                gain = (weights @ numerator @ weights) / (
                    prevalence[list(support)] @ weights
                )
                if gain > best:
                    best = gain
                    witness = np.zeros_like(prevalence)
                    witness[list(support)] = weights
    return best, witness


def select(batch):
    profiles = radial_profiles(batch.X)
    if profiles is None:
        return None
    prevalence = profiles.mean(axis=0)
    matrices = np.array([ap_matrix(score, profiles) for score in batch.scores])
    regrets = [
        max(maximum_gain(other - candidate, prevalence)[0] for other in matrices)
        for candidate in matrices
    ]
    return -np.round(regrets, 12)
