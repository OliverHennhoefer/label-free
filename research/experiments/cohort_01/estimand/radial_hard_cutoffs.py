"""Minimize worst AP regret over individual dyadic hard labels on sample Mahalanobis radius."""

import numpy as np

from benchmarks.protocol import top_membership


def select(batch):
    X = np.asarray(batch.X, dtype=float)
    n = len(X)
    if n < 2 or X.shape[1] == 0:
        return None
    left, singular, _ = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    rank = np.count_nonzero(singular > singular[0] * max(X.shape) * np.finfo(float).eps)
    if rank == 0 or rank >= n - 1:
        return None
    radius = np.round(np.sum(left[:, :rank] ** 2, axis=1), 12)
    if np.ptp(radius) == 0:
        return None
    budgets = np.unique(np.ceil(n / 2 ** np.arange(7)).astype(int))
    profiles = np.column_stack([top_membership(radius, k) for k in budgets])
    utilities = []
    for score in batch.scores:
        order = np.argsort(-score, kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(score[order])) + 1, n]
        cumulative = np.cumsum(profiles[order], axis=0)[ends - 1]
        increments = np.diff(cumulative, axis=0, prepend=np.zeros_like(cumulative[:1]))
        precision = cumulative / ends[:, None]
        utilities.append(np.sum(increments * precision, axis=0) / budgets)
    utilities = np.asarray(utilities)
    return -np.round(np.max(utilities.max(axis=0) - utilities, axis=1), 12)
