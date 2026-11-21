"""A useful detector of relation-breaking contamination can repair typical held-out feature predictions when its alerts are removed."""

import numpy as np

from benchmarks.protocol import TAIL, top_membership


def _folds(X):
    X = np.asarray(X, dtype=float)
    order = np.random.default_rng(1729).permutation(len(X))
    for test_idx in np.array_split(order, 3):
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[test_idx] = False
        train = X[train_mask]
        mean, scale = train.mean(axis=0), train.std(axis=0)
        scale[scale == 0] = 1.0
        yield train_mask, test_idx, (train - mean) / scale, (X[test_idx] - mean) / scale


def _row_losses(train, test, weights):
    """Squared errors for predicting each feature from the others, averaged per row."""
    center = np.average(train, axis=0, weights=weights)
    centered = train - center
    covariance = (centered.T * weights) @ centered / weights.sum()
    indices = np.diag_indices_from(covariance)
    covariance[indices] += 0.05
    precision = np.linalg.solve(covariance, np.eye(train.shape[1]))
    # The jth off-diagonal regression coefficient is -P[k,j] / P[j,j].
    residuals = ((test - center) @ precision) / precision.diagonal()
    return (residuals**2).mean(axis=1)


def _budget(n):
    return int(np.ceil(TAIL * n))


def select(batch):
    """Rank each candidate by the typical effect of deleting its own alerts."""
    if len(batch.X) < 5 or batch.X.shape[1] < 2:
        return None
    effects = np.empty(batch.scores.shape, dtype=float)
    rng = np.random.default_rng(1730)
    for train_mask, test_idx, train, test in _folds(batch.X):
        permutations = [rng.permutation(len(train)) for _ in range(4)]
        reference = {}
        for i, scores in enumerate(batch.scores):
            weights = 1.0 - top_membership(scores[train_mask], _budget(len(train)))
            pattern = np.sort(weights)
            key = pattern.tobytes()
            if key not in reference:
                reference[key] = np.mean(
                    [_row_losses(train, test, pattern[p]) for p in permutations],
                    axis=0,
                )
            effects[i, test_idx] = reference[key] - _row_losses(train, test, weights)
    return np.median(effects, axis=1)


def direct_residual(batch):
    """Rank each candidate by the mean cross-feature prediction loss of its alerts."""
    if len(batch.X) < 5 or batch.X.shape[1] < 2:
        return None
    losses = np.empty(len(batch.X))
    for _, test_idx, train, test in _folds(batch.X):
        losses[test_idx] = _row_losses(train, test, np.ones(len(train)))
    k = _budget(len(losses))
    return np.array([top_membership(s, k) @ losses / k for s in batch.scores])
