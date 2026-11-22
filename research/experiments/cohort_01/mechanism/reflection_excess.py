"""Estimate detector AP from one-sided excess over reflected nominal measurements.

One fold fits pairs of reflected cells. A second chooses the enriched side,
and a third measures its signed excess. The benchmark receives an abstention
when the implied fraction of positive rows falls outside the pilot's range.
"""

import numpy as np


def _assign(rows, prototypes):
    """Return signed cell IDs. Cell zero contains points closest to the origin."""
    if not len(prototypes):
        return np.zeros(len(rows), dtype=int)
    products = rows @ prototypes.T
    norms = np.einsum("ij,ij->i", prototypes, prototypes)
    relative_distance = norms - 2 * np.abs(products)
    closest = relative_distance.argmin(axis=1)
    index = np.arange(len(rows)), closest
    signs = np.sign(products[index]).astype(int)
    return np.where(relative_distance[index] < 0, signs * (closest + 1), 0)


def _fit_prototypes(rows, seed):
    rng = np.random.default_rng(seed)
    squared_norm = np.einsum("ij,ij->i", rows, rows)
    nearest = squared_norm.copy()
    prototypes = []
    limit = min(32, max(1, int(np.sqrt(len(rows) / 2))))
    for _ in range(limit):
        total = nearest.sum()
        if total <= np.finfo(float).eps * squared_norm.sum():
            break
        chosen = rng.choice(len(rows), p=nearest / total)
        prototype = rows[chosen]
        prototypes.append(prototype)
        distance = squared_norm + squared_norm[chosen] - 2 * np.abs(rows @ prototype)
        nearest = np.minimum(nearest, np.maximum(distance, 0))
    prototypes = np.asarray(prototypes, dtype=float).reshape(-1, rows.shape[1])
    for _ in range(12):
        cells = _assign(rows, prototypes)
        sums = np.zeros((len(prototypes) + 1, rows.shape[1]))
        np.add.at(sums, np.abs(cells), np.sign(cells[:, None]) * rows)
        counts = np.bincount(np.abs(cells), minlength=len(sums))
        updated = prototypes.copy()
        used = counts[1:] > 0
        updated[used] = sums[1:][used] / counts[1:][used, None]
        if np.array_equal(updated, prototypes):
            break
        prototypes = updated
    return prototypes


def anomaly_weights(features, *, clipped=False):
    """Estimate a signed positive mass per row across held-out partitions."""
    features = np.asarray(features, dtype=float)
    rows = len(features)
    weights = np.zeros(rows)
    if rows < 30 or features.shape[1] == 0:
        return weights
    permutation = np.random.default_rng(9137).permutation(rows)
    folds = np.array_split(permutation, 3)
    for fold, partition_rows in enumerate(folds):
        center = np.median(features[partition_rows], axis=0)
        prototypes = _fit_prototypes(features[partition_rows] - center, 9138 + fold)
        # Reversing the count array exchanges the two sides and fixes its midpoint.
        count_pairs = len(prototypes)
        cells = _assign(features - center, prototypes) + count_pairs
        other_folds = [folds[(fold + offset) % 3] for offset in (1, 2)]
        counts = [
            np.bincount(cells[indices], minlength=2 * count_pairs + 1)
            for indices in other_folds
        ]
        for position, indices in enumerate(other_folds):
            measured = counts[position]
            orientation = measured if clipped else counts[1 - position]
            oriented = orientation > orientation[::-1]
            excess = np.divide(
                measured - measured[::-1],
                measured,
                where=oriented & (measured > 0),
                out=np.zeros(len(measured), dtype=float),
            )
            weights[indices] += 0.5 * excess[cells[indices]]
    return weights


def average_precisions(scores, weights):
    """Integrate precision at every distinct score with signed recall increments."""
    mass = weights.sum()
    if mass <= 0:
        return np.full(len(scores), np.nan)
    values = []
    for candidate in scores:
        indices = np.argsort(-candidate, kind="stable")
        ordered = candidate[indices]
        ends = np.flatnonzero(np.r_[ordered[:-1] != ordered[1:], True])
        cumulative = np.cumsum(weights[indices])[ends]
        increments = np.diff(cumulative, prepend=0)
        values.append(np.sum(increments * cumulative / (ends + 1)) / mass)
    return np.asarray(values)


def rank_batch(batch, *, clipped):
    if len(batch.X) < 30:
        return None
    weights = anomaly_weights(batch.X, clipped=clipped)
    prevalence = weights.mean()
    if prevalence <= 2 / np.sqrt(len(weights)) or prevalence > 0.5:
        return None
    return average_precisions(batch.scores, weights)


def select(batch):
    return rank_batch(batch, clipped=False)
