"""Agreement on the same alerts beyond shared preference for sparse regions may improve selection."""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from benchmarks.protocol import TAIL, top_membership


def select(batch):
    """One vote per other detector family, corrected by a density-stratified null."""
    n = len(batch.X)
    k = max(1, int(np.ceil(TAIL * n)))
    distance = (
        NearestNeighbors(n_neighbors=min(10, n - 1)).fit(batch.X).kneighbors()[0][:, -1]
    )
    bins = np.searchsorted(
        np.unique(np.quantile(distance, [0.25, 0.5, 0.75])), distance
    )
    alerts = np.stack([top_membership(s, k) for s in batch.scores])
    expected = np.zeros((len(alerts), len(alerts)))
    for group in np.unique(bins):
        mask = bins == group
        counts = alerts[:, mask].sum(axis=1)
        expected += np.outer(counts, counts) / mask.sum()
    denominator = k - expected
    agreement = np.divide(
        alerts @ alerts.T - expected,
        denominator,
        out=np.zeros_like(expected),
        where=denominator > 1e-10,
    )
    families = np.array(batch.families)
    return [
        np.median(
            [np.median(row[families == f]) for f in np.unique(families) if f != family]
        )
        for row, family in zip(agreement, families, strict=True)
    ]
