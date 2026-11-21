"""Agreement on the same alerts may improve selection."""

import numpy as np

from benchmarks.protocol import TAIL, top_membership


def select(batch):
    """One vote per other detector family, corrected by a single-stratum null."""
    n = len(batch.X)
    k = max(1, int(np.ceil(TAIL * n)))
    bins = np.zeros(n, dtype=int)
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
