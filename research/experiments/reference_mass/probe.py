"""Quantify sampling error and reference mismatch at one mass-volume operating point."""

import numpy as np
from scipy.stats import binom

from benchmarks.prepare import oracle


def probability_of_choosing_a(probes):
    # With uniform probes, 25% hit A or B. Conditional on a hit, 80% land in A.
    hits = np.arange(probes + 1)
    half = hits // 2
    prefer_a = binom.sf(half, hits, 0.8)
    ties = (hits % 2 == 0) * binom.pmf(half, hits, 0.8)
    return binom.pmf(hits, probes, 0.25) @ (prefer_a + 0.5 * ties)


def main():
    counts = [500, 500, 9000]
    cell = np.repeat(np.arange(3), counts)
    scores = np.stack([cell == 0, cell == 1]).astype(float)
    utilities = []
    for positives in ([100, 25, 375], [25, 100, 375]):
        y = np.concatenate(
            [np.arange(n) < p for n, p in zip(counts, positives, strict=True)]
        )
        outcomes = oracle(y, scores)
        utilities.append(outcomes["precision_at_budget"])
    print(
        "uniform probes   P(choose A)   expected precision: uniform / shifted anomalies"
    )
    for budget in [8, 32, 128, 512]:
        p = probability_of_choosing_a(budget)
        chosen = [p * a + (1 - p) * b for a, b in utilities]
        print(f"{budget:14d}   {p:11.6f}   {chosen[0]:.6f} / {chosen[1]:.6f}")


if __name__ == "__main__":
    main()
