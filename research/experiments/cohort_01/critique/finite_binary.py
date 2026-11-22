"""Simultaneous two-cohort AP order certificates for prespecified binary scores."""

from itertools import combinations

import numpy as np
from scipy.stats import beta


def proportion_bounds(counts, sample_size, tail_probability):
    lower = np.zeros(counts.shape)
    upper = np.ones(counts.shape)
    positive = counts > 0
    k = counts[positive]
    lower[positive] = beta.ppf(tail_probability, k, sample_size - k + 1)
    below_full = counts < sample_size
    k = counts[below_full]
    upper[below_full] = beta.ppf(1 - tail_probability, k + 1, sample_size - k)
    return lower, upper


def coefficient_bounds(target_counts, high_counts, sample_size, alpha):
    candidates = target_counts.shape[1]
    tail = alpha / (4 * candidates)
    t_lo, t_hi = proportion_bounds(target_counts, sample_size, tail)
    h_lo, h_hi = proportion_bounds(high_counts, sample_size, tail)
    d_lo, d_hi = h_lo - t_hi, h_hi - t_lo
    nearest_zero = np.maximum(np.maximum(d_lo, -d_hi), 0)
    furthest_zero = np.maximum(np.abs(d_lo), np.abs(d_hi))
    q_lo = nearest_zero**2 / t_hi
    q_hi = np.divide(
        furthest_zero**2, t_lo, out=np.full_like(t_lo, np.inf), where=t_lo > 0
    )
    return np.stack((q_lo, q_lo + d_lo), axis=-1), np.stack(
        (q_hi, q_hi + d_hi), axis=-1
    )


def alert_counts(rng, sample_size, masses, repetitions):
    cells = rng.multinomial(sample_size, masses, size=repetitions)
    return np.column_stack((cells[:, 0], cells[:, 1], cells[:, :2].sum(axis=1)))


def experiment():
    target = np.array([0.05, 0.50, 0.45])
    high = np.array([0.055, 0.510, 0.435])
    repetitions = 4096
    sizes = (10**4, 10**6, 10**7)
    streams = np.random.SeedSequence(310071).spawn(len(sizes))
    true_t = np.array([target[0], target[1], target[:2].sum()])
    true_h = np.array([high[0], high[1], high[:2].sum()])
    true_d = true_h - true_t
    true_q = true_d**2 / true_t
    true_at_one = true_d + true_q
    true_at_thirty = true_d + 30 * true_q
    true_unbounded = (true_q[:, None] >= true_q) & (true_at_one[:, None] >= true_at_one)
    true_feasible = (true_at_thirty[:, None] >= true_at_thirty) & (
        true_at_one[:, None] >= true_at_one
    )

    for size, seed in zip(sizes, streams, strict=True):
        rng = np.random.default_rng(seed)
        target_counts = alert_counts(rng, size, target, repetitions)
        high_counts = alert_counts(rng, size, high, repetitions)
        lower, upper = coefficient_bounds(target_counts, high_counts, size, alpha=0.05)
        certified = np.all(lower[:, :, None, :] > upper[:, None, :, :], axis=-1)
        assert not np.any(certified & certified.transpose(0, 2, 1))
        no_order = ~(certified | certified.transpose(0, 2, 1))
        labels = ("A", "B", "C")
        print(f"n={size:,} per cohort")
        for i, j in combinations(range(3), 2):
            print(
                f"  {labels[i]}>{labels[j]}, {labels[j]}>{labels[i]}, abstain",
                certified[:, i, j].sum(),
                certified[:, j, i].sum(),
                no_order[:, i, j].sum(),
            )
        wrong_unbounded = np.any(certified & ~true_unbounded, axis=(1, 2))
        wrong_feasible = np.any(certified & ~true_feasible, axis=(1, 2))
        print("  Any wrong order for b >= 1", wrong_unbounded.sum())
        print("  Any wrong order for b in [1,30]", wrong_feasible.sum())


if __name__ == "__main__":
    experiment()
