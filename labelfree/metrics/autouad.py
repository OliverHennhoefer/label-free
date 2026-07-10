"""Score-distribution metrics from AutoUAD."""

from __future__ import annotations

import math

import numpy as np

from labelfree.utils.validation import orient_scores

EPSILON = 1e-6


def relative_top_median_score(
    scores,
    *,
    top_fraction: float = 0.05,
    score_polarity: str = "higher_is_anomalous",
    eps: float = EPSILON,
) -> float:
    """Relative Top-Median score-tail separation. Higher is better."""
    scores = orient_scores(scores, score_polarity=score_polarity)
    top = _top_scores(scores, top_fraction)
    median = float(np.median(scores))
    return (float(np.mean(top)) - median) / (median + eps)


def expected_anomaly_gap_score(
    scores,
    *,
    top_fraction: float = 0.2,
    score_polarity: str = "higher_is_anomalous",
    eps: float = EPSILON,
) -> float:
    """Expected Anomaly Gap over top-tail thresholds. Higher is better."""
    scores = np.sort(orient_scores(scores, score_polarity=score_polarity))[::-1]
    _check_fraction(top_fraction)
    n_samples = scores.size
    top_count = min(math.ceil(n_samples * top_fraction), n_samples - 1)
    if top_count < 1:
        raise ValueError("scores must contain at least two values")

    k = np.arange(1, top_count + 1, dtype=float)
    high_sum = np.cumsum(scores)[:top_count]
    high_sum_sq = np.cumsum(scores * scores)[:top_count]
    total = float(np.sum(scores))
    total_sq = float(np.sum(scores * scores))

    low_count = n_samples - k
    high_mean = high_sum / k
    low_sum = total - high_sum
    low_mean = low_sum / low_count
    high_var = np.maximum(high_sum_sq / k - high_mean * high_mean, 0.0)
    low_var = np.maximum(
        (total_sq - high_sum_sq) / low_count - low_mean * low_mean,
        0.0,
    )

    numerator = k * low_count * (high_mean - low_mean) ** 2
    denominator = n_samples * (k * high_var + low_count * low_var + eps)
    return float(np.mean(numerator / denominator))


def normalized_pseudo_discrepancy_score(
    validation_scores,
    generated_scores,
    *,
    score_polarity: str = "higher_is_anomalous",
    eps: float = EPSILON,
) -> float:
    """Discrepancy between validation and generated score vectors. Higher is better."""
    validation_scores = orient_scores(
        validation_scores,
        score_polarity=score_polarity,
    )
    generated_scores = orient_scores(
        generated_scores,
        score_polarity=score_polarity,
    )
    mean_gap = float(np.mean(generated_scores) - np.mean(validation_scores))
    variance_sum = float(np.var(generated_scores) + np.var(validation_scores))
    return mean_gap * mean_gap / (2 * variance_sum + eps)


def _top_scores(scores: np.ndarray, top_fraction: float) -> np.ndarray:
    _check_fraction(top_fraction)
    count = max(1, math.ceil(scores.size * top_fraction))
    threshold = np.partition(scores, -count)[-count]
    return scores[scores >= threshold]


def _check_fraction(value: float) -> None:
    if not 0 < value < 1:
        raise ValueError("top_fraction must be between 0 and 1")
