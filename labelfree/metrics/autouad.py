"""Score-distribution metrics from AutoUAD (Dai and Fan, ICLR 2025).

All three are internal criteria on detector scores. Dai and Fan compute RTM
and EAG on training-set scores and NPD on held-out scores against Gaussian
pseudo samples; this module evaluates precomputed score vectors only.
"""

from __future__ import annotations

import math

import numpy as np

from labelfree._validation import check_fraction, orient_scores

EPSILON = 1e-6


def relative_top_median_score(
    scores,
    *,
    top_fraction: float = 0.05,
    score_polarity: str = "higher_is_anomalous",
    eps: float = EPSILON,
) -> float:
    """Relative Top-Median (RTM), Definition 3 of Dai and Fan.

    ``RTM = (mean(s[s >= t]) - median(s)) / (median(s) + eps)``, where ``t`` is
    the score at the top ``top_fraction`` cutoff. Ties at ``t`` are included.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Detector scores.
    top_fraction : float
        Tail fraction ``tau``; the paper uses 5%.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores.
    eps : float
        Stabilizer in the denominator.

    Returns
    -------
    float
        Higher is better. Not invariant to score offsets, so compare it within
        one score convention.
    """
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
    """Expected Anomaly Gap (EAG), Definition 5 and Algorithm 1 of Dai and Fan.

    For every split of the descending scores into the top ``k`` and the rest,
    ``k = 1, ..., ceil(top_fraction * n)``, the anomaly gap is
    ``k (n - k) (mu_top - mu_rest)^2 / (n (k var_top + (n - k) var_rest + eps))``
    with population variances; EAG is the mean gap.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Detector scores.
    top_fraction : float
        Fraction ``rho`` of splits to average over; the paper uses 0.2.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores.
    eps : float
        Stabilizer in the denominator.

    Returns
    -------
    float
        Higher is better.
    """
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
    """Normalized Pseudo Discrepancy (NPD), Definition 6 of Dai and Fan.

    ``NPD = (mean(s_gen) - mean(s_val))^2 / (2 (var(s_gen) + var(s_val)) + eps)``
    with population variances. In the paper, ``s_val`` scores held-out
    training samples and ``s_gen`` scores samples drawn from
    ``N(mean, diag(var))`` of the detector's training features.

    Parameters
    ----------
    validation_scores : array-like of shape (n_validation,)
        Scores on held-out real samples.
    generated_scores : array-like of shape (n_generated,)
        Scores on generated pseudo samples.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores.
    eps : float
        Stabilizer in the denominator.

    Returns
    -------
    float
        Higher is better.
    """
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
    check_fraction(value, name="top_fraction")
