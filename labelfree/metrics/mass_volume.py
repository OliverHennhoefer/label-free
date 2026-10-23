"""Excess-Mass and Mass-Volume criteria (Goix, 2016).

Goix ("How to Evaluate the Quality of Unsupervised Anomaly Detection
Algorithms?", ICML Anomaly Detection Workshop 2016) scores a detector by the
level sets of its normality score ``s`` (higher means more normal):

- ``MV(alpha) = inf_u Leb(s >= u)`` subject to ``P_n(s(X) >= u) >= alpha``,
- ``EM(t) = sup_u P_n(s(X) >= u) - t * Leb(s >= u)``.

Lebesgue measures are estimated by Monte Carlo from detector scores on
samples drawn uniformly from a support of known volume. Unlike the paper,
these functions take oriented anomaly scores and negate them internally.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from labelfree._validation import as_1d_finite, as_2d_finite, orient_scores


def excess_mass_curve(
    scores,
    reference_scores,
    *,
    support_volume: float,
    levels=None,
    level_count: int = 1000,
    score_polarity: str = "higher_is_anomalous",
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical Excess-Mass curve, as in Goix's reference code.

    For every level ``t``, the supremum runs over the thresholds ``u`` at the
    observed data scores (with strict ``s > u`` as in the authors' code) and
    over the whole support, which contributes ``1 - t * support_volume``.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Detector scores on the data.
    reference_scores : array-like of shape (n_reference,)
        Detector scores on samples drawn uniformly from the support, for
        example the bounding box of the data (:func:`bounding_box_volume`).
        Goix uses 100,000 reference samples.
    support_volume : float
        Lebesgue volume of the sampled support.
    levels : array-like, optional
        Non-negative, strictly increasing levels ``t``. Defaults to
        ``level_count`` points on ``[0, 100 / support_volume]``, the paper's
        range.
    level_count : int
        Number of default levels.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores.

    Returns
    -------
    levels, curve : numpy.ndarray
        Levels and Excess-Mass values.
    """
    normal_scores, reference_normal_scores = _normality_scores(
        scores,
        reference_scores,
        score_polarity=score_polarity,
    )
    support_volume = _check_volume(support_volume)
    levels = _levels(levels, level_count=level_count, support_volume=support_volume)

    curve = np.maximum(0.0, 1.0 - levels * support_volume)
    for threshold in np.unique(normal_scores):
        mass = np.mean(normal_scores > threshold)
        volume = np.mean(reference_normal_scores > threshold) * support_volume
        curve = np.maximum(curve, mass - levels * volume)
    return levels, curve


def excess_mass_auc(
    scores,
    reference_scores,
    *,
    support_volume: float,
    levels=None,
    level_count: int = 1000,
    em_min: float | None = 0.9,
    score_polarity: str = "higher_is_anomalous",
) -> float:
    """Area under the Excess-Mass curve. Higher is better.

    Following Goix, the integral stops at the first level where the curve
    drops to ``em_min`` or below (the paper's ``t_max = 0.9``). If the curve
    never reaches ``em_min``, the whole level range is integrated.
    Parameters other than ``em_min`` match :func:`excess_mass_curve`;
    ``em_min=None`` integrates the full range.
    """
    levels, curve = excess_mass_curve(
        scores,
        reference_scores,
        support_volume=support_volume,
        levels=levels,
        level_count=level_count,
        score_polarity=score_polarity,
    )
    if em_min is not None:
        below = np.flatnonzero(curve <= em_min)
        if below.size:
            stop = below[0] + 1
            levels = levels[:stop]
            curve = curve[:stop]
    return float(trapezoid(curve, levels))


def mass_volume_curve(
    scores,
    reference_scores,
    *,
    support_volume: float,
    alpha_min: float = 0.9,
    alpha_max: float = 0.999,
    alpha_count: int = 1000,
    score_polarity: str = "higher_is_anomalous",
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical Mass-Volume curve, as in Goix's reference code.

    For every mass ``alpha``, the threshold is the smallest data score whose
    upper level set holds at least ``alpha`` of the data, and the volume is
    the reference fraction at or above it times ``support_volume``.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Detector scores on the data.
    reference_scores : array-like of shape (n_reference,)
        Detector scores on samples drawn uniformly from the support, for
        example the bounding box of the data (:func:`bounding_box_volume`).
        Goix uses 100,000 reference samples.
    support_volume : float
        Lebesgue volume of the sampled support.
    alpha_min, alpha_max : float
        Mass range; the paper uses ``[0.9, 0.999]``.
    alpha_count : int
        Number of masses.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores.

    Returns
    -------
    alphas, volumes : numpy.ndarray
        Masses and Mass-Volume values.
    """
    normal_scores, reference_normal_scores = _normality_scores(
        scores,
        reference_scores,
        score_polarity=score_polarity,
    )
    support_volume = _check_volume(support_volume)
    if not 0 < alpha_min <= alpha_max <= 1:
        raise ValueError("alpha_min and alpha_max must satisfy 0 < min <= max <= 1")
    if alpha_count < 1:
        raise ValueError("alpha_count must be positive")

    alphas = np.linspace(alpha_min, alpha_max, alpha_count)
    sorted_scores = np.sort(normal_scores)
    volumes = np.empty(alphas.size, dtype=float)
    for i, alpha in enumerate(alphas):
        count = min(int(np.ceil(alpha * sorted_scores.size)), sorted_scores.size)
        threshold = sorted_scores[-count]
        volumes[i] = np.mean(reference_normal_scores >= threshold) * support_volume
    return alphas, volumes


def mass_volume_auc(
    scores,
    reference_scores,
    *,
    support_volume: float,
    alpha_min: float = 0.9,
    alpha_max: float = 0.999,
    alpha_count: int = 1000,
    score_polarity: str = "higher_is_anomalous",
) -> float:
    """Area under the Mass-Volume curve. Lower is better.

    Parameters match :func:`mass_volume_curve`.
    """
    alphas, volumes = mass_volume_curve(
        scores,
        reference_scores,
        support_volume=support_volume,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_count=alpha_count,
        score_polarity=score_polarity,
    )
    return float(trapezoid(volumes, alphas))


def bounding_box_volume(X, *, offset: float = 1e-12) -> float:
    """Volume of the axis-aligned bounding box of ``X`` plus ``offset``.

    A convenience for ``support_volume``; drawing the uniform reference
    samples and scoring them is left to the caller.
    """
    X = as_2d_finite(X, name="X")
    return float(np.prod(X.max(axis=0) - X.min(axis=0)) + offset)


def _normality_scores(scores, reference_scores, *, score_polarity: str):
    scores = -orient_scores(scores, score_polarity=score_polarity)
    reference_scores = -orient_scores(reference_scores, score_polarity=score_polarity)
    return scores, reference_scores


def _check_volume(value: float) -> float:
    value = float(value)
    if value <= 0 or not np.isfinite(value):
        raise ValueError("support_volume must be a positive finite value")
    return value


def _levels(levels, *, level_count: int, support_volume: float) -> np.ndarray:
    if levels is None:
        if level_count < 2:
            raise ValueError("level_count must be at least 2")
        return np.linspace(0.0, 100.0 / support_volume, level_count)
    levels = as_1d_finite(levels, name="levels")
    if np.any(levels < 0) or np.any(np.diff(levels) <= 0):
        raise ValueError("levels must be non-negative and strictly increasing")
    return levels
