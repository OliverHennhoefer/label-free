"""Excess-Mass and Mass-Volume criteria."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from labelfree.utils.validation import as_1d_finite, as_2d_finite, orient_scores


def excess_mass_curve(
    scores,
    reference_scores,
    *,
    support_volume: float,
    levels=None,
    level_count: int = 1000,
    score_polarity: str = "higher_is_anomalous",
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical Excess-Mass curve from data and uniform-reference scores."""
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
    """Area under the empirical Excess-Mass curve. Higher is better."""
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
    """Empirical Mass-Volume curve from data and uniform-reference scores."""
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
    """Area under the empirical Mass-Volume curve. Lower is better."""
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
    """Volume of the axis-aligned bounding box containing X."""
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
