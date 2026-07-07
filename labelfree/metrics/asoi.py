"""ASI and ASOI metrics for predicted anomaly partitions."""

from __future__ import annotations

import math

import numpy as np

from labelfree.utils.validation import as_2d_finite, split_score_labels


def asi_score(
    X,
    scores,
    *,
    n_outliers: int | None = None,
    contamination: float | None = None,
    score_polarity: str = "higher_is_anomalous",
) -> float:
    """Anomaly Separation Index from Mahmud, Farou, and Lendak.

    Returns Eq. 3 from the paper: a multivariate standardized mean difference
    between predicted normal and anomaly groups. Higher is better.
    """
    normal, anomaly = _split_X(
        X,
        scores,
        n_outliers=n_outliers,
        contamination=contamination,
        score_polarity=score_polarity,
    )
    if len(normal) < 2 or len(anomaly) < 2:
        raise ValueError("ASI requires at least two normal and two anomaly samples")

    mean_distance = float(np.linalg.norm(normal.mean(axis=0) - anomaly.mean(axis=0)))
    pooled_var = (
        (len(normal) - 1) * normal.var(axis=0, ddof=1)
        + (len(anomaly) - 1) * anomaly.var(axis=0, ddof=1)
    ) / (len(normal) + len(anomaly) - 2)
    scale = float(np.sqrt(np.sum(pooled_var)))
    if scale == 0:
        return math.inf if mean_distance > 0 else 0.0
    return mean_distance / scale


def asoi_score(
    X,
    scores,
    *,
    n_outliers: int | None = None,
    contamination: float | None = None,
    score_polarity: str = "higher_is_anomalous",
    alpha: float = 0.5314,
    beta: float = 0.4686,
) -> float:
    """Anomaly Separation and Overlap Index.

    Combines normalized anomaly-to-normal-centroid separation with average
    feature-wise Hellinger distance. Higher is better.
    """
    _check_weights(alpha, beta)
    normal, anomaly = _split_X(
        X,
        scores,
        n_outliers=n_outliers,
        contamination=contamination,
        score_polarity=score_polarity,
    )
    normal_center = normal.mean(axis=0)
    separation = float(np.linalg.norm(anomaly - normal_center, axis=1).mean())
    max_distance = float(np.linalg.norm(anomaly.max(axis=0) - normal.min(axis=0)))
    separation_norm = 0.0 if max_distance == 0 else separation / max_distance
    separation_norm = float(np.clip(separation_norm, 0.0, 1.0))

    hellinger = float(
        np.mean(
            [
                _hellinger_feature(anomaly[:, feature], normal[:, feature])
                for feature in range(normal.shape[1])
            ]
        )
    )
    return alpha * separation_norm + beta * hellinger


def _split_X(
    X,
    scores,
    *,
    n_outliers: int | None,
    contamination: float | None,
    score_polarity: str,
) -> tuple[np.ndarray, np.ndarray]:
    X = as_2d_finite(X, name="X")
    _, labels = split_score_labels(
        scores,
        n_outliers=n_outliers,
        contamination=contamination,
        score_polarity=score_polarity,
    )
    if X.shape[0] != labels.size:
        raise ValueError("X and scores must have the same number of samples")
    return X[labels == 0], X[labels == 1]


def _hellinger_feature(anomaly: np.ndarray, normal: np.ndarray) -> float:
    low = min(float(anomaly.min()), float(normal.min()))
    high = max(float(anomaly.max()), float(normal.max()))
    if low == high:
        return 0.0

    bins = math.ceil(2 * (anomaly.size + normal.size) ** (1 / 3))
    anomaly_counts, edges = np.histogram(anomaly, bins=bins, range=(low, high))
    normal_counts, _ = np.histogram(normal, bins=edges)
    anomaly_mass = anomaly_counts / anomaly.size
    normal_mass = normal_counts / normal.size
    return float(np.linalg.norm(np.sqrt(anomaly_mass) - np.sqrt(normal_mass)) / math.sqrt(2))


def _check_weights(alpha: float, beta: float) -> None:
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")
    if not math.isclose(alpha + beta, 1.0):
        raise ValueError("alpha and beta must sum to 1")
