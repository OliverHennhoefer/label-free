"""ASI and ASOI for the score-induced anomaly partition.

Mahmud, Farou, and Lendak ("Anomaly separation and overlap index", Complex &
Intelligent Systems, 2025) rate how well the predicted anomalies separate from
the predicted normal samples in feature space.
"""

from __future__ import annotations

import math

import numpy as np

from labelfree._validation import as_2d_finite, split_score_labels


def asi_score(
    X,
    scores,
    *,
    n_outliers: int | None = None,
    contamination: float | None = None,
    score_polarity: str = "higher_is_anomalous",
) -> float:
    """Anomaly Separation Index (ASI), Eq. 3 of Mahmud et al.

    ``ASI = ||mean_N - mean_A|| / sqrt(sum_f pooled_var_f)``, a multivariate
    standardized mean difference with per-feature pooled sample variances.
    The paper mentions a normalization to ``[0, 1]`` without defining it, so
    none is applied.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    scores : array-like of shape (n_samples,)
        Detector scores; the top ``n_outliers`` form the anomaly group.
    n_outliers, contamination
        Size of the anomaly group. Pass exactly one.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores.

    Returns
    -------
    float
        Higher is better.
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
    """Anomaly Separation and Overlap Index (ASOI) of Mahmud et al.

    ``ASOI = alpha * S + beta * H`` where ``S`` is the mean Euclidean distance
    of predicted anomalies to the centroid of predicted normal samples and
    ``H`` is the mean feature-wise Hellinger distance between the anomaly and
    normal histograms (Rice rule, ``ceil(2 n^(1/3))`` shared bins). The
    default weights are the paper's mutual-information weights.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix. ``S`` is scale-dependent, so scale features first.
    scores : array-like of shape (n_samples,)
        Detector scores; the top ``n_outliers`` form the anomaly group.
    n_outliers, contamination
        Size of the anomaly group. Pass exactly one.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores.
    alpha, beta : float
        Non-negative weights summing to 1.

    Returns
    -------
    float
        Higher is better.
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

    hellinger = float(
        np.mean(
            [
                _hellinger_feature(anomaly[:, feature], normal[:, feature])
                for feature in range(normal.shape[1])
            ]
        )
    )
    return alpha * separation + beta * hellinger


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
    return float(
        np.linalg.norm(np.sqrt(anomaly_mass) - np.sqrt(normal_mass)) / math.sqrt(2)
    )


def _check_weights(alpha: float, beta: float) -> None:
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")
    if not math.isclose(alpha + beta, 1.0):
        raise ValueError("alpha and beta must sum to 1")
