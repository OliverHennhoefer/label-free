"""Shared validation helpers."""

import math

import numpy as np
from scipy.stats import rankdata


def as_1d_finite(values, *, name: str) -> np.ndarray:
    """Return a finite 1D float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def as_2d_finite(values, *, name: str) -> np.ndarray:
    """Return a finite 2D float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def orient_scores(scores, *, score_polarity: str) -> np.ndarray:
    """Normalize scores so larger values mean more anomalous."""
    scores = as_1d_finite(scores, name="scores")
    if score_polarity == "higher_is_anomalous":
        return scores
    if score_polarity == "higher_is_normal":
        return -scores
    raise ValueError(
        "score_polarity must be 'higher_is_anomalous' or 'higher_is_normal'"
    )


def split_score_labels(
    scores,
    *,
    n_outliers: int | None = None,
    contamination: float | None = None,
    score_polarity: str = "higher_is_anomalous",
) -> tuple[np.ndarray, np.ndarray]:
    """Orient scores and label the top score tail as anomalies."""
    oriented = orient_scores(scores, score_polarity=score_polarity)
    if (n_outliers is None) == (contamination is None):
        raise ValueError("pass exactly one of n_outliers or contamination")

    n_samples = oriented.size
    if contamination is not None:
        if not 0 < contamination < 1:
            raise ValueError("contamination must be between 0 and 1")
        n_outliers = math.ceil(n_samples * contamination)

    assert n_outliers is not None
    if not 1 <= n_outliers < n_samples:
        raise ValueError("n_outliers must be between 1 and len(scores) - 1")

    labels = np.zeros(n_samples, dtype=int)
    outlier_index = np.argsort(oriented, kind="mergesort")[-n_outliers:]
    labels[outlier_index] = 1
    return oriented, labels


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks, with larger values receiving larger ranks."""
    return rankdata(as_1d_finite(values, name="values"), method="average")
