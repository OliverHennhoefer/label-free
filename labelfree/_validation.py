"""Shared validation helpers."""

import math
import numbers

import numpy as np
from scipy.stats import rankdata

SCORE_POLARITIES = ("higher_is_anomalous", "higher_is_normal")


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
    """Return a finite 2D float array; 1D input is treated as one feature."""
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


def as_score_matrix(values, *, name: str, rows: str) -> np.ndarray:
    """Return a finite 2D matrix with at least two rows and two columns.

    ``rows`` names what the rows represent, for example ``"models"``.
    """
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D with shape (n_{rows}, n_samples)")
    if min(matrix.shape) < 2:
        raise ValueError(f"{name} must contain at least two {rows} and two samples")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def check_polarity(score_polarity: str) -> None:
    if score_polarity not in SCORE_POLARITIES:
        raise ValueError(
            "score_polarity must be 'higher_is_anomalous' or 'higher_is_normal'"
        )


def orient_scores(scores, *, score_polarity: str) -> np.ndarray:
    """Return scores as a float array in which larger values mean more anomalous.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Raw detector scores.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw values. ``"higher_is_normal"`` negates the scores,
        which suits APIs such as scikit-learn's ``decision_function``.
    """
    check_polarity(score_polarity)
    scores = as_1d_finite(scores, name="scores")
    return scores if score_polarity == "higher_is_anomalous" else -scores


def orient_matrix(score_matrix, *, score_polarity: str) -> np.ndarray:
    """Orient every row of a score matrix."""
    check_polarity(score_polarity)
    return score_matrix if score_polarity == "higher_is_anomalous" else -score_matrix


def resolve_outlier_count(
    n_samples: int,
    *,
    n_outliers: int | None,
    contamination: float | None,
) -> int:
    """Resolve an explicit split size from ``n_outliers`` or ``contamination``."""
    if (n_outliers is None) == (contamination is None):
        raise ValueError("pass exactly one of n_outliers or contamination")
    if contamination is not None:
        if not 0 < contamination < 1:
            raise ValueError("contamination must be between 0 and 1")
        n_outliers = math.ceil(n_samples * contamination)
    check_integer(n_outliers, name="n_outliers")
    if not 1 <= n_outliers < n_samples:
        raise ValueError("n_outliers must be between 1 and len(scores) - 1")
    return int(n_outliers)


def split_score_labels(
    scores,
    *,
    n_outliers: int | None = None,
    contamination: float | None = None,
    score_polarity: str = "higher_is_anomalous",
) -> tuple[np.ndarray, np.ndarray]:
    """Orient scores and label the ``n_outliers`` largest oriented scores as 1."""
    oriented = orient_scores(scores, score_polarity=score_polarity)
    n_outliers = resolve_outlier_count(
        oriented.size, n_outliers=n_outliers, contamination=contamination
    )
    labels = np.zeros(oriented.size, dtype=int)
    labels[np.argsort(oriented, kind="mergesort")[-n_outliers:]] = 1
    return oriented, labels


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks, with larger values receiving larger ranks."""
    return rankdata(as_1d_finite(values, name="values"), method="average")


def row_ranks(matrix: np.ndarray) -> np.ndarray:
    """Average ranks per row; the most anomalous sample gets rank ``n_samples``."""
    return rankdata(matrix, axis=1, method="average")


def check_integer(value, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")


def check_fraction(value: float, *, name: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1")


def check_positive(value: float, *, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive value")
    return value


def check_same_length(X: np.ndarray, scores: np.ndarray) -> None:
    if X.shape[0] != scores.size:
        raise ValueError("X and scores must have the same number of samples")
