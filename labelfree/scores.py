"""Score orientation and unification into outlier probabilities.

Implements the regularization and normalization framework of Kriegel,
Kröger, Schubert, and Zimek ("Interpreting and Unifying Outlier Scores",
SDM 2011), with the formulas of its reference implementation in ELKI.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf, gammainc

from labelfree._validation import as_1d_finite, check_polarity, orient_scores

__all__ = ["orient_scores", "unify_scores"]

UNIFY_METHODS = ("gaussian", "gamma", "linear")
INVERSIONS = ("linear", "log")


def unify_scores(
    scores,
    *,
    method: str = "gaussian",
    score_polarity: str = "higher_is_anomalous",
    inversion: str = "linear",
    baseline: float | None = None,
) -> np.ndarray:
    """Map detector scores to outlier probabilities in ``[0, 1]``.

    Scores are first regularized so that larger values mean more anomalous
    and inliers sit near zero, then normalized:

    - Inversion, for ``score_polarity="higher_is_normal"``: linear
      ``max(S) - S`` or logarithmic ``-log(S / max(S))``. The logarithmic form
      suits positive scores whose outliers have values near zero, such as
      ABOD.
    - Baseline, if given: ``max(0, S - baseline)``, for example
      ``baseline=1`` for LOF-type ratios whose inliers score about 1.
    - Normalization:

      * ``"gaussian"``: ``max(0, erf((S - mean) / (std * sqrt(2))))``.
      * ``"gamma"``: ``max(0, (F(S) - F(mean)) / (1 - F(mean)))`` with ``F``
        the Gamma CDF fitted by moments, ``k = mean^2 / var``,
        ``theta = var / mean``. Requires non-negative regularized scores.
      * ``"linear"``: ``(S - min) / (max - min)``.

    ``mean``, ``std``, and ``var`` are sample statistics of the regularized
    scores. The Gaussian and Gamma mappings send every score at or below the
    mean to 0, so only the upper tail receives probability mass.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Detector scores.
    method : {"gaussian", "gamma", "linear"}
        Normalization.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores.
    inversion : {"linear", "log"}
        Inversion used for ``"higher_is_normal"`` scores.
    baseline : float, optional
        Score of a typical inlier, subtracted before normalization.

    Returns
    -------
    numpy.ndarray of shape (n_samples,)
        Outlier probabilities, suitable for :func:`labelfree.metrics.ireos_score`
        and :func:`labelfree.metrics.sireos_score`.
    """
    if method not in UNIFY_METHODS:
        raise ValueError(f"method must be one of {UNIFY_METHODS}")
    if inversion not in INVERSIONS:
        raise ValueError(f"inversion must be one of {INVERSIONS}")
    check_polarity(score_polarity)
    values = as_1d_finite(scores, name="scores")

    if score_polarity == "higher_is_normal":
        if inversion == "linear":
            values = values.max() - values
        else:
            if np.any(values <= 0):
                raise ValueError("logarithmic inversion requires positive scores")
            values = -np.log(values / values.max())
    if baseline is not None:
        values = np.maximum(0.0, values - float(baseline))

    if method == "linear":
        low, high = values.min(), values.max()
        return np.zeros_like(values) if high == low else (values - low) / (high - low)

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if std == 0:
        return np.zeros_like(values)
    if method == "gaussian":
        return np.maximum(0.0, erf((values - mean) / (std * np.sqrt(2))))

    if np.any(values < 0) or mean <= 0:
        raise ValueError(
            "gamma scaling requires non-negative scores with a positive mean; "
            "use a baseline, an inversion, or method='gaussian'"
        )
    shape = mean**2 / std**2
    scale = std**2 / mean
    at_mean = gammainc(shape, mean / scale)
    return np.maximum(0.0, (gammainc(shape, values / scale) - at_mean) / (1 - at_mean))
