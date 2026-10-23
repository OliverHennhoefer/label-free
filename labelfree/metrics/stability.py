"""Ranking stability of an anomaly detector across refits (Perini et al., 2020)."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as beta_distribution

from labelfree._validation import as_score_matrix, orient_matrix, row_ranks


def ranking_stability_score(
    score_matrix,
    *,
    contamination: float,
    psi: float = 0.8,
    score_polarity: str = "higher_is_anomalous",
) -> float:
    """Stability of sample rankings across repeated runs of one detector.

    Implements the stability measure of Perini, Galvin, and Vercruyssen
    ("A Ranking Stability Measure for Quantifying the Robustness of Anomaly
    Detection Methods", ECML PKDD workshops 2020) for precomputed runs. Each
    row holds the scores of one refit (for example on a random training
    subsample) for the same evaluation samples. Rankings are normalized to
    ``(0, 1]``; each sample's instability is its rank standard deviation,
    weighted by the Beta-distribution mass between its minimum and maximum
    rank and divided by the standard deviation of a uniformly random ranking.
    The Beta parameters put ``psi`` of the mass on the top ``2 *
    contamination`` ranks (the authors' default ``beta_flavor=2``).

    Parameters
    ----------
    score_matrix : array-like of shape (n_runs, n_samples)
        One score row per run, all scoring the same samples in the same order.
    contamination : float
        Expected anomaly proportion, in ``(0, 0.5)``.
    psi : float
        Beta mass on the top ``2 * contamination`` ranks, in ``(0, 1)``.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores.

    Returns
    -------
    float
        Stability in ``[0, 1]``. Higher is better. Measures robustness, not
        accuracy.
    """
    matrix = as_score_matrix(score_matrix, name="score_matrix", rows="runs")
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5")
    if not 0 < psi < 1:
        raise ValueError("psi must be between 0 and 1")

    n_samples = matrix.shape[1]
    ranks = row_ranks(orient_matrix(matrix, score_polarity=score_polarity))
    ranks = ranks / n_samples

    alpha, beta = _beta_parameters(contamination, psi)
    weight = beta_distribution.cdf(ranks.max(axis=0), alpha, beta)
    weight -= beta_distribution.cdf(ranks.min(axis=0), alpha, beta)
    random_std = math.sqrt((n_samples + 1) * (n_samples - 1) / (12 * n_samples**2))
    instability = np.minimum(1.0, weight * ranks.std(axis=0) / random_std)
    return float(1.0 - instability.mean())


def _beta_parameters(contamination: float, psi: float) -> tuple[float, float]:
    """Fit Beta(alpha, beta) with mass ``psi`` above ``1 - 2 * contamination``.

    The linear constraint fixes the mode at ``1 - contamination``.
    """
    cutoff = 1 - 2 * contamination

    def objective(params: np.ndarray) -> float:
        alpha, beta = params
        return float(((1.0 - psi) - beta_distribution.cdf(cutoff, alpha, beta)) ** 2)

    constraint = {
        "type": "eq",
        "fun": lambda params: (
            contamination * params[0]
            - (1 - contamination) * params[1]
            - (2 * contamination - 1)
        ),
    }
    result = minimize(
        objective,
        x0=np.array([1.0, 1.0]),
        bounds=[(1.0, None), (1.0, None)],
        constraints=[constraint],
        method="SLSQP",
    )
    if not result.success:
        raise RuntimeError("failed to fit beta weighting parameters")
    return float(result.x[0]), float(result.x[1])
