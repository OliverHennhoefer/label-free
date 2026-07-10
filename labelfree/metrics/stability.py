"""Ranking stability metrics for repeated anomaly-score rankings."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as beta_distribution

from labelfree.utils.validation import average_ranks, orient_scores


def ranking_stability_score(
    score_matrix,
    *,
    contamination: float,
    psi: float = 0.8,
    score_polarity: str = "higher_is_anomalous",
) -> float:
    """Stability across repeated score rows for the same samples. Higher is better."""
    ranks = _normalized_rank_matrix(score_matrix, score_polarity=score_polarity)
    _check_contamination(contamination)
    if not 0 < psi < 1:
        raise ValueError("psi must be between 0 and 1")

    alpha, beta = _beta_parameters(contamination, psi)
    rank_min = ranks.min(axis=0)
    rank_max = ranks.max(axis=0)
    rank_std = ranks.std(axis=0)
    weight = beta_distribution.cdf(rank_max, alpha, beta) - beta_distribution.cdf(
        rank_min,
        alpha,
        beta,
    )
    n_samples = ranks.shape[1]
    random_std = math.sqrt((n_samples + 1) * (n_samples - 1) / (12 * n_samples**2))
    instability = np.minimum(1.0, weight * rank_std / random_std)
    return float(np.clip(1.0 - instability.mean(), 0.0, 1.0))


def top_k_stability_score(
    score_matrix,
    *,
    top_k: int | None = None,
    top_fraction: float | None = None,
    score_polarity: str = "higher_is_anomalous",
) -> float:
    """Pairwise top-k overlap across repeated score rows. Higher is better."""
    matrix = _score_matrix(score_matrix)
    if (top_k is None) == (top_fraction is None):
        raise ValueError("pass exactly one of top_k or top_fraction")
    if top_fraction is not None:
        if not 0 < top_fraction < 1:
            raise ValueError("top_fraction must be between 0 and 1")
        top_k = math.ceil(matrix.shape[1] * top_fraction)

    assert top_k is not None
    if not 1 <= top_k <= matrix.shape[1]:
        raise ValueError("top_k must be between 1 and the number of samples")

    top_sets = []
    for row in matrix:
        oriented = orient_scores(row, score_polarity=score_polarity)
        top_sets.append(set(np.argsort(oriented, kind="mergesort")[-top_k:]))

    overlaps = [
        len(left & right) / len(left | right)
        for left, right in combinations(top_sets, 2)
    ]
    return float(np.mean(overlaps))


def _normalized_rank_matrix(score_matrix, *, score_polarity: str) -> np.ndarray:
    matrix = _score_matrix(score_matrix)
    ranks = np.vstack(
        [
            average_ranks(orient_scores(row, score_polarity=score_polarity))
            for row in matrix
        ]
    )
    return ranks / matrix.shape[1]


def _score_matrix(score_matrix) -> np.ndarray:
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("score_matrix must be 2D with shape (n_runs, n_samples)")
    if min(matrix.shape) < 2:
        raise ValueError("score_matrix must contain at least two runs and two samples")
    if not np.isfinite(matrix).all():
        raise ValueError("score_matrix contains non-finite values")
    return matrix


def _check_contamination(contamination: float) -> None:
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5")


def _beta_parameters(contamination: float, psi: float) -> tuple[float, float]:
    cutoff = 1 - 2 * contamination

    def objective(params: np.ndarray) -> float:
        alpha, beta = params
        return float(((1.0 - psi) - beta_distribution.cdf(cutoff, alpha, beta)) ** 2)

    constraint = {
        "type": "eq",
        "fun": lambda params: contamination * params[0]
        - (1 - contamination) * params[1]
        - (2 * contamination - 1),
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
