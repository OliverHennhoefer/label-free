"""Consensus metrics over candidate anomaly-score matrices."""

from __future__ import annotations

import numpy as np

from labelfree.utils.validation import average_ranks, orient_scores


def model_centrality_scores(
    score_matrix,
    *,
    score_polarity: str = "higher_is_anomalous",
) -> np.ndarray:
    """Average pairwise rank agreement for model-by-sample scores. Higher is better."""
    ranks = _rank_matrix(score_matrix, score_polarity=score_polarity)
    n_models = ranks.shape[0]
    similarity = np.eye(n_models)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            corr = _pearson(ranks[i], ranks[j])
            similarity[i, j] = corr
            similarity[j, i] = corr
    return (similarity.sum(axis=1) - 1.0) / (n_models - 1)


def average_rank_consensus_scores(
    score_matrix,
    *,
    score_polarity: str = "higher_is_anomalous",
) -> np.ndarray:
    """Agreement with the average rank across model-by-sample scores. Higher is better."""
    ranks = _rank_matrix(score_matrix, score_polarity=score_polarity)
    consensus = ranks.mean(axis=0)
    return np.array([_pearson(row, consensus) for row in ranks], dtype=float)


def hits_model_scores(
    score_matrix,
    *,
    score_polarity: str = "higher_is_anomalous",
    max_iter: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    """HITS hubness for a model-by-sample score matrix. Higher is better."""
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    ranks = _rank_matrix(score_matrix, score_polarity=score_polarity)
    weights = ranks / ranks.shape[1]
    hub = np.full(ranks.shape[0], 1 / np.sqrt(ranks.shape[0]))

    for _ in range(max_iter):
        authority = _unit(weights.T @ hub)
        next_hub = _unit(weights @ authority)
        if np.linalg.norm(next_hub - hub) <= tol:
            hub = next_hub
            break
        hub = next_hub
    return hub


def _rank_matrix(score_matrix, *, score_polarity: str) -> np.ndarray:
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("score_matrix must be 2D with shape (n_models, n_samples)")
    if min(matrix.shape) < 2:
        raise ValueError(
            "score_matrix must contain at least two models and two samples"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("score_matrix contains non-finite values")
    return np.vstack(
        [
            average_ranks(orient_scores(row, score_polarity=score_polarity))
            for row in matrix
        ]
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator == 0:
        return 0.0
    return float(np.dot(x, y) / denominator)


def _unit(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm == 0:
        return values
    return values / norm
