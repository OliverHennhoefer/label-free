"""Consensus-based model evaluation over candidate score matrices.

Every function takes a ``(n_models, n_samples)`` matrix holding one score row
per candidate model on the same samples and returns one higher-is-better value
per model. Raw scores are never compared across models; only rankings are.
The strategies follow the unsupervised outlier model selection study of Ma et
al. (2023) and its UOMS reference code.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import kendalltau, rankdata
from sklearn.metrics import ndcg_score

from labelfree._validation import (
    as_score_matrix,
    check_integer,
    orient_matrix,
    row_ranks,
)

SIMILARITIES = ("kendall", "spearman", "ndcg")


def model_centrality_scores(
    score_matrix,
    *,
    similarity: str = "kendall",
    n_neighbors: int | None = None,
    n_peers: int | None = None,
    random_state=None,
    score_polarity: str = "higher_is_anomalous",
) -> np.ndarray:
    """Model Centrality (MC): average ranking similarity to other models.

    MC (Lin et al., 2020, adapted by Ma et al., 2023) scores model ``i`` with
    ``MC_i = 1 / (N - 1) * sum_{j != i} B_ij``, the mean similarity of its
    sample ranking to every other model's ranking.

    Two published variants restrict the peers:

    - ``n_neighbors=K`` averages over the ``K`` most similar models, the
      K-nearest-neighbor centrality of Goswami et al. (ICLR 2023).
    - ``n_peers=P`` averages over ``P`` peers drawn uniformly without
      replacement, the lightweight MCS variant of Ma et al. (2023), who use
      ``P = round(sqrt(N))``.

    Parameters
    ----------
    score_matrix : array-like of shape (n_models, n_samples)
        One score row per candidate model.
    similarity : {"kendall", "spearman", "ndcg"}
        Ranking similarity ``B_ij``. Kendall's tau-b and Spearman's rho are
        symmetric; NDCG is symmetrized by averaging both directions.
    n_neighbors : int, optional
        Number of most similar models to average over.
    n_peers : int, optional
        Number of random peers to average over. Mutually exclusive with
        ``n_neighbors``.
    random_state : int, numpy.random.Generator, or None
        Seed for ``n_peers`` sampling.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores.

    Returns
    -------
    numpy.ndarray of shape (n_models,)
        Centrality per model. Higher is better.
    """
    ranks = _rank_matrix(score_matrix, score_polarity=score_polarity)
    n_models = ranks.shape[0]
    if n_neighbors is not None and n_peers is not None:
        raise ValueError("pass at most one of n_neighbors or n_peers")

    if n_peers is not None:
        _check_count(n_peers, n_models - 1, name="n_peers")
        rng = np.random.default_rng(random_state)
        pair = _pair_similarity(ranks, similarity)
        result = np.empty(n_models)
        for i in range(n_models):
            others = np.delete(np.arange(n_models), i)
            peers = rng.choice(others, size=n_peers, replace=False)
            result[i] = np.mean([pair(i, j) for j in peers])
        return result

    matrix = similarity_matrix(ranks, similarity=similarity)
    off_diagonal = matrix[~np.eye(n_models, dtype=bool)].reshape(n_models, -1)
    if n_neighbors is None:
        return off_diagonal.mean(axis=1)
    _check_count(n_neighbors, n_models - 1, name="n_neighbors")
    nearest = -np.partition(-off_diagonal, n_neighbors - 1, axis=1)[:, :n_neighbors]
    return nearest.mean(axis=1)


def udr_scores(
    score_matrix,
    groups,
    *,
    similarity: str = "spearman",
    n_peers: int | None = None,
    random_state=None,
    score_polarity: str = "higher_is_anomalous",
) -> np.ndarray:
    """Unsupervised Disentanglement Ranking (UDR) adapted to outlier models.

    UDR (Duan et al., ICLR 2020) assumes that good settings give consistent
    results across random seeds. Ma et al. (2023) transfer it to outlier model
    selection by comparing each model with other configurations of the same
    detector family. Model ``i`` scores the median similarity to its peers.

    Parameters
    ----------
    score_matrix : array-like of shape (n_models, n_samples)
        One score row per candidate model.
    groups : array-like of shape (n_models,)
        Group label per model, for example the detector family. Every group
        needs at least two models.
    similarity : {"spearman", "kendall", "ndcg"}
        Ranking similarity. Duan et al. use Spearman's rho.
    n_peers : int, optional
        Number of same-group peers drawn without replacement per model. The
        default uses all same-group peers. Ma et al. use one less than the
        smallest group size.
    random_state : int, numpy.random.Generator, or None
        Seed for ``n_peers`` sampling.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores.

    Returns
    -------
    numpy.ndarray of shape (n_models,)
        Median same-group similarity per model. Higher is better.
    """
    ranks = _rank_matrix(score_matrix, score_polarity=score_polarity)
    groups = np.asarray(groups)
    if groups.shape != (ranks.shape[0],):
        raise ValueError("groups must contain one label per model")
    labels, counts = np.unique(groups, return_counts=True)
    if counts.min() < 2:
        raise ValueError(
            f"every group needs at least two models; group {labels[counts.argmin()]!r}"
            " has one"
        )
    if n_peers is not None:
        _check_count(n_peers, int(counts.min()) - 1, name="n_peers")

    rng = np.random.default_rng(random_state)
    pair = _pair_similarity(ranks, similarity)
    result = np.empty(ranks.shape[0])
    for i in range(ranks.shape[0]):
        peers = np.flatnonzero(groups == groups[i])
        peers = peers[peers != i]
        if n_peers is not None:
            peers = rng.choice(peers, size=n_peers, replace=False)
        result[i] = np.median([pair(i, j) for j in peers])
    return result


def hits_scores(
    score_matrix,
    *,
    max_iter: int = 500,
    tol: float = 1e-10,
    score_polarity: str = "higher_is_anomalous",
) -> np.ndarray:
    """HITS hub scores of models in a model-sample bipartite graph.

    Ma et al. (2023) connect every model ``i`` to every sample ``j`` with
    weight ``1 / r_ij``, where ``r_ij`` is the position of ``j`` in model
    ``i``'s ranking and the most anomalous sample has position 1. Hub and
    authority vectors (Kleinberg, 1999) are updated alternately until
    convergence; models are scored by their hub value.

    Parameters
    ----------
    score_matrix : array-like of shape (n_models, n_samples)
        One score row per candidate model.
    max_iter : int
        Maximum number of power iterations.
    tol : float
        Convergence tolerance on the Euclidean change of the hub vector.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores.

    Returns
    -------
    numpy.ndarray of shape (n_models,)
        Unit-norm hub scores. Higher is better.
    """
    check_integer(max_iter, name="max_iter")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be non-negative")
    weights = _inverse_rank_matrix(score_matrix, score_polarity=score_polarity)

    hub = np.full(weights.shape[0], 1 / np.sqrt(weights.shape[0]))
    for _ in range(max_iter):
        authority = _unit(weights.T @ hub)
        next_hub = _unit(weights @ authority)
        converged = np.linalg.norm(next_hub - hub) <= tol
        hub = next_hub
        if converged:
            break
    return hub


def ens_scores(
    score_matrix,
    *,
    similarity: str = "kendall",
    score_polarity: str = "higher_is_anomalous",
) -> np.ndarray:
    """Selective-ensemble consensus (ENS) scores.

    Algorithm 1 of Ma et al. (2023), adapted from the selective outlier
    ensembles of Rayana and Akoglu (2016). Scores become inverse ranks
    ``1 / r`` (most anomalous sample: ``r = 1``) and their average is the
    initial pseudo ground truth. Repeatedly, the remaining model ``m`` most
    similar to the target is accepted into the ensemble ``E`` when
    ``sim(avg(E + m), target) * |E| >= C``, where ``C`` sums the similarities
    of accepted models; the target then becomes ``avg(E)``. The loop stops at
    the first rejection. Each model scores its similarity to the final target.

    Parameters
    ----------
    score_matrix : array-like of shape (n_models, n_samples)
        One score row per candidate model.
    similarity : {"kendall", "spearman", "ndcg"}
        Rank correlation used for sorting, acceptance, and the final scores.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores.

    Returns
    -------
    numpy.ndarray of shape (n_models,)
        Similarity of each model to the pseudo ground truth. Higher is better.
    """
    _check_similarity(similarity)
    inverse_ranks = _inverse_rank_matrix(score_matrix, score_polarity=score_polarity)
    target = inverse_ranks.mean(axis=0)
    remaining = list(range(inverse_ranks.shape[0]))
    ensemble: list[int] = []
    accepted_similarity = 0.0

    while remaining:
        values = [_similarity(inverse_ranks[m], target, similarity) for m in remaining]
        best = int(np.argmax(values))
        candidate = remaining[best]
        merged = inverse_ranks[ensemble + [candidate]].mean(axis=0)
        if _similarity(merged, target, similarity) * len(ensemble) < (
            accepted_similarity
        ):
            break
        ensemble.append(candidate)
        remaining.pop(best)
        accepted_similarity += values[best]
        target = inverse_ranks[ensemble].mean(axis=0)

    return np.array(
        [_similarity(row, target, similarity) for row in inverse_ranks], dtype=float
    )


def similarity_matrix(ranks: np.ndarray, *, similarity: str) -> np.ndarray:
    """Pairwise ranking similarity between the rows of a rank matrix."""
    _check_similarity(similarity)
    n_models = ranks.shape[0]
    if similarity == "spearman":
        return _corrcoef(ranks)
    matrix = np.eye(n_models)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            matrix[i, j] = matrix[j, i] = _similarity(ranks[i], ranks[j], similarity)
    return matrix


def _similarity(left: np.ndarray, right: np.ndarray, similarity: str) -> float:
    """Rank-based similarity; inputs are ranked first, so any scale works."""
    if similarity == "kendall":
        value = kendalltau(left, right).statistic
        return 0.0 if np.isnan(value) else float(value)
    left, right = rankdata(left), rankdata(right)
    if similarity == "spearman":
        return float(_corrcoef(np.vstack([left, right]))[0, 1])
    return (_ndcg(left, right) + _ndcg(right, left)) / 2


def _ndcg(relevance_source: np.ndarray, ranking_source: np.ndarray) -> float:
    # UOMS uses min-max scaled ranks as graded relevance.
    relevance = relevance_source - relevance_source.min()
    if relevance.max() == 0:
        return 0.0
    relevance = relevance / relevance.max()
    return float(ndcg_score(relevance[None, :], ranking_source[None, :]))


def _corrcoef(rows: np.ndarray) -> np.ndarray:
    # Pearson correlation of rank rows; constant rows correlate as 0.
    centered = rows - rows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    safe = np.where(norms == 0, 1.0, norms)
    matrix = (centered @ centered.T) / np.outer(safe, safe)
    matrix[norms == 0, :] = 0.0
    matrix[:, norms == 0] = 0.0
    np.fill_diagonal(matrix, 1.0)
    return np.clip(matrix, -1.0, 1.0)


def _pair_similarity(ranks: np.ndarray, similarity: str) -> Callable[[int, int], float]:
    _check_similarity(similarity)
    cache: dict[tuple[int, int], float] = {}

    def pair(i: int, j: int) -> float:
        key = (min(i, j), max(i, j))
        if key not in cache:
            cache[key] = _similarity(ranks[key[0]], ranks[key[1]], similarity)
        return cache[key]

    return pair


def _rank_matrix(score_matrix, *, score_polarity: str) -> np.ndarray:
    matrix = as_score_matrix(score_matrix, name="score_matrix", rows="models")
    return row_ranks(orient_matrix(matrix, score_polarity=score_polarity))


def _inverse_rank_matrix(score_matrix, *, score_polarity: str) -> np.ndarray:
    ranks = _rank_matrix(score_matrix, score_polarity=score_polarity)
    return 1.0 / (ranks.shape[1] + 1 - ranks)


def _check_similarity(similarity: str) -> None:
    if similarity not in SIMILARITIES:
        raise ValueError(f"similarity must be one of {SIMILARITIES}")


def _check_count(value: int, upper: int, *, name: str) -> None:
    check_integer(value, name=name)
    if not 1 <= value <= upper:
        raise ValueError(f"{name} must be between 1 and {upper}")


def _unit(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return values if norm == 0 else values / norm
