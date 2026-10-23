from itertools import combinations

import numpy as np
import pytest
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import MinMaxScaler

from labelfree.metrics import (
    ens_scores,
    hits_scores,
    model_centrality_scores,
    udr_scores,
)

AGREEING = np.array(
    [
        [0.0, 1.0, 2.0, 3.0],
        [0.1, 1.1, 2.1, 3.1],
        [3.0, 2.0, 1.0, 0.0],
    ]
)
RANDOM = np.random.default_rng(3).normal(size=(6, 25))


def test_model_centrality_matches_uoms_kendall_definition():
    n_models = RANDOM.shape[0]
    expected = np.zeros(n_models)
    for i, j in combinations(range(n_models), 2):
        tau = kendalltau(RANDOM[i], RANDOM[j]).statistic
        expected[i] += tau
        expected[j] += tau
    expected /= n_models - 1

    assert model_centrality_scores(RANDOM) == pytest.approx(expected)


def test_model_centrality_spearman_matches_scipy():
    rho = spearmanr(RANDOM, axis=1).statistic
    expected = (rho.sum(axis=1) - 1) / (RANDOM.shape[0] - 1)

    assert model_centrality_scores(RANDOM, similarity="spearman") == pytest.approx(
        expected
    )


def test_model_centrality_ndcg_matches_uoms_relevance():
    # UOMS uses min-max scaled ranks as relevance and averages both directions.
    relevance = MinMaxScaler().fit_transform(rankdata(RANDOM, axis=1).T).T
    n_models = RANDOM.shape[0]
    expected = np.zeros(n_models)
    for i, j in combinations(range(n_models), 2):
        value = (
            ndcg_score(relevance[[i]], relevance[[j]])
            + ndcg_score(relevance[[j]], relevance[[i]])
        ) / 2
        expected[i] += value
        expected[j] += value
    expected /= n_models - 1

    assert model_centrality_scores(RANDOM, similarity="ndcg") == pytest.approx(expected)


def test_model_centrality_nearest_neighbors_follows_goswami():
    tau = np.array(
        [[kendalltau(a, b).statistic for b in RANDOM] for a in RANDOM], dtype=float
    )
    np.fill_diagonal(tau, -np.inf)
    expected = np.sort(tau, axis=1)[:, -2:].mean(axis=1)

    assert model_centrality_scores(RANDOM, n_neighbors=2) == pytest.approx(expected)


def test_model_centrality_sampled_peers():
    full = model_centrality_scores(RANDOM)
    all_peers = model_centrality_scores(RANDOM, n_peers=5, random_state=0)
    sampled = model_centrality_scores(RANDOM, n_peers=2, random_state=1)

    assert all_peers == pytest.approx(full)
    assert sampled == pytest.approx(
        model_centrality_scores(RANDOM, n_peers=2, random_state=1)
    )
    with pytest.raises(ValueError, match="at most one"):
        model_centrality_scores(RANDOM, n_peers=2, n_neighbors=2)
    with pytest.raises(ValueError, match="n_peers"):
        model_centrality_scores(RANDOM, n_peers=6)


def test_model_centrality_rewards_rank_agreement():
    scores = model_centrality_scores(AGREEING)

    assert scores[0] == pytest.approx(scores[1])
    assert scores[0] > scores[2]


def test_udr_is_median_same_group_similarity():
    groups = ["a", "a", "a", "b", "b", "b"]
    result = udr_scores(RANDOM, groups)

    expected = []
    for i, group in enumerate(groups):
        peers = [j for j, other in enumerate(groups) if other == group and j != i]
        expected.append(
            np.median([spearmanr(RANDOM[i], RANDOM[j]).statistic for j in peers])
        )
    assert result == pytest.approx(expected)


def test_udr_sampling_and_validation():
    groups = [0, 0, 0, 1, 1, 1]
    full = udr_scores(RANDOM, groups, similarity="kendall")

    assert udr_scores(
        RANDOM, groups, similarity="kendall", n_peers=2, random_state=0
    ) == pytest.approx(full)
    with pytest.raises(ValueError, match="at least two models"):
        udr_scores(RANDOM, [0, 0, 0, 0, 0, 1])
    with pytest.raises(ValueError, match="one label per model"):
        udr_scores(RANDOM, [0, 1])
    with pytest.raises(ValueError, match="n_peers"):
        udr_scores(RANDOM, groups, n_peers=3)


def _reference_hits(score_matrix, n_iter=500):
    # UOMS hits.py loop with edge weights 1 / r, r = 1 for the top outlier.
    rank_mat = rankdata(-score_matrix.T, axis=0)
    inv_rank_mat = 1 / rank_mat
    hub = np.full([score_matrix.shape[0], 1], 1 / score_matrix.shape[0])
    for _ in range(n_iter):
        auth = inv_rank_mat @ hub
        auth = auth / np.linalg.norm(auth)
        hub = inv_rank_mat.T @ auth
        hub = hub / np.linalg.norm(hub)
    return hub.ravel()


def test_hits_matches_inverse_rank_reference():
    assert hits_scores(RANDOM) == pytest.approx(_reference_hits(RANDOM), abs=1e-8)
    assert np.linalg.norm(hits_scores(RANDOM)) == pytest.approx(1.0)


def test_hits_emphasizes_agreement_on_top_outliers():
    # Models 0 and 1 agree on the top outlier; model 2 agrees on the inliers.
    score_matrix = np.array(
        [
            [9.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [9.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [9.0, 1.0, 5.0, 2.0, 4.0, 3.0],
        ]
    )
    scores = hits_scores(score_matrix)

    assert scores[2] < min(scores[0], scores[1], scores[3])


def test_hits_validates_iteration_controls():
    with pytest.raises(ValueError, match="max_iter"):
        hits_scores([[0.0, 1.0], [1.0, 0.0]], max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        hits_scores([[0.0, 1.0], [1.0, 0.0]], tol=-1.0)


def test_ens_selects_the_agreeing_majority():
    score_matrix = np.vstack([AGREEING, [[0.2, 1.2, 2.2, 3.2]]])
    scores = ens_scores(score_matrix)

    assert scores[[0, 1, 3]] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(-1.0)


def test_ens_follows_algorithm_one():
    inverse = 1 / rankdata(-RANDOM, axis=1)

    def corr(a, b):
        return kendalltau(a, b).statistic

    target = inverse.mean(axis=0)
    remaining, ensemble, accepted = list(range(len(RANDOM))), [], 0.0
    while remaining:
        values = [corr(inverse[m], target) for m in remaining]
        best = int(np.argmax(values))
        merged = inverse[ensemble + [remaining[best]]].mean(axis=0)
        if corr(merged, target) * len(ensemble) < accepted:
            break
        ensemble.append(remaining.pop(best))
        accepted += values[best]
        target = inverse[ensemble].mean(axis=0)
    expected = [corr(row, target) for row in inverse]

    assert ens_scores(RANDOM) == pytest.approx(expected)


@pytest.mark.parametrize(
    "function",
    [model_centrality_scores, hits_scores, ens_scores],
)
def test_consensus_metrics_apply_score_polarity(function):
    result = function(-RANDOM, score_polarity="higher_is_normal")

    assert result == pytest.approx(function(RANDOM))


def test_consensus_metrics_validate_inputs():
    with pytest.raises(ValueError, match="2D"):
        model_centrality_scores([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="similarity"):
        model_centrality_scores(RANDOM, similarity="pearson")
    with pytest.raises(ValueError, match="score_polarity"):
        ens_scores(RANDOM, score_polarity="up")
