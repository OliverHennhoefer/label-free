"""Regression tests for degenerate inputs, invariances, and metric directions."""

import math
import warnings

import numpy as np
import pytest

from labelfree import borda_scores, robust_borda_scores
from labelfree.metrics import (
    asi_score,
    asoi_score,
    excess_mass_auc,
    expected_anomaly_gap_score,
    ireos_score,
    laplacian_score,
    relative_top_median_score,
    score_cluster_metrics,
)

RNG = np.random.default_rng(0)
X = RNG.normal(size=(60, 3))
TIED_SCORES = np.round(RNG.normal(size=60), 1)


@pytest.mark.parametrize("metric", [asi_score, asoi_score])
def test_feature_split_metrics_ignore_row_order_under_ties(metric):
    order = np.random.default_rng(1).permutation(len(X))

    original = metric(X, TIED_SCORES, n_outliers=6)
    permuted = metric(X[order], TIED_SCORES[order], n_outliers=6)

    assert permuted == pytest.approx(original)


def test_split_puts_every_cutoff_tie_in_the_anomaly_group():
    X_line = [[0.0], [1.0], [2.0], [3.0], [10.0]]
    tied = asoi_score(X_line, [0, 0, 1, 1, 2], n_outliers=2)
    explicit = asoi_score(X_line, [0, 0, 1, 1, 2], n_outliers=3)

    assert tied == explicit


@pytest.mark.parametrize(
    "call",
    [
        lambda s: asoi_score(X, s, n_outliers=6),
        lambda s: asi_score(X, s, n_outliers=6),
        lambda s: score_cluster_metrics(s, n_outliers=6),
    ],
)
def test_constant_scores_do_not_induce_a_split(call):
    with pytest.raises(ValueError, match="do not split"):
        call(np.full(len(X), 0.3))


def test_calinski_harabasz_rates_a_perfect_split_as_best():
    perfect = score_cluster_metrics([0, 0, 0, 0, 0, 1, 1], n_outliers=2)
    noisy = score_cluster_metrics([0, 0.1, 0, 0.2, 0, 1, 1.1], n_outliers=2)

    assert perfect["calinski_harabasz"] == math.inf
    assert perfect["calinski_harabasz"] > noisy["calinski_harabasz"]


@pytest.mark.parametrize("offset", [1e4, 1e6, 1e8, 1e10])
def test_expected_anomaly_gap_is_offset_invariant(offset):
    scores = np.random.default_rng(2).normal(size=200)

    shifted = expected_anomaly_gap_score(scores + offset)

    assert shifted == pytest.approx(expected_anomaly_gap_score(scores), rel=1e-6)


@pytest.mark.parametrize("value", [0.0, 0.1, 0.3, 1.0, 7.7, -1e9])
def test_laplacian_score_is_worst_for_any_constant(value):
    assert laplacian_score(X, np.full(len(X), value)) == math.inf


def test_laplacian_score_is_affine_invariant():
    scores = np.random.default_rng(3).normal(size=len(X))

    base = laplacian_score(X, scores)

    assert laplacian_score(X, 1e6 * scores + 1e9) == pytest.approx(base)
    assert laplacian_score(X, 1e-9 * scores) == pytest.approx(base)


def test_relative_top_median_rejects_negative_median():
    scores = np.random.default_rng(4).normal(size=100) - 10

    with pytest.raises(ValueError, match="non-negative median"):
        relative_top_median_score(scores)


def test_ireos_gamma_search_skips_duplicated_candidates():
    data = np.random.default_rng(5).normal(size=(20, 2))
    data[1] = data[0]
    weights = np.zeros(20)
    weights[[0, 5]] = 1.0

    with pytest.warns(RuntimeWarning, match="duplicate another row"):
        result = ireos_score(data, weights)

    assert 0 <= result <= 1


def test_ireos_gamma_search_rejects_only_duplicated_candidates():
    data = np.random.default_rng(5).normal(size=(20, 2))
    data[1] = data[0]
    weights = np.zeros(20)
    weights[0] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="duplicates another row"):
            ireos_score(data, weights)


@pytest.mark.parametrize("aggregate", [borda_scores, robust_borda_scores])
@pytest.mark.parametrize("directions", [["higher", "lower"], [1, -1], [1.0, 0.0]])
def test_borda_rejects_non_boolean_directions(aggregate, directions):
    with pytest.raises(ValueError, match="bool"):
        aggregate([[1, 2, 3], [1, 2, 3]], higher_is_better=directions)


def test_borda_respects_mixed_directions():
    points = borda_scores([[1, 2, 3], [1, 2, 3]], higher_is_better=[True, False])

    assert np.all(points == points[0])


def test_borda_rejects_wrong_direction_count():
    with pytest.raises(ValueError, match="one direction per metric"):
        borda_scores([[1, 2, 3], [1, 2, 3]], higher_is_better=[True, False, True])


def test_excess_mass_auc_rejects_levels_starting_below_em_min():
    rng = np.random.default_rng(6)
    data, reference = rng.normal(size=200), rng.uniform(-4, 4, size=2000)

    with pytest.raises(ValueError, match="em_min"):
        excess_mass_auc(
            data, reference, support_volume=8.0, levels=np.linspace(0.2, 1, 50)
        )
