import numpy as np
import pytest
from scipy.optimize import minimize as reference_minimize
from scipy.spatial.distance import cdist
from scipy.special import expit

import labelfree.metrics.ireos as ireos_module
from labelfree.metrics import ireos_score, ireos_scores, ireos_significance


X = np.array([[0.0], [0.1], [0.2], [10.0]])


def test_kernel_logistic_probability_matches_reference_values():
    distances = cdist(X, X, metric="sqeuclidean")
    costs = np.full(len(X), 100.0)

    at_zero = ireos_module._kernel_logistic_probability(distances, 0.0, 3, costs)
    at_one = ireos_module._kernel_logistic_probability(distances, 1.0, 3, costs)

    assert at_zero == pytest.approx(0.25, abs=1e-7)

    kernel = np.exp(-distances)
    labels = np.array([-1.0, -1.0, -1.0, 1.0])

    def reference_objective(parameters):
        coefficients = parameters[:-1]
        scores = kernel @ coefficients + parameters[-1]
        return (
            0.5 * coefficients @ kernel @ coefficients
            + 100 * np.logaddexp(0, -labels * scores).sum()
        )

    reference = reference_minimize(
        reference_objective,
        np.ones(len(X) + 1),
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    assert reference.success
    expected = expit(kernel[3] @ reference.x[:-1] + reference.x[-1])
    assert at_one == pytest.approx(expected, abs=1e-6)


def test_ireos_rewards_an_isolated_high_probability_candidate():
    isolated = ireos_score(X, [0, 0, 0, 1], gamma_max=1.0)
    clustered = ireos_score(X, [1, 0, 0, 0], gamma_max=1.0)

    assert isolated == pytest.approx(0.902612344, abs=1e-6)
    assert isolated > clustered


def test_ireos_scores_reuses_curves_and_applies_score_weights(monkeypatch):
    calls = []

    def fake_probability(distances, gamma, candidate, costs):
        calls.append((gamma, candidate))
        return 0.2 + 0.1 * candidate

    monkeypatch.setattr(
        ireos_module,
        "_kernel_logistic_probability",
        fake_probability,
    )
    probabilities = np.array([[0.25, 0.75, 0, 0], [0.75, 0.25, 0, 0]], dtype=float)

    assert ireos_scores(X, probabilities, gamma_max=2.0) == pytest.approx(
        [0.275, 0.225]
    )
    assert len(calls) == 10


def test_ireos_applies_clump_costs():
    clump = np.array([[0.0], [0.1], [10.0], [10.1]])
    without_clumps = ireos_score(clump, [0, 0, 1, 1], gamma_max=1.0, max_clump_size=1)
    with_clumps = ireos_score(clump, [0, 0, 1, 1], gamma_max=1.0, max_clump_size=2)
    assert with_clumps > without_clumps


def test_ireos_scores_uses_one_gamma_range(monkeypatch):
    gamma_searches = []
    auc_ranges = []

    def fake_gamma(distances, probabilities, **kwargs):
        gamma_searches.append(probabilities.copy())
        return 2.0

    def fake_aucs(kernel_features, candidates, costs, *, gamma_max, **kwargs):
        auc_ranges.append(gamma_max)
        return np.full(costs.size, gamma_max)

    monkeypatch.setattr(ireos_module, "_find_gamma_max", fake_gamma)
    monkeypatch.setattr(ireos_module, "_candidate_aucs", fake_aucs)
    probabilities = np.array([[0, 0, 0, 1], [1, 0, 0, 0]], dtype=float)

    assert ireos_scores(X, probabilities) == pytest.approx([1.0, 1.0])
    assert len(gamma_searches) == 1
    assert np.array_equal(gamma_searches[0], probabilities)
    assert auc_ranges == [2.0]


def test_automatic_gamma_max_separates_candidates():
    distances = cdist(X, X, metric="sqeuclidean")
    probabilities = np.array([[0.0, 0.0, 0.0, 1.0]])
    gamma = ireos_module._find_gamma_max(
        ireos_module._KernelFeatures(distances),
        probabilities,
        n_features=1,
        penalty_cost=100.0,
    )

    probability = ireos_module._kernel_logistic_probability(
        distances, gamma, 3, np.full(len(X), 100.0)
    )
    assert gamma == pytest.approx(0.001)
    assert probability > 0.5


def test_automatic_gamma_max_reports_failure(monkeypatch):
    monkeypatch.setattr(ireos_module, "_MAX_GAMMA_STEPS", 2)
    monkeypatch.setattr(
        ireos_module,
        "_kernel_logistic_probability",
        lambda *args, **kwargs: 0.5,
    )
    distances = cdist(X, X, metric="sqeuclidean")

    with pytest.raises(RuntimeError, match="gamma_max search"):
        ireos_module._find_gamma_max(
            ireos_module._KernelFeatures(distances),
            np.array([[0.0, 0.0, 0.0, 1.0]]),
            n_features=1,
            penalty_cost=100.0,
        )


def test_adaptive_simpson_integrates_a_known_curve():
    result = ireos_module._adaptive_simpson(lambda x: x**4, 0.0, 1.0, 1e-10)
    assert result == pytest.approx(0.2, abs=1e-9)


def test_adaptive_simpson_reports_failure(monkeypatch):
    monkeypatch.setattr(ireos_module, "_MAX_QUADRATURE_DEPTH", 0)
    with pytest.raises(RuntimeError, match="Simpson integration"):
        ireos_module._adaptive_simpson(lambda x: x**4, 0.0, 1.0, 1e-15)


@pytest.mark.parametrize(
    ("probabilities", "kwargs", "message"),
    [
        ([0, 0, 0, 0], {"gamma_max": 1}, "positive total weight"),
        ([0, 0, 0, 2], {"gamma_max": 1}, "between 0 and 1"),
        ([0, 0, 0, 1], {"gamma_max": 0}, "gamma_max"),
        ([0, 0, 0, 1], {"gamma_max": 1, "max_clump_size": 2}, "binary"),
        ([0.1, 0.1, 0.1, 0.1], {}, "at least 0.5"),
    ],
)
def test_ireos_validates_inputs(probabilities, kwargs, message):
    with pytest.raises(ValueError, match=message):
        ireos_score(X, probabilities, **kwargs)


def test_ireos_reports_optimizer_failure(monkeypatch):
    monkeypatch.setattr(ireos_module, "_MAX_NEWTON_STEPS", 1)
    distances = cdist(X, X, metric="sqeuclidean")

    with pytest.raises(RuntimeError, match="did not converge"):
        ireos_module._kernel_logistic_probability(
            distances, 1.0, 3, np.full(len(X), 100.0)
        )


def test_kernel_logistic_regression_is_stable_for_tiny_gamma():
    data = np.random.default_rng(0).normal(size=(150, 3))
    distances = cdist(data, data, metric="sqeuclidean")

    probability = ireos_module._kernel_logistic_probability(
        distances, 1e-5, 0, np.full(len(data), 100.0)
    )

    assert probability == pytest.approx(1 / 150, abs=1e-3)


def test_automatic_gamma_max_follows_author_candidate_rule(monkeypatch):
    # The authors treat probabilities of exactly 0.5 as candidates and train
    # with uniform costs, whatever the clump size.
    seen = []

    def fake_probability(distances, gamma, candidate, costs):
        seen.append((candidate, costs.copy()))
        return 0.9

    monkeypatch.setattr(ireos_module, "_kernel_logistic_probability", fake_probability)
    distances = cdist(X, X, metric="sqeuclidean")
    probabilities = np.array([[0.5, 0.0, 0.2, 0.0], [0.0, 0.0, 0.0, 0.7]])

    gamma = ireos_module._find_gamma_max(
        ireos_module._KernelFeatures(distances),
        probabilities,
        n_features=1,
        penalty_cost=100.0,
    )

    assert gamma == pytest.approx(0.001)
    assert [candidate for candidate, _ in seen] == [0, 3]
    assert all(np.all(costs == 100.0) for _, costs in seen)


def _fake_curves(monkeypatch):
    def fake_probability(distances, gamma, candidate, costs):
        return min(1.0, (0.1 + 0.2 * candidate) * (1 + gamma) * costs.mean() / 100)

    monkeypatch.setattr(ireos_module, "_kernel_logistic_probability", fake_probability)


def test_ireos_significance_exact_moments_match_permutation_enumeration(monkeypatch):
    from itertools import permutations

    _fake_curves(monkeypatch)
    weights = np.array([0.9, 0.0, 0.3, 0.6])

    result = ireos_significance(X, weights, gamma_max=1.0)

    null = [ireos_score(X, list(perm), gamma_max=1.0) for perm in permutations(weights)]
    assert result.method == "exact_moments"
    assert result.index == pytest.approx(ireos_score(X, weights, gamma_max=1.0))
    assert result.expected == pytest.approx(np.mean(null))
    assert result.std == pytest.approx(np.std(null))
    assert result.adjusted == pytest.approx(
        (result.index - result.expected) / (1 - result.expected)
    )
    assert result.z_score == pytest.approx(
        (result.index - result.expected) / result.std
    )
    assert 0 < result.p_value < 1


def test_ireos_significance_p_value_respects_discrete_null(monkeypatch):
    # One of four samples is the outlier: a random solution matches the best
    # one with probability 1/4, which a normal approximation understates.
    _fake_curves(monkeypatch)

    result = ireos_significance(X, [0, 0, 0, 1], gamma_max=1.0, random_state=0)

    assert result.p_value == pytest.approx(0.25, abs=0.02)


def test_ireos_significance_monte_carlo_for_clumps(monkeypatch):
    _fake_curves(monkeypatch)
    weights = np.array([0.0, 0.0, 1.0, 1.0])

    result = ireos_significance(
        X, weights, gamma_max=1.0, max_clump_size=2, n_permutations=20, random_state=0
    )
    again = ireos_significance(
        X, weights, gamma_max=1.0, max_clump_size=2, n_permutations=20, random_state=0
    )

    assert result == again
    assert result.method == "monte_carlo"
    assert result.index == pytest.approx(
        ireos_score(X, weights, gamma_max=1.0, max_clump_size=2)
    )
    assert 1 / 21 <= result.p_value <= 1
    with pytest.raises(ValueError, match="n_permutations"):
        ireos_significance(
            X, weights, gamma_max=1.0, max_clump_size=2, n_permutations=1
        )


def test_ireos_significance_flags_an_isolated_outlier():
    data = np.vstack([np.linspace(0, 1, 8)[:, None], [[8.0]]])
    weights = np.zeros(9)
    weights[-1] = 1.0

    result = ireos_significance(data, weights, gamma_max=1.0)

    assert result.index > result.expected
    assert result.adjusted > 0
    assert result.z_score > 2
    # Only the one random solution that picks the outlier ties it.
    assert result.p_value == pytest.approx(1 / 9, abs=0.02)
