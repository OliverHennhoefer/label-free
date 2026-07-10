from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize as reference_minimize
from scipy.spatial.distance import cdist
from scipy.special import expit

import labelfree.metrics.ireos as ireos_module
from labelfree.metrics import ireos_score, ireos_scores


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

    def fake_aucs(distances, candidates, costs, *, gamma_max, **kwargs):
        auc_ranges.append(gamma_max)
        return np.full(distances.shape[0], gamma_max)

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
        distances,
        probabilities,
        n_features=1,
        max_clump_size=1,
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
            distances,
            np.array([[0.0, 0.0, 0.0, 1.0]]),
            n_features=1,
            max_clump_size=1,
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
        ([0.1, 0.1, 0.1, 0.1], {}, "above 0.5"),
    ],
)
def test_ireos_validates_inputs(probabilities, kwargs, message):
    with pytest.raises(ValueError, match=message):
        ireos_score(X, probabilities, **kwargs)


def test_ireos_reports_optimizer_failure(monkeypatch):
    monkeypatch.setattr(
        ireos_module,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False, message="no convergence"
        ),
    )
    distances = cdist(X, X, metric="sqeuclidean")

    with pytest.raises(RuntimeError, match="no convergence"):
        ireos_module._kernel_logistic_probability(
            distances, 1.0, 3, np.full(len(X), 100.0)
        )
