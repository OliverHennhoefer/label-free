import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.stats import beta

from labelfree.metrics import ranking_stability_score


def _perini_reference(score_matrix, gamma, psi=0.8):
    """Stability from Perini et al.'s stability.py (beta_flavor=2) on given runs."""
    point_rankings = np.zeros((score_matrix.shape[1], score_matrix.shape[0]))
    for i, scores in enumerate(score_matrix):
        for rank, index in enumerate(np.argsort(scores)):
            point_rankings[index, i] = rank + 1
    nte = score_matrix.shape[1]
    point_rankings = point_rankings / nte

    def f(p):
        return ((1.0 - psi) - beta.cdf(1.0 - 2 * gamma, p[0], p[1])) ** 2

    res = minimize(
        f,
        np.array([1.0, 1.0]),
        method="trust-constr",
        constraints=[
            LinearConstraint(
                [[gamma, gamma - 1.0]], [2 * gamma - 1.0], [2 * gamma - 1.0]
            )
        ],
        bounds=Bounds([1.0, 1.0], [np.inf, np.inf]),
    )
    alpha_param, beta_param = res.x
    random_stdev = np.sqrt((nte + 1) * (nte - 1) / (12 * nte**2))
    stabilities = np.zeros(nte)
    for ii in range(nte):
        p_min = np.min(point_rankings[ii])
        p_max = np.max(point_rankings[ii])
        p_std = np.std(point_rankings[ii])
        area = beta.cdf(p_max, alpha_param, beta_param) - beta.cdf(
            p_min, alpha_param, beta_param
        )
        stabilities[ii] = area * p_std
    return 1.0 - np.mean(np.minimum(1, stabilities / random_stdev))


@pytest.mark.parametrize("noise", [0.1, 1.0, 5.0])
def test_ranking_stability_matches_author_implementation(noise):
    rng = np.random.default_rng(11)
    base = rng.normal(size=60)
    score_matrix = base + noise * rng.normal(size=(8, 60))

    result = ranking_stability_score(score_matrix, contamination=0.1)

    assert result == pytest.approx(_perini_reference(score_matrix, 0.1), abs=1e-4)


def test_ranking_stability_is_one_for_identical_rankings():
    score_matrix = [[0.0, 1.0, 2.0, 3.0, 4.0]] * 3

    assert ranking_stability_score(score_matrix, contamination=0.2) == pytest.approx(
        1.0
    )


def test_ranking_stability_drops_for_variable_rankings():
    stable = [[0.0, 1.0, 2.0, 3.0, 4.0]] * 3
    variable = [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0, 0.0],
        [0.0, 2.0, 4.0, 1.0, 3.0],
    ]

    assert ranking_stability_score(variable, contamination=0.2) < (
        ranking_stability_score(stable, contamination=0.2)
    )


def test_ranking_stability_applies_score_polarity():
    rng = np.random.default_rng(0)
    score_matrix = rng.normal(size=(4, 30))

    assert ranking_stability_score(
        -score_matrix, contamination=0.1, score_polarity="higher_is_normal"
    ) == pytest.approx(ranking_stability_score(score_matrix, contamination=0.1))


def test_ranking_stability_validates_inputs():
    with pytest.raises(ValueError, match="contamination"):
        ranking_stability_score([[0.0, 1.0], [1.0, 0.0]], contamination=0.5)
    with pytest.raises(ValueError, match="psi"):
        ranking_stability_score([[0.0, 1.0], [1.0, 0.0]], contamination=0.1, psi=1.0)
    with pytest.raises(ValueError, match="two runs"):
        ranking_stability_score([[0.0, 1.0]], contamination=0.1)
