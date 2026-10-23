import numpy as np
import pytest
from scipy.special import erf
from scipy.stats import gamma

from labelfree import orient_scores, unify_scores

SCORES = np.array([0.9, 1.0, 1.1, 1.05, 0.95, 1.2, 3.5, 6.0])


def test_gaussian_scaling_matches_kriegel_formula():
    mean, std = SCORES.mean(), SCORES.std(ddof=1)
    expected = np.maximum(0, erf((SCORES - mean) / (std * np.sqrt(2))))

    assert unify_scores(SCORES) == pytest.approx(expected)


def test_gamma_scaling_matches_moment_fitted_cdf():
    mean, var = SCORES.mean(), SCORES.var(ddof=1)
    shape, scale = mean**2 / var, var / mean
    at_mean = gamma.cdf(mean, shape, scale=scale)
    expected = np.maximum(
        0, (gamma.cdf(SCORES, shape, scale=scale) - at_mean) / (1 - at_mean)
    )

    assert unify_scores(SCORES, method="gamma") == pytest.approx(expected)


def test_linear_scaling_and_baseline_regularization():
    assert unify_scores(SCORES, method="linear") == pytest.approx(
        (SCORES - 0.9) / (6.0 - 0.9)
    )
    regularized = np.maximum(0, SCORES - 1.0)
    assert unify_scores(SCORES, method="linear", baseline=1.0) == pytest.approx(
        regularized / regularized.max()
    )


def test_inversions_for_higher_is_normal_scores():
    normal = np.array([0.5, 1.0, 2.0, 4.0])

    linear = unify_scores(normal, method="linear", score_polarity="higher_is_normal")
    log = unify_scores(
        normal, method="linear", score_polarity="higher_is_normal", inversion="log"
    )

    assert linear == pytest.approx((4.0 - normal) / 3.5)
    inverted = -np.log(normal / 4.0)
    assert log == pytest.approx(inverted / inverted.max())


def test_unified_scores_are_probabilities_with_inliers_at_zero():
    probabilities = unify_scores(SCORES)

    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.all(probabilities[SCORES <= SCORES.mean()] == 0)
    assert np.argmax(probabilities) == np.argmax(SCORES)


def test_constant_scores_map_to_zero():
    assert unify_scores([1.0, 1.0, 1.0]) == pytest.approx([0.0, 0.0, 0.0])
    assert unify_scores([1.0, 1.0], method="linear") == pytest.approx([0.0, 0.0])


def test_unify_scores_validates_inputs():
    with pytest.raises(ValueError, match="method"):
        unify_scores(SCORES, method="sigmoid")
    with pytest.raises(ValueError, match="inversion"):
        unify_scores(SCORES, inversion="square")
    with pytest.raises(ValueError, match="positive scores"):
        unify_scores([-1.0, 1.0], score_polarity="higher_is_normal", inversion="log")
    with pytest.raises(ValueError, match="baseline"):
        unify_scores(SCORES, baseline=np.nan)
    with pytest.raises(ValueError, match="gamma scaling"):
        unify_scores([-1.0, 0.0, 2.0], method="gamma")


def test_orient_scores_is_public():
    assert orient_scores([1.0, -2.0], score_polarity="higher_is_normal") == (
        pytest.approx([-1.0, 2.0])
    )
    with pytest.raises(ValueError, match="score_polarity"):
        orient_scores([1.0], score_polarity="lower")
