import numpy as np
import pytest

from labelfree.metrics import (
    bounding_box_volume,
    excess_mass_auc,
    excess_mass_curve,
    mass_volume_auc,
    mass_volume_curve,
)


def test_mass_volume_curve_matches_empirical_level_sets():
    scores = [5.0, 4.0, 3.0, 1.0, 0.0]
    reference_scores = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]

    alphas, volumes = mass_volume_curve(
        scores,
        reference_scores,
        support_volume=6.0,
        alpha_min=0.4,
        alpha_max=0.8,
        alpha_count=2,
        score_polarity="higher_is_normal",
    )

    assert alphas.tolist() == pytest.approx([0.4, 0.8])
    assert volumes.tolist() == pytest.approx([2.0, 5.0])


def test_mass_volume_auc_integrates_curve():
    result = mass_volume_auc(
        [5.0, 4.0, 3.0, 1.0, 0.0],
        [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        support_volume=6.0,
        alpha_min=0.4,
        alpha_max=0.8,
        alpha_count=2,
        score_polarity="higher_is_normal",
    )

    assert result == pytest.approx(1.4)


def test_excess_mass_curve_matches_empirical_supremum():
    levels, curve = excess_mass_curve(
        [5.0, 4.0, 3.0, 1.0, 0.0],
        [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        support_volume=6.0,
        levels=[0.0, 0.5],
        score_polarity="higher_is_normal",
    )

    assert levels.tolist() == pytest.approx([0.0, 0.5])
    assert curve.tolist() == pytest.approx([1.0, 0.0])


def test_excess_mass_curve_does_not_assume_first_level_is_zero():
    _, curve = excess_mass_curve(
        [5.0, 4.0],
        [5.0, 4.0],
        support_volume=2.0,
        levels=[0.25],
        score_polarity="higher_is_normal",
    )

    assert curve.tolist() == pytest.approx([0.5])


def test_excess_mass_levels_must_be_ordered_and_non_negative():
    with pytest.raises(ValueError, match="non-negative and strictly increasing"):
        excess_mass_curve([1.0, 0.0], [1.0, 0.0], support_volume=1.0, levels=[0.5, 0.0])


def test_excess_mass_auc_integrates_truncated_curve():
    result = excess_mass_auc(
        [5.0, 4.0, 3.0, 1.0, 0.0],
        [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        support_volume=6.0,
        levels=[0.0, 0.5],
        score_polarity="higher_is_normal",
    )

    assert result == pytest.approx(0.25)


def test_mass_volume_applies_score_polarity():
    normal_scores = np.array([5.0, 4.0, 3.0, 1.0, 0.0])
    reference = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])

    result = mass_volume_auc(
        -normal_scores,
        -reference,
        support_volume=6.0,
        alpha_min=0.4,
        alpha_max=0.8,
        alpha_count=2,
        score_polarity="higher_is_anomalous",
    )
    expected = mass_volume_auc(
        normal_scores,
        reference,
        support_volume=6.0,
        alpha_min=0.4,
        alpha_max=0.8,
        alpha_count=2,
        score_polarity="higher_is_normal",
    )

    assert result == pytest.approx(expected)


def test_bounding_box_volume_returns_axis_aligned_volume():
    assert bounding_box_volume([[0.0, 2.0], [1.0, 4.0]], offset=0.0) == 2.0


def _goix_em(t, volume_support, s_unif, s_X, n_generated):
    """em() from Goix's EMMV_benchmarks, without the AUC step."""
    EM_t = np.zeros(t.shape[0])
    n_samples = s_X.shape[0]
    EM_t[0] = 1.0
    for u in np.unique(s_X):
        EM_t = np.maximum(
            EM_t,
            1.0 / n_samples * (s_X > u).sum()
            - t * (s_unif > u).sum() / n_generated * volume_support,
        )
    return EM_t


def _goix_mv(axis_alpha, volume_support, s_unif, s_X, n_generated):
    """mv() from Goix's EMMV_benchmarks, without the AUC step."""
    n_samples = s_X.shape[0]
    s_X_argsort = s_X.argsort()
    mass = 0
    cpt = 0
    mv = np.zeros(axis_alpha.shape[0])
    for i in range(axis_alpha.shape[0]):
        while mass < axis_alpha[i]:
            cpt += 1
            u = s_X[s_X_argsort[-cpt]]
            mass = 1.0 / n_samples * cpt
        mv[i] = float((s_unif >= u).sum()) / n_generated * volume_support
    return mv


def _goix_fixture():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 2))
    low, high = X.min(axis=0), X.max(axis=0)
    unif = rng.uniform(low, high, size=(5_000, 2))
    volume = float(np.prod(high - low))
    # Normality score: negative distance to the origin.
    return -np.linalg.norm(X, axis=1), -np.linalg.norm(unif, axis=1), volume


def test_mass_volume_curve_matches_goix_reference():
    s_X, s_unif, volume = _goix_fixture()
    alphas = np.linspace(0.9, 0.999, 50)

    _, volumes = mass_volume_curve(
        s_X,
        s_unif,
        support_volume=volume,
        alpha_count=50,
        score_polarity="higher_is_normal",
    )

    assert volumes == pytest.approx(_goix_mv(alphas, volume, s_unif, s_X, s_unif.size))


def test_excess_mass_curve_adds_the_full_support_to_goix_reference():
    s_X, s_unif, volume = _goix_fixture()
    levels = np.linspace(0.0, 100 / volume, 200)

    _, curve = excess_mass_curve(
        s_X,
        s_unif,
        support_volume=volume,
        levels=levels,
        score_polarity="higher_is_normal",
    )

    reference = _goix_em(levels, volume, s_unif, s_X, s_unif.size)
    assert curve == pytest.approx(np.maximum(reference, 1 - levels * volume))
