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
