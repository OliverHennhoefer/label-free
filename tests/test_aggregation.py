import numpy as np
import pytest

from labelfree import borda_scores, robust_borda_scores


def test_borda_awards_n_minus_position_points():
    metrics = np.array(
        [
            [0.9, 0.5, 0.1],  # higher is better: ranks 1, 2, 3
            [3.0, 1.0, 2.0],  # lower is better: ranks 3, 1, 2
        ]
    )

    result = borda_scores(metrics, higher_is_better=[True, False])

    assert result == pytest.approx([2 + 0, 1 + 2, 0 + 1])


def test_partial_borda_only_rewards_top_k():
    metrics = np.array([[0.9, 0.5, 0.1], [0.1, 0.9, 0.5]])

    assert borda_scores(metrics, top_k=1) == pytest.approx([2.0, 2.0, 0.0])


def test_borda_shares_tied_positions():
    assert borda_scores([[1.0, 1.0, 0.0]]) == pytest.approx([1.5, 1.5, 0.0])


def test_robust_borda_discards_an_influential_adversarial_metric():
    good = np.arange(8, dtype=float)
    metrics = np.vstack(
        [
            good,
            good + np.array([0.0, 0, 0, 0, 0, 1.5, 0, 0]),
            good + np.array([0.0, 0, 0, 0, 0, 0, 1.5, 0]),
            good[::-1] * 10,
        ]
    )

    trimmed = robust_borda_scores(metrics)

    assert trimmed == pytest.approx(borda_scores(metrics[:3]))
    assert np.argmax(trimmed) == 7


def test_robust_borda_keeps_small_or_uniform_sets():
    metrics = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])

    assert robust_borda_scores(metrics) == pytest.approx(borda_scores(metrics))
    same = np.array([[0.1, 0.2, 0.3]] * 3)
    assert robust_borda_scores(same) == pytest.approx(borda_scores(same))


def test_aggregation_validates_inputs():
    with pytest.raises(ValueError, match="n_models"):
        borda_scores([[1.0]])
    with pytest.raises(ValueError, match="top_k"):
        borda_scores([[1.0, 2.0]], top_k=3)
    with pytest.raises(ValueError, match="non-finite"):
        robust_borda_scores([[1.0, np.nan]])
