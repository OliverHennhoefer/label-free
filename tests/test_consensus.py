import numpy as np
import pytest

from labelfree.metrics import (
    average_rank_consensus_scores,
    hits_model_scores,
    model_centrality_scores,
)


def test_model_centrality_rewards_rank_agreement():
    score_matrix = [
        [0.0, 1.0, 2.0, 3.0],
        [0.1, 1.1, 2.1, 3.1],
        [3.0, 2.0, 1.0, 0.0],
    ]

    scores = model_centrality_scores(score_matrix)

    assert scores[0] == pytest.approx(scores[1])
    assert scores[0] > scores[2]


def test_average_rank_consensus_rewards_majority_ranking():
    score_matrix = [
        [0.0, 1.0, 2.0, 3.0],
        [0.1, 1.1, 2.1, 3.1],
        [3.0, 2.0, 1.0, 0.0],
    ]

    scores = average_rank_consensus_scores(score_matrix)

    assert scores[0] == pytest.approx(scores[1])
    assert scores[0] > scores[2]


def test_hits_model_scores_rewards_models_pointing_to_consensus_authorities():
    score_matrix = [
        [0.0, 1.0, 2.0, 3.0],
        [0.1, 1.1, 2.1, 3.1],
        [3.0, 2.0, 1.0, 0.0],
    ]

    scores = hits_model_scores(score_matrix)

    assert scores[0] == pytest.approx(scores[1])
    assert scores[0] > scores[2]
    assert np.linalg.norm(scores) == pytest.approx(1.0)


def test_consensus_metrics_apply_score_polarity():
    high_is_anomalous = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.1, 1.1, 2.1, 3.1],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    high_is_normal = -high_is_anomalous

    result = model_centrality_scores(
        high_is_normal,
        score_polarity="higher_is_normal",
    )
    expected = model_centrality_scores(high_is_anomalous)

    assert result == pytest.approx(expected)


def test_consensus_metrics_validate_matrix_shape():
    with pytest.raises(ValueError, match="2D"):
        model_centrality_scores([1.0, 2.0, 3.0])
