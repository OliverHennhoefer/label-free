import pytest
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from labelfree.metrics import score_cluster_metrics


def test_score_cluster_metrics_match_reference_indices():
    scores = [0.0, 0.2, 0.3, 4.0, 4.2]
    labels = [0, 0, 0, 1, 1]
    values = [[score] for score in scores]

    result = score_cluster_metrics(scores, n_outliers=2)

    assert result["silhouette"] == pytest.approx(silhouette_score(values, labels))
    assert result["calinski_harabasz"] == pytest.approx(
        calinski_harabasz_score(values, labels)
    )
    assert result["davies_bouldin"] == pytest.approx(
        davies_bouldin_score(values, labels)
    )
    compactness = (
        (0.0 - 1 / 6) ** 2
        + (0.2 - 1 / 6) ** 2
        + (0.3 - 1 / 6) ** 2
        + (4.0 - 4.1) ** 2
        + (4.2 - 4.1) ** 2
    )
    separation = (4.1 - 1 / 6) ** 2
    assert result["xie_beni"] == pytest.approx(compactness / (5 * separation))


def test_score_polarity_controls_outlier_tail():
    low_is_anomalous = [-4.2, -4.0, -0.3, -0.2, 0.0]
    high_is_anomalous = [4.2, 4.0, 0.3, 0.2, 0.0]

    result = score_cluster_metrics(
        low_is_anomalous,
        n_outliers=2,
        score_polarity="higher_is_normal",
    )
    expected = score_cluster_metrics(high_is_anomalous, n_outliers=2)

    assert result == pytest.approx(expected)


def test_split_size_must_be_explicit():
    with pytest.raises(ValueError, match="exactly one"):
        score_cluster_metrics([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="exactly one"):
        score_cluster_metrics([0.0, 1.0, 2.0], n_outliers=1, contamination=0.1)


def test_contamination_uses_top_ceiled_fraction():
    result = score_cluster_metrics([0.0, 0.2, 0.3, 4.0, 4.2], contamination=0.21)
    expected = score_cluster_metrics([0.0, 0.2, 0.3, 4.0, 4.2], n_outliers=2)

    assert result == pytest.approx(expected)
