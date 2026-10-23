import math

import numpy as np
import pytest
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    r2_score,
    silhouette_score,
)

from labelfree.metrics import SCORE_CLUSTER_INDICES, score_cluster_metrics

RNG_SCORES = np.random.default_rng(0).gamma(2.0, 1.0, size=40)


def _split(scores, n_outliers):
    x = np.asarray(scores, dtype=float)
    labels = np.zeros(x.size, dtype=int)
    labels[np.argsort(x, kind="mergesort")[-n_outliers:]] = 1
    c_normal = x[labels == 0].mean()
    c_anomaly = x[labels == 1].mean()
    centers = np.where(labels == 1, c_anomaly, c_normal)
    return x, labels, centers, c_normal, c_anomaly


# Reference functions transcribed from UOMS stand-alone/ten_metrics_mat.py
# (Ma et al., 2023), which implements the indices of Nguyen et al.
def _uoms_hubert(x, c):
    n = len(x)
    diff = abs(x - x[:, None])
    diff_c = abs(c - c[:, None])
    mask = np.zeros_like(diff, dtype=bool)
    mask[np.tril_indices_from(mask)] = True
    return ((mask * diff) * diff_c * 2).sum() / (n * (n - 1))


def _uoms_i_index(x, c, c_normal, c_anomaly):
    max_d = abs(c_normal - c_anomaly)
    numerator = abs(x - x.mean()).sum()
    denominator = abs(x - c).sum()
    return ((numerator * max_d) / (denominator * 2)) ** 2


def _uoms_dunn(x, c_binary):
    normal_mask = np.tile(c_binary, (len(c_binary), 1))
    anomaly_mask = np.tile(1 - c_binary.reshape(-1, 1).T, (len(c_binary), 1))
    distance = abs(x - x[:, None])
    diff_inter = distance * normal_mask * anomaly_mask.T
    diff_intra_1 = distance * normal_mask * normal_mask.T
    diff_intra_2 = distance * anomaly_mask * anomaly_mask.T
    min_value = np.min(diff_inter[np.nonzero(diff_inter)])
    max_value = max(np.max(diff_intra_1), np.max(diff_intra_2))
    return min_value / max_value


def _uoms_xb(x, c, c_normal, c_anomaly):
    return sum((x - c) ** 2) / (len(x) * ((c_normal - c_anomaly) ** 2))


@pytest.mark.parametrize(
    ("scores", "n_outliers"),
    [([0.0, 0.2, 0.3, 0.35, 0.5, 0.9, 4.0, 4.2], 2), (RNG_SCORES, 5)],
)
def test_score_cluster_metrics_match_reference_implementations(scores, n_outliers):
    x, labels, centers, c_normal, c_anomaly = _split(scores, n_outliers)
    values = x.reshape(-1, 1)

    result = score_cluster_metrics(scores, n_outliers=n_outliers)

    assert list(result) == list(SCORE_CLUSTER_INDICES)
    assert result["silhouette"] == pytest.approx(silhouette_score(values, labels))
    assert result["calinski_harabasz"] == pytest.approx(
        calinski_harabasz_score(values, labels)
    )
    assert result["davies_bouldin"] == pytest.approx(
        davies_bouldin_score(values, labels)
    )
    assert result["xie_beni"] == pytest.approx(
        _uoms_xb(x, centers, c_normal, c_anomaly)
    )
    assert result["r_squared"] == pytest.approx(r2_score(x, centers))
    assert result["rmsstd"] == pytest.approx(
        math.sqrt(np.sum((x - centers) ** 2) / (x.size - 2))
    )
    assert result["hubert_gamma"] == pytest.approx(_uoms_hubert(x, centers))
    assert result["i_index"] == pytest.approx(
        _uoms_i_index(x, centers, c_normal, c_anomaly)
    )
    assert result["dunn"] == pytest.approx(_uoms_dunn(x, labels))


def test_s_dbw_matches_halkidi_reference_values():
    # Reference values from the s_dbw package (0.4.0):
    # S_Dbw(x[:, None], labels, method="Halkidi", centr="mean", nearest_centr=False)
    small = score_cluster_metrics(
        [0.0, 0.2, 0.3, 0.35, 0.5, 0.9, 4.0, 4.2], n_outliers=2, indices=["s_dbw"]
    )
    gamma = score_cluster_metrics(RNG_SCORES, n_outliers=5, indices=["s_dbw"])

    assert small["s_dbw"] == pytest.approx(0.11628518831641563)
    assert gamma["s_dbw"] == pytest.approx(0.8350817518562825)


def test_xie_beni_matches_nguyen_equation():
    result = score_cluster_metrics([0.0, 0.2, 0.3, 4.0, 4.2], n_outliers=2)

    compactness = (
        (0.0 - 1 / 6) ** 2
        + (0.2 - 1 / 6) ** 2
        + (0.3 - 1 / 6) ** 2
        + (4.0 - 4.1) ** 2
        + (4.2 - 4.1) ** 2
    )
    separation = (4.1 - 1 / 6) ** 2
    assert result["xie_beni"] == pytest.approx(compactness / (5 * separation))


def test_index_subset_and_validation():
    result = score_cluster_metrics(
        [0.0, 0.2, 0.3, 4.0, 4.2], n_outliers=2, indices=["dunn", "silhouette"]
    )
    assert list(result) == ["dunn", "silhouette"]

    with pytest.raises(ValueError, match="unknown"):
        score_cluster_metrics([0.0, 0.2, 0.3], n_outliers=1, indices=["nope"])


def test_degenerate_splits_are_finite_or_infinite_by_definition():
    result = score_cluster_metrics([0.0, 0.0, 0.0, 5.0, 5.0], n_outliers=2)

    assert math.isinf(result["dunn"])
    assert math.isinf(result["i_index"])
    assert result["xie_beni"] == 0.0
    assert result["r_squared"] == pytest.approx(1.0)


def test_score_polarity_controls_outlier_tail():
    low_is_anomalous = [-4.2, -4.0, -0.3, -0.2, 0.0]
    high_is_anomalous = [4.2, 4.0, 0.3, 0.2, 0.0]

    result = score_cluster_metrics(
        low_is_anomalous, n_outliers=2, score_polarity="higher_is_normal"
    )
    expected = score_cluster_metrics(high_is_anomalous, n_outliers=2)

    assert result == pytest.approx(expected)


def test_split_size_must_be_explicit():
    with pytest.raises(ValueError, match="exactly one"):
        score_cluster_metrics([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="exactly one"):
        score_cluster_metrics([0.0, 1.0, 2.0], n_outliers=1, contamination=0.1)

    with pytest.raises(ValueError, match="integer"):
        score_cluster_metrics([0.0, 1.0, 2.0], n_outliers=1.5)


def test_contamination_uses_top_ceiled_fraction():
    result = score_cluster_metrics([0.0, 0.2, 0.3, 4.0, 4.2], contamination=0.21)
    expected = score_cluster_metrics([0.0, 0.2, 0.3, 4.0, 4.2], n_outliers=2)

    assert result == pytest.approx(expected)


def test_score_cluster_requires_enough_samples_for_internal_indices():
    with pytest.raises(ValueError, match="at least three"):
        score_cluster_metrics([0.0, 1.0], n_outliers=1)
