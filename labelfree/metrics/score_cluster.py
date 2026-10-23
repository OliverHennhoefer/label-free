"""Internal clustering indices on the score-induced normal/anomaly split.

Nguyen et al. treat the scores of a detector as a one-dimensional dataset, split
it into the ``n_outliers`` highest scores and the rest, and rate the split with
ten internal clustering validation indices. Ma et al. (2023) use the same ten
indices in their unsupervised outlier model selection benchmark (UOMS).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from labelfree._validation import split_score_labels

#: Direction of every index returned by :func:`score_cluster_metrics`.
SCORE_CLUSTER_INDICES: dict[str, str] = {
    "silhouette": "higher",
    "calinski_harabasz": "higher",
    "davies_bouldin": "lower",
    "xie_beni": "lower",
    "r_squared": "higher",
    "rmsstd": "lower",
    "hubert_gamma": "higher",
    "i_index": "higher",
    "dunn": "higher",
    "s_dbw": "lower",
}


def score_cluster_metrics(
    scores,
    *,
    n_outliers: int | None = None,
    contamination: float | None = None,
    score_polarity: str = "higher_is_anomalous",
    indices: Iterable[str] | None = None,
) -> dict[str, float]:
    """Rate the score-induced normal/anomaly split with clustering indices.

    The top ``n_outliers`` oriented scores form the anomaly cluster and all
    remaining scores form the normal cluster. Every index is computed on the
    one-dimensional scores, not on the features.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Detector scores.
    n_outliers, contamination
        Size of the anomaly cluster. Pass exactly one; ``contamination`` uses
        ``ceil(n_samples * contamination)``.
    score_polarity : {"higher_is_anomalous", "higher_is_normal"}
        Meaning of larger raw scores.
    indices : iterable of str, optional
        Subset of :data:`SCORE_CLUSTER_INDICES` to compute. Defaults to all.

    Returns
    -------
    dict of str to float
        Index values keyed by name. Directions differ per index; see
        :data:`SCORE_CLUSTER_INDICES`.
    """
    selected = list(SCORE_CLUSTER_INDICES if indices is None else indices)
    unknown = sorted(set(selected) - set(SCORE_CLUSTER_INDICES))
    if unknown:
        raise ValueError(f"unknown score-cluster indices: {unknown}")

    oriented, labels = split_score_labels(
        scores,
        n_outliers=n_outliers,
        contamination=contamination,
        score_polarity=score_polarity,
    )
    if oriented.size < 3:
        raise ValueError("score-cluster metrics require at least three samples")
    split = _Split(oriented, labels)
    return {name: float(_INDEX_FUNCTIONS[name](split)) for name in selected}


class _Split:
    """Shared statistics of a two-cluster split of one-dimensional scores."""

    def __init__(self, values: np.ndarray, labels: np.ndarray):
        self.values = values
        self.labels = labels
        self.normal = values[labels == 0]
        self.anomaly = values[labels == 1]
        self.normal_center = float(self.normal.mean())
        self.anomaly_center = float(self.anomaly.mean())
        self.centers = np.where(labels == 1, self.anomaly_center, self.normal_center)
        self.center_distance = abs(self.anomaly_center - self.normal_center)
        self.ssw = float(np.sum((values - self.centers) ** 2))
        self.sst = float(np.sum((values - values.mean()) ** 2))


def _silhouette(split: _Split) -> float:
    return silhouette_score(split.values.reshape(-1, 1), split.labels)


def _calinski_harabasz(split: _Split) -> float:
    return calinski_harabasz_score(split.values.reshape(-1, 1), split.labels)


def _davies_bouldin(split: _Split) -> float:
    return davies_bouldin_score(split.values.reshape(-1, 1), split.labels)


def _xie_beni(split: _Split) -> float:
    # Nguyen et al., Eq. 4 in Ma et al. (2023).
    if split.center_distance == 0:
        return math.inf
    return split.ssw / (split.values.size * split.center_distance**2)


def _r_squared(split: _Split) -> float:
    # RS = (SST - SSW) / SST.
    if split.sst == 0:
        return 0.0
    return (split.sst - split.ssw) / split.sst


def _rmsstd(split: _Split) -> float:
    # RMSSTD = sqrt(SSW / (P * sum_k (n_k - 1))) with P = 1 score dimension.
    return math.sqrt(split.ssw / (split.values.size - 2))


def _hubert_gamma(split: _Split) -> float:
    # Gamma = (1 / NP) * sum_{i<j} d(x_i, x_j) * d(c_i, c_j), NP = n(n-1)/2.
    # d(c_i, c_j) vanishes within a cluster, so only cross-cluster pairs count.
    n_samples = split.values.size
    pairs = n_samples * (n_samples - 1) / 2
    cross = _cross_absolute_difference_sum(split.anomaly, split.normal)
    return split.center_distance * cross / pairs


def _i_index(split: _Split) -> float:
    # Maulik and Bandyopadhyay: ((1 / K) * (E_1 / E_K) * D_K) ** p, K = p = 2.
    within = float(np.sum(np.abs(split.values - split.centers)))
    total = float(np.sum(np.abs(split.values - split.values.mean())))
    if within == 0:
        return math.inf if split.center_distance > 0 else 0.0
    return (0.5 * total / within * split.center_distance) ** 2


def _dunn(split: _Split) -> float:
    # Minimum inter-cluster distance over maximum cluster diameter. The
    # anomaly cluster is the upper score tail, so its closest pair to the
    # normal cluster is (min anomaly score, max normal score).
    separation = float(split.anomaly.min() - split.normal.max())
    diameter = max(float(np.ptp(split.anomaly)), float(np.ptp(split.normal)))
    if diameter == 0:
        return math.inf if separation > 0 else 0.0
    return separation / diameter


def _s_dbw(split: _Split) -> float:
    # Halkidi and Vazirgiannis (2001): Scat + Dens_bw, population deviations.
    total_std = float(np.std(split.values))
    if total_std == 0:
        return math.inf
    cluster_stds = (float(np.std(split.normal)), float(np.std(split.anomaly)))
    scat = sum(cluster_stds) / (2 * total_std)

    stdev = math.sqrt(sum(cluster_stds)) / 2
    density_normal = np.count_nonzero(
        np.abs(split.normal - split.normal_center) <= stdev
    )
    density_anomaly = np.count_nonzero(
        np.abs(split.anomaly - split.anomaly_center) <= stdev
    )
    midpoint = (split.normal_center + split.anomaly_center) / 2
    density_between = np.count_nonzero(np.abs(split.values - midpoint) <= stdev)
    reference = max(density_normal, density_anomaly)
    if reference == 0:
        return math.inf
    return scat + density_between / reference


def _cross_absolute_difference_sum(left: np.ndarray, right: np.ndarray) -> float:
    """Return sum_{a in left, b in right} |a - b| in O((n + m) log m)."""
    ordered = np.sort(right)
    prefix = np.concatenate([[0.0], np.cumsum(ordered)])
    position = np.searchsorted(ordered, left, side="right")
    below = left * position - prefix[position]
    above = (prefix[-1] - prefix[position]) - left * (ordered.size - position)
    return float(np.sum(below + above))


_INDEX_FUNCTIONS = {
    "silhouette": _silhouette,
    "calinski_harabasz": _calinski_harabasz,
    "davies_bouldin": _davies_bouldin,
    "xie_beni": _xie_beni,
    "r_squared": _r_squared,
    "rmsstd": _rmsstd,
    "hubert_gamma": _hubert_gamma,
    "i_index": _i_index,
    "dunn": _dunn,
    "s_dbw": _s_dbw,
}
