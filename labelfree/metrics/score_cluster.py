"""Internal score-cluster metrics for unlabeled anomaly scores."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from labelfree.utils.validation import split_score_labels


def score_cluster_metrics(
    scores,
    *,
    n_outliers: int | None = None,
    contamination: float | None = None,
    score_polarity: str = "higher_is_anomalous",
) -> dict[str, float]:
    """Evaluate the score-induced normal/anomaly split with cluster indices.

    The score vector is treated as a one-dimensional embedding. The top
    `n_outliers` oriented scores form the anomaly cluster; all remaining scores
    form the normal cluster.
    """
    oriented, labels = split_score_labels(
        scores,
        n_outliers=n_outliers,
        contamination=contamination,
        score_polarity=score_polarity,
    )
    values = oriented.reshape(-1, 1)

    return {
        "silhouette": float(silhouette_score(values, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(values, labels)),
        "davies_bouldin": float(davies_bouldin_score(values, labels)),
        "xie_beni": _xie_beni_score(values, labels),
    }

def _xie_beni_score(values: np.ndarray, labels: np.ndarray) -> float:
    centers = np.array([values[labels == label].mean(axis=0) for label in (0, 1)])
    compactness = sum(
        float(np.sum((values[labels == label] - centers[label]) ** 2))
        for label in (0, 1)
    )
    separation = float(np.sum((centers[0] - centers[1]) ** 2))
    if separation == 0:
        return math.inf
    return compactness / (values.shape[0] * separation)
