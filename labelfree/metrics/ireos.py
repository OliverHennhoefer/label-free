"""Internal Relative Evaluation of Outlier Solutions (IREOS)."""

import numpy as np
from typing import Tuple, Optional
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from labelfree.utils import validate_scores, validate_data


def ireos(
    scores: np.ndarray,
    data: np.ndarray,
    n_splits: int = 5,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute IREOS (Internal Relative Evaluation of Outlier Solutions).

    IREOS measures the separability between high-scoring (anomalous) and
    low-scoring (normal) points using a classifier. No labels required.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Anomaly scores from detector.
    data : array-like of shape (n_samples, n_features)
        Original features corresponding to scores.
    n_splits : int, default=5
        Number of cross-validation splits.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ireos_score : float
        Separability score (higher is better), typically in [0.5, 1].
    p_value : float
        Approximate p-value for the score.
    """
    scores = validate_scores(scores)
    data = validate_data(data)

    if len(scores) != len(data):
        raise ValueError(
            f"Length mismatch: {len(scores)} scores vs {len(data)} data points"
        )

    # Create binary labels using median split
    median_score = np.median(scores)
    binary_labels = (scores > median_score).astype(int)

    # Check for degenerate cases
    n_positive = binary_labels.sum()
    if n_positive == 0 or n_positive == len(binary_labels):
        return 0.5, 1.0  # No separation possible

    # Train SVM and evaluate via cross-validation
    svm = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=random_state)
    cv_scores = cross_val_score(
        svm, data, binary_labels, cv=n_splits, scoring="roc_auc"
    )

    ireos_score = float(cv_scores.mean())

    # Approximate p-value (simplified)
    z_score = (ireos_score - 0.5) / (cv_scores.std() + 1e-10)
    p_value = 2 * (1 - min(0.9999, 0.5 + 0.5 * np.tanh(z_score)))

    return ireos_score, p_value


def sireos(
    scores: np.ndarray, data: np.ndarray, similarity: str = "euclidean"
) -> float:
    """
    Compute SIREOS (Similarity-based IREOS).

    Faster variant using similarity ratios instead of classification.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Anomaly scores.
    data : array-like of shape (n_samples, n_features)
        Original features.
    similarity : {'euclidean', 'cosine'}, default='euclidean'
        Similarity metric to use.

    Returns
    -------
    sireos_score : float
        Separability score in range [0, 2] where:
        - 0: No separation (inter-group similarity > intra-group)
        - 1: Equal intra and inter-group similarities
        - 2: Perfect separation (high intra-group, zero inter-group)
        Higher values indicate better anomaly detection.
    """
    scores = validate_scores(scores)
    data = validate_data(data)

    # Split based on median
    median_score = np.median(scores)
    high_mask = scores > median_score
    low_mask = ~high_mask

    if not any(high_mask) or not any(low_mask):
        return 1.0  # Perfect separation (degenerate case)

    # Compute similarity matrix
    if similarity == "euclidean":
        from scipy.spatial.distance import cdist

        distances = cdist(data, data, metric="euclidean")
        # Convert distances to similarities using Gaussian kernel
        # Normalize by median distance for stability
        median_dist = np.median(distances[distances > 0])
        similarities = np.exp(-distances / median_dist)
    else:  # cosine
        from scipy.spatial.distance import cdist

        # Cosine distance is in [0, 2], where 0 means identical
        cosine_distances = cdist(data, data, metric="cosine")
        # Convert to similarity: 1 - distance/2 gives range [0, 1]
        similarities = 1 - cosine_distances / 2

    # Compute intra-group and inter-group similarities
    # Exclude diagonal for intra-group to avoid self-similarity bias
    high_high_mask = np.outer(high_mask, high_mask)
    low_low_mask = np.outer(low_mask, low_mask)
    high_low_mask = np.outer(high_mask, low_mask)

    # Remove diagonal
    np.fill_diagonal(high_high_mask, False)
    np.fill_diagonal(low_low_mask, False)

    # Calculate mean similarities
    high_high = similarities[high_high_mask].mean() if high_high_mask.any() else 0
    low_low = similarities[low_low_mask].mean() if low_low_mask.any() else 0
    high_low = similarities[high_low_mask].mean() if high_low_mask.any() else 0

    # SIREOS score: ratio of intra to inter similarity
    intra_sim = (high_high + low_low) / 2
    inter_sim = high_low

    # Add small epsilon and ensure non-negative
    sireos_score = max(0, (intra_sim - inter_sim) / (intra_sim + inter_sim + 1e-10) + 1)

    return float(sireos_score)
