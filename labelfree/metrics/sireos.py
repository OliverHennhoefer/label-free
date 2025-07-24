"""Similarity-based Internal Relative Evaluation of Outlier Solutions (SIREOS)."""

import numpy as np
from typing import Optional
import faiss
from labelfree.utils import validate_scores, validate_data


def sireos(
    scores: np.ndarray,
    data: np.ndarray,
    quantile: float = 0.01,
) -> float:
    """
    Compute SIREOS (Similarity-based Internal Relative Evaluation of Outlier Solutions).

    SIREOS evaluates outlier detection quality by measuring how well outlier scores
    represent the underlying data distribution using similarity-based neighborhood
    characteristics with a heat kernel approach.

    This is the neighborhood-based SIREOS implementation from the original paper.
    It computes local similarity for each point and weights it by the normalized
    anomaly score, providing a measure of how well scores align with local
    data structure.

    References
    ----------
    .. [1] Marques, H.O., Campello, R.J.G.B., Zimek, A., Sander, J. (2022).
           "Similarity-based Internal Relative Evaluation of Outlier Solutions (SIREOS)."
           In: Similarity Search and Applications. SISAP 2022.
           https://link.springer.com/chapter/10.1007/978-3-031-17849-8_19

    .. [2] GitHub repository: https://github.com/homarques/SIREOS

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Anomaly scores from detector.
    data : array-like of shape (n_samples, n_features)
        Original data points corresponding to scores.
    quantile : float, default=0.01
        Quantile used to compute the heat kernel threshold parameter.
        Controls the locality of the similarity computation.
    random_state : int, optional
        Random seed for reproducibility (currently unused).

    Returns
    -------
    float
        SIREOS score. Higher values indicate better outlier detection quality.
        Score reflects how well anomaly scores align with local neighborhood structure.
    """
    scores = validate_scores(scores)
    data = validate_data(data)

    if len(scores) != len(data):
        raise ValueError(
            f"Length mismatch: {len(scores)} scores vs {len(data)} data points"
        )

    n_samples, n_features = data.shape

    if n_samples <= 1:
        return 0.0

    # Normalize scores using sum normalization (probability distribution)
    # This matches the reference implementation: X = X/X.sum()
    scores_sum = scores.sum()
    if scores_sum == 0:
        scores_normalized = np.ones_like(scores) / len(scores)
    else:
        scores_normalized = scores / scores_sum

    # Compute pairwise distances using FAISS (corrected for proper matrix)
    data_float32 = data.astype(np.float32)

    # Create FAISS index for L2 distance
    index = faiss.IndexFlatL2(data_float32.shape[1])
    index.add(data_float32)

    # Build proper distance matrix (FAISS returns sorted results)
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        query_point = data_float32[i : i + 1]
        distances_sq_i, indices_i = index.search(query_point, n_samples)
        distances_i = np.sqrt(distances_sq_i[0])
        # Place distances in correct positions based on indices
        distances[i, indices_i[0]] = distances_i

    # Compute heat kernel threshold parameter
    non_zero_distances = distances[distances > 0]
    if len(non_zero_distances) == 0:
        threshold = 1.0
    else:
        threshold = np.quantile(non_zero_distances, quantile)

    # Compute SIREOS score
    total_score = 0.0

    # For each sample, compute its contribution to the SIREOS score
    for i in range(n_samples):
        # Get distances from point i to all other points
        point_distances = distances[i]

        # Get indices of other points (excluding self)
        other_indices = [j for j in range(n_samples) if j != i]
        distances_to_others = point_distances[other_indices]

        # Compute exponential kernel similarities to other points only
        similarities = np.exp(-(distances_to_others**2) / (2 * threshold**2))

        # Compute mean similarity (matches reference: np.mean(similarities))
        mean_similarity = np.mean(similarities)

        # Weight by normalized score and add to total
        total_score += mean_similarity * scores_normalized[i]

    # Return total score (no division by n_samples in reference implementation)
    return float(total_score)


def sireos_separation(
    scores: np.ndarray,
    data: np.ndarray,
    similarity: str = "euclidean",
    random_state: Optional[int] = None,
) -> float:
    """
    Compute SIREOS Separation (Group-based SIREOS).

    This variant uses median split to create groups and measures separation
    between high-scoring and low-scoring groups using similarity ratios.
    Unlike the original SIREOS, this approach focuses on group-level separation
    rather than individual point-neighborhood alignment.

    This method is particularly useful for evaluating one-class classifiers
    where the algorithm was trained only on normal data but needs to distinguish
    between normal and anomalous samples in the test set.

    Key differences from sireos():
    - Uses median split to create artificial binary groups
    - Measures intra-group vs inter-group similarity ratios
    - Returns score in [0, 2] range instead of [0, 1]
    - Focuses on group separation rather than local neighborhood structure

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Anomaly scores from detector.
    data : array-like of shape (n_samples, n_features)
        Original features corresponding to scores.
    similarity : {'euclidean', 'cosine'}, default='euclidean'
        Similarity metric to use for computing pairwise similarities.
        - 'euclidean': Uses Gaussian kernel on Euclidean distances
        - 'cosine': Uses cosine similarity
    random_state : int, optional
        Random seed for reproducibility (currently unused).

    Returns
    -------
    float
        Separability score in range [0, 2] where:
        - 0: No separation (inter-group similarity > intra-group)
        - 1: Equal intra and inter-group similarities
        - 2: Perfect separation (high intra-group, zero inter-group)
        Higher values indicate better anomaly detection quality.

    Notes
    -----
    The median split approach assumes that the anomaly detector produces
    meaningful scores where higher values indicate more anomalous samples.
    If the detector's scores are poorly calibrated, the median split may
    not create meaningful groups.
    """
    scores = validate_scores(scores)
    data = validate_data(data)

    if len(scores) != len(data):
        raise ValueError(
            f"Length mismatch: {len(scores)} scores vs {len(data)} data points"
        )

    # Split based on median
    median_score = np.median(scores)
    high_mask = scores > median_score
    low_mask = ~high_mask

    if not any(high_mask) or not any(low_mask):
        return 1.0  # Perfect separation (degenerate case)

    # Compute similarity matrix
    if similarity == "euclidean":
        # Use FAISS for euclidean distances
        data_float32 = data.astype(np.float32)
        index = faiss.IndexFlatL2(data_float32.shape[1])
        index.add(data_float32)
        distances_sq, _ = index.search(data_float32, data_float32.shape[0])
        distances = np.sqrt(distances_sq)

        # Convert distances to similarities using Gaussian kernel
        median_dist = np.median(distances[distances > 0])
        if median_dist == 0:
            median_dist = 1.0
        similarities = np.exp(-distances / median_dist)
    else:  # cosine
        from scipy.spatial.distance import cdist

        # Cosine distance is in [0, 2], where 0 means identical
        cosine_distances = cdist(data, data, metric="cosine")
        # Convert to similarity: 1 - distance/2 gives range [0, 1]
        similarities = 1 - cosine_distances / 2

    # Compute intra-group and inter-group similarities
    high_high_mask = np.outer(high_mask, high_mask)
    low_low_mask = np.outer(low_mask, low_mask)
    high_low_mask = np.outer(high_mask, low_mask)

    # Remove diagonal for intra-group similarities
    np.fill_diagonal(high_high_mask, False)
    np.fill_diagonal(low_low_mask, False)

    # Calculate mean similarities
    high_high = similarities[high_high_mask].mean() if high_high_mask.any() else 0
    low_low = similarities[low_low_mask].mean() if low_low_mask.any() else 0
    high_low = similarities[high_low_mask].mean() if high_low_mask.any() else 0

    # SIREOS separation score: ratio of intra to inter similarity
    intra_sim = (high_high + low_low) / 2
    inter_sim = high_low

    # Add small epsilon and ensure non-negative
    sireos_separation = max(
        0, (intra_sim - inter_sim) / (intra_sim + inter_sim + 1e-10) + 1
    )

    return float(sireos_separation)
