"""Similarity-based Internal Relative Evaluation of Outlier Solutions (SIREOS)."""

import numpy as np
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
