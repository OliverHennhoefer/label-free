"""Ranking stability metrics."""

import numpy as np
from typing import Dict, List, Callable, Optional
from scipy.stats import kendalltau, spearmanr
from labelfree.utils import validate_data


def ranking_stability(
    score_func: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    n_subsamples: int = 20,
    subsample_ratio: float = 0.8,
    method: str = "kendall",
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Measure stability of anomaly rankings under data perturbation.

    Parameters
    ----------
    score_func : callable
        Function that takes data and returns anomaly scores.
    data : array-like of shape (n_samples, n_features)
        Dataset to evaluate on.
    n_subsamples : int, default=20
        Number of subsamples to generate.
    subsample_ratio : float, default=0.8
        Fraction of data in each subsample.
    method : {'kendall', 'spearman'}, default='kendall'
        Correlation method to use.
    random_state : int, optional
        Random seed.

    Returns
    -------
    dict with keys:
        - 'mean': Mean correlation between rankings
        - 'std': Standard deviation of correlations
        - 'min': Minimum correlation observed
    """
    data = validate_data(data)
    rng = np.random.default_rng(random_state)

    n_samples = len(data)
    subsample_size = int(n_samples * subsample_ratio)

    # Generate scores for each subsample
    all_scores = []
    all_indices = []

    for _ in range(n_subsamples):
        indices = rng.choice(n_samples, size=subsample_size, replace=False)
        indices = np.sort(indices)  # Sort for easier lookup later
        scores = score_func(data[indices])

        all_scores.append(scores)
        all_indices.append(indices)

    # Compute pairwise correlations
    correlations = []

    for i in range(n_subsamples):
        for j in range(i + 1, n_subsamples):
            # Find common indices
            common = np.intersect1d(all_indices[i], all_indices[j])

            if len(common) < 10:  # Skip if too little overlap
                continue

            # Get positions of common indices in each subsample
            # Since indices are sorted, we can use searchsorted
            pos_i = np.searchsorted(all_indices[i], common)
            pos_j = np.searchsorted(all_indices[j], common)

            # Verify all common indices were found
            if np.any(pos_i >= len(all_indices[i])) or np.any(
                pos_j >= len(all_indices[j])
            ):
                continue

            scores_i = all_scores[i][pos_i]
            scores_j = all_scores[j][pos_j]

            # Compute correlation
            if method == "kendall":
                corr, _ = kendalltau(scores_i, scores_j)
            else:
                corr, _ = spearmanr(scores_i, scores_j)

            if not np.isnan(corr):
                correlations.append(corr)

    if not correlations:
        # No valid correlations computed
        return {"mean": 0.0, "std": 0.0, "min": 0.0}

    correlations = np.array(correlations)

    return {
        "mean": float(correlations.mean()),
        "std": float(correlations.std()),
        "min": float(correlations.min()),
    }


def top_k_stability(
    score_func: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    k_values: List[int] = [10, 50, 100],
    n_subsamples: int = 20,
    subsample_ratio: float = 0.8,
    random_state: Optional[int] = None,
) -> Dict[int, float]:
    """
    Measure stability of top-k anomaly detection.

    Returns
    -------
    dict mapping k to average Jaccard similarity of top-k sets.
    """
    data = validate_data(data)
    rng = np.random.default_rng(random_state)

    n_samples = len(data)
    subsample_size = int(n_samples * subsample_ratio)

    # Collect top-k for each subsample
    top_k_sets = {k: [] for k in k_values}

    for _ in range(n_subsamples):
        indices = rng.choice(n_samples, size=subsample_size, replace=False)
        scores = score_func(data[indices])

        # Get top-k indices (highest scores)
        sorted_idx = np.argsort(scores)[::-1]
        top_indices = indices[sorted_idx]

        for k in k_values:
            if k <= len(top_indices):
                top_k_sets[k].append(set(top_indices[:k]))

    # Compute Jaccard similarities
    results = {}

    for k in k_values:
        if not top_k_sets[k]:
            continue

        similarities = []
        sets = top_k_sets[k]

        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                intersection = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j])
                similarities.append(intersection / union if union > 0 else 0)

        results[k] = float(np.mean(similarities)) if similarities else 0.0

    return results
