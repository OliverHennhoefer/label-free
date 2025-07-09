"""Mass-Volume curve implementation."""

import numpy as np
from typing import Optional, Dict, Callable
from .utils import validate_scores, validate_data, compute_auc


def mass_volume_curve(
    scores: np.ndarray,
    data: np.ndarray,
    n_thresholds: int = 100,
    n_mc_samples: int = 10000,
    random_state: Optional[int] = None,
    scoring_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute Mass-Volume curve for anomaly detection evaluation.

    The MV curve shows the trade-off between mass (fraction of data captured)
    and volume (fraction of space occupied) at different score thresholds.

    This implementation follows the algorithm from Goix et al. (EMMV_benchmarks).

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Anomaly scores where higher values indicate anomalies.
    data : array-like of shape (n_samples, n_features)
        Original data points corresponding to scores.
    n_thresholds : int, default=100
        Number of mass levels (alpha values) to evaluate.
    n_mc_samples : int, default=10000
        Number of Monte Carlo samples for volume estimation.
    random_state : int, optional
        Random seed for reproducibility.
    scoring_function : callable, optional
        Function that takes data points and returns anomaly scores.
        If not provided, a simulation based on nearest neighbors will be used.

    Returns
    -------
    dict with keys:
        - 'mass': Mass values (alpha) from 0 to 1
        - 'volume': Volume values at each mass level
        - 'auc': Area under the MV curve (lower is better)
        - 'axis_alpha': The alpha values used (same as mass)
    """
    scores = validate_scores(scores)
    data = validate_data(data)

    if len(scores) != len(data):
        raise ValueError(
            f"Length mismatch: {len(scores)} scores vs {len(data)} data points"
        )

    rng = np.random.default_rng(random_state)

    # Generate uniform samples in data bounding box
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    volume_support = np.prod(data_max - data_min)
    uniform_samples = rng.uniform(
        data_min, data_max, size=(n_mc_samples, data.shape[1])
    )

    # Generate scores for uniform samples
    if scoring_function is not None:
        # Use provided scoring function
        uniform_scores = scoring_function(uniform_samples)
    else:
        # Simulate scores based on nearest neighbors (for testing/demo purposes)
        uniform_scores = _simulate_uniform_scores(uniform_samples, data, scores, rng)

    # Define target mass levels (alpha values)
    axis_alpha = np.linspace(0, 1, n_thresholds)

    # Sort scores in descending order (higher scores = more anomalous)
    n_samples = len(scores)
    scores_argsort = scores.argsort()[::-1]  # Descending order

    # Compute mass-volume curve following reference implementation
    masses = np.zeros(n_thresholds)
    volumes = np.zeros(n_thresholds)

    mass = 0
    cpt = 0
    threshold = scores[scores_argsort[0]] if n_samples > 0 else 0

    for i in range(n_thresholds):
        # Special case for alpha = 0
        if axis_alpha[i] == 0:
            masses[i] = 0
            # Use highest threshold (includes no points)
            volumes[i] = (
                (uniform_scores > scores[scores_argsort[0]]).sum() / n_mc_samples
                if n_samples > 0
                else 0
            )
        else:
            # Find threshold corresponding to target mass
            while mass < axis_alpha[i] and cpt < n_samples:
                threshold = scores[scores_argsort[cpt]]
                cpt += 1
                mass = cpt / n_samples

            masses[i] = mass
            # Volume: fraction of uniform samples with score >= threshold
            volumes[i] = (uniform_scores >= threshold).sum() / n_mc_samples

    # Compute area under curve
    auc = compute_auc(masses, volumes)

    return {"mass": masses, "volume": volumes, "auc": auc, "axis_alpha": axis_alpha}


def _simulate_uniform_scores(uniform_samples, data, scores, rng):
    """Simulate scores for uniform samples based on nearest neighbors."""
    # Simple approach: assign score based on distance to nearest data point
    from scipy.spatial import cKDTree

    tree = cKDTree(data)
    distances, indices = tree.query(uniform_samples, k=1)

    # Add noise to avoid exact copies
    base_scores = scores[indices]
    noise = rng.normal(0, 0.1 * scores.std(), size=len(uniform_samples))
    return base_scores + noise
