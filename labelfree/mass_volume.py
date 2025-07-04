"""Mass-Volume curve implementation."""
import numpy as np
from typing import Tuple, Optional, Dict, Callable, Union
from .utils import validate_scores, validate_data, compute_auc


def mass_volume_curve(
    scores: np.ndarray,
    data: np.ndarray,
    n_thresholds: int = 100,
    n_mc_samples: int = 10000,
    random_state: Optional[int] = None,
    scoring_function: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute Mass-Volume curve for anomaly detection evaluation.
    
    The MV curve shows the trade-off between mass (fraction of data captured)
    and volume (fraction of space occupied) at different score thresholds.
    
    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Anomaly scores where higher values indicate anomalies.
    data : array-like of shape (n_samples, n_features)
        Original data points corresponding to scores.
    n_thresholds : int, default=100
        Number of thresholds to evaluate.
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
        - 'mass': Mass values at each threshold
        - 'volume': Volume values at each threshold
        - 'auc': Area under the MV curve (lower is better)
        - 'thresholds': Score thresholds used
    """
    scores = validate_scores(scores)
    data = validate_data(data)
    
    if len(scores) != len(data):
        raise ValueError(f"Length mismatch: {len(scores)} scores vs {len(data)} data points")
    
    rng = np.random.default_rng(random_state)
    
    # Compute thresholds based on score quantiles
    quantiles = np.linspace(0, 100, n_thresholds)
    thresholds = np.percentile(scores, quantiles)
    
    # Generate uniform samples in data bounding box
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    uniform_samples = rng.uniform(data_min, data_max, size=(n_mc_samples, data.shape[1]))
    
    # Generate scores for uniform samples
    if scoring_function is not None:
        # Use provided scoring function
        uniform_scores = scoring_function(uniform_samples)
    else:
        # Simulate scores based on nearest neighbors (for testing/demo purposes)
        uniform_scores = _simulate_uniform_scores(uniform_samples, data, scores, rng)
    
    # Compute masses and volumes
    masses = np.zeros(n_thresholds)
    volumes = np.zeros(n_thresholds)
    
    for i, threshold in enumerate(thresholds):
        # Mass: fraction of data with score >= threshold
        masses[i] = (scores >= threshold).mean()
        
        # Volume: estimated by fraction of uniform samples with score >= threshold
        volumes[i] = (uniform_scores >= threshold).mean()
    
    # Compute area under curve
    auc = compute_auc(masses, volumes)
    
    return {
        'mass': masses,
        'volume': volumes,
        'auc': auc,
        'thresholds': thresholds
    }


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