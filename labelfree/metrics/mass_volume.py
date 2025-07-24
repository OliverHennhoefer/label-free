import numpy as np
from typing import Optional, Dict, Callable
from labelfree.utils import validate_scores, validate_data, compute_auc


def mass_volume_auc(
    scores: np.ndarray,
    data: np.ndarray,
    alpha_min: float = 0.9,
    alpha_max: float = 0.999,
    n_thresholds: int = 1000,
    n_mc_samples: int = 10000,
    random_state: Optional[int] = None,
    scoring_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute Mass-Volume curve for anomaly detection evaluation.

    The MV curve shows the trade-off between mass (fraction of data captured)
    and volume (absolute volume of space occupied) at different score thresholds.

    This implementation follows the algorithm from Goix et al. (EMMV_benchmarks)
    and is designed for evaluation in the high-mass region (typically 0.9-0.999).

    The volume support (bounding box volume) is computed automatically from the data.

    **Important:** Data should be scaled using MinMaxScaler or similar to [0,1] range
    for best results. AUC values scale with the bounding box volume, so unscaled
    data can produce very large, hard-to-interpret values. The function will warn
    when volume support exceeds 100, indicating potential scaling issues.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Anomaly scores where higher values indicate anomalies.
    data : array-like of shape (n_samples, n_features)
        Original data points corresponding to scores.
    alpha_min : float, default=0.9
        Minimum mass level (fraction of data) to evaluate.
    alpha_max : float, default=0.999
        Maximum mass level (fraction of data) to evaluate.
    n_thresholds : int, default=1000
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
        - 'mass': Mass values (alpha) from alpha_min to alpha_max
        - 'volume': Volume values at each mass level (in data units)
        - 'auc': Area under the MV curve
        - 'axis_alpha': The alpha values used (same as mass)

    Examples
    --------
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from sklearn.ensemble import IsolationForest
    >>> # Scale data for optimal results
    >>> scaler = MinMaxScaler()
    >>> X_scaled = scaler.fit_transform(X)
    >>> # Fit anomaly detector and compute MV-AUC
    >>> model = IsolationForest()
    >>> model.fit(X_scaled)
    >>> scores = -model.score_samples(X_scaled)
    >>> result = mass_volume_auc(scores, X_scaled)
    >>> print(f"MV-AUC: {result['auc']:.3f}")

    Notes
    -----
    The volume values are in absolute data units (not normalized fractions).
    This means AUC values will scale with the data's bounding box volume and are not
    directly comparable across datasets with different scales.
    """
    scores = validate_scores(scores)
    data = validate_data(data)

    if len(scores) != len(data):
        raise ValueError(
            f"Length mismatch: {len(scores)} scores vs {len(data)} data points"
        )

    # Compute volume support (bounding box volume) internally
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    ranges = data_max - data_min
    # Handle zero ranges (all values identical in a dimension)
    ranges = np.maximum(ranges, 1e-60)
    volume_support = float(np.prod(ranges)) + 1e-60

    # Validation for parameters
    if not 0 <= alpha_min < alpha_max <= 1:
        raise ValueError(
            f"Invalid alpha range: alpha_min={alpha_min}, alpha_max={alpha_max}"
        )

    # Warn about potential scaling issues
    if volume_support > 100:
        import warnings

        warnings.warn(
            f"Large volume support ({volume_support:.2f}) detected. "
            "Consider normalizing your data for better interpretability.",
            UserWarning,
        )

    rng = np.random.default_rng(random_state)

    # Generate uniform samples in data bounding box (reuse computed min/max)
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

    # Define target mass levels (alpha values) - focus on high-mass region
    axis_alpha = np.linspace(alpha_min, alpha_max, n_thresholds)

    # Sort scores in ascending order with negative indexing (matches external implementation)
    n_samples = len(scores)
    scores_argsort = scores.argsort()  # Ascending order

    # Compute mass-volume curve following reference implementation
    masses = np.zeros(n_thresholds)
    volumes = np.zeros(n_thresholds)

    mass = 0
    cpt = 0
    threshold = (
        scores[scores_argsort[-1]] if n_samples > 0 else 0
    )  # Highest score initially

    for i in range(n_thresholds):
        # Find threshold corresponding to target mass
        while mass < axis_alpha[i] and cpt < n_samples:
            cpt += 1
            threshold = scores[scores_argsort[-cpt]]  # Negative indexing from highest
            mass = cpt / n_samples

        masses[i] = mass
        # Volume: absolute volume in data units (matches external implementation)
        volumes[i] = (uniform_scores >= threshold).sum() / n_mc_samples * volume_support

    # Compute area under curve using target masses (axis_alpha) for compatibility with reference
    # This matches the reference implementation which uses axis_alpha for AUC calculation
    auc = compute_auc(axis_alpha, volumes)

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
