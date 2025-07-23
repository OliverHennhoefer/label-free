"""Internal Relative Evaluation of Outlier Solutions (IREOS)."""

import numpy as np
from typing import Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.integrate import simpson
import warnings
from labelfree.utils import validate_scores, validate_data


def ireos(
    scores: np.ndarray,
    data: np.ndarray,
    n_outliers: Optional[int] = None,
    percentile: float = 90.0,
    gamma_min: float = 0.1,
    gamma_max: Optional[float] = None,
    n_gamma: int = 50,
    penalty: float = 100.0,
    adjustment: bool = True,
    n_monte_carlo: int = 200,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute IREOS (Internal Relative Evaluation of Outlier Solutions).

    Reference implementation following the continuous-score IREOS algorithm
    from the original Java implementation. Measures separability of selected
    outliers across multiple gamma parameters using kernel logistic regression.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Continuous anomaly scores from detector.
    data : array-like of shape (n_samples, n_features)
        Original features corresponding to scores.
    n_outliers : int, optional
        Number of top-scoring points to select as outliers.
        If None, uses percentile threshold.
    percentile : float, default=90.0
        Percentile threshold for outlier selection if n_outliers is None.
    gamma_min : float, default=0.1
        Minimum gamma value for RBF kernel.
    gamma_max : float, optional
        Maximum gamma value. If None, estimated from data.
    n_gamma : int, default=50
        Number of gamma values to sample.
    penalty : float, default=100.0
        Penalty parameter C for logistic regression.
    adjustment : bool, default=True
        Whether to apply statistical adjustment for chance.
    n_monte_carlo : int, default=200
        Number of Monte Carlo runs for statistical estimation.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ireos_score : float
        IREOS index (adjusted if adjustment=True). Higher values indicate
        better anomaly detection quality.
    p_value : float
        Statistical significance p-value against random outlier selection.
    """
    scores = validate_scores(scores)
    data = validate_data(data)

    if len(scores) != len(data):
        raise ValueError(
            f"Length mismatch: {len(scores)} scores vs {len(data)} data points"
        )

    rng = np.random.default_rng(random_state)
    
    # Standardize data
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    
    # Select outliers from continuous scores
    if n_outliers is not None:
        outlier_indices = np.argpartition(scores, -n_outliers)[-n_outliers:]
    else:
        threshold = np.percentile(scores, percentile)
        outlier_indices = np.where(scores >= threshold)[0]
    
    if len(outlier_indices) == 0:
        return 0.0, 1.0
    
    # Estimate gamma range if not provided
    if gamma_max is None:
        gamma_max = _estimate_gamma_max(X, outlier_indices, gamma_min, rng)
    
    # Create gamma values using logarithmic spacing
    gamma_values = np.logspace(
        np.log10(gamma_min), 
        np.log10(gamma_max), 
        n_gamma
    )
    
    # Compute separability curve
    separabilities = []
    for gamma in gamma_values:
        # Compute separability for each outlier at this gamma
        outlier_seps = []
        for outlier_idx in outlier_indices:
            sep = _compute_separability(X, outlier_idx, gamma, penalty, random_state)
            outlier_seps.append(sep)
        
        # Average separability across all outliers
        avg_sep = np.mean(outlier_seps)
        separabilities.append(avg_sep)
    
    separabilities = np.array(separabilities)
    
    # Compute IREOS index using numerical integration
    gamma_range = gamma_max - gamma_min
    ireos_raw = simpson(separabilities, x=gamma_values) / gamma_range
    
    # Statistical adjustment if requested
    if adjustment:
        # Estimate expected value using Monte Carlo
        expected_ireos = _estimate_expected_ireos(
            X, len(outlier_indices), gamma_values, penalty, n_monte_carlo//4, rng
        )
        
        # Apply adjustment formula: (I - E{I}) / (1 - E{I})
        ireos_adjusted = (ireos_raw - expected_ireos) / (1 - expected_ireos + 1e-10)
        ireos_score = max(0.0, ireos_adjusted)
    else:
        ireos_score = ireos_raw
    
    # Compute p-value using Monte Carlo
    p_value = _compute_p_value(
        X, len(outlier_indices), gamma_values, penalty, ireos_score, n_monte_carlo//2, rng
    )
    
    return float(ireos_score), float(p_value)


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """Compute RBF kernel matrix efficiently."""
    X_norm = np.sum(X**2, axis=1, keepdims=True)
    Y_norm = np.sum(Y**2, axis=1, keepdims=True)
    K = -2 * np.dot(X, Y.T) + X_norm + Y_norm.T
    return np.exp(-gamma * K)


def _estimate_gamma_max(
    X: np.ndarray, 
    outlier_indices: np.ndarray, 
    gamma_min: float, 
    rng: np.random.Generator
) -> float:
    """Estimate maximum gamma value where outliers become separable."""
    gamma = gamma_min
    max_attempts = 30
    
    # Test with a representative outlier
    outlier_idx = outlier_indices[0]
    
    for _ in range(max_attempts):
        separability = _compute_separability(X, outlier_idx, gamma, 100.0, None)
        if separability >= 0.95:
            break
        gamma *= 1.5
    
    return min(gamma, 1000.0)


def _compute_separability(
    X: np.ndarray, 
    outlier_idx: int, 
    gamma: float, 
    penalty: float,
    random_state: Optional[int]
) -> float:
    """Compute separability probability for a single outlier at given gamma."""
    n_samples = len(X)
    
    # Create binary labels: outlier vs all others
    y = np.full(n_samples, 0)
    y[outlier_idx] = 1
    
    # Check for degenerate case
    if np.sum(y) == 0 or np.sum(y) == n_samples:
        return 0.5
    
    try:
        # Create RBF features using Nystroem-like sampling for efficiency
        n_components = min(150, n_samples)
        rng_local = np.random.default_rng(random_state)
        sample_indices = rng_local.choice(n_samples, n_components, replace=False)
        X_sample = X[sample_indices]
        
        # Compute kernel features for all points
        K = _rbf_kernel(X, X_sample, gamma)
        
        # Train logistic regression with high regularization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                C=penalty,
                random_state=random_state,
                max_iter=500,
                solver='liblinear'
            )
            clf.fit(K, y)
            
            # Get probability for the outlier
            prob = clf.predict_proba(K[outlier_idx:outlier_idx+1])
            return float(prob[0, 1])
            
    except Exception:
        return 0.5


def _estimate_expected_ireos(
    X: np.ndarray, 
    n_outliers: int, 
    gamma_values: np.ndarray, 
    penalty: float,
    n_runs: int, 
    rng: np.random.Generator
) -> float:
    """Estimate expected IREOS for random solutions using Monte Carlo."""
    random_scores = []
    
    # Use coarse gamma grid for efficiency
    gamma_subset = gamma_values[::max(1, len(gamma_values)//15)]
    
    for _ in range(n_runs):
        # Select random outliers
        random_outliers = rng.choice(len(X), n_outliers, replace=False)
        
        # Compute separability curve for random selection
        separabilities = []
        for gamma in gamma_subset:
            outlier_seps = [
                _compute_separability(X, idx, gamma, penalty, None) 
                for idx in random_outliers
            ]
            separabilities.append(np.mean(outlier_seps))
        
        # Compute raw IREOS for random selection
        gamma_range = gamma_values[-1] - gamma_values[0]
        random_ireos = simpson(separabilities, x=gamma_subset) / gamma_range
        random_scores.append(random_ireos)
    
    return np.mean(random_scores) if random_scores else 0.5


def _compute_p_value(
    X: np.ndarray, 
    n_outliers: int, 
    gamma_values: np.ndarray, 
    penalty: float,
    observed_ireos: float, 
    n_runs: int, 
    rng: np.random.Generator
) -> float:
    """Compute statistical significance using Monte Carlo."""
    random_scores = []
    
    # Use even coarser gamma grid for p-value computation
    gamma_subset = gamma_values[::max(1, len(gamma_values)//10)]
    
    for _ in range(n_runs):
        # Select random outliers
        random_outliers = rng.choice(len(X), n_outliers, replace=False)
        
        # Compute separability curve for random selection
        separabilities = []
        for gamma in gamma_subset:
            outlier_seps = [
                _compute_separability(X, idx, gamma, penalty, None) 
                for idx in random_outliers
            ]
            separabilities.append(np.mean(outlier_seps))
        
        gamma_range = gamma_values[-1] - gamma_values[0]
        random_ireos = simpson(separabilities, x=gamma_subset) / gamma_range
        random_scores.append(random_ireos)
    
    if len(random_scores) == 0:
        return 1.0
    
    # Compute p-value: proportion of random scores >= observed
    p_value = np.mean(np.array(random_scores) >= observed_ireos)
    return max(p_value, 1e-6)


def sireos_legacy(
    scores: np.ndarray, data: np.ndarray, similarity: str = "euclidean"
) -> float:
    """
    Compute SIREOS Legacy (Similarity-based IREOS).

    Legacy implementation of SIREOS that was part of the IREOS module.
    For the official SIREOS implementation, use labelfree.metrics.sireos.sireos().

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
