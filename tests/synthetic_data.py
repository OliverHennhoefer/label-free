"""Generate synthetic data for testing label-free metrics."""
import numpy as np
from typing import Tuple, Optional


def make_blobs_with_anomalies(
    n_samples: int = 1000,
    n_features: int = 2,
    n_anomalies: int = 50,
    centers: int = 3,
    anomaly_factor: float = 3.0,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate blob data with anomalies far from centers.
    
    Returns
    -------
    X : array of shape (n_samples + n_anomalies, n_features)
        Data points.
    y : array of shape (n_samples + n_anomalies,)
        Labels where 0=normal, 1=anomaly.
    """
    from sklearn.datasets import make_blobs
    
    # Generate normal data
    X_normal, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        random_state=random_state
    )
    
    # Generate anomalies far from normal data
    rng = np.random.default_rng(random_state)
    
    # Find data range
    data_min = X_normal.min(axis=0)
    data_max = X_normal.max(axis=0)
    data_range = data_max - data_min
    
    # Place anomalies outside normal range
    X_anomalies = rng.uniform(
        data_min - anomaly_factor * data_range,
        data_max + anomaly_factor * data_range,
        size=(n_anomalies, n_features)
    )
    
    # Combine data
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([np.zeros(n_samples), np.ones(n_anomalies)])
    
    return X, y


def make_moons_with_noise(
    n_samples: int = 1000,
    noise_level: float = 0.3,
    n_anomalies: int = 50,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two moons data with uniform noise as anomalies.
    
    Returns
    -------
    X : array of shape (n_samples + n_anomalies, 2)
        Data points.
    y : array of shape (n_samples + n_anomalies,)
        Labels where 0=normal, 1=anomaly.
    """
    from sklearn.datasets import make_moons
    
    # Generate moons
    X_normal, _ = make_moons(
        n_samples=n_samples,
        noise=noise_level,
        random_state=random_state
    )
    
    # Generate uniform anomalies
    rng = np.random.default_rng(random_state)
    X_anomalies = rng.uniform(-3, 3, size=(n_anomalies, 2))
    
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([np.zeros(n_samples), np.ones(n_anomalies)])
    
    return X, y


def make_anomaly_scores(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'distance',
    noise_level: float = 0.1,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic anomaly scores for testing.
    
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Data points.
    y : array of shape (n_samples,)
        True labels (0=normal, 1=anomaly).
    method : {'distance', 'random', 'perfect'}, default='distance'
        How to generate scores.
    noise_level : float, default=0.1
        Amount of noise to add.
        
    Returns
    -------
    scores : array of shape (n_samples,)
        Anomaly scores (higher = more anomalous).
    """
    rng = np.random.default_rng(random_state)
    
    if method == 'distance':
        # Score based on distance from center of normal points
        normal_center = X[y == 0].mean(axis=0)
        scores = np.linalg.norm(X - normal_center, axis=1)
        
    elif method == 'random':
        # Random scores
        scores = rng.random(len(X))
        
    elif method == 'perfect':
        # Perfect scores based on labels
        scores = y.astype(float)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Add noise
    if noise_level > 0:
        scores += rng.normal(0, noise_level * scores.std(), size=len(scores))
    
    return scores