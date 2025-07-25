import numpy as np
from typing import Tuple, Optional
from sklearn.datasets import fetch_openml
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def load_shuttle_data(
    n_samples: int = 1000,
    n_anomalies: int = 50,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load shuttle dataset and convert to anomaly detection format.
    
    The shuttle dataset contains space shuttle operational data with natural
    anomaly classes. Class 1 (normal operations) vs classes 2-7 (anomalies).
    
    Parameters
    ----------
    n_samples : int, default=1000
        Total number of samples to return (subsampled from full dataset).
    n_anomalies : int, default=50
        Number of anomaly samples to include.
    random_state : int, optional
        Random seed for reproducible sampling.
        
    Returns
    -------
    X : array of shape (n_samples, 9)
        Data points from shuttle dataset.
    y : array of shape (n_samples,)
        Labels where 0=normal (class 1), 1=anomaly (classes 2-7).
    """
    # Load full shuttle dataset
    X_full, y_full = fetch_openml("shuttle", version=1, return_X_y=True, as_frame=False)
    
    # Convert to numeric and handle any missing values
    X_full = X_full.astype(np.float32)
    y_full = y_full.astype(int)
    
    # Class 1 = normal operations, classes 2-7 = various anomalies
    normal_mask = (y_full == 1)
    anomaly_mask = (y_full != 1)
    
    X_normal = X_full[normal_mask]
    X_anomaly = X_full[anomaly_mask]
    
    rng = np.random.default_rng(random_state)
    
    # Calculate number of normal samples needed
    n_normal = n_samples - n_anomalies
    
    # Sample normal data
    if len(X_normal) < n_normal:
        # If not enough normal samples, use all and adjust anomaly count
        normal_indices = np.arange(len(X_normal))
        n_normal = len(X_normal)
        n_anomalies = n_samples - n_normal
    else:
        normal_indices = rng.choice(len(X_normal), size=n_normal, replace=False)
    
    # Sample anomaly data
    if len(X_anomaly) < n_anomalies:
        # If not enough anomaly samples, use all available
        anomaly_indices = np.arange(len(X_anomaly))
        n_anomalies = len(X_anomaly)
    else:
        anomaly_indices = rng.choice(len(X_anomaly), size=n_anomalies, replace=False)
    
    # Combine selected samples
    X_selected_normal = X_normal[normal_indices]
    X_selected_anomaly = X_anomaly[anomaly_indices]
    
    X = np.vstack([X_selected_normal, X_selected_anomaly])
    y = np.hstack([np.zeros(len(X_selected_normal)), np.ones(len(X_selected_anomaly))])
    
    # Shuffle the data
    indices = np.arange(len(X))
    rng.shuffle(indices)
    
    return X[indices], y[indices]


def generate_anomaly_scores(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "distance",
    noise_level: float = 0.1,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate anomaly scores for shuttle dataset.
    
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Data points from shuttle dataset.
    y : array of shape (n_samples,)
        True labels (0=normal, 1=anomaly).
    method : {'distance', 'random', 'perfect', 'isolation_forest', 'lof'}, default='distance'
        How to generate scores.
    noise_level : float, default=0.1
        Amount of noise to add to scores.
    random_state : int, optional
        Random seed.
        
    Returns
    -------
    scores : array of shape (n_samples,)
        Anomaly scores (higher = more anomalous).
    """
    rng = np.random.default_rng(random_state)
    
    if method == "distance":
        # Score based on distance from center of normal points
        normal_center = X[y == 0].mean(axis=0)
        scores = np.linalg.norm(X - normal_center, axis=1)
        
    elif method == "random":
        # Random scores
        scores = rng.random(len(X))
        
    elif method == "perfect":
        # Perfect scores based on labels
        scores = y.astype(float)
        
    elif method == "isolation_forest":
        # Use IsolationForest for realistic scores
        clf = IsolationForest(random_state=random_state, contamination=0.1)
        scores = -clf.fit(X).decision_function(X)  # Invert so higher = more anomalous
        
    elif method == "lof":
        # Use Local Outlier Factor for realistic scores  
        clf = LocalOutlierFactor(contamination=0.1, novelty=False)
        scores = -clf.fit_predict(X)  # Convert to positive scores
        # Get actual LOF scores
        scores = -clf.negative_outlier_factor_
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Add noise
    if noise_level > 0 and method not in ["perfect"]:
        scores += rng.normal(0, noise_level * scores.std(), size=len(scores))
    
    # Ensure positive scores and normalize
    scores = scores - scores.min() + 1e-8
    
    # Normalize scores based on method
    if method == "distance":
        # Normalize to [0,1] range for IREOS compatibility
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
        scores = scores_normalized * 5.0  # Scale for reasonable range
    elif method in ["isolation_forest", "lof"]:
        # Keep realistic algorithm scores but ensure positive range
        scores = scores - scores.min() + 0.1
    elif method == "perfect":
        # Perfect scores should clearly separate anomalies from normal points
        scores = y.astype(float) * 2.0 + rng.normal(0, 0.01, len(y))
    
    # Final normalization for IREOS: ensure scores are in [0,1] range
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    return scores


