"""Utility functions for label-free metrics."""

import numpy as np


def validate_scores(scores: np.ndarray, name: str = "scores") -> np.ndarray:
    """Ensure scores are valid 1D array."""
    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError(f"{name} must be 1D array, got shape {scores.shape}")
    if len(scores) == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(scores).all():
        raise ValueError(f"{name} contains non-finite values")
    return scores


def validate_data(data: np.ndarray, name: str = "data") -> np.ndarray:
    """Ensure data is valid 2D array."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2:
        raise ValueError(f"{name} must be 2D array, got shape {data.shape}")
    if len(data) == 0:
        raise ValueError(f"{name} cannot be empty")
    return data


def compute_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute area under curve using trapezoidal rule."""
    # Sort by x values for proper integration
    idx = np.argsort(x)
    return float(np.trapezoid(y[idx], x[idx]))


def compute_volume_support(data: np.ndarray, offset: float = 1e-60) -> float:
    """
    Compute the volume of the bounding box containing all data points.
    
    This is used for Mass-Volume curve calculations where the volume represents
    the absolute size of the data space in original units.
    
    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Data points to compute bounding box for.
    offset : float, default=1e-60
        Small offset added to prevent division by zero in edge cases.
        
    Returns
    -------
    float
        Volume of the bounding box (product of feature ranges).
        
    Examples
    --------
    >>> data = np.array([[0, 0], [1, 2], [3, 1]])
    >>> compute_volume_support(data)
    6.0
    """
    data = validate_data(data)
    
    # Compute range for each feature
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    ranges = data_max - data_min
    
    # Handle zero ranges (all values identical in a dimension)
    ranges = np.maximum(ranges, offset)
    
    # Volume is product of all ranges
    return float(np.prod(ranges)) + offset
