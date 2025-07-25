"""Input validation utilities for label-free metrics."""

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
