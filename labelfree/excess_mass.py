"""Excess-Mass curve implementation."""
import numpy as np
from typing import Dict, Optional
from .utils import validate_scores, compute_auc


def excess_mass_curve(
    scores: np.ndarray,
    volume_scores: np.ndarray,
    n_levels: int = 100,
    volume: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Compute Excess-Mass curve for anomaly detection evaluation.
    
    The Excess-Mass at level t measures how well the scoring function
    captures high-density regions: EM(t) = P(score > s) - t * V(score > s)
    where V is the volume measure.
    
    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Anomaly scores on actual data.
    volume_scores : array-like of shape (n_uniform_samples,)
        Anomaly scores on uniform samples (for volume estimation).
    n_levels : int, default=100
        Number of levels to evaluate.
    volume : float, default=1.0
        Total volume of the data space.
        
    Returns
    -------
    dict with keys:
        - 'levels': Level values t
        - 'excess_mass': EM values at each level
        - 'auc': Area under EM curve (higher is better)
        - 'max_em': Maximum excess mass achieved
    """
    scores = validate_scores(scores)
    volume_scores = validate_scores(volume_scores, "volume_scores")
    
    # Generate levels
    levels = np.linspace(0, 100.0 / volume, n_levels)
    
    # Find unique score thresholds from data
    unique_thresholds = np.unique(scores)
    
    # Compute excess mass for each level
    excess_masses = np.zeros(n_levels)
    
    for i, level in enumerate(levels):
        # Find optimal threshold for this level
        max_em = -np.inf
        
        for threshold in unique_thresholds:
            # P(score > threshold) on data
            p_data = (scores > threshold).mean()
            
            # P(score > threshold) on uniform
            p_uniform = (volume_scores > threshold).mean()
            
            # Excess mass at this threshold and level
            em = p_data - level * p_uniform * volume
            max_em = max(max_em, em)
        
        excess_masses[i] = max_em
    
    # Area under curve
    auc = compute_auc(levels, excess_masses)
    
    return {
        'levels': levels,
        'excess_mass': excess_masses,
        'auc': auc,
        'max_em': excess_masses.max()
    }