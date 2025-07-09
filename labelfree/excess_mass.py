"""Excess-Mass curve implementation."""

import numpy as np
from typing import Dict
from .utils import validate_scores, compute_auc


def excess_mass_curve(
    scores: np.ndarray,
    volume_scores: np.ndarray,
    volume_support: float = 1.0,
    t_max: float = 0.9,
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
    volume_support : float, default=1.0
        Total volume of the data space support.
    t_max : float, default=0.9
        Maximum level value for AUC computation.

    Returns
    -------
    dict with keys:
        - 't': Level values
        - 'em': EM values at each level
        - 'auc': Area under EM curve up to t_max
        - 'amax': Index where EM <= t_max
    """
    scores = validate_scores(scores)
    volume_scores = validate_scores(volume_scores, "volume_scores")

    n_samples = len(scores)
    n_generated = len(volume_scores)

    # Generate levels t following reference implementation
    # t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
    t = np.arange(0, 100 / volume_support, 0.01 / volume_support)

    # Find unique score thresholds from data
    scores_unique = np.unique(scores)

    # Initialize EM array following reference
    em = np.zeros(len(t))
    em[0] = 1.0

    # Compute excess mass following reference implementation
    for u in scores_unique:
        # Vectorized computation over all t values
        em = np.maximum(
            em,
            (scores > u).sum() / n_samples
            - t * (volume_scores > u).sum() / n_generated * volume_support,
        )

    # Find amax: index where EM first drops below or equals t_max
    # This follows the reference implementation exactly
    amax = np.argmax(em <= t_max) + 1
    if amax == 1:
        # Failed to achieve t_max (EM never drops below t_max)
        # Use full range in this case
        amax = len(t)

    # Compute AUC
    auc = compute_auc(t[:amax], em[:amax])

    return {"t": t, "em": em, "auc": auc, "amax": amax}
