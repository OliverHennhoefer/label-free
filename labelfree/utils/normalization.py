"""Score normalization utilities for outlier detection algorithms."""

import numpy as np
import warnings
from typing import Literal, Optional
from scipy import stats
from .validation import validate_scores


def normalize_outlier_scores(
    scores: np.ndarray,
    method: Literal["linear", "gaussian", "minmax", "sigmoid"] = "gaussian",
    invert: bool = False,
    clip_range: Optional[tuple] = None,
) -> np.ndarray:
    """
    Normalize outlier scores to [0, 1] probability-like values.

    Implements normalization strategies based on Kriegel et al. (2011) framework
    to make outlier scores interpretable as probability values and comparable
    across different outlier detection algorithms.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Raw outlier scores from any outlier detection algorithm.
    method : {"linear", "gaussian", "minmax", "sigmoid"}, default="gaussian"
        Normalization method:
        - "linear": Linear scaling based on percentiles (robust to outliers)
        - "gaussian": Statistical scaling assuming normal distribution
        - "minmax": Simple min-max scaling to [0, 1]
        - "sigmoid": Sigmoid transformation with mean centering
    invert : bool, default=False
        If True, inverts scores (1 - normalized_score). Use when lower raw
        scores indicate higher anomaly probability (e.g., negative log-likelihood).
    clip_range : tuple, optional
        If provided, clips normalized scores to (min, max) range.
        Default is (1e-6, 1-1e-6) to avoid exact 0/1 values.

    Returns
    -------
    np.ndarray
        Normalized scores in [0, 1] range where higher values indicate
        higher anomaly probability.

    Examples
    --------
    >>> # IsolationForest returns negative scores (lower = more anomalous)
    >>> from sklearn.ensemble import IsolationForest
    >>> model = IsolationForest()
    >>> raw_scores = model.score_samples(data)
    >>> normalized = normalize_outlier_scores(raw_scores, invert=True)

    >>> # LOF returns values around 1 (higher = more anomalous)
    >>> from sklearn.neighbors import LocalOutlierFactor
    >>> model = LocalOutlierFactor(novelty=True)
    >>> raw_scores = model.score_samples(data)
    >>> normalized = normalize_outlier_scores(raw_scores, method="gaussian")

    Notes
    -----
    Based on Kriegel et al. "Interpreting and Unifying Outlier Scores" (SDM 2011).
    The normalization makes scores from different algorithms comparable and
    interpretable as anomaly probabilities for use with IREOS and other metrics.
    """
    scores = validate_scores(scores, "scores")

    if method == "minmax":
        # Simple min-max normalization
        score_min, score_max = scores.min(), scores.max()
        if score_max == score_min:
            # All scores are identical
            normalized = np.full_like(scores, 0.5)
        else:
            normalized = (scores - score_min) / (score_max - score_min)

    elif method == "linear":
        # Robust linear scaling using percentiles (handles outliers better)
        q25, q75 = np.percentile(scores, [25, 75])
        if q75 == q25:
            # No variance in scores
            normalized = np.full_like(scores, 0.5)
        else:
            # Scale using IQR, then clip to [0, 1]
            normalized = (scores - q25) / (q75 - q25)
            normalized = np.clip(normalized, 0, 1)

    elif method == "gaussian":
        # Statistical scaling assuming Gaussian distribution
        mean, std = scores.mean(), scores.std()
        if std == 0:
            # No variance in scores
            normalized = np.full_like(scores, 0.5)
        else:
            # Convert to z-scores, then to probabilities via CDF
            z_scores = (scores - mean) / std
            normalized = stats.norm.cdf(z_scores)

    elif method == "sigmoid":
        # Sigmoid transformation with mean centering
        mean = scores.mean()
        # Use MAD (median absolute deviation) for robust scaling
        mad = np.median(np.abs(scores - np.median(scores)))
        if mad == 0:
            normalized = np.full_like(scores, 0.5)
        else:
            # Sigmoid with robust scaling factor
            scale = 1.4826 * mad  # MAD to std conversion factor
            normalized = 1 / (1 + np.exp(-(scores - mean) / scale))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Invert scores if needed (for algorithms where lower = more anomalous)
    if invert:
        normalized = 1 - normalized

    # Apply clipping to avoid exact 0/1 values
    if clip_range is None:
        clip_range = (1e-6, 1 - 1e-6)

    if clip_range is not None:
        normalized = np.clip(normalized, clip_range[0], clip_range[1])

    return normalized


def auto_normalize_scores(
    scores: np.ndarray,
    algorithm_name: Optional[str] = None,
    auto_invert: bool = True,
) -> np.ndarray:
    """
    Automatically normalize outlier scores with algorithm-specific handling.

    Provides convenience function that automatically selects appropriate
    normalization method and inversion based on the outlier detection algorithm.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Raw outlier scores from outlier detection algorithm.
    algorithm_name : str, optional
        Name of the algorithm that generated scores. Used to automatically
        determine appropriate normalization and inversion. Supported:
        - "isolation_forest", "iforest": Uses gaussian, inverts (negative scores)
        - "lof", "local_outlier_factor": Uses gaussian, no inversion
        - "one_class_svm", "ocsvm": Uses sigmoid, inverts if negative
        - "knn": Uses linear, no inversion
        If None, uses heuristics based on score distribution.
    auto_invert : bool, default=True
        If True and algorithm_name is None, automatically determines whether
        to invert scores based on their distribution.

    Returns
    -------
    np.ndarray
        Normalized scores in [0, 1] range.

    Examples
    --------
    >>> # Automatic handling for IsolationForest
    >>> normalized = auto_normalize_scores(raw_scores, "isolation_forest")

    >>> # Automatic detection without providing algorithm name
    >>> normalized = auto_normalize_scores(raw_scores)
    """
    scores = validate_scores(scores, "scores")

    # Algorithm-specific settings
    algorithm_configs = {
        "isolation_forest": {"method": "gaussian", "invert": True},
        "iforest": {"method": "gaussian", "invert": True},
        "lof": {"method": "gaussian", "invert": False},
        "local_outlier_factor": {"method": "gaussian", "invert": False},
        "one_class_svm": {"method": "sigmoid", "invert": None},  # Auto-detect
        "ocsvm": {"method": "sigmoid", "invert": None},
        "knn": {"method": "linear", "invert": False},
        "k_nearest_neighbors": {"method": "linear", "invert": False},
    }

    if algorithm_name and algorithm_name.lower() in algorithm_configs:
        config = algorithm_configs[algorithm_name.lower()]
        method = config["method"]
        invert = config["invert"]

        # Auto-detect inversion for some algorithms
        if invert is None:
            invert = auto_invert and _should_invert_scores(scores)
    else:
        # Use heuristics to determine normalization method
        method = "gaussian"  # Generally robust default
        invert = auto_invert and _should_invert_scores(scores)

        if algorithm_name:
            warnings.warn(
                f"Unknown algorithm '{algorithm_name}'. Using default normalization. "
                f"Supported algorithms: {list(algorithm_configs.keys())}"
            )

    return normalize_outlier_scores(scores, method=method, invert=invert)


def _should_invert_scores(scores: np.ndarray) -> bool:
    """
    Heuristically determine if scores should be inverted.

    Returns True if scores appear to be "distance-like" (lower = more anomalous)
    rather than "similarity-like" (higher = more anomalous).
    """
    # Check if scores are predominantly negative
    if np.mean(scores < 0) > 0.7:
        return True

    # Check score distribution characteristics
    # Distance-like scores often have skewed distributions
    skewness = stats.skew(scores)

    # If scores are heavily left-skewed, they might be inverted
    if skewness < -1.0:
        return True

    return False
