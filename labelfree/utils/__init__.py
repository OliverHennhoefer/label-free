"""Utility functions for label-free metrics.

This module provides input validation, mathematical computation, and score
normalization utilities used across the label-free metrics package.
"""

# Import all functions from specialized modules for backward compatibility
from .validation import validate_scores, validate_data
from .computation import compute_auc, compute_volume_support
from .normalization import (
    normalize_outlier_scores,
    auto_normalize_scores,
    _should_invert_scores,
)

# Expose all functions at package level
__all__ = [
    # Validation functions
    "validate_scores",
    "validate_data",
    # Computation functions
    "compute_auc",
    "compute_volume_support",
    # Normalization functions
    "normalize_outlier_scores",
    "auto_normalize_scores",
]