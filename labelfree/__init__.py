"""Label-Free Metrics for Anomaly Detection.

This package provides label-free evaluation metrics for anomaly detection:
- Mass-Volume curves: Trade-off between data coverage and space volume
- Excess-Mass curves: Measure of density capture effectiveness
- IREOS: Internal Relative Evaluation of Outlier Solutions
- SIREOS: Similarity-based IREOS (faster variant)
- Ranking stability: Consistency of rankings under data perturbation
- Top-K stability: Consistency of top anomalies

All metrics work without ground truth labels.
"""

from labelfree.metrics import (
    mass_volume_auc,
    mass_exceedance_auc,
    ranking_stability,
    top_k_stability,
    ireos,
    sireos,
    sireos_separation,
)
from .utils import validate_scores, validate_data, compute_auc, compute_volume_support

__version__ = "0.0.1"

__all__ = [
    # Core metrics
    "mass_volume_auc",
    "mass_exceedance_auc",
    "ireos",
    "sireos",
    "sireos_separation",
    "ranking_stability",
    "top_k_stability",
    # Utilities
    "validate_scores",
    "validate_data",
    "compute_auc",
    "compute_volume_support",
]
