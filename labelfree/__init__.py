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

from labelfree.metrics.mass_volume import mass_volume_auc
from labelfree.metrics.mass_exceedance import mass_exceedance_auc
from labelfree.metrics.ireos import ireos, sireos
from labelfree.metrics.stability import ranking_stability, top_k_stability
from labelfree.metrics.base import LabelFreeMetric
from .utils import validate_scores, validate_data, compute_auc

__version__ = "0.0.1"

__all__ = [
    # Core metrics
    "mass_volume_auc",
    "mass_exceedance_auc",
    "ireos",
    "sireos",
    "ranking_stability",
    "top_k_stability",
    # Base classes
    "LabelFreeMetric",
    # Utilities
    "validate_scores",
    "validate_data",
    "compute_auc",
]
