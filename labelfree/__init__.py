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

from .mass_volume import mass_volume_curve
from .excess_mass import excess_mass_curve
from .ireos import ireos, sireos
from .stability import ranking_stability, top_k_stability
from .base import LabelFreeMetric
from .utils import validate_scores, validate_data, compute_auc

__version__ = "0.0.1"

__all__ = [
    # Core metrics
    "mass_volume_curve",
    "excess_mass_curve",
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
