"""Label-free metrics for unsupervised anomaly detector evaluation."""

from labelfree import metrics
from labelfree.aggregation import borda_scores, robust_borda_scores
from labelfree.metrics import *  # noqa: F403
from labelfree.metrics import __all__ as _metric_names
from labelfree.scores import orient_scores, unify_scores

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "borda_scores",
    "metrics",
    "orient_scores",
    "robust_borda_scores",
    "unify_scores",
    *_metric_names,
]
