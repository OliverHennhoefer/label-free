"""Abstract base class for label-free metrics."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional


class LabelFreeMetric(ABC):
    """
    Abstract base class for label-free anomaly detection metrics.

    This class provides a common interface for all label-free metrics,
    ensuring consistent behavior and validation across different metric types.
    """

    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the metric.

        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility.
        """
        self.random_state = random_state
        self.results_ = None

    @abstractmethod
    def compute(self, scores: np.ndarray, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Compute the metric.

        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Anomaly scores.
        data : array-like of shape (n_samples, n_features)
            Original data points.
        **kwargs : dict
            Additional parameters specific to the metric.

        Returns
        -------
        dict
            Dictionary containing metric results.
        """
        pass

    def fit(self, scores: np.ndarray, data: np.ndarray, **kwargs) -> "LabelFreeMetric":
        """
        Compute and store the metric results.

        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Anomaly scores.
        data : array-like of shape (n_samples, n_features)
            Original data points.
        **kwargs : dict
            Additional parameters specific to the metric.

        Returns
        -------
        self
        """
        self.results_ = self.compute(scores, data, **kwargs)
        return self

    def score(self) -> float:
        """
        Get the primary metric score.

        Returns
        -------
        float
            Primary metric score.
        """
        if self.results_ is None:
            raise ValueError("Must call fit() first")
        return self._get_primary_score()

    @abstractmethod
    def _get_primary_score(self) -> float:
        """
        Get the primary score from results.

        Returns
        -------
        float
            Primary metric score.
        """
        pass

    def get_results(self) -> Dict[str, Any]:
        """
        Get all metric results.

        Returns
        -------
        dict
            Dictionary containing all metric results.
        """
        if self.results_ is None:
            raise ValueError("Must call fit() first")
        return self.results_
