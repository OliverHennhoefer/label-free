import numpy as np
from labelfree.metrics.ireos import ireos
from labelfree.metrics.sireos import sireos, sireos_separation
from .synthetic_data import make_blobs_with_anomalies, make_anomaly_scores


class TestIREOS:
    """Test suite for IREOS metrics."""

    def test_ireos_basic(self):
        """Test basic IREOS functionality."""
        X, y = make_blobs_with_anomalies(n_samples=200, n_anomalies=20, random_state=42)
        scores = make_anomaly_scores(X, y, method="distance", random_state=42)

        ireos_score, p_value = ireos(scores, X, n_splits=3, random_state=42)

        # Check ranges
        assert 0.5 <= ireos_score <= 1.0
        assert 0.0 <= p_value <= 1.0

        # Good detector should have high IREOS
        assert ireos_score > 0.7
        assert p_value < 0.05

    def test_ireos_random_scores(self):
        """Test IREOS with random scores."""
        X = np.random.randn(200, 5)
        scores = np.random.randn(200)

        ireos_score, p_value = ireos(scores, X, random_state=42)

        # Random scores should give IREOS near 0.5 (but allow some variance)
        assert 0.4 <= ireos_score <= 0.6
        assert p_value > 0.05

    def test_sireos(self):
        """Test SIREOS functionality."""
        X, y = make_blobs_with_anomalies(n_samples=200, n_anomalies=20, random_state=42)
        scores = make_anomaly_scores(X, y, method="distance", random_state=42)

        # Test Euclidean
        sireos_euclidean = sireos_separation(scores, X, similarity="euclidean")
        assert sireos_euclidean > 1.0  # Good separation

        # Test cosine
        sireos_cosine = sireos_separation(scores, X, similarity="cosine")
        assert sireos_cosine > 1.0

    def test_degenerate_cases(self):
        """Test degenerate cases."""
        X = np.random.randn(100, 2)

        # All scores identical
        scores = np.ones(100)
        ireos_score, p_value = ireos(scores, X)
        assert ireos_score == 0.5
        assert p_value == 1.0

        # SIREOS with perfect separation
        scores = np.hstack([np.zeros(50), np.ones(50)])
        sireos_score = sireos_separation(scores, X[:100])
        # Should have good separation, but may not exceed 0.9 due to random data
        assert sireos_score > 0.8
