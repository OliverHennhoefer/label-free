import numpy as np
import pytest
from labelfree.metrics.mass_exceedance import mass_exceedance_auc
from .synthetic_data import make_blobs_with_anomalies, make_anomaly_scores


class TestExcessMass:
    """Test suite for Excess-Mass curve."""

    def test_basic_functionality(self):
        """Test basic EM curve computation."""
        # Generate data and scores
        X, y = make_blobs_with_anomalies(n_samples=300, n_anomalies=30, random_state=42)
        scores = make_anomaly_scores(X, y, method="distance", random_state=42)

        # Generate volume scores (scores on uniform samples)
        rng = np.random.default_rng(42)
        volume_scores = rng.standard_normal(1000)

        # Compute EM curve
        result = mass_exceedance_auc(scores, volume_scores)

        # Check output
        assert set(result.keys()) == {"t", "em", "auc", "amax"}
        assert len(result["t"]) > 0  # Length depends on volume_support
        assert len(result["em"]) == len(result["t"])
        assert isinstance(result["auc"], float)
        assert isinstance(result["amax"], (int, np.integer))

        # Check that first EM value is 1.0
        assert result["em"][0] == 1.0

    def test_edge_cases(self):
        """Test edge cases."""
        # All scores identical
        scores = np.ones(100)
        volume_scores = np.ones(100)

        result = mass_exceedance_auc(scores, volume_scores)

        # Should still run without errors
        assert result["auc"] >= 0
        assert result["em"][0] == 1.0

        # Empty volume scores
        with pytest.raises(ValueError):
            mass_exceedance_auc(scores, [])
