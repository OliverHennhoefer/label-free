import numpy as np
import pytest
from labelfree.metrics.mass_volume import mass_volume_auc
from labelfree.utils import compute_volume_support
from .synthetic_data import make_blobs_with_anomalies, make_anomaly_scores


class TestMassVolume:
    """Test suite for Mass-Volume curve.

    Note: The implementation follows Goix et al. algorithm and is designed
    for high-mass region evaluation (typically alpha_min=0.9, alpha_max=0.999).
    """

    def test_basic_functionality(self):
        """Test basic MV curve computation."""
        # Generate simple data
        X, y = make_blobs_with_anomalies(n_samples=200, n_anomalies=20, random_state=42)
        scores = make_anomaly_scores(X, y, method="distance", random_state=42)
        volume_support = compute_volume_support(X)

        # Compute MV curve with default high-mass region
        result = mass_volume_auc(
            scores, X, volume_support=volume_support,
            alpha_min=0.9, alpha_max=0.999, n_thresholds=50, n_mc_samples=1000
        )

        # Check output structure
        assert set(result.keys()) == {"mass", "volume", "auc", "axis_alpha"}
        assert len(result["mass"]) == 50
        assert len(result["volume"]) == 50
        assert len(result["axis_alpha"]) == 50

        # Check value ranges for high-mass region
        assert np.all((result["mass"] >= 0.9) & (result["mass"] <= 1.0))
        assert np.all(result["volume"] >= 0)  # Volume can be > 1 due to volume_support scaling
        assert result["auc"] >= 0  # AUC can be > 1 due to volume_support scaling

        # Mass should be monotonic non-decreasing
        assert np.all(np.diff(result["mass"]) >= 0)
        # Check that axis_alpha range is correct
        expected_alpha = np.linspace(0.9, 0.999, 50)
        np.testing.assert_allclose(result["axis_alpha"], expected_alpha, rtol=1e-10)

    def test_perfect_detector(self):
        """Test MV curve for perfect anomaly detector."""
        X, y = make_blobs_with_anomalies(n_samples=500, n_anomalies=50, random_state=42)
        scores = make_anomaly_scores(X, y, method="perfect", noise_level=0)
        volume_support = compute_volume_support(X)

        result = mass_volume_auc(
            scores, X, volume_support=volume_support, n_thresholds=100
        )

        # Check basic properties
        assert result["auc"] >= 0
        # Mass should be in high-mass region
        assert np.all(result["mass"] >= 0.9)
        assert np.all(result["mass"] <= 1.0)

    def test_random_detector(self):
        """Test MV curve for random detector."""
        X, y = make_blobs_with_anomalies(n_samples=500, n_anomalies=50, random_state=42)
        scores = make_anomaly_scores(X, y, method="random")
        volume_support = compute_volume_support(X)

        result = mass_volume_auc(
            scores, X, volume_support=volume_support, n_thresholds=100
        )

        # Random detector should produce reasonable results
        assert result["auc"] >= 0

    def test_input_validation(self):
        """Test input validation."""
        X = np.random.randn(100, 2)
        scores = np.random.randn(100)
        volume_support = compute_volume_support(X)

        # Mismatched lengths
        with pytest.raises(ValueError):
            mass_volume_auc(scores[:50], X, volume_support)

        # Empty inputs
        with pytest.raises(ValueError):
            mass_volume_auc([], X, volume_support)

        # Invalid dimensions
        with pytest.raises(ValueError):
            mass_volume_auc(scores.reshape(-1, 1), X, volume_support)
        
        # Invalid alpha range
        with pytest.raises(ValueError):
            mass_volume_auc(scores, X, volume_support, alpha_min=0.99, alpha_max=0.9)
        
        # Invalid volume_support
        with pytest.raises(ValueError):
            mass_volume_auc(scores, X, volume_support=-1.0)

    def test_reproducibility(self):
        """Test that results are reproducible with fixed seed."""
        X, y = make_blobs_with_anomalies(random_state=42)
        scores = make_anomaly_scores(X, y, random_state=42)
        volume_support = compute_volume_support(X)

        result1 = mass_volume_auc(scores, X, volume_support, random_state=42)
        result2 = mass_volume_auc(scores, X, volume_support, random_state=42)

        np.testing.assert_array_equal(result1["mass"], result2["mass"])
        np.testing.assert_array_equal(result1["volume"], result2["volume"])
        assert result1["auc"] == result2["auc"]
    
    def test_volume_support_warning(self):
        """Test that large volume_support triggers warning."""
        X = np.random.randn(100, 2) * 100  # Large scale data
        scores = np.random.randn(100)
        volume_support = compute_volume_support(X)  # Will be large
        
        with pytest.warns(UserWarning, match="Large volume_support"):
            mass_volume_auc(scores, X, volume_support)
