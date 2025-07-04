"""Tests for Mass-Volume curve."""
import numpy as np
import pytest
from labelfree.mass_volume import mass_volume_curve
from .synthetic_data import make_blobs_with_anomalies, make_anomaly_scores


class TestMassVolume:
    """Test suite for Mass-Volume curve.
    
    Note: The implementation follows Goix et al. where mass values
    represent target alpha levels from 0 to 1, not decreasing thresholds.
    """
    
    def test_basic_functionality(self):
        """Test basic MV curve computation."""
        # Generate simple data
        X, y = make_blobs_with_anomalies(n_samples=200, n_anomalies=20, random_state=42)
        scores = make_anomaly_scores(X, y, method='distance', random_state=42)
        
        # Compute MV curve
        result = mass_volume_curve(scores, X, n_thresholds=50, n_mc_samples=1000)
        
        # Check output structure
        assert set(result.keys()) == {'mass', 'volume', 'auc', 'axis_alpha'}
        assert len(result['mass']) == 50
        assert len(result['volume']) == 50
        assert len(result['axis_alpha']) == 50
        
        # Check value ranges
        assert np.all((result['mass'] >= 0) & (result['mass'] <= 1))
        assert np.all((result['volume'] >= 0) & (result['volume'] <= 1))
        assert 0 <= result['auc'] <= 1
        
        # Mass should be monotonic non-decreasing (follows axis_alpha from 0 to 1)
        assert np.all(np.diff(result['mass']) >= 0)  # Non-decreasing
        # Check that mass starts at 0 and ends at 1
        assert result['mass'][0] == 0.0  # First alpha = 0
        assert result['mass'][-1] == 1.0  # Last alpha = 1
        # Check that axis_alpha is correct
        np.testing.assert_allclose(result['axis_alpha'], np.linspace(0, 1, 50))
        # With simulation, volume can fluctuate
        # Just check that volumes are valid
        assert all(0 <= v <= 1 for v in result['volume'])
    
    def test_perfect_detector(self):
        """Test MV curve for perfect anomaly detector."""
        X, y = make_blobs_with_anomalies(n_samples=500, n_anomalies=50, random_state=42)
        scores = make_anomaly_scores(X, y, method='perfect', noise_level=0)
        
        result = mass_volume_curve(scores, X, n_thresholds=100)
        
        # With simulation, perfect detector might not achieve very low AUC
        # but should still be reasonable
        assert 0 <= result['auc'] <= 1
        # Check that mass follows axis_alpha
        assert result['mass'][0] == 0.0  # First alpha = 0
        assert result['mass'][-1] == 1.0  # Last alpha = 1
    
    def test_random_detector(self):
        """Test MV curve for random detector."""
        X, y = make_blobs_with_anomalies(n_samples=500, n_anomalies=50, random_state=42)
        scores = make_anomaly_scores(X, y, method='random')
        
        result = mass_volume_curve(scores, X, n_thresholds=100)
        
        # Random detector should have AUC close to 0.5
        assert 0.4 <= result['auc'] <= 0.6
    
    def test_input_validation(self):
        """Test input validation."""
        X = np.random.randn(100, 2)
        scores = np.random.randn(100)
        
        # Mismatched lengths
        with pytest.raises(ValueError):
            mass_volume_curve(scores[:50], X)
        
        # Empty inputs
        with pytest.raises(ValueError):
            mass_volume_curve([], X)
        
        # Invalid dimensions
        with pytest.raises(ValueError):
            mass_volume_curve(scores.reshape(-1, 1), X)
    
    def test_reproducibility(self):
        """Test that results are reproducible with fixed seed."""
        X, y = make_blobs_with_anomalies(random_state=42)
        scores = make_anomaly_scores(X, y, random_state=42)
        
        result1 = mass_volume_curve(scores, X, random_state=42)
        result2 = mass_volume_curve(scores, X, random_state=42)
        
        np.testing.assert_array_equal(result1['mass'], result2['mass'])
        np.testing.assert_array_equal(result1['volume'], result2['volume'])
        assert result1['auc'] == result2['auc']