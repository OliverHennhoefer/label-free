"""Tests for Excess-Mass curve."""
import numpy as np
import pytest
from labelfree.excess_mass import excess_mass_curve
from .synthetic_data import make_blobs_with_anomalies, make_anomaly_scores


class TestExcessMass:
    """Test suite for Excess-Mass curve."""
    
    def test_basic_functionality(self):
        """Test basic EM curve computation."""
        # Generate data and scores
        X, y = make_blobs_with_anomalies(n_samples=300, n_anomalies=30, random_state=42)
        scores = make_anomaly_scores(X, y, method='distance', random_state=42)
        
        # Generate volume scores (scores on uniform samples)
        volume_scores = np.random.randn(1000)
        
        # Compute EM curve
        result = excess_mass_curve(scores, volume_scores, n_levels=50)
        
        # Check output
        assert set(result.keys()) == {'levels', 'excess_mass', 'auc', 'max_em'}
        assert len(result['levels']) == 50
        assert len(result['excess_mass']) == 50
        assert isinstance(result['auc'], float)
        assert isinstance(result['max_em'], float)
        
        # Check that max_em is actually the maximum
        assert result['max_em'] == pytest.approx(result['excess_mass'].max())
    
    def test_edge_cases(self):
        """Test edge cases."""
        # All scores identical
        scores = np.ones(100)
        volume_scores = np.ones(100)
        
        result = excess_mass_curve(scores, volume_scores)
        
        # Should still run without errors
        assert result['max_em'] >= 0
        
        # Empty volume scores
        with pytest.raises(ValueError):
            excess_mass_curve(scores, [])