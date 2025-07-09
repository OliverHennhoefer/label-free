"""Tests for ranking stability metrics."""

import numpy as np
from labelfree.stability import ranking_stability, top_k_stability
from .synthetic_data import make_blobs_with_anomalies


class TestStability:
    """Test suite for stability metrics."""

    def test_ranking_stability(self):
        """Test ranking stability measurement."""
        X, y = make_blobs_with_anomalies(n_samples=500, random_state=42)

        # Simple scoring function
        def score_func(data):
            center = data.mean(axis=0)
            return np.linalg.norm(data - center, axis=1)

        result = ranking_stability(
            score_func, X, n_subsamples=10, subsample_ratio=0.8, random_state=42
        )

        # Check output
        assert set(result.keys()) == {"mean", "std", "min"}
        assert 0 <= result["mean"] <= 1
        assert result["std"] >= 0
        assert result["min"] <= result["mean"]

        # Stable scoring should have high correlation
        assert result["mean"] > 0.7

    def test_top_k_stability(self):
        """Test top-k stability measurement."""
        X, y = make_blobs_with_anomalies(n_samples=500, random_state=42)

        def score_func(data):
            center = data.mean(axis=0)
            return np.linalg.norm(data - center, axis=1)

        result = top_k_stability(
            score_func, X, k_values=[10, 20, 50], n_subsamples=10, random_state=42
        )

        # Check output
        assert set(result.keys()) == {10, 20, 50}

        # Jaccard similarity should be in [0, 1]
        for k, similarity in result.items():
            assert 0 <= similarity <= 1

        # Smaller k should be more stable
        assert result[10] >= result[50]

    def test_unstable_scoring(self):
        """Test with unstable scoring function."""
        X = np.random.randn(200, 5)

        # Random scoring function (unstable)
        def score_func(data):
            return np.random.randn(len(data))

        result = ranking_stability(score_func, X, n_subsamples=10)

        # Should have very low stability
        assert result["mean"] < 0.1
        assert result["min"] < 0.1
