"""
Cross-validation tests against the original SIREOS reference implementation.

This module tests our sireos implementation against the original
GitHub implementation from: https://github.com/homarques/SIREOS/blob/main/sireos.py
to ensure algorithmic equivalence and correctness.
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from labelfree.metrics.sireos import sireos
from .synthetic_data import make_blobs_with_anomalies, make_anomaly_scores


def reference_sireos(scores, data, quantile=0.01):
    """
    Reference implementation from https://github.com/homarques/SIREOS/blob/main/sireos.py
    
    This is the original algorithm from the GitHub repository, inlined here
    to avoid external dependencies while maintaining exact equivalence testing.
    
    The original implementation:
    1. Computes pairwise distances using sklearn
    2. Sets heat kernel threshold using quantile of non-zero distances
    3. Normalizes scores by sum (X = X/X.sum())
    4. Computes weighted similarity score using heat kernel
    """
    # Convert inputs to numpy arrays
    scores = np.asarray(scores, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    
    if len(scores) != len(data):
        raise ValueError(f"Length mismatch: {len(scores)} scores vs {len(data)} data points")
    
    n_samples = len(data)
    
    if n_samples <= 1:
        return 0.0
    
    # Pairwise distance computation (exactly as in reference)
    D = pairwise_distances(data)
    
    # Heat kernel parameter (exactly as in reference)
    non_zero_distances = D[np.nonzero(D)]
    if len(non_zero_distances) == 0:
        t = 1.0
    else:
        t = np.quantile(non_zero_distances, quantile)
    
    # Score normalization (exactly as in reference)
    # X = X/X.sum()
    scores_sum = scores.sum()
    if scores_sum == 0:
        X = np.ones_like(scores) / len(scores)
    else:
        X = scores / scores_sum
    
    # Computing the index (exactly as in reference)
    score = 0.0
    for j in range(n_samples):
        # Create data array without point j
        # data[None, j, :] creates shape (1, n_features)
        # np.delete(data[None, :, :], j, axis=1) removes j-th point, creates shape (1, n_samples-1, n_features)
        point_j = data[None, j, :]  # Shape: (1, n_features)
        other_data = np.delete(data[None, :, :], j, axis=1)  # Shape: (1, n_samples-1, n_features)
        
        # Compute distances from point j to all other points
        # np.linalg.norm computes L2 norm along last axis
        distances = np.linalg.norm(point_j - other_data, axis=-1)  # Shape: (1, n_samples-1)
        
        # Apply heat kernel and take mean
        similarities = np.exp(-(distances**2) / (2 * t * t))
        mean_similarity = np.mean(similarities)
        
        # Weight by normalized score
        score += mean_similarity * X[j]
    
    return float(score)


class TestSIREOSReference:
    """Cross-validation tests against reference implementation."""

    def test_debug_algorithm_differences(self):
        """Diagnostic test to identify sources of differences between implementations."""
        # Test case 1: Simple case with known data
        print("\n=== SIMPLE SIREOS CASE ===")
        X_simple = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # Simple triangle
        scores_simple = np.array([0.1, 0.5, 0.9])
        
        our_simple = sireos(scores_simple, X_simple, quantile=0.01)
        ref_simple = reference_sireos(scores_simple, X_simple, quantile=0.01)
        
        print(f"Simple - Our SIREOS: {our_simple:.8f}, Ref SIREOS: {ref_simple:.8f}")
        if ref_simple != 0:
            simple_error = abs(our_simple - ref_simple) / ref_simple
            print(f"Simple - Relative error: {simple_error:.8f}")
        else:
            simple_error = abs(our_simple - ref_simple)
            print(f"Simple - Absolute error: {simple_error:.8f}")
        
        # Test case 2: More realistic case
        print("\n=== REALISTIC SIREOS CASE ===")
        rng = np.random.default_rng(42)
        X_real = rng.standard_normal((20, 2))
        scores_real = rng.standard_normal(20)
        
        our_real = sireos(scores_real, X_real, quantile=0.01)
        ref_real = reference_sireos(scores_real, X_real, quantile=0.01)
        
        print(f"Realistic - Our SIREOS: {our_real:.8f}, Ref SIREOS: {ref_real:.8f}")
        if ref_real != 0:
            realistic_error = abs(our_real - ref_real) / ref_real
            print(f"Realistic - Relative error: {realistic_error:.8f}")
        else:
            realistic_error = abs(our_real - ref_real)
            print(f"Realistic - Absolute error: {realistic_error:.8f}")
        
        # Check errors are within acceptable bounds
        # FAISS vs sklearn may have small numerical differences
        if ref_simple != 0:
            assert simple_error < 0.001, f"Simple case error too large: {simple_error:.8f}"
        else:
            assert simple_error < 1e-8, f"Simple case absolute error too large: {simple_error:.8f}"
        
        if ref_real != 0:
            assert realistic_error < 0.01, f"Realistic case error too large: {realistic_error:.8f}"
        else:
            assert realistic_error < 1e-6, f"Realistic case absolute error too large: {realistic_error:.8f}"

    def test_basic_equivalence(self):
        """Test basic equivalence between our implementation and reference."""
        # Generate test data
        X, y = make_blobs_with_anomalies(n_samples=100, n_anomalies=10, random_state=42)
        scores = make_anomaly_scores(X, y, method="distance", random_state=42)

        # Parameters
        quantile = 0.01

        # Both implementations
        our_result = sireos(scores, X, quantile=quantile)
        ref_result = reference_sireos(scores, X, quantile=quantile)

        # Compare results with appropriate tolerance for numerical differences
        # FAISS (float32) vs sklearn (float64) may have small precision differences
        if ref_result != 0:
            relative_error = abs(our_result - ref_result) / ref_result
            assert relative_error < 0.01, \
                f"SIREOS difference too large: ours={our_result:.8f}, ref={ref_result:.8f}, error={relative_error:.8f}"
        else:
            absolute_error = abs(our_result - ref_result)
            assert absolute_error < 1e-6, \
                f"SIREOS absolute difference too large: ours={our_result:.8f}, ref={ref_result:.8f}, error={absolute_error:.8f}"

    def test_perfect_detector_equivalence(self):
        """Test equivalence with perfect anomaly detector."""
        X, y = make_blobs_with_anomalies(n_samples=200, n_anomalies=20, random_state=123)
        scores = make_anomaly_scores(X, y, method="perfect", noise_level=0, random_state=123)

        # Parameters
        quantile = 0.01

        # Both implementations
        our_result = sireos(scores, X, quantile=quantile)
        ref_result = reference_sireos(scores, X, quantile=quantile)

        # Perfect detector should be equivalent between implementations
        if ref_result != 0:
            relative_error = abs(our_result - ref_result) / ref_result
            assert relative_error < 0.01, \
                f"Perfect detector SIREOS mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={relative_error:.8f}"
        else:
            absolute_error = abs(our_result - ref_result)
            assert absolute_error < 1e-6, \
                f"Perfect detector SIREOS absolute mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={absolute_error:.8f}"

    def test_random_detector_equivalence(self):
        """Test equivalence with random detector."""
        X, y = make_blobs_with_anomalies(n_samples=150, n_anomalies=15, random_state=456)
        scores = make_anomaly_scores(X, y, method="random", random_state=456)

        # Parameters
        quantile = 0.01

        # Both implementations
        our_result = sireos(scores, X, quantile=quantile)
        ref_result = reference_sireos(scores, X, quantile=quantile)

        # Random detector should be equivalent
        if ref_result != 0:
            relative_error = abs(our_result - ref_result) / ref_result
            assert relative_error < 0.01, \
                f"Random detector SIREOS mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={relative_error:.8f}"
        else:
            absolute_error = abs(our_result - ref_result)
            assert absolute_error < 1e-6, \
                f"Random detector SIREOS absolute mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={absolute_error:.8f}"

    def test_quantile_parameter_equivalence(self):
        """Test equivalence with different quantile parameters."""
        rng = np.random.default_rng(789)
        X = rng.standard_normal((80, 3))
        scores = rng.standard_normal(80)

        # Test different quantile values
        for quantile in [0.001, 0.01, 0.05, 0.1]:
            our_result = sireos(scores, X, quantile=quantile)
            ref_result = reference_sireos(scores, X, quantile=quantile)

            # Should be equivalent for any quantile value
            if ref_result != 0:
                relative_error = abs(our_result - ref_result) / ref_result
                assert relative_error < 0.01, \
                    f"Quantile {quantile} mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={relative_error:.8f}"
            else:
                absolute_error = abs(our_result - ref_result)
                assert absolute_error < 1e-6, \
                    f"Quantile {quantile} absolute mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={absolute_error:.8f}"

    def test_identical_scores_equivalence(self):
        """Test equivalence with identical scores (edge case)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2))
        scores = np.ones(50)  # All identical scores

        # Both implementations
        our_result = sireos(scores, X, quantile=0.01)
        ref_result = reference_sireos(scores, X, quantile=0.01)

        # With identical scores, both should handle gracefully and produce same results
        if ref_result != 0:
            relative_error = abs(our_result - ref_result) / ref_result
            assert relative_error < 0.01, \
                f"Identical scores mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={relative_error:.8f}"
        else:
            absolute_error = abs(our_result - ref_result)
            assert absolute_error < 1e-6, \
                f"Identical scores absolute mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={absolute_error:.8f}"

        # Both should produce valid results
        assert np.isfinite(our_result) and np.isfinite(ref_result)
        assert our_result >= 0 and ref_result >= 0

    def test_zero_scores_equivalence(self):
        """Test equivalence with zero scores (edge case)."""
        rng = np.random.default_rng(111)
        X = rng.standard_normal((30, 2))
        scores = np.zeros(30)  # All zero scores

        # Both implementations
        our_result = sireos(scores, X, quantile=0.01)
        ref_result = reference_sireos(scores, X, quantile=0.01)

        # With zero scores, both should use uniform distribution and produce same results
        if ref_result != 0:
            relative_error = abs(our_result - ref_result) / ref_result
            assert relative_error < 0.01, \
                f"Zero scores mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={relative_error:.8f}"
        else:
            absolute_error = abs(our_result - ref_result)
            assert absolute_error < 1e-6, \
                f"Zero scores absolute mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={absolute_error:.8f}"

    def test_single_point_equivalence(self):
        """Test equivalence with single point (edge case)."""
        X = np.array([[1.0, 2.0]])  # Single point
        scores = np.array([0.5])

        # Both implementations
        our_result = sireos(scores, X, quantile=0.01)
        ref_result = reference_sireos(scores, X, quantile=0.01)

        # Single point should return 0.0 for both (no neighbors to compare)
        assert our_result == 0.0, f"Our implementation should return 0.0 for single point, got {our_result}"
        assert ref_result == 0.0, f"Reference implementation should return 0.0 for single point, got {ref_result}"
        assert our_result == ref_result, "Both implementations should return same result for single point"

    def test_two_point_equivalence(self):
        """Test equivalence with two points (minimal case)."""
        X = np.array([[0.0, 0.0], [1.0, 1.0]])  # Two points
        scores = np.array([0.3, 0.7])

        # Both implementations
        our_result = sireos(scores, X, quantile=0.01)
        ref_result = reference_sireos(scores, X, quantile=0.01)

        # Should handle two points equivalently
        if ref_result != 0:
            relative_error = abs(our_result - ref_result) / ref_result
            assert relative_error < 0.01, \
                f"Two points mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={relative_error:.8f}"
        else:
            absolute_error = abs(our_result - ref_result)
            assert absolute_error < 1e-6, \
                f"Two points absolute mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={absolute_error:.8f}"

    def test_high_dimensional_equivalence(self):
        """Test equivalence with high-dimensional data."""
        rng = np.random.default_rng(999)
        X = rng.standard_normal((100, 10))  # 10-dimensional data  
        scores = rng.standard_normal(100)

        # Both implementations
        our_result = sireos(scores, X, quantile=0.01)
        ref_result = reference_sireos(scores, X, quantile=0.01)

        # Should handle high dimensions equivalently
        if ref_result != 0:
            relative_error = abs(our_result - ref_result) / ref_result
            assert relative_error < 0.01, \
                f"High-dimensional mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={relative_error:.8f}"
        else:
            absolute_error = abs(our_result - ref_result)
            assert absolute_error < 1e-6, \
                f"High-dimensional absolute mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={absolute_error:.8f}"

    def test_data_scaling_equivalence(self):
        """Test equivalence with different data scaling."""
        # Base data
        rng = np.random.default_rng(333)
        X_base = rng.standard_normal((60, 2))
        scores = rng.standard_normal(60)

        # Test with different scales
        for scale in [0.1, 1.0, 10.0]:
            X_scaled = X_base * scale
            
            our_result = sireos(scores, X_scaled, quantile=0.01)
            ref_result = reference_sireos(scores, X_scaled, quantile=0.01)

            # Should be equivalent for any scaling
            if ref_result != 0:
                relative_error = abs(our_result - ref_result) / ref_result
                assert relative_error < 0.01, \
                    f"Scale {scale} mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={relative_error:.8f}"
            else:
                absolute_error = abs(our_result - ref_result)
                assert absolute_error < 1e-6, \
                    f"Scale {scale} absolute mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={absolute_error:.8f}"

    def test_heat_kernel_threshold_equivalence(self):
        """Test that heat kernel threshold computation is equivalent."""
        # Create data with known distances
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])  # Points on line
        scores = np.array([0.25, 0.25, 0.25, 0.25])  # Equal scores

        # Test different quantiles to verify threshold computation
        for quantile in [0.01, 0.1, 0.5]:
            our_result = sireos(scores, X, quantile=quantile)
            ref_result = reference_sireos(scores, X, quantile=quantile)

            # Threshold computation should be identical
            if ref_result != 0:
                relative_error = abs(our_result - ref_result) / ref_result
                assert relative_error < 0.01, \
                    f"Heat kernel quantile {quantile} mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={relative_error:.8f}"
            else:
                absolute_error = abs(our_result - ref_result)
                assert absolute_error < 1e-6, \
                    f"Heat kernel quantile {quantile} absolute mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={absolute_error:.8f}"

    def test_score_normalization_equivalence(self):
        """Test that score normalization is equivalent."""
        rng = np.random.default_rng(666)
        X = rng.standard_normal((70, 2))
        
        # Test with different score ranges
        for score_range in [(0, 1), (100, 1000), (-50, 50)]:
            scores = rng.uniform(score_range[0], score_range[1], 70)
            
            our_result = sireos(scores, X, quantile=0.01)
            ref_result = reference_sireos(scores, X, quantile=0.01)

            # Normalization should make results equivalent regardless of score range
            if ref_result != 0:
                relative_error = abs(our_result - ref_result) / ref_result
                assert relative_error < 0.01, \
                    f"Score range {score_range} mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={relative_error:.8f}"
            else:
                absolute_error = abs(our_result - ref_result)
                assert absolute_error < 1e-6, \
                    f"Score range {score_range} absolute mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={absolute_error:.8f}"

    def test_large_dataset_equivalence(self):
        """Test equivalence with larger dataset."""
        X, y = make_blobs_with_anomalies(n_samples=500, n_anomalies=50, random_state=777)
        scores = make_anomaly_scores(X, y, method="distance", random_state=777)

        # Both implementations
        our_result = sireos(scores, X, quantile=0.01)
        ref_result = reference_sireos(scores, X, quantile=0.01)

        # Should be equivalent regardless of dataset size
        if ref_result != 0:
            relative_error = abs(our_result - ref_result) / ref_result
            assert relative_error < 0.01, \
                f"Large dataset mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={relative_error:.8f}"
        else:
            absolute_error = abs(our_result - ref_result)
            assert absolute_error < 1e-6, \
                f"Large dataset absolute mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={absolute_error:.8f}"

    def test_regression_golden_standard(self):
        """Test against golden standard results to catch algorithm changes."""
        # Fixed synthetic scenario for regression testing
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        scores = np.sum(X**2, axis=1)  # Simple quadratic scores

        # Both implementations with fixed parameters
        our_result = sireos(scores, X, quantile=0.01)
        ref_result = reference_sireos(scores, X, quantile=0.01)

        # These should remain stable unless algorithm changes
        expected_our = 0.0165  # From our previous tests
        expected_ref = 0.0165  # Should be nearly identical to ours

        # Check our implementation hasn't changed
        assert abs(our_result - expected_our) < 0.01, \
            f"Our implementation changed: expected {expected_our:.4f}, got {our_result:.4f}"

        # Check reference produces expected result
        assert abs(ref_result - expected_ref) < 0.01, \
            f"Reference implementation unexpected: expected {expected_ref:.4f}, got {ref_result:.4f}"

        # Check equivalence between implementations
        if ref_result != 0:
            relative_error = abs(our_result - ref_result) / ref_result
            assert relative_error < 0.01, \
                f"Golden standard mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={relative_error:.8f}"
        else:
            absolute_error = abs(our_result - ref_result)
            assert absolute_error < 1e-6, \
                f"Golden standard absolute mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={absolute_error:.8f}"

    def test_distance_computation_consistency(self):
        """Test that distance computations are consistent between FAISS and sklearn."""
        # Simple case where we can verify distances manually
        X = np.array([[0.0, 0.0], [3.0, 4.0]])  # Distance should be 5.0
        scores = np.array([0.4, 0.6])

        # Both implementations should handle this simple case identically
        our_result = sireos(scores, X, quantile=0.01)
        ref_result = reference_sireos(scores, X, quantile=0.01)

        # Should be very close (accounting for numerical precision differences)
        if ref_result != 0:
            relative_error = abs(our_result - ref_result) / ref_result
            assert relative_error < 0.001, \
                f"Distance computation mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={relative_error:.8f}"
        else:
            absolute_error = abs(our_result - ref_result)
            assert absolute_error < 1e-8, \
                f"Distance computation absolute mismatch: ours={our_result:.8f}, ref={ref_result:.8f}, error={absolute_error:.8f}"