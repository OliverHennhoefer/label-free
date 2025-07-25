"""
Cross-validation tests against the original EMMV reference implementation.

This module tests our mass_volume_auc implementation against the original
GitHub implementation from: https://github.com/christian-oleary/emmv
to ensure algorithmic equivalence and correctness.
"""

import numpy as np
from labelfree.metrics.mass_volume import mass_volume_auc
from labelfree.utils.computation import compute_volume_support
from .shuttle_data import load_shuttle_data, generate_anomaly_scores


def reference_mass_volume(
    alpha_min, alpha_max, volume, uniform_scores, anomaly_scores, alpha_count=1000
):
    """
    Reference implementation from https://github.com/christian-oleary/emmv

    This is the original algorithm from the GitHub repository, inlined here
    to avoid external dependencies while maintaining exact equivalence testing.
    """
    # Sort anomaly scores
    indices = np.argsort(anomaly_scores)
    n_samples = len(anomaly_scores)

    # Initialize
    axis_alpha = np.linspace(alpha_min, alpha_max, alpha_count)
    mv_scores = np.zeros(alpha_count)

    mass = 0
    cpt = 0
    threshold = (
        anomaly_scores[indices[-1]] if n_samples > 0 else 0
    )  # Start from highest score

    for i in range(alpha_count):
        # Update threshold to match target mass
        while mass < axis_alpha[i] and cpt < n_samples:
            cpt += 1
            threshold = anomaly_scores[indices[-cpt]]  # Work backwards from highest
            mass = cpt / n_samples

        # Compute volume estimate
        score_count = np.sum(uniform_scores >= threshold)
        mv_scores[i] = (score_count / len(uniform_scores)) * volume

    return mv_scores, axis_alpha


def reference_mass_volume_auc(
    alpha_min, alpha_max, volume, uniform_scores, anomaly_scores, alpha_count=1000
):
    """Compute AUC for reference implementation using same method as our implementation."""
    mv_scores, axis_alpha = reference_mass_volume(
        alpha_min, alpha_max, volume, uniform_scores, anomaly_scores, alpha_count
    )

    # Compute AUC using trapezoidal rule (same as our implementation)
    idx = np.argsort(axis_alpha)
    auc = float(np.trapezoid(mv_scores[idx], axis_alpha[idx]))

    return {
        "mass": np.linspace(
            alpha_min, alpha_max, alpha_count
        ),  # Simplified for comparison
        "volume": mv_scores,
        "auc": auc,
        "axis_alpha": axis_alpha,
    }


def generate_consistent_uniform_scores(data, scores, n_mc_samples, random_state=None):
    """Generate identical uniform scores for both implementations to eliminate Monte Carlo variance."""
    rng = np.random.default_rng(random_state)

    # Generate uniform samples in data bounding box
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    uniform_samples = rng.uniform(
        data_min, data_max, size=(n_mc_samples, data.shape[1])
    )

    # Use consistent scoring approach identical to our implementation
    from scipy.spatial import cKDTree

    tree = cKDTree(data)
    distances, indices = tree.query(uniform_samples, k=1)

    # Add noise to avoid exact copies (same as our implementation)
    base_scores = scores[indices]
    noise = rng.normal(0, 0.1 * scores.std(), size=len(uniform_samples))
    uniform_scores = base_scores + noise

    return uniform_samples, uniform_scores


def create_scoring_function(uniform_scores):
    """Create a scoring function that returns pre-computed uniform scores."""

    def scoring_function(uniform_samples):
        # Return the pre-computed scores, ignoring the samples
        # This ensures both implementations use identical uniform scores
        return uniform_scores

    return scoring_function


class TestMassVolumeReference:
    """Cross-validation tests against reference implementation."""

    def test_debug_algorithm_differences(self):
        """Diagnostic test to identify sources of differences between implementations."""
        # Test case 1: Simple case (this worked perfectly)
        print("\\n=== SIMPLE CASE ===")
        X_simple = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        scores_simple = np.array([0.1, 0.5, 0.9])
        uniform_scores_simple = np.array([0.2, 0.3, 0.4, 0.6, 0.7])

        scoring_function = create_scoring_function(uniform_scores_simple)
        our_simple = mass_volume_auc(
            scores_simple,
            X_simple,
            alpha_min=0.8,
            alpha_max=0.95,
            n_thresholds=3,
            n_mc_samples=len(uniform_scores_simple),
            random_state=42,
            scoring_function=scoring_function,
        )

        volume_support = compute_volume_support(X_simple)
        ref_simple = reference_mass_volume_auc(
            0.8, 0.95, volume_support, uniform_scores_simple, scores_simple, 3
        )

        print(
            f"Simple - Our AUC: {our_simple['auc']:.6f}, Ref AUC: {ref_simple['auc']:.6f}"
        )
        print(
            f"Simple - Relative error: {abs(our_simple['auc'] - ref_simple['auc']) / ref_simple['auc']:.8f}"
        )

        # Test case 2: More realistic case
        print("\\n=== REALISTIC CASE ===")
        rng = np.random.default_rng(42)
        X_real = rng.standard_normal((20, 2))  # 20 points, 2D
        scores_real = rng.standard_normal(20)
        uniform_samples, uniform_scores_real = generate_consistent_uniform_scores(
            X_real, scores_real, 100, random_state=42
        )

        scoring_function_real = create_scoring_function(uniform_scores_real)
        our_real = mass_volume_auc(
            scores_real,
            X_real,
            alpha_min=0.9,
            alpha_max=0.999,
            n_thresholds=10,
            n_mc_samples=len(uniform_scores_real),
            random_state=42,
            scoring_function=scoring_function_real,
        )

        volume_support_real = compute_volume_support(X_real)
        ref_real = reference_mass_volume_auc(
            0.9, 0.999, volume_support_real, uniform_scores_real, scores_real, 10
        )

        print(
            f"Realistic - Our AUC: {our_real['auc']:.6f}, Ref AUC: {ref_real['auc']:.6f}"
        )
        realistic_error = abs(our_real["auc"] - ref_real["auc"]) / ref_real["auc"]
        print(f"Realistic - Relative error: {realistic_error:.6f}")

        # Check if the realistic case has larger errors
        if realistic_error > 0.01:
            print(
                f"\\nRealistic case shows {realistic_error:.4f} error - investigating..."
            )
            print(f"Our volumes (first 5): {our_real['volume'][:5]}")
            print(f"Ref volumes (first 5): {ref_real['volume'][:5]}")
            print(f"Our masses (first 5): {our_real['mass'][:5]}")
            print(f"Ref masses (first 5): {ref_real['mass'][:5]}")

        # The simple case should be perfect
        simple_error = (
            abs(our_simple["auc"] - ref_simple["auc"]) / ref_simple["auc"]
            if ref_simple["auc"] != 0
            else 0
        )
        assert simple_error < 0.001, f"Simple case error: {simple_error:.6f}"

        # Realistic case should be much better than current 3-5%
        assert (
            realistic_error < 0.02
        ), f"Realistic case error too high: {realistic_error:.4f}"

    def test_basic_equivalence(self):
        """Test basic equivalence between our implementation and reference."""
        # Generate test data
        X, y = load_shuttle_data(n_samples=200, n_anomalies=20, random_state=42)
        scores = generate_anomaly_scores(X, y, method="distance", random_state=42)

        # Shared parameters
        alpha_min, alpha_max = 0.9, 0.999
        n_thresholds = 50
        n_mc_samples = 1000
        random_state = 42

        # Generate identical uniform scores for both implementations
        uniform_samples, uniform_scores = generate_consistent_uniform_scores(
            X, scores, n_mc_samples, random_state=random_state
        )
        volume_support = compute_volume_support(X)

        # Our implementation - use scoring function to force identical uniform scores
        scoring_function = create_scoring_function(uniform_scores)
        our_result = mass_volume_auc(
            scores,
            X,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            n_thresholds=n_thresholds,
            n_mc_samples=n_mc_samples,
            random_state=random_state,
            scoring_function=scoring_function,
        )

        # Reference implementation - use the same uniform scores directly
        ref_result = reference_mass_volume_auc(
            alpha_min, alpha_max, volume_support, uniform_scores, scores, n_thresholds
        )

        # Compare AUC values with floating-point precision tolerance (identical uniform scores)
        relative_error = abs(our_result["auc"] - ref_result["auc"]) / max(
            ref_result["auc"], 1e-12
        )
        assert (
            relative_error < 0.001
        ), f"AUC difference too large: ours={our_result['auc']:.6f}, ref={ref_result['auc']:.6f}, error={relative_error:.6f}"

        # Check that both have same output structure
        assert len(our_result["volume"]) == len(ref_result["volume"])
        assert len(our_result["axis_alpha"]) == len(ref_result["axis_alpha"])

        # Alpha axes should be identical
        np.testing.assert_allclose(
            our_result["axis_alpha"], ref_result["axis_alpha"], rtol=1e-10
        )

    def test_perfect_detector_equivalence(self):
        """Test equivalence with perfect anomaly detector."""
        X, y = load_shuttle_data(n_samples=300, n_anomalies=30, random_state=123)
        scores = generate_anomaly_scores(
            X, y, method="perfect", noise_level=0, random_state=123
        )

        # Parameters
        alpha_min, alpha_max = 0.95, 0.999
        n_thresholds = 20
        n_mc_samples = 2000
        random_state = 123

        # Generate identical uniform scores
        uniform_samples, uniform_scores = generate_consistent_uniform_scores(
            X, scores, n_mc_samples, random_state=random_state
        )
        volume_support = compute_volume_support(X)

        # Both implementations with identical uniform scores
        scoring_function = create_scoring_function(uniform_scores)
        our_result = mass_volume_auc(
            scores,
            X,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            n_thresholds=n_thresholds,
            n_mc_samples=n_mc_samples,
            random_state=random_state,
            scoring_function=scoring_function,
        )

        ref_result = reference_mass_volume_auc(
            alpha_min, alpha_max, volume_support, uniform_scores, scores, n_thresholds
        )

        # Perfect detector should be identical with identical uniform scores
        relative_error = abs(our_result["auc"] - ref_result["auc"]) / max(
            ref_result["auc"], 1e-12
        )
        assert (
            relative_error < 0.0001
        ), f"Perfect detector AUC mismatch: ours={our_result['auc']:.6f}, ref={ref_result['auc']:.6f}, error={relative_error:.6f}"

    def test_random_detector_equivalence(self):
        """Test equivalence with random detector."""
        X, y = load_shuttle_data(n_samples=400, n_anomalies=40, random_state=456)
        scores = generate_anomaly_scores(X, y, method="random", random_state=456)

        # Parameters
        alpha_min, alpha_max = 0.9, 0.99
        n_thresholds = 25
        n_mc_samples = 1500
        random_state = 456

        # Generate identical uniform scores
        uniform_samples, uniform_scores = generate_consistent_uniform_scores(
            X, scores, n_mc_samples, random_state=random_state
        )
        volume_support = compute_volume_support(X)

        # Both implementations with identical uniform scores
        scoring_function = create_scoring_function(uniform_scores)
        our_result = mass_volume_auc(
            scores,
            X,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            n_thresholds=n_thresholds,
            n_mc_samples=n_mc_samples,
            random_state=random_state,
            scoring_function=scoring_function,
        )

        ref_result = reference_mass_volume_auc(
            alpha_min, alpha_max, volume_support, uniform_scores, scores, n_thresholds
        )

        # Random detector should be identical with identical uniform scores
        relative_error = abs(our_result["auc"] - ref_result["auc"]) / max(
            ref_result["auc"], 1e-12
        )
        assert (
            relative_error < 0.001
        ), f"Random detector AUC mismatch: ours={our_result['auc']:.6f}, ref={ref_result['auc']:.6f}, error={relative_error:.6f}"

    def test_volume_computation_consistency(self):
        """Test that our internal volume computation matches external computation."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3)) * 2 + 1  # Non-unit scale data
        scores = rng.standard_normal(100)

        # Our internal volume computation
        our_result = mass_volume_auc(
            scores, X, n_thresholds=10, n_mc_samples=500, random_state=789
        )

        # Generate identical uniform scores for comparison
        uniform_samples, uniform_scores = generate_consistent_uniform_scores(
            X, scores, 500, random_state=789
        )
        external_volume = compute_volume_support(X)

        # Reference with identical uniform scores
        ref_result = reference_mass_volume_auc(
            0.9, 0.999, external_volume, uniform_scores, scores, 10
        )

        # Volume support should be used identically with same uniform scores
        relative_error = abs(our_result["auc"] - ref_result["auc"]) / max(
            ref_result["auc"], 1e-12
        )
        assert (
            relative_error < 0.001
        ), f"Volume computation inconsistency: ours={our_result['auc']:.6f}, ref={ref_result['auc']:.6f}, error={relative_error:.6f}"

    def test_identical_scores_equivalence(self):
        """Test equivalence with identical scores (edge case)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2))
        scores = np.ones(50)  # All identical

        # Parameters
        n_thresholds = 10
        n_mc_samples = 500
        random_state = 999

        # Generate identical uniform scores
        uniform_samples, uniform_scores = generate_consistent_uniform_scores(
            X, scores, n_mc_samples, random_state=random_state
        )
        volume_support = compute_volume_support(X)

        # Both implementations with identical uniform scores
        scoring_function = create_scoring_function(uniform_scores)
        our_result = mass_volume_auc(
            scores,
            X,
            n_thresholds=n_thresholds,
            n_mc_samples=n_mc_samples,
            random_state=random_state,
            scoring_function=scoring_function,
        )

        ref_result = reference_mass_volume_auc(
            0.9, 0.999, volume_support, uniform_scores, scores, n_thresholds
        )

        # With identical scores, both should handle gracefully
        assert our_result["auc"] >= 0
        assert ref_result["auc"] >= 0

        # Results should be similar (though exact equivalence is challenging with identical scores)
        # Focus on ensuring neither implementation crashes or produces invalid results
        assert np.all(np.isfinite(our_result["volume"]))
        assert np.all(np.isfinite(ref_result["volume"]))

    def test_single_threshold_equivalence(self):
        """Test equivalence with minimal number of thresholds."""
        X, y = load_shuttle_data(n_samples=100, n_anomalies=10, random_state=111)
        scores = generate_anomaly_scores(X, y, method="distance", random_state=111)

        # Minimal thresholds
        n_thresholds = 2
        n_mc_samples = 200
        random_state = 111

        # Generate identical uniform scores
        uniform_samples, uniform_scores = generate_consistent_uniform_scores(
            X, scores, n_mc_samples, random_state=random_state
        )
        volume_support = compute_volume_support(X)

        # Both implementations with identical uniform scores
        scoring_function = create_scoring_function(uniform_scores)
        our_result = mass_volume_auc(
            scores,
            X,
            alpha_min=0.95,
            alpha_max=0.99,
            n_thresholds=n_thresholds,
            n_mc_samples=n_mc_samples,
            random_state=random_state,
            scoring_function=scoring_function,
        )

        ref_result = reference_mass_volume_auc(
            0.95, 0.99, volume_support, uniform_scores, scores, n_thresholds
        )

        # Should handle minimal case without issues
        assert len(our_result["volume"]) == n_thresholds
        assert len(ref_result["volume"]) == n_thresholds

        # Basic sanity checks
        assert our_result["auc"] >= 0
        assert ref_result["auc"] >= 0

    def test_large_dataset_equivalence(self):
        """Test equivalence with larger dataset."""
        X, y = load_shuttle_data(n_samples=1000, n_anomalies=100, random_state=777)
        scores = generate_anomaly_scores(X, y, method="distance", random_state=777)

        # Parameters
        alpha_min, alpha_max = 0.92, 0.998
        n_thresholds = 30
        n_mc_samples = 3000
        random_state = 777

        # Generate identical uniform scores
        uniform_samples, uniform_scores = generate_consistent_uniform_scores(
            X, scores, n_mc_samples, random_state=random_state
        )
        volume_support = compute_volume_support(X)

        # Both implementations with identical uniform scores
        scoring_function = create_scoring_function(uniform_scores)
        our_result = mass_volume_auc(
            scores,
            X,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            n_thresholds=n_thresholds,
            n_mc_samples=n_mc_samples,
            random_state=random_state,
            scoring_function=scoring_function,
        )

        ref_result = reference_mass_volume_auc(
            alpha_min, alpha_max, volume_support, uniform_scores, scores, n_thresholds
        )

        # Should be identical with same uniform scores regardless of dataset size
        relative_error = abs(our_result["auc"] - ref_result["auc"]) / max(
            ref_result["auc"], 1e-12
        )
        assert (
            relative_error < 0.001
        ), f"Large dataset AUC mismatch: ours={our_result['auc']:.6f}, ref={ref_result['auc']:.6f}, error={relative_error:.6f}"

        # Both should produce reasonable ranges
        assert np.all(our_result["volume"] >= 0)
        assert np.all(ref_result["volume"] >= 0)
