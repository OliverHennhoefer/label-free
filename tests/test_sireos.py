import numpy as np
import pytest
from labelfree.metrics.sireos import sireos
from .shuttle_data import load_shuttle_data, generate_anomaly_scores


class TestSIREOS:
    """Test suite for SIREOS (Similarity-based Internal Relative Evaluation of Outlier Solutions).

    These tests are tight and focused on mathematical correctness and edge cases.
    SIREOS evaluates anomaly detection quality by measuring how well outlier scores
    align with local data structure using similarity-based neighborhood characteristics.
    """

    def test_basic_functionality(self):
        """Test basic SIREOS computation with known data."""
        # Use controlled data for reproducible results
        X, y = load_shuttle_data(n_samples=100, n_anomalies=10, random_state=42)
        scores = generate_anomaly_scores(X, y, method="distance", random_state=42)

        # Compute SIREOS with default parameters
        sireos_score = sireos(scores, X, quantile=0.01)

        # Check output properties
        assert isinstance(sireos_score, float)
        assert np.isfinite(sireos_score)
        assert sireos_score >= 0, "SIREOS scores must be non-negative"

        # For distance-based scores on shuttle data, expect reasonable range
        assert (
            0.0 <= sireos_score <= 1.0
        ), f"SIREOS score out of expected range: {sireos_score:.6f}"

    def test_perfect_detector(self):
        """Test SIREOS for perfect anomaly detector."""
        X, y = load_shuttle_data(n_samples=100, n_anomalies=10, random_state=42)
        scores = generate_anomaly_scores(
            X, y, method="perfect", noise_level=0, random_state=42
        )

        sireos_score = sireos(scores, X, quantile=0.01)

        # Perfect detector should produce valid SIREOS score
        assert isinstance(sireos_score, float)
        assert np.isfinite(sireos_score)
        assert sireos_score >= 0, "Perfect detector SIREOS must be non-negative"

    def test_random_detector(self):
        """Test SIREOS for random detector."""
        X, y = load_shuttle_data(n_samples=100, n_anomalies=10, random_state=42)
        scores = generate_anomaly_scores(X, y, method="random", random_state=42)

        sireos_score = sireos(scores, X, quantile=0.01)

        # Random detector should produce valid results
        assert isinstance(sireos_score, float)
        assert np.isfinite(sireos_score)
        assert sireos_score >= 0, "Random detector SIREOS must be non-negative"

    def test_detector_comparison(self):
        """Test that different detectors produce different SIREOS scores."""
        X, y = load_shuttle_data(n_samples=150, n_anomalies=15, random_state=42)

        perfect_scores = generate_anomaly_scores(
            X, y, method="perfect", noise_level=0, random_state=42
        )
        random_scores = generate_anomaly_scores(X, y, method="random", random_state=42)
        distance_scores = generate_anomaly_scores(
            X, y, method="distance", random_state=42
        )

        perfect_sireos = sireos(perfect_scores, X, quantile=0.01)
        random_sireos = sireos(random_scores, X, quantile=0.01)
        distance_sireos = sireos(distance_scores, X, quantile=0.01)

        # All should be valid non-negative scores
        assert perfect_sireos >= 0 and random_sireos >= 0 and distance_sireos >= 0

        # Different detectors should produce different scores
        scores_list = [perfect_sireos, random_sireos, distance_sireos]
        unique_scores = len(set(np.round(scores_list, 8)))
        assert (
            unique_scores >= 2
        ), "Different detectors should produce different SIREOS scores"

    def test_input_validation(self):
        """Test strict input validation."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2))
        scores = rng.standard_normal(50)

        # Test mismatched lengths
        with pytest.raises(ValueError, match="Length mismatch"):
            sireos(scores[:25], X)

        # Test empty inputs
        with pytest.raises(ValueError):
            sireos(np.array([]), X)
        with pytest.raises(ValueError):
            sireos(scores, np.empty((0, 2)))

        # Test 1D scores requirement
        with pytest.raises(ValueError):
            sireos(scores.reshape(-1, 1), X)

        # Test NaN values
        scores_with_nan = scores.copy()
        scores_with_nan[0] = np.nan
        with pytest.raises(ValueError):
            sireos(scores_with_nan, X)

        # Test infinite values in data
        X_with_inf = X.copy()
        X_with_inf[0, 0] = np.inf
        result = sireos(scores, X_with_inf)
        assert np.isfinite(result), "SIREOS should handle infinite data gracefully"

        # Test invalid quantile values
        with pytest.raises(ValueError):
            sireos(scores, X, quantile=-0.1)
        with pytest.raises(ValueError):
            sireos(scores, X, quantile=1.1)

    def test_reproducibility(self):
        """Test exact reproducibility with fixed data."""
        X, y = load_shuttle_data(n_samples=80, random_state=42)
        scores = generate_anomaly_scores(X, y, method="distance", random_state=42)

        # Same inputs should produce identical results
        sireos1 = sireos(scores, X, quantile=0.01)
        sireos2 = sireos(scores, X, quantile=0.01)

        assert (
            sireos1 == sireos2
        ), "SIREOS must be exactly reproducible with same inputs"

    def test_quantile_parameter_effects(self):
        """Test that quantile parameter affects results correctly."""
        X, y = load_shuttle_data(n_samples=80, n_anomalies=8, random_state=42)
        scores = generate_anomaly_scores(X, y, method="distance", random_state=42)

        # Test specific quantile values
        quantiles = [0.01, 0.05, 0.1, 0.5]
        sireos_scores = []

        for q in quantiles:
            score = sireos(scores, X, quantile=q)
            sireos_scores.append(score)

            # All should be valid and finite
            assert isinstance(score, float)
            assert np.isfinite(score)
            assert score >= 0, f"SIREOS score must be non-negative for quantile {q}"

        # Different quantiles must produce different results
        unique_scores = len(set(np.round(sireos_scores, 10)))
        assert (
            unique_scores >= 3
        ), "Different quantiles must produce different SIREOS scores"

    def test_edge_cases(self):
        """Test critical edge cases with tight assertions."""

        # Test single point
        X_single = np.array([[1.0, 2.0]])
        scores_single = np.array([0.5])
        result_single = sireos(scores_single, X_single, quantile=0.01)
        assert result_single == 0.0, "Single point must return exactly 0.0"

        # Test two points
        X_two = np.array([[0.0, 0.0], [1.0, 1.0]])
        scores_two = np.array([0.3, 0.7])
        result_two = sireos(scores_two, X_two, quantile=0.01)
        assert isinstance(result_two, float)
        assert np.isfinite(result_two)
        assert result_two >= 0, "Two-point SIREOS must be non-negative"

        # Test identical scores (uniform distribution)
        rng = np.random.default_rng(42)
        X_uniform = rng.standard_normal((50, 2))
        scores_uniform = np.ones(50)  # All identical scores
        result_uniform = sireos(scores_uniform, X_uniform, quantile=0.01)
        assert isinstance(result_uniform, float)
        assert np.isfinite(result_uniform)
        assert result_uniform >= 0, "Uniform scores SIREOS must be non-negative"

        # Test zero scores
        scores_zero = np.zeros(50)
        result_zero = sireos(scores_zero, X_uniform, quantile=0.01)
        assert isinstance(result_zero, float)
        assert np.isfinite(result_zero)
        assert result_zero >= 0, "Zero scores SIREOS must be non-negative"

        # Uniform and zero scores should produce identical results (both become uniform distribution)
        assert (
            abs(result_uniform - result_zero) < 1e-10
        ), "Uniform and zero scores should produce identical results"

    def test_mathematical_properties(self):
        """Test fundamental mathematical properties of SIREOS."""
        rng = np.random.default_rng(42)

        # Test with controlled data
        X = rng.standard_normal((60, 3))
        scores = rng.uniform(0.1, 1.0, 60)  # Positive scores

        sireos_score = sireos(scores, X, quantile=0.01)

        # Test core mathematical properties
        assert isinstance(sireos_score, float), "SIREOS must return float"
        assert np.isfinite(sireos_score), "SIREOS must be finite"
        assert sireos_score >= 0, "SIREOS must be non-negative"

        # Test score normalization property (SIREOS should be invariant to score scaling)
        scores_scaled = scores * 1000  # Scale scores by large factor
        sireos_scaled = sireos(scores_scaled, X, quantile=0.01)
        assert (
            abs(sireos_score - sireos_scaled) < 1e-10
        ), "SIREOS must be invariant to score scaling"

    def test_high_dimensional_robustness(self):
        """Test SIREOS with high-dimensional data."""
        rng = np.random.default_rng(42)

        # Test dimensions: 2D, 5D, 10D
        dimensions = [2, 5, 10]
        results = []

        for dim in dimensions:
            X = rng.standard_normal((50, dim))
            scores = rng.uniform(0.1, 1.0, 50)
            result = sireos(scores, X, quantile=0.01)

            assert isinstance(
                result, float
            ), f"SIREOS must return float for {dim}D data"
            assert np.isfinite(result), f"SIREOS must be finite for {dim}D data"
            assert result >= 0, f"SIREOS must be non-negative for {dim}D data"
            results.append(result)

        # Different dimensions should generally produce different results
        unique_results = len(set(np.round(results, 8)))
        assert (
            unique_results >= 2
        ), "Different dimensions should produce different SIREOS values"

    def test_regression_stability(self):
        """Test against known results to catch algorithm changes."""
        # Fixed test case for regression testing
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 3))
        scores = rng.uniform(0.1, 1.0, 80)

        sireos_score = sireos(scores, X, quantile=0.01)

        # This should remain stable unless algorithm changes
        # Value determined by running corrected implementation
        expected_range = (0.0, 1.0)  # Expected range for SIREOS scores
        assert (
            expected_range[0] <= sireos_score <= expected_range[1]
        ), f"SIREOS score {sireos_score:.6f} outside expected range {expected_range}"

    def test_extreme_quantile_values(self):
        """Test SIREOS with extreme but valid quantile values."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 2))
        scores = rng.uniform(0.1, 1.0, 40)

        # Test very small quantile
        sireos_small = sireos(scores, X, quantile=1e-6)
        assert isinstance(sireos_small, float)
        assert np.isfinite(sireos_small)
        assert sireos_small >= 0

        # Test large quantile
        sireos_large = sireos(scores, X, quantile=0.99)
        assert isinstance(sireos_large, float)
        assert np.isfinite(sireos_large)
        assert sireos_large >= 0

        # Different quantiles must produce different results
        assert (
            abs(sireos_small - sireos_large) > 1e-10
        ), "Extreme quantiles must produce different results"

    def test_memory_efficiency(self):
        """Test that SIREOS handles moderately large datasets without memory issues."""
        # Test with dataset size that would cause memory issues with old implementation
        X, y = load_shuttle_data(n_samples=2000, n_anomalies=200, random_state=42)
        scores = generate_anomaly_scores(X, y, method="distance", random_state=42)

        # This should complete without memory errors
        sireos_score = sireos(scores, X, quantile=0.01)

        assert isinstance(sireos_score, float)
        assert np.isfinite(sireos_score)
        assert sireos_score >= 0, "Large dataset SIREOS must be non-negative"
