import numpy as np
import pytest
from labelfree.metrics.mass_exceedance import mass_exceedance_auc
from .synthetic_data import make_blobs_with_anomalies, make_anomaly_scores


class TestExcessMass:
    """Test suite for Excess-Mass curve.

    Note: Excess-Mass measures how well the scoring function captures
    high-density regions: EM(t) = P(score > s) - t * V(score > s).
    Lower EM-AUC indicates better anomaly detection performance (less area under curve
    means the curve drops faster, indicating better separation).
    """

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

        # Check output structure
        assert set(result.keys()) == {"t", "em", "auc", "amax"}
        assert len(result["t"]) > 0  # Length depends on volume_support
        assert len(result["em"]) == len(result["t"])
        assert isinstance(result["auc"], float)
        assert isinstance(result["amax"], (int, np.integer))

        # Check fundamental EM properties
        assert result["em"][0] == 1.0  # EM(0) = 1.0 by definition
        assert result["auc"] >= 0  # AUC should be non-negative
        assert np.all(result["em"] >= 0)  # EM values should be non-negative
        assert result["amax"] > 0  # Should have valid threshold index

        # EM curve should generally decrease (with some Monte Carlo noise)
        em_increases = np.sum(np.diff(result["em"]) > 0.01)
        assert em_increases <= len(result["em"]) * 0.3, "EM should generally decrease"

    def test_perfect_detector(self):
        """Test EM curve for perfect anomaly detector."""
        X, y = make_blobs_with_anomalies(n_samples=500, n_anomalies=50, random_state=42)
        scores = make_anomaly_scores(
            X, y, method="perfect", noise_level=0, random_state=42
        )

        # Generate volume scores
        rng = np.random.default_rng(42)
        volume_scores = rng.standard_normal(2000)

        result = mass_exceedance_auc(scores, volume_scores)

        # Perfect detector should have good EM properties
        assert result["em"][0] == 1.0
        assert result["auc"] > 0  # Should have positive AUC

        # Perfect detector should have quick EM decay
        # (anomalies are clearly separated, so EM drops quickly)
        low_em_ratio = np.sum(result["em"] < 0.1) / len(result["em"])
        assert (
            low_em_ratio > 0.8
        ), f"Perfect detector should have quick EM decay, got {low_em_ratio:.3f} low ratio"

        # Should reach t_max threshold reasonably
        assert (
            1 < result["amax"] < len(result["em"])
        ), "Should have meaningful threshold detection"

    def test_random_detector(self):
        """Test EM curve for random detector."""
        X, y = make_blobs_with_anomalies(n_samples=500, n_anomalies=50, random_state=42)
        scores = make_anomaly_scores(X, y, method="random", random_state=42)

        # Generate volume scores
        rng = np.random.default_rng(42)
        volume_scores = rng.standard_normal(2000)

        result = mass_exceedance_auc(scores, volume_scores)

        # Random detector should still produce valid results
        assert result["em"][0] == 1.0
        assert result["auc"] >= 0

        # Random detector should have lower performance (quick EM decay)
        # Since random scores don't distinguish anomalies well
        em_decay_point = np.argmax(result["em"] < 0.5)
        if em_decay_point > 0:  # If EM does drop below 0.5
            decay_ratio = em_decay_point / len(result["em"])
            assert decay_ratio < 0.5, "Random detector should have quick EM decay"

    def test_perfect_vs_random_comparison(self):
        """Test that perfect detector significantly outperforms random detector."""
        X, y = make_blobs_with_anomalies(n_samples=400, n_anomalies=40, random_state=42)

        perfect_scores = make_anomaly_scores(
            X, y, method="perfect", noise_level=0, random_state=42
        )
        random_scores = make_anomaly_scores(X, y, method="random", random_state=42)

        # Use same volume scores for fair comparison
        rng = np.random.default_rng(42)
        volume_scores = rng.standard_normal(1500)

        perfect_result = mass_exceedance_auc(perfect_scores, volume_scores)
        random_result = mass_exceedance_auc(random_scores, volume_scores)

        # Perfect detector should have lower EM-AUC (better performance)
        assert (
            perfect_result["auc"] < random_result["auc"]
        ), f"Perfect detector AUC ({perfect_result['auc']:.3f}) should be lower than random ({random_result['auc']:.3f})"

        # The difference should be meaningful
        improvement_ratio = random_result["auc"] / max(perfect_result["auc"], 1e-12)
        assert (
            improvement_ratio > 2.0
        ), f"Perfect detector should be significantly better, got improvement ratio {improvement_ratio:.3f}"

    def test_input_validation(self):
        """Test comprehensive input validation."""
        rng = np.random.default_rng(42)
        scores = rng.standard_normal(100)
        volume_scores = rng.standard_normal(500)

        # Empty inputs
        with pytest.raises(ValueError, match="scores cannot be empty"):
            mass_exceedance_auc([], volume_scores)

        with pytest.raises(ValueError, match="volume_scores cannot be empty"):
            mass_exceedance_auc(scores, [])

        # Invalid dimensions
        with pytest.raises(ValueError, match="scores must be 1D array"):
            mass_exceedance_auc(scores.reshape(-1, 1), volume_scores)

        with pytest.raises(ValueError, match="volume_scores must be 1D array"):
            mass_exceedance_auc(scores, volume_scores.reshape(-1, 1))

        # Non-finite values
        scores_with_nan = scores.copy()
        scores_with_nan[0] = np.nan
        with pytest.raises(ValueError, match="scores contains non-finite values"):
            mass_exceedance_auc(scores_with_nan, volume_scores)

        scores_with_inf = scores.copy()
        scores_with_inf[0] = np.inf
        with pytest.raises(ValueError, match="scores contains non-finite values"):
            mass_exceedance_auc(scores_with_inf, volume_scores)

        # Invalid parameters
        with pytest.raises(ValueError):
            mass_exceedance_auc(scores, volume_scores, volume_support=-1.0)

        with pytest.raises(ValueError):
            mass_exceedance_auc(scores, volume_scores, t_max=-0.1)

    def test_reproducibility(self):
        """Test that results are reproducible with fixed parameters."""
        X, y = make_blobs_with_anomalies(n_samples=200, random_state=42)
        scores = make_anomaly_scores(X, y, method="distance", random_state=42)

        # Generate same volume scores
        rng = np.random.default_rng(42)
        volume_scores = rng.standard_normal(1000)

        # Same inputs should produce identical results
        result1 = mass_exceedance_auc(
            scores, volume_scores, volume_support=10.0, t_max=0.8
        )
        result2 = mass_exceedance_auc(
            scores, volume_scores, volume_support=10.0, t_max=0.8
        )

        np.testing.assert_array_equal(result1["t"], result2["t"])
        np.testing.assert_array_equal(result1["em"], result2["em"])
        assert result1["auc"] == result2["auc"]
        assert result1["amax"] == result2["amax"]

    def test_level_generation_accuracy(self):
        """Test that level generation follows the correct formula."""
        rng = np.random.default_rng(42)
        scores = rng.standard_normal(50)
        volume_scores = rng.standard_normal(200)

        # Test with known volume support
        volume_support = 5.0
        result = mass_exceedance_auc(
            scores, volume_scores, volume_support=volume_support
        )

        # Verify level generation: t = np.arange(0, 100/volume_support, 0.01/volume_support)
        expected_max_t = 100.0 / volume_support
        expected_step = 0.01 / volume_support
        expected_t = np.arange(0, expected_max_t, expected_step)

        np.testing.assert_allclose(result["t"], expected_t, rtol=1e-10)
        assert len(result["t"]) == len(expected_t)

    def test_threshold_detection_accuracy(self):
        """Test amax calculation (where EM drops below t_max)."""
        rng = np.random.default_rng(42)
        scores = rng.standard_normal(100)
        volume_scores = rng.standard_normal(500)

        # Test with different t_max values
        for t_max in [0.5, 0.7, 0.9]:
            result = mass_exceedance_auc(scores, volume_scores, t_max=t_max)

            if result["amax"] < len(result["em"]):
                # If threshold was found, verify it's correct
                amax_idx = result["amax"] - 1  # Convert to 0-based index
                if amax_idx > 0:
                    assert (
                        result["em"][amax_idx] <= t_max
                    ), f"EM at amax should be <= t_max ({t_max})"
                    if amax_idx > 1:
                        assert (
                            result["em"][amax_idx - 1] > t_max
                        ), f"EM before amax should be > t_max ({t_max})"

    def test_identical_scores(self):
        """Test behavior with identical scores (edge case)."""
        # All identical scores
        scores = np.ones(100)
        volume_scores = np.ones(200)  # Also identical

        result = mass_exceedance_auc(scores, volume_scores)

        # Should handle identical scores gracefully
        assert result["em"][0] == 1.0
        assert result["auc"] >= 0
        assert np.all(np.isfinite(result["em"]))

        # With identical scores, EM behavior depends on the specific values
        # But algorithm should still work without crashes
        assert len(result["t"]) > 0
        assert result["amax"] > 0

    def test_single_score_data(self):
        """Test with minimal data (edge case)."""
        scores = np.array([0.5])  # Single score
        volume_scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7])  # Few volume scores

        result = mass_exceedance_auc(scores, volume_scores, volume_support=1.0)

        # Should handle single score gracefully
        assert result["em"][0] == 1.0
        assert result["auc"] >= 0
        assert np.all(np.isfinite(result["em"]))
        assert len(result["t"]) > 0

    def test_volume_support_scaling(self):
        """Test behavior with different volume support values."""
        rng = np.random.default_rng(42)
        scores = rng.standard_normal(100)
        volume_scores = rng.standard_normal(500)

        # Test with different volume support values
        vol_small = 0.5
        vol_medium = 5.0
        vol_large = 50.0

        result_small = mass_exceedance_auc(
            scores, volume_scores, volume_support=vol_small
        )
        result_medium = mass_exceedance_auc(
            scores, volume_scores, volume_support=vol_medium
        )
        result_large = mass_exceedance_auc(
            scores, volume_scores, volume_support=vol_large
        )

        # Level arrays should have different ranges
        assert (
            result_small["t"].max() > result_medium["t"].max() > result_large["t"].max()
        )

        # AUC should scale appropriately with volume support
        # (smaller volume support = finer level resolution = potentially different AUC)
        assert result_small["auc"] >= 0
        assert result_medium["auc"] >= 0
        assert result_large["auc"] >= 0

    def test_degenerate_volume_scores(self):
        """Test with volume scores that have no variation."""
        rng = np.random.default_rng(42)
        scores = rng.standard_normal(100)
        volume_scores = np.zeros(500)  # All zero volume scores

        result = mass_exceedance_auc(scores, volume_scores)

        # Should handle degenerate volume scores
        assert result["em"][0] == 1.0
        assert result["auc"] >= 0
        assert np.all(np.isfinite(result["em"]))

        # With zero volume scores, the uniform term becomes zero
        # So EM should equal the anomaly fraction
        # This is a valid edge case that should not crash

    def test_regression_golden_standard(self):
        """Test against golden standard results to catch algorithm changes."""
        # Fixed synthetic scenario for regression testing
        rng = np.random.default_rng(42)
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Simple increasing scores
        volume_scores = rng.standard_normal(100)  # Fixed random volume scores

        result = mass_exceedance_auc(
            scores, volume_scores, volume_support=2.0, t_max=0.8
        )

        # These values computed with current implementation
        # Should remain stable unless algorithm changes
        expected_auc = 0.004499  # Fixed value from current implementation
        assert (
            abs(result["auc"] - expected_auc) < 0.001
        ), f"AUC changed significantly from expected {expected_auc:.6f}, got {result['auc']:.6f}"

        # Basic sanity checks
        assert result["em"][0] == 1.0
        assert result["amax"] > 0
        assert (
            len(result["t"]) == 10000
        )  # Should be 10000 levels for volume_support=2.0 (100/2.0 / 0.01/2.0)

    def test_em_curve_mathematical_properties(self):
        """Test fundamental mathematical properties of EM curve."""
        X, y = make_blobs_with_anomalies(n_samples=300, n_anomalies=30, random_state=42)
        scores = make_anomaly_scores(X, y, method="distance", random_state=42)

        rng = np.random.default_rng(42)
        volume_scores = rng.standard_normal(1000)

        result = mass_exceedance_auc(scores, volume_scores)

        # EM(0) = 1.0 by definition
        assert result["em"][0] == 1.0

        # EM values should be bounded (can be negative in theory, but typically >= 0)
        assert np.all(result["em"] >= -1.0), "EM values should be reasonably bounded"

        # Level values should be non-negative and increasing
        assert np.all(result["t"] >= 0)
        assert np.all(
            np.diff(result["t"]) > 0
        ), "Level values should be strictly increasing"

        # AUC should be finite and non-negative
        assert np.isfinite(result["auc"])
        assert result["auc"] >= 0
