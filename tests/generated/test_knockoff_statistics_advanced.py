"""
Advanced test coverage for knockoff statistical methods.
Addresses gaps in statistical property validation and method reliability.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import sys
sys.path.insert(0, 'src')

from loveslide.knockoff.stats.base import (
    swap_columns, correct_for_swap, compute_difference_stat,
    compute_signed_max_stat, standardize
)
from loveslide.knockoff.stats.lasso import stat_lasso_coefdiff, stat_lasso_lambdadiff
from loveslide.knockoff.stats.sqrt_lasso import stat_sqrt_lasso
from loveslide.knockoff.stats.random_forest import stat_random_forest
from loveslide.knockoff.stats.stability import stat_stability_selection


class TestKnockoffStatisticalProperties:
    """Test fundamental statistical properties of knockoff methods."""

    def test_swap_columns_basic(self):
        """Test basic column swapping functionality."""
        X = np.random.randn(50, 10)
        Xk = np.random.randn(50, 10)

        # Test random swapping
        X_new, Xk_new, swap = swap_columns(X, Xk, seed=42)

        assert X_new.shape == X.shape
        assert Xk_new.shape == Xk.shape
        assert len(swap) == X.shape[1]
        assert all(s in [0, 1] for s in swap)

    def test_swap_columns_deterministic(self):
        """Test deterministic swapping with fixed seed."""
        X = np.random.randn(50, 10)
        Xk = np.random.randn(50, 10)

        # Same seed should produce same swaps
        _, _, swap1 = swap_columns(X, Xk, seed=42)
        _, _, swap2 = swap_columns(X, Xk, seed=42)

        assert_array_almost_equal(swap1, swap2)

    def test_swap_columns_edge_cases(self):
        """Test edge cases in column swapping."""
        # Single column
        X = np.random.randn(50, 1)
        Xk = np.random.randn(50, 1)

        X_new, Xk_new, swap = swap_columns(X, Xk)
        assert len(swap) == 1

        # Empty matrix
        with pytest.raises(ValueError):
            swap_columns(np.array([]), np.array([]))

    def test_correct_for_swap_antisymmetry(self):
        """Test that swap correction preserves antisymmetry property."""
        W = np.array([1.5, -0.8, 2.3, -1.1, 0.9])
        swap = np.array([1, 0, 1, 0, 1])  # Swap 1st, 3rd, 5th

        W_corrected = correct_for_swap(W, swap)

        # After correction, antisymmetry should be preserved
        # TODO: Validate antisymmetry property
        assert len(W_corrected) == len(W)

    def test_compute_difference_stat_properties(self):
        """Test statistical properties of difference statistic."""
        X = np.random.randn(100, 20)
        Xk = np.random.randn(100, 20)
        y = np.random.randn(100)

        W = compute_difference_stat(X, Xk, y, method='lasso')

        # Should have correct dimension
        assert len(W) == X.shape[1]

        # Test with different methods
        W_sqrt = compute_difference_stat(X, Xk, y, method='sqrt_lasso')
        assert len(W_sqrt) == X.shape[1]

    def test_compute_signed_max_stat(self):
        """Test signed max statistic computation."""
        X = np.random.randn(100, 20)
        Xk = np.random.randn(100, 20)
        y = np.random.randn(100)

        W = compute_signed_max_stat(X, Xk, y, method='lasso')

        # Should have correct dimension
        assert len(W) == X.shape[1]

    def test_standardize_functionality(self):
        """Test feature standardization."""
        X = np.random.randn(100, 10) * 5 + 3  # Non-standard data

        X_std = standardize(X)

        # Should be standardized (mean ~0, std ~1)
        assert abs(np.mean(X_std)) < 1e-10
        assert abs(np.std(X_std, axis=0).mean() - 1) < 1e-10

    def test_standardize_edge_cases(self):
        """Test standardization edge cases."""
        # Constant columns
        X = np.ones((50, 3))
        X_std = standardize(X)

        # Should handle constant columns gracefully
        assert not np.any(np.isnan(X_std))


class TestMethodComparison:
    """Test consistency and comparison between different knockoff methods."""

    def test_lasso_methods_consistency(self):
        """Test consistency between different LASSO-based methods."""
        X = np.random.randn(100, 20)
        Xk = np.random.randn(100, 20)
        y = np.random.randn(100)

        W_coef = stat_lasso_coefdiff(X, Xk, y)
        W_lambda = stat_lasso_lambdadiff(X, Xk, y)

        # Both should have same dimension
        assert len(W_coef) == len(W_lambda) == X.shape[1]

        # Correlation should be positive (testing same underlying signal)
        correlation = np.corrcoef(W_coef, W_lambda)[0, 1]
        # TODO: Define acceptable correlation threshold

    def test_sqrt_lasso_vs_regular_lasso(self):
        """Test relationship between sqrt-lasso and regular lasso."""
        X = np.random.randn(100, 20)
        Xk = np.random.randn(100, 20)
        y = np.random.randn(100)

        W_lasso = stat_lasso_coefdiff(X, Xk, y)
        W_sqrt = stat_sqrt_lasso(X, Xk, y)

        # Should have same dimension
        assert len(W_lasso) == len(W_sqrt)

        # Should capture similar signal patterns
        # TODO: Define similarity metrics

    def test_random_forest_basic(self):
        """Test random forest knockoff statistics."""
        X = np.random.randn(100, 20)
        Xk = np.random.randn(100, 20)
        y = np.random.choice([0, 1], 100)  # Binary classification

        W_rf = stat_random_forest(X, Xk, y)

        assert len(W_rf) == X.shape[1]
        assert not np.any(np.isnan(W_rf))

    def test_stability_selection(self):
        """Test stability selection method."""
        X = np.random.randn(100, 20)
        Xk = np.random.randn(100, 20)
        y = np.random.randn(100)

        W_stability = stat_stability_selection(X, Xk, y)

        assert len(W_stability) == X.shape[1]
        # Stability selection scores should be in [0, 1]
        assert np.all((W_stability >= 0) & (W_stability <= 1))


class TestStatisticalValidation:
    """Test statistical validation and properties."""

    def test_null_hypothesis_behavior(self):
        """Test behavior under null hypothesis (no signal)."""
        np.random.seed(42)

        # Generate pure noise data
        X = np.random.randn(200, 50)
        Xk = np.random.randn(200, 50)
        y = np.random.randn(200)  # No relationship to X

        W = stat_lasso_coefdiff(X, Xk, y)

        # Under null, expect roughly balanced positive/negative statistics
        pos_count = np.sum(W > 0)
        neg_count = np.sum(W < 0)

        # Should be roughly balanced (allowing some randomness)
        assert abs(pos_count - neg_count) < 15  # Allow some variation

    def test_signal_detection_power(self):
        """Test power to detect true signal."""
        np.random.seed(42)

        # Generate data with signal in first 5 features
        X = np.random.randn(200, 20)
        true_coeffs = np.zeros(20)
        true_coeffs[:5] = np.array([2, -1.5, 3, -2, 1.8])  # Strong signal

        y = X @ true_coeffs + 0.1 * np.random.randn(200)
        Xk = np.random.randn(200, 20)

        W = stat_lasso_coefdiff(X, Xk, y)

        # Should detect signal in first 5 features
        signal_stats = W[:5]
        noise_stats = W[5:]

        # Signal features should tend to have higher absolute statistics
        mean_signal = np.mean(np.abs(signal_stats))
        mean_noise = np.mean(np.abs(noise_stats))

        assert mean_signal > mean_noise

    def test_method_robustness(self):
        """Test robustness to outliers and data quality issues."""
        np.random.seed(42)

        # Base data
        X = np.random.randn(100, 10)
        Xk = np.random.randn(100, 10)
        y = np.random.randn(100)

        # Clean statistics
        W_clean = stat_lasso_coefdiff(X, Xk, y)

        # Add outliers
        X_outlier = X.copy()
        X_outlier[0] = 10  # Extreme outlier
        y_outlier = y.copy()
        y_outlier[0] = 10

        W_outlier = stat_lasso_coefdiff(X_outlier, Xk, y_outlier)

        # Methods should be reasonably robust
        correlation = np.corrcoef(W_clean, W_outlier)[0, 1]
        assert correlation > 0.5  # Should maintain reasonable similarity


class TestComputationalEfficiency:
    """Test computational efficiency and scalability."""

    def test_large_problem_scalability(self):
        """Test behavior with large problems."""
        # Test with moderately large problem
        n, p = 500, 100
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)
        y = np.random.randn(n)

        # Should complete without excessive time/memory
        W = stat_lasso_coefdiff(X, Xk, y)
        assert len(W) == p

    def test_high_dimensional_regime(self):
        """Test high-dimensional regime (p > n)."""
        # High-dimensional setting
        n, p = 50, 200
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)
        y = np.random.randn(n)

        # Methods should handle high-dimensional case
        W = stat_lasso_coefdiff(X, Xk, y)
        assert len(W) == p
        assert not np.any(np.isnan(W))