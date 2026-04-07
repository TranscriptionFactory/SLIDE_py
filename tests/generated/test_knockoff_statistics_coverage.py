"""
Test coverage for knockoff statistics computation functions.
Addresses: Various knockoff statistics, statistical properties, edge cases
"""
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import warnings

from loveslide.knockoff.stats.lasso import stat_lasso_lambdadiff, stat_lasso_lambdasmax, stat_lasso_coefdiff
from loveslide.knockoff.stats.lasso_bin import (
    stat_lasso_lambdadiff_bin, stat_lasso_lambdasmax_bin, stat_lasso_coefdiff_bin
)
from loveslide.knockoff.stats.sqrt_lasso import stat_sqrt_lasso
from loveslide.knockoff.stats.random_forest import stat_random_forest
from loveslide.knockoff.stats.stability import stat_stability_selection
from loveslide.knockoff.stats.forward import stat_forward_selection
from loveslide.knockoff.stats.base import (
    swap_columns, correct_for_swap, compute_difference_stat,
    compute_signed_max_stat, standardize
)


class TestLassoStatistics:
    """Test LASSO-based knockoff statistics."""

    def test_stat_lasso_lambdadiff_basic(self):
        """Test LASSO lambda difference statistic."""
        n, p = 100, 10
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)  # Knockoffs
        y = X[:, 0] + X[:, 1] + np.random.randn(n) * 0.5  # Sparse signal

        W = stat_lasso_lambdadiff(X, Xk, y)

        # Basic properties
        assert len(W) == p
        assert np.all(np.isfinite(W))

        # Should have some positive values for signal variables
        assert np.sum(W > 0) >= 1

    def test_stat_lasso_lambdadiff_edge_cases(self):
        """Test LASSO lambda difference with edge cases."""
        # Perfect separation case
        n, p = 50, 5
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)
        y = np.sign(X[:, 0])  # Binary response

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W = stat_lasso_lambdadiff(X, Xk, y)

        assert len(W) == p
        assert np.all(np.isfinite(W))

        # No signal case
        y = np.random.randn(n)
        W = stat_lasso_lambdadiff(X, Xk, y)
        # Should have approximately balanced positive/negative values
        pos_ratio = np.sum(W > 0) / len(W)
        assert 0.2 <= pos_ratio <= 0.8

    def test_stat_lasso_lambdasmax_basic(self):
        """Test LASSO lambda max statistic."""
        n, p = 80, 8
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)
        y = 2 * X[:, 0] - X[:, 2] + np.random.randn(n) * 0.3

        W = stat_lasso_lambdasmax(X, Xk, y)

        assert len(W) == p
        assert np.all(np.isfinite(W))

        # Signal variables should have higher statistics
        assert W[0] > np.percentile(W, 50)  # First variable has signal

    def test_stat_lasso_coefdiff_basic(self):
        """Test LASSO coefficient difference statistic."""
        n, p = 60, 6
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)
        y = X[:, 1] + X[:, 3] + np.random.randn(n) * 0.4

        W = stat_lasso_coefdiff(X, Xk, y)

        assert len(W) == p
        assert np.all(np.isfinite(W))

    def test_lasso_statistics_consistency(self):
        """Test consistency between different LASSO statistics."""
        n, p = 100, 12
        np.random.seed(42)
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)
        y = X[:, 0] + 0.5 * X[:, 2] + np.random.randn(n) * 0.3

        W1 = stat_lasso_lambdadiff(X, Xk, y)
        W2 = stat_lasso_lambdasmax(X, Xk, y)
        W3 = stat_lasso_coefdiff(X, Xk, y)

        # All should identify similar signal variables
        signal_vars = [0, 2]
        for var in signal_vars:
            # Signal variables should be above median in most statistics
            above_median_count = sum([
                W1[var] > np.median(W1),
                W2[var] > np.median(W2),
                W3[var] > np.median(W3)
            ])
            assert above_median_count >= 2


class TestBinaryLassoStatistics:
    """Test binary LASSO statistics."""

    def test_stat_lasso_lambdadiff_bin_basic(self):
        """Test binary LASSO lambda difference."""
        n, p = 120, 8
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)

        # Binary response
        linear_comb = 2 * X[:, 0] - X[:, 3]
        y = (linear_comb > np.median(linear_comb)).astype(int)

        W = stat_lasso_lambdadiff_bin(X, Xk, y)

        assert len(W) == p
        assert np.all(np.isfinite(W))

    def test_stat_lasso_lambdasmax_bin_basic(self):
        """Test binary LASSO lambda max."""
        n, p = 100, 6
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)
        y = (X[:, 1] + X[:, 4] > 0).astype(int)

        W = stat_lasso_lambdasmax_bin(X, Xk, y)

        assert len(W) == p
        assert np.all(np.isfinite(W))

    def test_stat_lasso_coefdiff_bin_basic(self):
        """Test binary LASSO coefficient difference."""
        n, p = 90, 5
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)
        y = (X[:, 0] - X[:, 2] > np.random.randn(n) * 0.5).astype(int)

        W = stat_lasso_coefdiff_bin(X, Xk, y)

        assert len(W) == p
        assert np.all(np.isfinite(W))

    def test_binary_lasso_extreme_cases(self):
        """Test binary LASSO with extreme class imbalance."""
        n, p = 100, 4
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)

        # Extreme imbalance: 95% class 0, 5% class 1
        y = np.zeros(n)
        y[:5] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W = stat_lasso_lambdadiff_bin(X, Xk, y)

        assert len(W) == p
        assert np.all(np.isfinite(W))


class TestSqrtLassoStatistics:
    """Test Square-root LASSO statistics."""

    def test_stat_sqrt_lasso_basic(self):
        """Test square-root LASSO statistic."""
        n, p = 80, 6
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)
        y = X[:, 0] + 0.8 * X[:, 2] + np.random.randn(n) * 0.4

        W = stat_sqrt_lasso(X, Xk, y)

        assert len(W) == p
        assert np.all(np.isfinite(W))

    def test_sqrt_lasso_vs_regular_lasso(self):
        """Compare sqrt LASSO with regular LASSO."""
        n, p = 100, 8
        np.random.seed(123)
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)
        y = X[:, 1] + X[:, 4] + np.random.randn(n) * 0.3

        W_sqrt = stat_sqrt_lasso(X, Xk, y)
        W_lasso = stat_lasso_lambdadiff(X, Xk, y)

        # Both should identify signal variables
        signal_vars = [1, 4]
        for var in signal_vars:
            assert W_sqrt[var] > np.percentile(W_sqrt, 60)
            assert W_lasso[var] > np.percentile(W_lasso, 60)


class TestRandomForestStatistics:
    """Test Random Forest statistics."""

    def test_stat_random_forest_basic(self):
        """Test Random Forest statistic."""
        n, p = 150, 10
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)

        # Non-linear signal
        y = X[:, 0]**2 + np.sin(X[:, 3]) + np.random.randn(n) * 0.2

        W = stat_random_forest(X, Xk, y)

        assert len(W) == p
        assert np.all(np.isfinite(W))

        # Should detect non-linear patterns
        assert np.max(W) > 0

    def test_random_forest_classification(self):
        """Test Random Forest with classification."""
        n, p = 120, 6
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)

        # Classification with non-linear boundary
        y = ((X[:, 0] * X[:, 1]) > 0).astype(int)

        W = stat_random_forest(X, Xk, y, classification=True)

        assert len(W) == p
        assert np.all(np.isfinite(W))


class TestStabilitySelection:
    """Test Stability Selection statistics."""

    def test_stat_stability_selection_basic(self):
        """Test stability selection statistic."""
        n, p = 100, 8
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)
        y = X[:, 0] + X[:, 3] + np.random.randn(n) * 0.4

        # Use small number of subsamples for testing
        W = stat_stability_selection(X, Xk, y, n_bootstrap=10)

        assert len(W) == p
        assert np.all(np.isfinite(W))
        assert np.all(W >= 0)  # Stability scores are non-negative


class TestForwardSelection:
    """Test Forward Selection statistics."""

    def test_stat_forward_selection_basic(self):
        """Test forward selection statistic."""
        n, p = 60, 5
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)
        y = 2 * X[:, 1] + np.random.randn(n) * 0.3

        W = stat_forward_selection(X, Xk, y)

        assert len(W) == p
        assert np.all(np.isfinite(W))

    def test_forward_selection_max_features(self):
        """Test forward selection with maximum features limit."""
        n, p = 80, 10
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)
        y = X[:, 0] + X[:, 2] + X[:, 5] + np.random.randn(n) * 0.2

        # Limit to fewer features than signal
        W = stat_forward_selection(X, Xk, y, max_features=2)

        assert len(W) == p
        assert np.all(np.isfinite(W))

        # Should select some features
        assert np.sum(W > 0) <= 2 * 2  # At most 2 original + 2 knockoff


class TestBaseStatisticsFunctions:
    """Test base statistics utility functions."""

    def test_swap_columns_basic(self):
        """Test column swapping functionality."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Xk = np.array([[7, 8, 9], [10, 11, 12]])

        X_swap, Xk_swap, swap = swap_columns(X, Xk, randomize=False)

        # With randomize=False, should be identity
        assert_array_equal(X_swap, X)
        assert_array_equal(Xk_swap, Xk)
        assert_array_equal(swap, np.arange(3))

    def test_swap_columns_randomized(self):
        """Test randomized column swapping."""
        np.random.seed(42)
        X = np.random.randn(50, 6)
        Xk = np.random.randn(50, 6)

        X_swap, Xk_swap, swap = swap_columns(X, Xk, randomize=True)

        # Shapes should be preserved
        assert X_swap.shape == X.shape
        assert Xk_swap.shape == Xk.shape
        assert len(swap) == X.shape[1]

        # Some columns should be swapped
        swapped_count = np.sum(swap != np.arange(len(swap)))
        assert swapped_count >= 1  # At least some swapping should occur

    def test_correct_for_swap_basic(self):
        """Test correction for column swapping."""
        W = np.array([1, -2, 3, -4, 5])
        swap = np.array([0, 1, 1, 0, 1])  # Swap columns 1, 2, 4

        W_corrected = correct_for_swap(W, swap)

        # Swapped columns should have sign flipped
        expected = np.array([1, 2, -3, -4, -5])
        assert_array_equal(W_corrected, expected)

    def test_compute_difference_stat_basic(self):
        """Test difference statistic computation."""
        Z = np.array([1, -2, 3, -4, 5, -6])  # 3 original, 3 knockoff

        W = compute_difference_stat(Z)

        # Should compute X - X_knockoff
        expected = np.array([1 - (-4), -2 - 5, 3 - (-6)])  # [5, -7, 9]
        assert_array_equal(W, expected)

    def test_compute_signed_max_stat_basic(self):
        """Test signed max statistic computation."""
        Z = np.array([2, -1, 4, -3, 1, -5])

        W = compute_signed_max_stat(Z)

        # Should compute sign(X - X_k) * max(|X|, |X_k|)
        expected = np.array([
            np.sign(2 - (-3)) * max(abs(2), abs(-3)),  # 1 * 3 = 3
            np.sign(-1 - 1) * max(abs(-1), abs(1)),    # -1 * 1 = -1
            np.sign(4 - (-5)) * max(abs(4), abs(-5))   # 1 * 5 = 5
        ])
        assert_array_equal(W, expected)

    def test_standardize_basic(self):
        """Test feature standardization."""
        X = np.array([[1, 4], [2, 5], [3, 6]], dtype=float)

        X_std = standardize(X)

        # Should have mean 0 and std 1
        assert_array_almost_equal(np.mean(X_std, axis=0), 0, decimal=10)
        assert_array_almost_equal(np.std(X_std, axis=0, ddof=0), 1, decimal=10)

    def test_standardize_edge_cases(self):
        """Test standardization edge cases."""
        # Constant column
        X = np.array([[1, 2], [1, 3], [1, 4]], dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_std = standardize(X)

        # Constant column should become zero
        assert_array_almost_equal(X_std[:, 0], 0)

        # Single row
        X = np.array([[1, 2, 3]], dtype=float)
        X_std = standardize(X)
        assert X_std.shape == (1, 3)


class TestStatisticsIntegration:
    """Test integration between different statistic methods."""

    def test_statistic_method_comparison(self):
        """Compare different statistic methods on same data."""
        n, p = 100, 6
        np.random.seed(789)
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)
        y = 2 * X[:, 0] + X[:, 3] + np.random.randn(n) * 0.3

        # Compute different statistics
        W_lasso = stat_lasso_lambdadiff(X, Xk, y)
        W_sqrt = stat_sqrt_lasso(X, Xk, y)
        W_rf = stat_random_forest(X, Xk, y)

        # All should identify signal variables
        signal_vars = [0, 3]
        for var in signal_vars:
            # Signal variables should be above median in most methods
            above_median = [
                W_lasso[var] > np.median(W_lasso),
                W_sqrt[var] > np.median(W_sqrt),
                W_rf[var] > np.median(W_rf)
            ]
            assert sum(above_median) >= 2

    def test_swap_correction_consistency(self):
        """Test that swap correction works consistently."""
        n, p = 50, 4
        X = np.random.randn(n, p)
        Xk = np.random.randn(n, p)

        # Generate swaps
        X_swap, Xk_swap, swap = swap_columns(X, Xk, randomize=True, seed=456)

        # Compute statistic with swapped data
        y = X_swap[:, 0] + np.random.randn(n) * 0.2
        W_swap = stat_lasso_lambdadiff(X_swap, Xk_swap, y)

        # Correct for swaps
        W_corrected = correct_for_swap(W_swap, swap)

        assert len(W_corrected) == p
        assert np.all(np.isfinite(W_corrected))

    def test_robustness_across_methods(self):
        """Test robustness of different methods to data characteristics."""
        n, p = 80, 5

        # Test with different data characteristics
        data_types = {
            'gaussian': lambda: np.random.randn(n, p),
            'heavy_tailed': lambda: np.random.standard_t(3, size=(n, p)),
            'correlated': lambda: np.random.multivariate_normal(
                np.zeros(p), 0.5 * np.ones((p, p)) + 0.5 * np.eye(p), size=n
            )
        }

        for data_name, data_gen in data_types.items():
            X = data_gen()
            Xk = data_gen()
            y = X[:, 0] + np.random.randn(n) * 0.3

            # All methods should produce finite statistics
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                W_lasso = stat_lasso_lambdadiff(X, Xk, y)
                W_rf = stat_random_forest(X, Xk, y)

                assert np.all(np.isfinite(W_lasso)), f"LASSO failed on {data_name}"
                assert np.all(np.isfinite(W_rf)), f"RF failed on {data_name}"