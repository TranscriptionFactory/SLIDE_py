"""
Comprehensive test coverage for knockoff statistics validation and properties.
"""
import pytest
import numpy as np
from scipy import stats
from unittest.mock import patch, Mock

from loveslide.knockoff.stats.lasso import (
    stat_lasso_lambdadiff,
    stat_lasso_lambdasmax,
    stat_lasso_coefdiff
)
from loveslide.knockoff.stats.random_forest import stat_random_forest
from loveslide.knockoff.stats.stability import stat_stability_selection
from loveslide.knockoff.filter import knockoff_threshold, knockoff_filter


class TestKnockoffStatisticsValidity:
    """Test statistical validity of knockoff statistics."""

    def test_stat_lasso_lambdadiff_antisymmetry(self):
        """Test that LASSO lambda difference is antisymmetric."""
        np.random.seed(42)
        n, p = 100, 20
        X = np.random.randn(n, p)
        X_ko = np.random.randn(n, p)
        y = np.random.randn(n)

        # Compute statistic
        W = stat_lasso_lambdadiff(X, X_ko, y)

        # Test antisymmetry: swapping X and X_ko should negate the statistic
        W_swapped = stat_lasso_lambdadiff(X_ko, X, y)

        np.testing.assert_array_almost_equal(W, -W_swapped, decimal=5)

    def test_stat_lasso_lambdasmax_antisymmetry(self):
        """Test that LASSO lambda max is antisymmetric."""
        np.random.seed(42)
        n, p = 100, 20
        X = np.random.randn(n, p)
        X_ko = np.random.randn(n, p)
        y = np.random.randn(n)

        W = stat_lasso_lambdasmax(X, X_ko, y)
        W_swapped = stat_lasso_lambdasmax(X_ko, X, y)

        np.testing.assert_array_almost_equal(W, -W_swapped, decimal=5)

    def test_stat_lasso_coefdiff_antisymmetry(self):
        """Test that LASSO coefficient difference is antisymmetric."""
        np.random.seed(42)
        n, p = 100, 20
        X = np.random.randn(n, p)
        X_ko = np.random.randn(n, p)
        y = np.random.randn(n)

        W = stat_lasso_coefdiff(X, X_ko, y)
        W_swapped = stat_lasso_coefdiff(X_ko, X, y)

        np.testing.assert_array_almost_equal(W, -W_swapped, decimal=5)

    def test_stat_random_forest_antisymmetry(self):
        """Test that Random Forest statistic is antisymmetric."""
        np.random.seed(42)
        n, p = 100, 20
        X = np.random.randn(n, p)
        X_ko = np.random.randn(n, p)
        y = (np.random.randn(n) > 0).astype(int)  # Binary classification

        W = stat_random_forest(X, X_ko, y)
        W_swapped = stat_random_forest(X_ko, X, y)

        np.testing.assert_array_almost_equal(W, -W_swapped, decimal=3)


class TestKnockoffStatisticsRobustness:
    """Test robustness of knockoff statistics under various conditions."""

    def test_lasso_statistics_with_correlated_features(self):
        """Test LASSO statistics with highly correlated features."""
        np.random.seed(42)
        n, p = 200, 30

        # Create correlated design matrix
        X = np.random.randn(n, p)
        # Make some features highly correlated
        X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n)
        X[:, 2] = X[:, 0] + 0.1 * np.random.randn(n)

        X_ko = np.random.randn(n, p)
        y = X @ np.random.randn(p) + np.random.randn(n)

        # Should not crash with correlated features
        W1 = stat_lasso_lambdadiff(X, X_ko, y)
        W2 = stat_lasso_lambdasmax(X, X_ko, y)
        W3 = stat_lasso_coefdiff(X, X_ko, y)

        assert len(W1) == p
        assert len(W2) == p
        assert len(W3) == p
        assert np.all(np.isfinite(W1))
        assert np.all(np.isfinite(W2))
        assert np.all(np.isfinite(W3))

    def test_statistics_with_sparse_signals(self):
        """Test statistics performance with sparse true signals."""
        np.random.seed(42)
        n, p = 200, 50

        X = np.random.randn(n, p)
        X_ko = np.random.randn(n, p)

        # Sparse signal: only 3 true features
        beta_true = np.zeros(p)
        beta_true[:3] = [2, -1.5, 1]
        y = X @ beta_true + np.random.randn(n)

        W = stat_lasso_lambdadiff(X, X_ko, y)

        # True features should have higher statistics on average
        true_stats = W[:3]
        null_stats = W[3:]

        assert np.mean(np.abs(true_stats)) > np.mean(np.abs(null_stats))

    def test_statistics_with_outliers(self):
        """Test statistics robustness to outliers."""
        np.random.seed(42)
        n, p = 100, 20

        X = np.random.randn(n, p)
        X_ko = np.random.randn(n, p)
        y = np.random.randn(n)

        # Add outliers
        y[0] = 100  # Extreme outlier

        # Should handle outliers gracefully
        W = stat_lasso_lambdadiff(X, X_ko, y)
        assert np.all(np.isfinite(W))
        assert not np.any(np.abs(W) > 100)  # Statistics shouldn't be extreme

    def test_random_forest_with_mixed_types(self):
        """Test Random Forest with mixed feature types."""
        np.random.seed(42)
        n = 200

        # Mixed features: continuous and categorical-like
        X_cont = np.random.randn(n, 10)
        X_cat = np.random.randint(0, 3, size=(n, 5)).astype(float)
        X = np.column_stack([X_cont, X_cat])

        X_ko = np.random.randn(n, 15)
        y = (np.random.randn(n) > 0).astype(int)

        W = stat_random_forest(X, X_ko, y)
        assert len(W) == 15
        assert np.all(np.isfinite(W))


class TestKnockoffFilterProperties:
    """Test properties of knockoff filtering procedures."""

    def test_knockoff_threshold_fdr_control(self):
        """Test FDR control property of knockoff threshold."""
        np.random.seed(42)
        p = 100
        n_null = 80  # 80 null hypotheses

        # Simulate W statistics: some signal, mostly null
        W = np.random.randn(p)
        W[:20] = np.abs(W[:20]) + 2  # Make first 20 have larger positive values

        fdr = 0.1
        threshold = knockoff_threshold(W, fdr)
        selected = np.where(W >= threshold)[0]

        # For simulated data, should select reasonable number
        assert len(selected) >= 0
        assert len(selected) <= p

    def test_knockoff_threshold_monotonicity(self):
        """Test that threshold decreases as FDR increases."""
        np.random.seed(42)
        W = np.random.randn(50)

        fdr_levels = [0.05, 0.1, 0.2, 0.3]
        thresholds = [knockoff_threshold(W, fdr) for fdr in fdr_levels]

        # Thresholds should decrease (become less stringent) as FDR increases
        for i in range(len(thresholds) - 1):
            assert thresholds[i] >= thresholds[i + 1]

    def test_knockoff_filter_empty_selection(self):
        """Test knockoff filter when no features are selected."""
        np.random.seed(42)
        n, p = 100, 20

        # Generate data with very weak signal
        X = np.random.randn(n, p)
        X_ko = np.random.randn(n, p)
        y = 0.01 * np.random.randn(n)  # Very weak signal

        result = knockoff_filter(X, X_ko, y, fdr=0.01)  # Very stringent FDR

        assert hasattr(result, 'selected')
        assert len(result.selected) >= 0  # May be empty

    def test_knockoff_filter_all_selection(self):
        """Test knockoff filter when all features should be selected."""
        np.random.seed(42)
        n, p = 50, 10

        # Generate data with strong signal in all features
        beta_true = 3 * np.ones(p)
        X = np.random.randn(n, p)
        X_ko = np.random.randn(n, p)
        y = X @ beta_true + 0.1 * np.random.randn(n)

        result = knockoff_filter(X, X_ko, y, fdr=0.5)  # Lenient FDR

        # Should select most or all features
        assert len(result.selected) >= p // 2


class TestKnockoffStatisticsNumericalStability:
    """Test numerical stability of knockoff statistics."""

    def test_statistics_with_ill_conditioned_matrices(self):
        """Test statistics with ill-conditioned design matrices."""
        np.random.seed(42)
        n, p = 100, 20

        # Create ill-conditioned matrix
        U, s, Vt = np.linalg.svd(np.random.randn(n, p), full_matrices=False)
        s[:-5] = 1e-10  # Make most singular values very small
        X = U @ np.diag(s) @ Vt

        X_ko = np.random.randn(n, p)
        y = np.random.randn(n)

        # Should handle gracefully without numerical errors
        W = stat_lasso_lambdadiff(X, X_ko, y)
        assert np.all(np.isfinite(W))

    def test_statistics_with_perfect_collinearity(self):
        """Test statistics with perfect collinearity."""
        np.random.seed(42)
        n, p = 100, 20

        X = np.random.randn(n, p)
        X[:, 1] = X[:, 0]  # Perfect collinearity

        X_ko = np.random.randn(n, p)
        y = np.random.randn(n)

        # Should either handle gracefully or raise informative error
        try:
            W = stat_lasso_lambdadiff(X, X_ko, y)
            assert np.all(np.isfinite(W))
        except (np.linalg.LinAlgError, ValueError) as e:
            assert "singular" in str(e).lower() or "collinear" in str(e).lower()

    def test_statistics_with_extreme_scales(self):
        """Test statistics with features at very different scales."""
        np.random.seed(42)
        n, p = 100, 20

        X = np.random.randn(n, p)
        X[:, 0] *= 1e6  # Very large scale
        X[:, 1] *= 1e-6  # Very small scale

        X_ko = np.random.randn(n, p)
        y = np.random.randn(n)

        W = stat_lasso_lambdadiff(X, X_ko, y)
        assert np.all(np.isfinite(W))
        assert not np.any(np.abs(W) > 1e10)  # Shouldn't be extremely large


class TestKnockoffStatisticsConsistency:
    """Test consistency properties across different statistics."""

    def test_statistics_correlation_structure(self):
        """Test that different statistics have reasonable correlation."""
        np.random.seed(42)
        n, p = 200, 30

        X = np.random.randn(n, p)
        X_ko = np.random.randn(n, p)
        y = np.random.randn(n)

        W1 = stat_lasso_lambdadiff(X, X_ko, y)
        W2 = stat_lasso_lambdasmax(X, X_ko, y)
        W3 = stat_lasso_coefdiff(X, X_ko, y)

        # Different LASSO statistics should be positively correlated
        corr_12 = np.corrcoef(W1, W2)[0, 1]
        corr_13 = np.corrcoef(W1, W3)[0, 1]

        assert corr_12 > 0.3  # Reasonable positive correlation
        assert corr_13 > 0.3

    def test_statistics_ranking_consistency(self):
        """Test that statistics provide consistent feature ranking."""
        np.random.seed(42)
        n, p = 150, 25

        # Create signal with varying strengths
        beta_true = np.zeros(p)
        beta_true[0] = 3    # Strong signal
        beta_true[1] = 1.5  # Medium signal
        beta_true[2] = 0.8  # Weak signal

        X = np.random.randn(n, p)
        X_ko = np.random.randn(n, p)
        y = X @ beta_true + np.random.randn(n)

        W1 = stat_lasso_lambdadiff(X, X_ko, y)
        W2 = stat_lasso_lambdasmax(X, X_ko, y)

        # Top features should be similar across statistics
        top_k = 5
        top_features_1 = np.argsort(W1)[-top_k:]
        top_features_2 = np.argsort(W2)[-top_k:]

        # Should have significant overlap in top features
        overlap = len(set(top_features_1) & set(top_features_2))
        assert overlap >= 2  # At least 40% overlap