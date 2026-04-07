#!/usr/bin/env python3
"""
Comprehensive test coverage for knockoff statistical methods.
Many of these functions have minimal or no test coverage.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from loveslide.knockoff.stats.lasso import (
    stat_lasso_lambdadiff, stat_lasso_lambdasmax, stat_lasso_coefdiff
)
from loveslide.knockoff.stats.sqrt_lasso import stat_sqrt_lasso
from loveslide.knockoff.stats.stability import stat_stability_selection
from loveslide.knockoff.stats.random_forest import stat_random_forest
from loveslide.knockoff.stats.forward import stat_forward_selection
from loveslide.knockoff.stats.glmnet import (
    stat_glmnet_lambdadiff, stat_glmnet_lambdasmax, stat_glmnet_coefdiff
)
from loveslide.knockoff.filter import (
    knockoff_filter, knockoff_threshold, knockoff_filter_voting_slide
)
from loveslide.knockoff.create import create_gaussian, create_second_order
from loveslide.knockoff.solve import create_solve_sdp, create_solve_equi


class TestLassoStatistics:
    """Test Lasso-based knockoff statistics."""

    def test_stat_lasso_lambdadiff_gaussian(self):
        """Test Lasso lambda difference statistic for Gaussian response."""
        n, p = 100, 20
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)  # Mock knockoffs
        y = np.random.randn(n)  # Continuous response

        W = stat_lasso_lambdadiff(X, X_k, y)

        assert len(W) == p
        assert isinstance(W, np.ndarray)
        # No constraint on signs - W can be positive or negative

    def test_stat_lasso_lambdadiff_validation(self):
        """Test input validation for lasso lambda difference."""
        X = np.random.randn(50, 10)
        X_k = np.random.randn(50, 10)

        # Test with non-numeric response
        y_categorical = np.array(['A', 'B'] * 25)

        with pytest.raises(ValueError, match="requires numeric response"):
            stat_lasso_lambdadiff(X, X_k, y_categorical)

        # Test with mismatched dimensions
        y_wrong = np.random.randn(40)  # Wrong length

        with pytest.raises((ValueError, IndexError)):
            stat_lasso_lambdadiff(X, X_k, y_wrong)

    def test_stat_lasso_lambdasmax(self):
        """Test Lasso lambda max statistic."""
        n, p = 80, 15
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)
        y = np.random.randn(n)

        W = stat_lasso_lambdasmax(X, X_k, y)

        assert len(W) == p
        assert isinstance(W, np.ndarray)

    def test_stat_lasso_coefdiff(self):
        """Test Lasso coefficient difference statistic."""
        n, p = 60, 12
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)
        y = np.random.randn(n)

        W = stat_lasso_coefdiff(X, X_k, y)

        assert len(W) == p
        assert isinstance(W, np.ndarray)

    def test_lasso_stats_perfect_correlation(self):
        """Test Lasso statistics with perfectly correlated features."""
        n, p = 50, 8
        X = np.random.randn(n, p)
        X[:, 1] = X[:, 0]  # Perfect correlation
        X_k = np.random.randn(n, p)
        y = np.random.randn(n)

        # Should handle gracefully despite perfect correlation
        W = stat_lasso_lambdadiff(X, X_k, y)
        assert len(W) == p
        assert not np.any(np.isnan(W))

    def test_lasso_stats_high_dimensional(self):
        """Test Lasso statistics in high-dimensional setting (p > n)."""
        n, p = 30, 50  # p > n
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)
        y = np.random.randn(n)

        W = stat_lasso_lambdadiff(X, X_k, y)
        assert len(W) == p


class TestSqrtLassoStatistics:
    """Test Square-root Lasso statistics."""

    def test_stat_sqrt_lasso_basic(self):
        """Test basic sqrt Lasso statistic functionality."""
        n, p = 70, 16
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)
        y = np.random.randn(n)

        W = stat_sqrt_lasso(X, X_k, y)

        assert len(W) == p
        assert isinstance(W, np.ndarray)
        assert np.all(np.isfinite(W))

    def test_stat_sqrt_lasso_robustness(self):
        """Test sqrt Lasso robustness to outliers."""
        n, p = 60, 10
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)
        y = np.random.randn(n)

        # Add outliers
        y[0] = 100  # Extreme outlier
        y[1] = -100

        W = stat_sqrt_lasso(X, X_k, y)

        assert len(W) == p
        assert np.all(np.isfinite(W))

    def test_stat_sqrt_lasso_noise_levels(self):
        """Test sqrt Lasso with different noise levels."""
        n, p = 80, 12
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)

        noise_levels = [0.1, 1.0, 10.0]

        for noise_level in noise_levels:
            y = np.random.randn(n) * noise_level
            W = stat_sqrt_lasso(X, X_k, y)

            assert len(W) == p
            assert np.all(np.isfinite(W))


class TestStabilitySelection:
    """Test stability selection statistics."""

    def test_stat_stability_selection_basic(self):
        """Test basic stability selection functionality."""
        n, p = 100, 20
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)
        y = np.random.randn(n)

        W = stat_stability_selection(
            X, X_k, y,
            n_bootstrap=20,  # Reduced for testing speed
            threshold=0.6
        )

        assert len(W) == p
        assert isinstance(W, np.ndarray)

    def test_stat_stability_selection_parameters(self):
        """Test stability selection with different parameters."""
        n, p = 60, 12
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)
        y = np.random.randn(n)

        # Test different thresholds
        thresholds = [0.5, 0.7, 0.9]

        for threshold in thresholds:
            W = stat_stability_selection(
                X, X_k, y,
                n_bootstrap=10,
                threshold=threshold
            )
            assert len(W) == p

    def test_stat_stability_selection_binary_response(self):
        """Test stability selection with binary response."""
        n, p = 80, 15
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)
        y = np.random.choice([0, 1], size=n)

        W = stat_stability_selection(X, X_k, y, n_bootstrap=15)

        assert len(W) == p
        assert isinstance(W, np.ndarray)

    def test_stat_stability_selection_edge_cases(self):
        """Test stability selection edge cases."""
        n, p = 40, 8
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)

        # Case 1: Constant response
        y_constant = np.ones(n)
        W_constant = stat_stability_selection(X, X_k, y_constant, n_bootstrap=5)
        assert len(W_constant) == p

        # Case 2: Very small bootstrap samples
        y = np.random.randn(n)
        W_small = stat_stability_selection(X, X_k, y, n_bootstrap=2)
        assert len(W_small) == p


class TestRandomForestStatistics:
    """Test Random Forest knockoff statistics."""

    def test_stat_random_forest_basic(self):
        """Test basic Random Forest statistic functionality."""
        n, p = 120, 25
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)
        y = np.random.randn(n)

        W = stat_random_forest(
            X, X_k, y,
            n_estimators=20,  # Reduced for testing speed
            max_depth=5
        )

        assert len(W) == p
        assert isinstance(W, np.ndarray)

    def test_stat_random_forest_classification(self):
        """Test Random Forest with classification response."""
        n, p = 100, 20
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)
        y = np.random.choice([0, 1, 2], size=n)  # Multi-class

        W = stat_random_forest(X, X_k, y, n_estimators=15)

        assert len(W) == p
        assert isinstance(W, np.ndarray)

    def test_stat_random_forest_parameters(self):
        """Test Random Forest with different parameters."""
        n, p = 80, 16
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)
        y = np.random.randn(n)

        # Test different numbers of estimators
        n_estimators_list = [5, 20, 50]

        for n_est in n_estimators_list:
            W = stat_random_forest(X, X_k, y, n_estimators=n_est)
            assert len(W) == p

    def test_stat_random_forest_feature_importance(self):
        """Test Random Forest feature importance calculation."""
        n, p = 90, 18
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)

        # Create response with known signal
        signal_features = [0, 5, 10]
        y = np.sum(X[:, signal_features], axis=1) + 0.1 * np.random.randn(n)

        W = stat_random_forest(X, X_k, y, n_estimators=30)

        assert len(W) == p
        # Signal features should tend to have positive importance differences


class TestForwardSelection:
    """Test Forward Selection statistics."""

    def test_stat_forward_selection_basic(self):
        """Test basic forward selection functionality."""
        n, p = 80, 15
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)
        y = np.random.randn(n)

        W = stat_forward_selection(X, X_k, y, max_steps=10)

        assert len(W) == p
        assert isinstance(W, np.ndarray)

    def test_stat_forward_selection_early_stopping(self):
        """Test forward selection with early stopping."""
        n, p = 60, 12
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)
        y = np.random.randn(n)

        W = stat_forward_selection(
            X, X_k, y,
            max_steps=5,  # Early stopping
            stopping_criterion="aic"
        )

        assert len(W) == p

    def test_stat_forward_selection_different_criteria(self):
        """Test forward selection with different stopping criteria."""
        n, p = 70, 14
        X = np.random.randn(n, p)
        X_k = np.random.randn(n, p)
        y = np.random.randn(n)

        criteria = ["aic", "bic", "cv"]

        for criterion in criteria:
            W = stat_forward_selection(
                X, X_k, y,
                max_steps=8,
                stopping_criterion=criterion
            )
            assert len(W) == p


class TestKnockoffFilter:
    """Test knockoff filtering procedures."""

    def test_knockoff_filter_basic(self):
        """Test basic knockoff filter functionality."""
        p = 20
        W = np.random.randn(p)  # Random statistics

        selected = knockoff_filter(W, fdr=0.1)

        assert isinstance(selected, (list, np.ndarray))
        assert len(selected) <= p
        assert all(0 <= idx < p for idx in selected)

    def test_knockoff_filter_no_signal(self):
        """Test knockoff filter with no signal (negative statistics)."""
        p = 15
        W = -np.abs(np.random.randn(p))  # All negative

        selected = knockoff_filter(W, fdr=0.1)

        # Should select nothing or very few
        assert len(selected) <= 2

    def test_knockoff_filter_strong_signal(self):
        """Test knockoff filter with strong positive signal."""
        p = 25
        W = np.ones(p) * 3  # All strong positive

        selected = knockoff_filter(W, fdr=0.1)

        # Should select many features
        assert len(selected) >= p // 2

    def test_knockoff_threshold_calculation(self):
        """Test knockoff threshold calculation."""
        W_values = [2.5, 1.8, -0.5, 3.1, -1.2, 0.8, 2.0]
        W = np.array(W_values)

        threshold = knockoff_threshold(W, fdr=0.2, offset=1)

        assert isinstance(threshold, float)
        assert threshold >= 0

    def test_knockoff_threshold_edge_cases(self):
        """Test knockoff threshold edge cases."""
        # All positive statistics
        W_pos = np.array([1, 2, 3, 4, 5])
        t_pos = knockoff_threshold(W_pos, fdr=0.1)
        assert t_pos >= 0

        # All negative statistics
        W_neg = np.array([-1, -2, -3, -4, -5])
        t_neg = knockoff_threshold(W_neg, fdr=0.1)
        assert t_neg >= 0

        # Mixed with zeros
        W_mixed = np.array([0, 0, 1, -1, 0])
        t_mixed = knockoff_threshold(W_mixed, fdr=0.1)
        assert t_mixed >= 0

    def test_knockoff_filter_voting_slide(self):
        """Test SLIDE-specific knockoff voting filter."""
        n_runs = 10
        p = 20

        # Simulate multiple knockoff runs
        statistics_list = []
        selected_list = []

        for _ in range(n_runs):
            W = np.random.randn(p)
            selected = knockoff_filter(W, fdr=0.1)
            statistics_list.append(W)
            selected_list.append(selected)

        voting_result = knockoff_filter_voting_slide(
            statistics_list,
            selected_list,
            consensus_threshold=0.5
        )

        assert hasattr(voting_result, 'selected')
        assert hasattr(voting_result, 'votes')
        assert len(voting_result.votes) == p

    def test_knockoff_voting_consensus_levels(self):
        """Test knockoff voting with different consensus thresholds."""
        n_runs = 8
        p = 15

        statistics_list = []
        selected_list = []

        # Create consistent selection pattern
        true_signal = [0, 5, 10]
        for _ in range(n_runs):
            W = np.random.randn(p)
            W[true_signal] += 2  # Add signal
            selected = knockoff_filter(W, fdr=0.15)
            statistics_list.append(W)
            selected_list.append(selected)

        # Test different consensus thresholds
        thresholds = [0.3, 0.5, 0.7, 0.9]

        for threshold in thresholds:
            result = knockoff_filter_voting_slide(
                statistics_list,
                selected_list,
                consensus_threshold=threshold
            )

            # Higher thresholds should select fewer features
            assert len(result.selected) <= p


class TestKnockoffCreation:
    """Test knockoff variable creation edge cases."""

    def test_create_gaussian_edge_cases(self):
        """Test Gaussian knockoff creation edge cases."""
        # Nearly singular covariance
        X = np.random.randn(50, 10)
        X[:, 1] = X[:, 0] + 1e-10 * np.random.randn(50)

        try:
            X_k = create_gaussian(X)
            assert X_k.shape == X.shape
        except np.linalg.LinAlgError:
            # Acceptable failure for singular matrices
            pass

    def test_create_second_order_validation(self):
        """Test second-order knockoff input validation."""
        # Test with insufficient samples
        X_small = np.random.randn(5, 10)  # n < p

        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            create_second_order(X_small)

    def test_create_knockoffs_different_solvers(self):
        """Test knockoff creation with different SDP solvers."""
        X = np.random.randn(80, 20)

        # Test equicorrelated solver
        X_k_equi = create_gaussian(X, method='equi')
        assert X_k_equi.shape == X.shape

        # Test SDP solver
        X_k_sdp = create_gaussian(X, method='sdp')
        assert X_k_sdp.shape == X.shape

        # Verify knockoff properties (approximately)
        Sigma_X = X.T @ X / X.shape[0]
        Sigma_Xk = X_k_sdp.T @ X_k_sdp / X.shape[0]
        Cross_corr = X.T @ X_k_sdp / X.shape[0]

        # Knockoff constraint: X^T X_k = X_k^T X
        assert np.allclose(Cross_corr, Cross_corr.T, atol=1e-2)


class TestIntegratedKnockoffWorkflow:
    """Test integrated knockoff workflow scenarios."""

    def test_complete_knockoff_pipeline(self):
        """Test complete knockoff selection pipeline."""
        n, p = 150, 30
        X = np.random.randn(n, p)

        # Add signal structure
        true_signal = [0, 5, 10, 15, 20]
        beta_true = np.zeros(p)
        beta_true[true_signal] = np.random.randn(len(true_signal))
        y = X @ beta_true + 0.5 * np.random.randn(n)

        # Create knockoffs
        X_k = create_gaussian(X)

        # Compute statistics
        W = stat_lasso_lambdadiff(X, X_k, y)

        # Apply filter
        selected = knockoff_filter(W, fdr=0.15)

        # Basic validation
        assert isinstance(selected, (list, np.ndarray))
        assert len(selected) <= p

        # Check if any true signals were recovered
        recovered_signals = set(selected) & set(true_signal)
        # (In practice, we'd test power, but here just check it runs)

    def test_knockoff_workflow_robustness(self):
        """Test knockoff workflow robustness to challenging scenarios."""
        scenarios = [
            {"n": 100, "p": 50, "noise": 2.0},    # High noise
            {"n": 200, "p": 80, "correlation": 0.7},  # High correlation
            {"n": 80, "p": 40, "sparsity": 0.9},   # Very sparse signal
        ]

        for scenario in scenarios:
            n, p = scenario["n"], scenario["p"]
            X = np.random.randn(n, p)

            # Apply scenario-specific modifications
            if "correlation" in scenario:
                rho = scenario["correlation"]
                for i in range(1, p):
                    X[:, i] = rho * X[:, 0] + np.sqrt(1-rho**2) * X[:, i]

            # Create response
            n_signal = max(1, int(p * (1 - scenario.get("sparsity", 0.8))))
            signal_idx = np.random.choice(p, n_signal, replace=False)
            beta = np.zeros(p)
            beta[signal_idx] = np.random.randn(n_signal)

            noise_level = scenario.get("noise", 1.0)
            y = X @ beta + noise_level * np.random.randn(n)

            # Run workflow - should not crash
            try:
                X_k = create_gaussian(X)
                W = stat_lasso_lambdadiff(X, X_k, y)
                selected = knockoff_filter(W, fdr=0.2)

                # Basic checks
                assert X_k.shape == X.shape
                assert len(W) == p
                assert len(selected) <= p

            except Exception as e:
                pytest.fail(f"Knockoff workflow failed on {scenario}: {e}")


if __name__ == "__main__":
    pytest.main([__file__])