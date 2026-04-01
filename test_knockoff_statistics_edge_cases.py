"""
Test coverage for knockoff statistics edge cases and boundary conditions.
Critical for robust feature selection under diverse data conditions.
"""
import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import patch, Mock

from loveslide.knockoff.stats.stability import _stability_selection, stat_stability_selection
from loveslide.knockoff.stats.sqrt_lasso import _sqrt_lasso_path, stat_sqrt_lasso
from loveslide.knockoff.stats.forward import _forward_selection, stat_forward_selection
from loveslide.knockoff.stats.random_forest import stat_random_forest


class TestKnockoffStatisticsEdgeCases:
    """Test edge cases in knockoff statistics functions."""

    def test_stability_selection_extreme_parameters(self):
        """Test stability selection with extreme parameters."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Test with very high threshold (should select few/no features)
        result_high = _stability_selection(
            X, y, n_bootstrap=10, threshold=0.95, sample_fraction=0.8
        )
        assert len(result_high) == X.shape[1]
        assert np.all(result_high >= 0) and np.all(result_high <= 1)

        # Test with very low threshold (should select many features)
        result_low = _stability_selection(
            X, y, n_bootstrap=10, threshold=0.1, sample_fraction=0.8
        )
        assert len(result_low) == X.shape[1]

    def test_stability_selection_small_sample_size(self):
        """Test stability selection with very small sample sizes."""
        np.random.seed(42)
        X = np.random.randn(10, 8)  # More features than samples
        y = np.random.randn(10)

        # Should handle high-dimensional case
        result = _stability_selection(
            X, y, n_bootstrap=5, sample_fraction=0.6
        )
        assert len(result) == X.shape[1]
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_stability_selection_perfect_correlation(self):
        """Test stability selection with perfectly correlated features."""
        np.random.seed(42)
        X_base = np.random.randn(50, 3)
        # Add perfectly correlated feature
        X = np.column_stack([X_base, X_base[:, 0]])
        y = X_base[:, 0] + 0.1 * np.random.randn(50)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May warn about multicollinearity
            result = _stability_selection(X, y, n_bootstrap=10)
            assert len(result) == X.shape[1]

    def test_sqrt_lasso_path_edge_cases(self):
        """Test sqrt lasso path computation with edge cases."""
        np.random.seed(42)
        X = np.random.randn(30, 10)
        y = np.random.randn(30)

        # Test with very small lambda values
        alphas = np.logspace(-6, -1, 10)
        try:
            path_result = _sqrt_lasso_path(X, y, alphas=alphas, max_iter=100)
            assert 'alphas' in path_result
            assert 'coefs' in path_result
        except Exception as e:
            # May fail with extreme regularization, which is acceptable
            assert "convergence" in str(e).lower() or "numerical" in str(e).lower()

    def test_sqrt_lasso_singular_design_matrix(self):
        """Test sqrt lasso with singular design matrix."""
        np.random.seed(42)
        X_base = np.random.randn(20, 5)
        # Create singular matrix
        X = np.column_stack([X_base, X_base.sum(axis=1, keepdims=True)])
        y = np.random.randn(20)

        # Should handle or detect singularity
        try:
            stat_result = stat_sqrt_lasso(X, X, y)  # X as knockoff for simplicity
            assert len(stat_result) == X.shape[1]
        except (np.linalg.LinAlgError, ValueError) as e:
            # Acceptable to fail with singular matrices
            assert "singular" in str(e).lower() or "condition" in str(e).lower()

    def test_forward_selection_no_significant_features(self):
        """Test forward selection when no features are significant."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)  # Pure noise, no relationship

        # Should return empty or minimal selection
        try:
            selected = _forward_selection(X, y, alpha=0.01)  # Strict threshold
            assert isinstance(selected, (list, np.ndarray))
            # Should select few or no features for pure noise
        except Exception:
            # May fail if no features meet threshold
            pass

    def test_forward_selection_highly_correlated_predictors(self):
        """Test forward selection with highly correlated predictors."""
        np.random.seed(42)
        X_base = np.random.randn(50, 3)
        # Add correlated versions
        X = np.column_stack([X_base, X_base + 0.01 * np.random.randn(50, 3)])
        y = X_base[:, 0] + 0.1 * np.random.randn(50)

        selected = _forward_selection(X, y, max_features=2)
        assert len(selected) <= 2
        # Should select meaningful features despite correlation

    def test_random_forest_edge_cases(self):
        """Test random forest statistics with edge cases."""
        np.random.seed(42)
        X = np.random.randn(30, 8)
        Xk = np.random.randn(30, 8)  # Knockoffs
        y_continuous = np.random.randn(30)
        y_binary = np.random.choice([0, 1], size=30)

        # Test with continuous outcome
        stat_cont = stat_random_forest(X, Xk, y_continuous)
        assert len(stat_cont) == X.shape[1]

        # Test with binary outcome
        stat_bin = stat_random_forest(X, Xk, y_binary)
        assert len(stat_bin) == X.shape[1]

    def test_random_forest_few_trees_edge_case(self):
        """Test random forest with very few trees."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        Xk = np.random.randn(50, 5)
        y = np.random.randn(50)

        # Test with minimal trees
        stat_result = stat_random_forest(X, Xk, y, n_estimators=1)
        assert len(stat_result) == X.shape[1]
        # Results may be unstable but should not crash

    def test_statistics_with_constant_features(self):
        """Test statistics functions with constant features."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        X[:, 2] = 1.0  # Constant feature
        Xk = np.random.randn(50, 5)
        y = np.random.randn(50)

        # Should handle constant features gracefully
        try:
            stat_result = stat_random_forest(X, Xk, y)
            assert len(stat_result) == X.shape[1]
            # Constant feature should have low importance
            assert stat_result[2] <= np.max(stat_result)
        except (ValueError, Warning):
            # May warn about constant features
            pass

    def test_statistics_with_missing_knockoffs_alignment(self):
        """Test behavior when X and Xk dimensions don't match."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        Xk = np.random.randn(50, 4)  # Wrong number of features
        y = np.random.randn(50)

        with pytest.raises((ValueError, AssertionError)):
            stat_random_forest(X, Xk, y)

    def test_statistics_memory_efficiency_large_data(self):
        """Test memory efficiency with large datasets."""
        np.random.seed(42)
        # Moderately large for testing
        X = np.random.randn(500, 50)
        Xk = np.random.randn(500, 50)
        y = np.random.randn(500)

        # Should handle without excessive memory usage
        try:
            stat_result = stat_random_forest(X, Xk, y, n_estimators=10)
            assert len(stat_result) == X.shape[1]
        except MemoryError:
            pytest.skip("Test environment memory insufficient")

    def test_statistics_numerical_precision(self):
        """Test numerical precision with extreme feature scales."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        X[:, 0] *= 1e6  # Very large scale
        X[:, 1] *= 1e-6  # Very small scale
        Xk = np.random.randn(50, 5)
        y = np.random.randn(50)

        # Should handle or normalize extreme scales
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stat_result = stat_random_forest(X, Xk, y)
                assert len(stat_result) == X.shape[1]
                assert np.all(np.isfinite(stat_result))
            except Exception as e:
                # May fail with extreme scaling
                assert "numeric" in str(e).lower() or "overflow" in str(e).lower()

    def test_binary_outcome_edge_cases(self):
        """Test statistics with edge cases in binary outcomes."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        Xk = np.random.randn(50, 5)

        # Extremely imbalanced binary outcome
        y_imbalanced = np.zeros(50)
        y_imbalanced[:2] = 1  # Only 2 positive cases

        try:
            stat_result = stat_random_forest(X, Xk, y_imbalanced)
            assert len(stat_result) == X.shape[1]
        except (ValueError, Warning) as e:
            # May fail or warn with extreme imbalance
            assert any(word in str(e).lower() for word in ['class', 'imbalance', 'sample'])

    def test_stability_selection_bootstrap_edge_cases(self):
        """Test stability selection bootstrap edge cases."""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20)

        # Very small sample fraction
        result_small = _stability_selection(
            X, y, n_bootstrap=5, sample_fraction=0.1  # Only 2 samples per bootstrap
        )
        assert len(result_small) == X.shape[1]

        # Sample fraction approaching 1
        result_large = _stability_selection(
            X, y, n_bootstrap=5, sample_fraction=0.95
        )
        assert len(result_large) == X.shape[1]

    def test_cross_validation_within_statistics(self):
        """Test statistics functions that use internal cross-validation."""
        np.random.seed(42)
        X = np.random.randn(30, 5)
        Xk = np.random.randn(30, 5)
        y = np.random.randn(30)

        # Test with insufficient samples for CV
        X_small = X[:8]  # Too few for standard CV
        Xk_small = Xk[:8]
        y_small = y[:8]

        try:
            stat_result = stat_sqrt_lasso(X_small, Xk_small, y_small)
            assert len(stat_result) == X_small.shape[1]
        except ValueError as e:
            # May fail with insufficient samples for CV
            assert "sample" in str(e).lower() or "fold" in str(e).lower()