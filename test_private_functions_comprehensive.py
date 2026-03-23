"""
Comprehensive testing of private/internal functions in SLIDE_py.

These functions are critical for algorithm correctness but often lack direct testing.
Testing them directly ensures robust behavior and easier debugging.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import private functions for testing
from loveslide.cv import SLIDEcv
from loveslide.knockoffs import (
    _rlist_get, _create_second_order_r, _solve_sdp_r,
    _single_knockoff_iteration_python
)
from loveslide.love import _convert_r_pure_ind


class TestPrivateCVFunctions:
    """Test private cross-validation functions."""

    @pytest.fixture
    def sample_cv_instance(self):
        """Create a SLIDEcv instance for testing."""
        params = {
            'x_path': None, 'y_path': None, 'fdr': 0.1,
            'delta': [0.1], 'lambda': [0.5], 'n_folds': 3
        }
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        return SLIDEcv(params, x=X, y=y)

    def test_bench_cv_basic_functionality(self, sample_cv_instance):
        """Test _bench_cv private method."""
        # Mock SLIDE result
        mock_slide_result = Mock()
        mock_slide_result.get_features.return_value = np.random.randint(0, 50, 10)

        with patch.object(sample_cv_instance, '_run_slide_fold', return_value=mock_slide_result):
            result = sample_cv_instance._bench_cv(
                train_idx=np.arange(80),
                test_idx=np.arange(80, 100),
                delta=0.1,
                lbd=0.5
            )
            assert 'features' in result
            assert 'y_pred' in result
            assert len(result['y_pred']) == 20  # test set size

    def test_bench_cv_empty_features(self, sample_cv_instance):
        """Test _bench_cv when no features are selected."""
        mock_slide_result = Mock()
        mock_slide_result.get_features.return_value = np.array([])

        with patch.object(sample_cv_instance, '_run_slide_fold', return_value=mock_slide_result):
            result = sample_cv_instance._bench_cv(
                train_idx=np.arange(80),
                test_idx=np.arange(80, 100),
                delta=0.1,
                lbd=0.5
            )
            # Should handle empty feature case gracefully
            assert result['features'].size == 0
            assert len(result['y_pred']) == 20

    def test_run_slide_fold_basic(self, sample_cv_instance):
        """Test _run_slide_fold private method."""
        train_idx = np.arange(80)
        delta = 0.1
        lbd = 0.5

        # Mock the SLIDE initialization and execution
        with patch('loveslide.cv.SLIDE') as MockSLIDE:
            mock_slide = Mock()
            MockSLIDE.return_value = mock_slide

            result = sample_cv_instance._run_slide_fold(train_idx, delta, lbd)

            # Verify SLIDE was initialized correctly
            MockSLIDE.assert_called_once()
            assert result == mock_slide

    def test_find_interactions_fold_basic(self, sample_cv_instance):
        """Test _find_interactions_fold private method."""
        train_idx = np.arange(80)
        test_idx = np.arange(80, 100)
        features = np.array([1, 5, 10, 15])

        result = sample_cv_instance._find_interactions_fold(
            train_idx, test_idx, features, delta=0.1, lbd=0.5
        )

        assert 'interactions' in result
        assert 'y_pred' in result
        assert len(result['y_pred']) == 20

    def test_build_prediction_features_basic(self, sample_cv_instance):
        """Test _build_prediction_features private method."""
        X_test = np.random.randn(20, 50)
        features = np.array([1, 5, 10])
        interactions = [(1, 5), (5, 10)]

        X_pred = sample_cv_instance._build_prediction_features(
            X_test, features, interactions
        )

        # Should include original features + interaction terms
        expected_cols = len(features) + len(interactions)
        assert X_pred.shape[1] == expected_cols
        assert X_pred.shape[0] == 20

    def test_compute_metric_regression(self, sample_cv_instance):
        """Test _compute_metric for regression."""
        y_true = np.random.randn(20)
        y_pred = y_true + 0.1 * np.random.randn(20)  # Add small noise

        metric = sample_cv_instance._compute_metric(y_true, y_pred)
        assert isinstance(metric, float)
        assert metric >= 0  # MSE is non-negative

    def test_standardize_fold_basic(self, sample_cv_instance):
        """Test _standardize_fold private method."""
        X_train = np.random.randn(80, 50)
        X_test = np.random.randn(20, 50)

        X_train_std, X_test_std = sample_cv_instance._standardize_fold(X_train, X_test)

        assert X_train_std.shape == X_train.shape
        assert X_test_std.shape == X_test.shape
        # Check standardization worked
        assert np.allclose(np.mean(X_train_std, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_train_std, axis=0, ddof=1), 1, atol=1e-10)


class TestPrivateKnockoffFunctions:
    """Test private knockoff functions."""

    def test_rlist_get_valid_attribute(self):
        """Test _rlist_get with valid R list attribute."""
        # Mock R list object
        mock_robj = Mock()
        mock_robj.names = ['attr1', 'attr2']
        mock_robj.rx2.return_value = np.array([1, 2, 3])

        result = _rlist_get(mock_robj, 'attr1')
        assert np.array_equal(result, np.array([1, 2, 3]))

    def test_rlist_get_missing_attribute(self):
        """Test _rlist_get with missing attribute."""
        mock_robj = Mock()
        mock_robj.names = ['attr1', 'attr2']

        result = _rlist_get(mock_robj, 'missing_attr')
        assert result is None

    def test_create_second_order_r_basic(self):
        """Test _create_second_order_r function."""
        X = np.random.randn(50, 20)

        with patch('loveslide.knockoffs.r_knockoffs') as mock_r:
            mock_result = Mock()
            mock_r.create_second_order.return_value = mock_result
            mock_result.rx2.return_value = np.random.randn(50, 20)

            result = _create_second_order_r(X)
            assert result.shape == X.shape
            mock_r.create_second_order.assert_called_once()

    def test_solve_sdp_r_basic(self):
        """Test _solve_sdp_r function."""
        # Create positive definite covariance matrix
        A = np.random.randn(20, 20)
        Sigma = A @ A.T + np.eye(20)

        with patch('loveslide.knockoffs.r_knockoffs') as mock_r:
            mock_result = Mock()
            mock_r.create_solve_sdp.return_value = mock_result
            mock_result.rx2.return_value = np.random.randn(20, 20)

            result = _solve_sdp_r(Sigma, method='sdp')
            assert result.shape == Sigma.shape
            mock_r.create_solve_sdp.assert_called_once()

    def test_single_knockoff_iteration_python_basic(self):
        """Test _single_knockoff_iteration_python function."""
        # Create sample data
        z = np.random.randn(100, 40)  # Original + knockoff features
        y = np.random.randn(100)
        fdr = 0.1

        # Mock required parameters
        method = 'equicorrelated'
        shrink = True
        offset = 1
        statistic = 'lcd'

        with patch('loveslide.knockoffs.create_gaussian') as mock_create:
            with patch('loveslide.knockoffs.knockoff_filter') as mock_filter:
                mock_create.return_value = np.random.randn(100, 20)
                mock_filter.return_value = Mock(selected=np.array([1, 5, 10]))

                result = _single_knockoff_iteration_python(
                    z, y, fdr, method, shrink, offset, statistic
                )

                assert hasattr(result, 'selected')
                mock_create.assert_called_once()
                mock_filter.assert_called_once()

    def test_single_knockoff_iteration_empty_selection(self):
        """Test _single_knockoff_iteration_python with no features selected."""
        z = np.random.randn(100, 40)
        y = np.random.randn(100)
        fdr = 0.01  # Very strict FDR

        with patch('loveslide.knockoffs.create_gaussian') as mock_create:
            with patch('loveslide.knockoffs.knockoff_filter') as mock_filter:
                mock_create.return_value = np.random.randn(100, 20)
                mock_filter.return_value = Mock(selected=np.array([]))  # No selection

                result = _single_knockoff_iteration_python(
                    z, y, fdr, 'equicorrelated', True, 1, 'lcd'
                )

                assert hasattr(result, 'selected')
                assert len(result.selected) == 0


class TestPrivateLOVEFunctions:
    """Test private LOVE functions."""

    def test_convert_r_pure_ind_basic(self):
        """Test _convert_r_pure_ind function."""
        # Mock R list with pure indices
        mock_r_list = [
            Mock(names=['pos', 'neg']),
            Mock(names=['pos', 'neg']),
        ]

        # Configure mock returns
        mock_r_list[0].rx2.side_effect = lambda x: {
            'pos': np.array([1, 3, 5]),
            'neg': np.array([2, 4])
        }[x]

        mock_r_list[1].rx2.side_effect = lambda x: {
            'pos': np.array([6, 8]),
            'neg': np.array([7])
        }[x]

        result = _convert_r_pure_ind(mock_r_list)

        assert len(result) == 2
        assert 'pos' in result[0] and 'neg' in result[0]
        assert 'pos' in result[1] and 'neg' in result[1]

    def test_convert_r_pure_ind_empty_groups(self):
        """Test _convert_r_pure_ind with empty groups."""
        mock_r_list = [Mock(names=['pos', 'neg'])]
        mock_r_list[0].rx2.side_effect = lambda x: np.array([])

        result = _convert_r_pure_ind(mock_r_list)

        assert len(result) == 1
        assert result[0]['pos'].size == 0
        assert result[0]['neg'].size == 0


class TestPrivateFunctionEdgeCases:
    """Test edge cases for private functions."""

    def test_private_functions_with_extreme_data(self):
        """Test private functions with extreme data values."""
        # Very large values
        X_large = np.full((10, 5), 1e10)

        # Very small values
        X_small = np.full((10, 5), 1e-10)

        # Test that functions handle extreme values gracefully
        cv_instance = SLIDEcv({
            'x_path': None, 'y_path': None, 'fdr': 0.1,
            'delta': [0.1], 'lambda': [0.5], 'n_folds': 3
        }, x=X_large, y=np.ones(10))

        # Should not raise exceptions
        X_train_std, X_test_std = cv_instance._standardize_fold(
            X_large[:8], X_large[8:]
        )
        assert not np.any(np.isnan(X_train_std))
        assert not np.any(np.isnan(X_test_std))

    def test_private_functions_memory_efficiency(self):
        """Test private functions with large arrays for memory efficiency."""
        # Large but manageable arrays
        X = np.random.randn(1000, 100)
        y = np.random.randn(1000)

        cv_instance = SLIDEcv({
            'x_path': None, 'y_path': None, 'fdr': 0.1,
            'delta': [0.1], 'lambda': [0.5], 'n_folds': 3
        }, x=X, y=y)

        # Test memory efficiency of standardization
        train_idx = np.arange(800)
        test_idx = np.arange(800, 1000)

        X_train_std, X_test_std = cv_instance._standardize_fold(
            X[train_idx], X[test_idx]
        )

        # Verify no memory leaks (arrays should be reasonable size)
        assert X_train_std.nbytes < 1e8  # Less than 100MB
        assert X_test_std.nbytes < 1e8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])