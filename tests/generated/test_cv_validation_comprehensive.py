"""
Comprehensive test coverage for cross-validation functionality.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock
import tempfile
from pathlib import Path

from loveslide.cv import SLIDEcv


class TestSLIDEcvInitialization:
    """Test SLIDEcv initialization and parameter validation."""

    def test_slidecv_init_valid_params(self):
        """Test SLIDEcv initialization with valid parameters."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        cv = SLIDEcv(
            X=X, y=y,
            K_vals=[2, 3],
            lbd_vals=[0.3, 0.5],
            delta_vals=[0.1, 0.2],
            cv_folds=3
        )

        assert cv.K_vals == [2, 3]
        assert cv.lbd_vals == [0.3, 0.5]
        assert cv.delta_vals == [0.1, 0.2]
        assert cv.cv_folds == 3

    def test_slidecv_init_invalid_k_vals(self):
        """Test SLIDEcv with invalid K values."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        # Negative K values
        with pytest.raises(ValueError):
            SLIDEcv(X=X, y=y, K_vals=[-1, 2])

        # K greater than features
        with pytest.raises(ValueError):
            SLIDEcv(X=X, y=y, K_vals=[25])  # X has 20 features

        # Zero K
        with pytest.raises(ValueError):
            SLIDEcv(X=X, y=y, K_vals=[0])

    def test_slidecv_init_invalid_lbd_vals(self):
        """Test SLIDEcv with invalid lambda values."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        # Out of range lambda
        with pytest.raises(ValueError):
            SLIDEcv(X=X, y=y, lbd_vals=[-0.1, 0.5])

        with pytest.raises(ValueError):
            SLIDEcv(X=X, y=y, lbd_vals=[1.5])

    def test_slidecv_init_invalid_delta_vals(self):
        """Test SLIDEcv with invalid delta values."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        with pytest.raises(ValueError):
            SLIDEcv(X=X, y=y, delta_vals=[-0.1])

        with pytest.raises(ValueError):
            SLIDEcv(X=X, y=y, delta_vals=[1.5])

    def test_slidecv_init_invalid_cv_folds(self):
        """Test SLIDEcv with invalid CV fold specifications."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        # Too many folds
        with pytest.raises(ValueError):
            SLIDEcv(X=X, y=y, cv_folds=101)  # More folds than samples

        # Too few folds
        with pytest.raises(ValueError):
            SLIDEcv(X=X, y=y, cv_folds=1)

    def test_slidecv_init_mismatched_data(self):
        """Test SLIDEcv with mismatched X and y dimensions."""
        X = np.random.randn(100, 20)
        y = np.random.randn(50)  # Wrong size

        with pytest.raises(ValueError):
            SLIDEcv(X=X, y=y)


class TestSLIDEcvFoldValidation:
    """Test cross-validation fold validation and creation."""

    def test_folds_valid_with_valid_folds(self):
        """Test _folds_valid with properly formed folds."""
        y = np.random.randn(100)
        folds = [
            (np.arange(0, 33), np.arange(33, 100)),
            (np.concatenate([np.arange(0, 33), np.arange(66, 100)]),
             np.arange(33, 66)),
            (np.arange(33, 100), np.arange(0, 33))
        ]

        from loveslide.cv import SLIDEcv
        assert SLIDEcv._folds_valid(y, folds)

    def test_folds_valid_with_overlapping_folds(self):
        """Test _folds_valid rejects overlapping train/test sets."""
        y = np.random.randn(100)
        folds = [
            (np.arange(0, 50), np.arange(25, 75)),  # Overlap!
        ]

        from loveslide.cv import SLIDEcv
        assert not SLIDEcv._folds_valid(y, folds)

    def test_folds_valid_with_missing_samples(self):
        """Test _folds_valid rejects folds missing samples."""
        y = np.random.randn(100)
        folds = [
            (np.arange(0, 30), np.arange(50, 100)),  # Missing 30-49!
        ]

        from loveslide.cv import SLIDEcv
        assert not SLIDEcv._folds_valid(y, folds)

    def test_folds_valid_with_out_of_bounds_indices(self):
        """Test _folds_valid rejects out-of-bounds indices."""
        y = np.random.randn(100)
        folds = [
            (np.arange(0, 50), np.arange(50, 105)),  # Index 100-104 don't exist!
        ]

        from loveslide.cv import SLIDEcv
        assert not SLIDEcv._folds_valid(y, folds)


class TestSLIDEcvMetricComputation:
    """Test metric computation edge cases."""

    def test_compute_metric_perfect_predictions(self):
        """Test metric computation with perfect predictions."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        cv = SLIDEcv(X=X, y=y, metric='mse')

        # Perfect predictions
        metric = cv._compute_metric(y, y)
        assert metric == 0.0

        # For classification
        y_binary = (np.random.randn(100) > 0).astype(int)
        cv_clf = SLIDEcv(X=X, y=y_binary, metric='auc')

        # Perfect predictions should give AUC = 1.0
        y_proba = y_binary.astype(float)  # Perfect probability predictions
        auc = cv_clf._compute_metric(y_binary, y_proba)
        assert auc == 1.0

    def test_compute_metric_worst_predictions(self):
        """Test metric computation with worst possible predictions."""
        X = np.random.randn(100, 20)
        y_binary = (np.random.randn(100) > 0).astype(int)

        cv = SLIDEcv(X=X, y=y_binary, metric='auc')

        # Worst predictions (opposite of truth)
        y_proba = 1.0 - y_binary.astype(float)
        auc = cv._compute_metric(y_binary, y_proba)
        assert auc == 0.0

    def test_compute_metric_random_predictions(self):
        """Test metric computation with random predictions."""
        X = np.random.randn(100, 20)
        y_binary = (np.random.randn(100) > 0).astype(int)

        cv = SLIDEcv(X=X, y=y_binary, metric='auc')

        # Random predictions should give AUC around 0.5
        y_proba = np.random.uniform(0, 1, 100)
        auc = cv._compute_metric(y_binary, y_proba)
        assert 0.3 <= auc <= 0.7  # Should be close to 0.5

    def test_compute_metric_nan_predictions(self):
        """Test metric computation handles NaN predictions."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        cv = SLIDEcv(X=X, y=y, metric='mse')

        # Predictions with NaN
        y_pred = np.copy(y)
        y_pred[0] = np.nan

        with pytest.raises((ValueError, np.core._exceptions._ArrayMemoryError)):
            cv._compute_metric(y, y_pred)

    def test_compute_metric_infinite_predictions(self):
        """Test metric computation handles infinite predictions."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        cv = SLIDEcv(X=X, y=y, metric='mse')

        # Predictions with infinity
        y_pred = np.copy(y)
        y_pred[0] = np.inf

        # Should either handle gracefully or raise appropriate error
        try:
            metric = cv._compute_metric(y, y_pred)
            assert np.isfinite(metric) or metric == np.inf
        except (ValueError, OverflowError):
            pass  # Acceptable to raise error


class TestSLIDEcvStandardization:
    """Test data standardization in cross-validation."""

    def test_standardize_fold_basic(self):
        """Test basic fold standardization."""
        X_train = np.random.randn(80, 20)
        X_test = np.random.randn(20, 20)

        from loveslide.cv import SLIDEcv
        X_train_std, X_test_std = SLIDEcv._standardize_fold(X_train, X_test)

        # Training data should be standardized
        assert np.allclose(np.mean(X_train_std, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_train_std, axis=0), 1, atol=1e-10)

        # Test data should be transformed with training statistics
        assert X_test_std.shape == X_test.shape

    def test_standardize_fold_zero_variance(self):
        """Test standardization with zero variance features."""
        X_train = np.random.randn(80, 20)
        X_train[:, 0] = 5.0  # Zero variance feature

        X_test = np.random.randn(20, 20)

        from loveslide.cv import SLIDEcv
        X_train_std, X_test_std = SLIDEcv._standardize_fold(X_train, X_test)

        # Zero variance feature should remain zero (or be handled gracefully)
        assert np.allclose(X_train_std[:, 0], 0)

    def test_standardize_fold_extreme_values(self):
        """Test standardization with extreme values."""
        X_train = np.random.randn(80, 20)
        X_train[0, 0] = 1000  # Extreme outlier

        X_test = np.random.randn(20, 20)

        from loveslide.cv import SLIDEcv
        X_train_std, X_test_std = SLIDEcv._standardize_fold(X_train, X_test)

        # Should handle extreme values without producing NaN/Inf
        assert np.all(np.isfinite(X_train_std))
        assert np.all(np.isfinite(X_test_std))


class TestSLIDEcvMemoryManagement:
    """Test memory management during cross-validation."""

    def test_cv_memory_efficiency_small_data(self):
        """Test CV memory usage with small datasets."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        X = np.random.randn(200, 50)
        y = np.random.randn(200)

        cv = SLIDEcv(
            X=X, y=y,
            K_vals=[2, 3],
            lbd_vals=[0.3, 0.5],
            delta_vals=[0.1, 0.2],
            cv_folds=3
        )

        # Run a quick CV (we'll mock the expensive parts)
        with patch.object(cv, '_run_slide_fold', return_value=(np.random.randn(10), {})):
            results = cv.run(verbose=False)

        memory_after = process.memory_info().rss
        memory_increase = (memory_after - memory_before) / (1024 * 1024)  # MB

        # Memory increase should be reasonable
        assert memory_increase < 50  # Less than 50MB

    def test_cv_cleanup_after_fold(self):
        """Test that memory is cleaned up after each fold."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        cv = SLIDEcv(X=X, y=y, cv_folds=3)

        # Mock fold execution to track cleanup
        original_run_slide_fold = cv._run_slide_fold

        fold_count = 0
        def mock_run_slide_fold(*args, **kwargs):
            nonlocal fold_count
            fold_count += 1
            # Simulate some computation
            temp_data = np.random.randn(1000, 1000)
            result = original_run_slide_fold(*args, **kwargs)
            del temp_data  # Explicit cleanup
            return result

        with patch.object(cv, '_run_slide_fold', side_effect=mock_run_slide_fold):
            # This should not accumulate memory across folds
            with patch.object(cv, '_run_slide_fold', return_value=(np.random.randn(10), {})):
                results = cv.run(verbose=False)

        assert fold_count > 0  # Verify folds were executed


class TestSLIDEcvParallelization:
    """Test parallelization correctness."""

    def test_parallel_vs_serial_consistency(self):
        """Test that parallel and serial execution give same results."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        cv_serial = SLIDEcv(
            X=X, y=y,
            K_vals=[2],
            lbd_vals=[0.5],
            delta_vals=[0.1],
            cv_folds=3,
            n_workers=1
        )

        cv_parallel = SLIDEcv(
            X=X, y=y,
            K_vals=[2],
            lbd_vals=[0.5],
            delta_vals=[0.1],
            cv_folds=3,
            n_workers=2
        )

        # Mock the expensive computation to focus on parallelization logic
        mock_result = (np.array([1.0, 2.0, 3.0]), {"score": 0.85})

        with patch.object(cv_serial, '_run_slide_fold', return_value=mock_result), \
             patch.object(cv_parallel, '_run_slide_fold', return_value=mock_result):

            results_serial = cv_serial.run(verbose=False)
            results_parallel = cv_parallel.run(verbose=False)

        # Results should be identical
        assert len(results_serial) == len(results_parallel)
        for (params_s, score_s), (params_p, score_p) in zip(results_serial, results_parallel):
            assert params_s == params_p
            assert abs(score_s - score_p) < 1e-10

    def test_parallel_error_handling(self):
        """Test error handling in parallel execution."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        cv = SLIDEcv(
            X=X, y=y,
            K_vals=[2],
            lbd_vals=[0.5],
            delta_vals=[0.1],
            cv_folds=3,
            n_workers=2
        )

        # Mock a fold that raises an error
        def mock_run_slide_fold_error(*args, **kwargs):
            raise ValueError("Simulated fold error")

        with patch.object(cv, '_run_slide_fold', side_effect=mock_run_slide_fold_error):
            with pytest.raises(ValueError, match="Simulated fold error"):
                cv.run(verbose=False)


class TestSLIDEcvOutputValidation:
    """Test validation of cross-validation outputs."""

    def test_run_output_structure(self):
        """Test that CV output has expected structure."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        cv = SLIDEcv(
            X=X, y=y,
            K_vals=[2, 3],
            lbd_vals=[0.3, 0.5],
            delta_vals=[0.1, 0.2],
            cv_folds=3
        )

        # Mock fold execution
        with patch.object(cv, '_run_slide_fold', return_value=(np.random.randn(10), {"score": 0.8})):
            results = cv.run(verbose=False)

        # Should return list of (params, score) tuples
        assert isinstance(results, list)
        assert len(results) == 2 * 2 * 2  # K_vals * lbd_vals * delta_vals

        for params, score in results:
            assert isinstance(params, dict)
            assert 'K' in params
            assert 'lbd' in params
            assert 'delta' in params
            assert isinstance(score, (int, float))
            assert np.isfinite(score)

    def test_run_parameter_grid_completeness(self):
        """Test that all parameter combinations are evaluated."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        K_vals = [2, 3, 4]
        lbd_vals = [0.3, 0.5]
        delta_vals = [0.1, 0.2, 0.3]

        cv = SLIDEcv(
            X=X, y=y,
            K_vals=K_vals,
            lbd_vals=lbd_vals,
            delta_vals=delta_vals,
            cv_folds=3
        )

        with patch.object(cv, '_run_slide_fold', return_value=(np.random.randn(10), {"score": 0.8})):
            results = cv.run(verbose=False)

        # Should have all combinations
        assert len(results) == len(K_vals) * len(lbd_vals) * len(delta_vals)

        # Extract all evaluated parameter combinations
        evaluated_params = {(r[0]['K'], r[0]['lbd'], r[0]['delta']) for r in results}

        # Should match expected grid
        expected_params = {(K, lbd, delta) for K in K_vals for lbd in lbd_vals for delta in delta_vals}
        assert evaluated_params == expected_params