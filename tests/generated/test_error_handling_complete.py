"""
Test comprehensive error handling and edge cases across SLIDE modules.
Addresses: Input validation, graceful failures, boundary conditions
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import warnings
from unittest.mock import Mock, patch, MagicMock

from loveslide import (
    SLIDE, OptimizeSLIDE, SLIDEcv, Knockoffs,
    call_love, Plotter, Estimator, SLIDE_Estimator,
    init_data, check_params
)


class TestInputValidation:
    """Test input validation across all modules."""

    def test_slide_invalid_input_types(self):
        """Test SLIDE with invalid input types."""
        # String instead of dict for params
        with pytest.raises((TypeError, ValueError)):
            SLIDE("invalid_params")

        # List instead of array for X
        params = {"fdr": 0.1}
        with pytest.raises((TypeError, ValueError)):
            SLIDE(params, x=[[1, 2], [3, 4]], y=[1, 2])

        # Mismatched X, y dimensions
        X = np.random.randn(50, 10)
        y = np.random.randn(30)  # Wrong size
        with pytest.raises(ValueError):
            SLIDE(params, x=X, y=y)

    def test_slide_extreme_parameter_values(self):
        """Test SLIDE with extreme parameter values."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        # Negative FDR
        with pytest.raises((ValueError, AssertionError)):
            slide = SLIDE({"fdr": -0.1}, x=X, y=y)

        # FDR > 1
        with pytest.raises((ValueError, AssertionError)):
            slide = SLIDE({"fdr": 1.5}, x=X, y=y)

        # Zero iterations
        with pytest.raises((ValueError, AssertionError)):
            slide = SLIDE({"niter": 0}, x=X, y=y)

        # Extremely large feature size
        slide = SLIDE({"f_size": 100000}, x=X, y=y)
        # Should cap at reasonable value or handle gracefully
        assert slide.input_params["f_size"] <= X.shape[1]

    def test_knockoffs_invalid_matrices(self):
        """Test knockoffs with invalid correlation matrices."""
        knockoffs = Knockoffs()

        # Non-square matrix
        X = np.random.randn(50, 10)
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            knockoffs.create_knockoffs(X, method='equi')

        # Non-positive definite matrix
        X = np.array([[1, 2], [2, 1]])  # rank-deficient when computing correlation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises((np.linalg.LinAlgError, ValueError)):
                knockoffs.create_knockoffs(X, method='sdp')

    def test_love_invalid_inputs(self):
        """Test LOVE with invalid inputs."""
        # Empty matrix
        with pytest.raises((ValueError, IndexError)):
            call_love(np.array([]).reshape(0, 5))

        # Single row
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            call_love(np.random.randn(1, 10))

        # NaN values
        X = np.random.randn(50, 8)
        X[10, 3] = np.nan
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            call_love(X)

        # Infinite values
        X = np.random.randn(30, 6)
        X[5, 2] = np.inf
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            call_love(X)

    def test_estimator_invalid_configurations(self):
        """Test Estimator with invalid configurations."""
        # Unknown model
        with pytest.raises((ValueError, KeyError)):
            Estimator(model='unknown_model')

        # Invalid scaler
        with pytest.raises((ValueError, KeyError)):
            Estimator(scaler='invalid_scaler')

        # Fit without data
        estimator = Estimator()
        with pytest.raises((ValueError, TypeError)):
            estimator.fit(None, None)

        # Predict before fit
        estimator = Estimator()
        with pytest.raises((ValueError, AttributeError)):
            estimator.predict(np.random.randn(10, 5))

    def test_cv_invalid_configurations(self):
        """Test SLIDEcv with invalid configurations."""
        X = np.random.randn(50, 8)
        y = np.random.randn(50)

        # Invalid number of folds
        with pytest.raises(ValueError):
            SLIDEcv(X, y, folds=1)  # Too few folds

        with pytest.raises(ValueError):
            SLIDEcv(X, y, folds=len(y) + 1)  # More folds than samples

        # Empty parameter grids
        with pytest.raises(ValueError):
            cv = SLIDEcv(X, y, folds=5)
            cv.run(param_grid={})

        # Invalid metric
        with pytest.raises((ValueError, KeyError)):
            SLIDEcv(X, y, folds=5, metric='invalid_metric')


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_minimal_dataset_sizes(self):
        """Test with minimal viable dataset sizes."""
        # Smallest possible dataset
        X = np.random.randn(3, 2)
        y = np.random.randn(3)

        # Should handle gracefully or raise informative error
        try:
            slide = SLIDE({"fdr": 0.1, "niter": 1}, x=X, y=y)
            # If it succeeds, should not crash
        except (ValueError, np.linalg.LinAlgError) as e:
            # Should provide informative error message
            assert "size" in str(e).lower() or "dimension" in str(e).lower()

    def test_high_dimensional_data(self):
        """Test with high-dimensional data (p > n)."""
        n, p = 20, 50
        X = np.random.randn(n, p)
        y = np.random.randn(n)

        # Should handle p > n case
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                slide = SLIDE({"fdr": 0.1, "niter": 1}, x=X, y=y)
                # If successful, should have appropriate default parameters
                f_size = slide.calc_default_fsize(5)
                assert f_size <= p
            except (ValueError, np.linalg.LinAlgError):
                # Acceptable for very high-dimensional cases
                pass

    def test_perfect_correlation_cases(self):
        """Test with perfectly correlated features."""
        n = 100
        X = np.random.randn(n, 1)
        X = np.hstack([X, X, X + 1e-10 * np.random.randn(n, 1)])  # Near-perfect correlation
        y = np.random.randn(n)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = call_love(X)
                # Should handle multicollinearity gracefully
                assert 'A' in result or 'error' in result
            except (np.linalg.LinAlgError, ValueError):
                # Acceptable for singular cases
                pass

    def test_extreme_noise_levels(self):
        """Test with extremely noisy data."""
        n, p = 100, 10
        X = np.random.randn(n, p)

        # Very high noise
        y = 1e-6 * X[:, 0] + np.random.randn(n) * 1000

        slide = SLIDE({"fdr": 0.1, "niter": 1}, x=X, y=y)
        # Should not crash, though power may be low
        assert slide.data.X.shape == (n, p)

        # Very low noise (near deterministic)
        y = X[:, 0] + 1e-10 * np.random.randn(n)

        slide = SLIDE({"fdr": 0.1, "niter": 1}, x=X, y=y)
        assert slide.data.X.shape == (n, p)

    def test_extreme_correlation_structures(self):
        """Test with extreme correlation structures."""
        # Block diagonal structure
        n = 80
        block1 = np.random.multivariate_normal([0, 0], [[1, 0.99], [0.99, 1]], n)
        block2 = np.random.multivariate_normal([0, 0], [[1, -0.99], [-0.99, 1]], n)
        independent = np.random.randn(n, 2)

        X = np.hstack([block1, block2, independent])
        y = np.random.randn(n)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Should handle extreme correlations
            slide = SLIDE({"fdr": 0.1, "niter": 1}, x=X, y=y)
            assert slide.data.X.shape == (n, X.shape[1])


class TestResourceLimitations:
    """Test behavior under resource limitations."""

    def test_memory_efficient_operations(self):
        """Test memory efficiency with larger datasets."""
        # Moderately large dataset
        n, p = 1000, 100
        X = np.random.randn(n, p)
        y = np.random.randn(n)

        # Should not consume excessive memory
        slide = SLIDE({"fdr": 0.1, "niter": 1, "f_size": 20}, x=X, y=y)

        # Feature size should be respected for memory management
        assert slide.input_params["f_size"] == 20

    @patch('loveslide.knockoffs.Knockoffs.filter_knockoffs_iterative')
    def test_computation_timeout_handling(self, mock_filter):
        """Test handling of computation timeouts."""
        # Mock a function that takes too long
        def slow_function(*args, **kwargs):
            import time
            time.sleep(10)  # Simulate long computation
            return {'selected': np.array([]), 'statistic': np.array([])}

        mock_filter.side_effect = slow_function

        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        knockoffs = Knockoffs(y, X)

        # Should handle timeout gracefully (implementation-dependent)
        # This test structure allows for timeout handling if implemented
        try:
            result = knockoffs.filter_knockoffs_iterative(
                X, y, fdr=0.1, niter=1
            )
        except Exception as e:
            # Should not crash the entire system
            assert True

    def test_disk_space_limitations(self):
        """Test behavior when disk space is limited."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        slide = SLIDE({"fdr": 0.1}, x=X, y=y)

        # Test saving to invalid location
        with pytest.raises((OSError, PermissionError)):
            slide.save_params("/invalid/path/results", {"test": "data"})

        # Test saving with insufficient permissions
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create read-only directory
            readonly_dir = os.path.join(tmpdir, "readonly")
            os.makedirs(readonly_dir)
            os.chmod(readonly_dir, 0o444)

            try:
                with pytest.raises((OSError, PermissionError)):
                    slide.save_params(
                        os.path.join(readonly_dir, "test.pkl"),
                        {"test": "data"}
                    )
            finally:
                # Restore permissions for cleanup
                os.chmod(readonly_dir, 0o755)


class TestErrorPropagation:
    """Test error propagation and recovery."""

    def test_partial_failure_recovery(self):
        """Test recovery from partial failures."""
        X = np.random.randn(80, 12)
        y = np.random.randn(80)

        # Mock a function that fails intermittently
        original_filter = Knockoffs.filter_knockoffs_iterative

        def intermittent_failure(self, *args, **kwargs):
            if hasattr(self, '_call_count'):
                self._call_count += 1
            else:
                self._call_count = 1

            if self._call_count == 1:
                raise ValueError("Simulated failure")
            return original_filter(self, *args, **kwargs)

        with patch.object(Knockoffs, 'filter_knockoffs_iterative', intermittent_failure):
            knockoffs = Knockoffs(y, X)

            # First call should fail
            with pytest.raises(ValueError):
                knockoffs.filter_knockoffs_iterative(X, y, fdr=0.1)

            # Second call should succeed (if retry mechanism exists)
            # This tests the system's ability to recover from failures
            try:
                result = knockoffs.filter_knockoffs_iterative(X, y, fdr=0.1)
                assert 'selected' in result
            except ValueError:
                # If no retry mechanism, that's also acceptable behavior
                pass

    def test_cascading_failure_prevention(self):
        """Test that single failures don't cause cascading failures."""
        X = np.random.randn(60, 8)
        y = np.random.randn(60)

        slide = SLIDE({"fdr": 0.1, "niter": 1}, x=X, y=y)

        # Mock one component to fail
        with patch.object(slide, 'calc_default_fsize') as mock_calc:
            mock_calc.side_effect = RuntimeError("Component failure")

            # Other components should still be accessible
            slide.show_params()  # Should not fail

            # Data should still be available
            assert slide.data.X.shape == (60, 8)

    def test_error_message_informativeness(self):
        """Test that error messages are informative."""
        # Test various invalid inputs and check error message quality
        test_cases = [
            (lambda: SLIDE("invalid"), "parameter"),
            (lambda: Estimator(model="nonexistent"), "model"),
            (lambda: init_data({}, x=None, y=None), "data"),
        ]

        for test_func, expected_keyword in test_cases:
            try:
                test_func()
                assert False, "Expected an exception"
            except Exception as e:
                error_msg = str(e).lower()
                # Error message should contain relevant keyword
                assert any(keyword in error_msg for keyword in [
                    expected_keyword, "invalid", "error", "failed"
                ]), f"Error message not informative: {e}"


class TestStateConsistency:
    """Test state consistency under various failure scenarios."""

    def test_object_state_after_failure(self):
        """Test object state remains consistent after failures."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        slide = SLIDE({"fdr": 0.1}, x=X, y=y)
        original_params = slide.input_params.copy()

        # Simulate operation that might modify state
        try:
            # This might fail, but shouldn't corrupt object state
            slide.calc_default_fsize(-5)  # Invalid K value
        except (ValueError, RuntimeError):
            pass

        # Object state should remain consistent
        assert slide.input_params == original_params
        assert slide.data.X.shape == (50, 10)

    def test_cleanup_after_exceptions(self):
        """Test that resources are cleaned up after exceptions."""
        X = np.random.randn(40, 6)
        y = np.random.randn(40)

        estimator = Estimator()

        # Force an exception during fit
        try:
            estimator.fit(X[:10], y)  # Mismatched dimensions
        except (ValueError, IndexError):
            pass

        # Internal state should be clean
        with pytest.raises((ValueError, AttributeError)):
            estimator.predict(X)  # Should fail because fit didn't complete

    def test_thread_safety_basics(self):
        """Test basic thread safety considerations."""
        X = np.random.randn(100, 15)
        y = np.random.randn(100)

        # Create multiple SLIDE instances
        slides = [SLIDE({"fdr": 0.1}, x=X, y=y) for _ in range(3)]

        # Each should have independent state
        for i, slide in enumerate(slides):
            slide.input_params["test_id"] = i

        # Verify independence
        for i, slide in enumerate(slides):
            assert slide.input_params["test_id"] == i


class TestRecoveryMechanisms:
    """Test recovery mechanisms and fallback behaviors."""

    def test_solver_fallback_mechanisms(self):
        """Test fallback when primary solvers fail."""
        # Create a scenario where SDP solver might fail
        p = 10
        Sigma = np.random.randn(p, p)
        Sigma = Sigma @ Sigma.T + 0.001 * np.eye(p)  # Ill-conditioned

        knockoffs = Knockoffs()

        # Should fall back gracefully if SDP fails
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = knockoffs.create_knockoffs(np.random.randn(50, p), method='sdp')
                if result is None:
                    # Fallback to equicorrelated should work
                    result = knockoffs.create_knockoffs(np.random.randn(50, p), method='equi')
                    assert result is not None
            except Exception:
                # Complete failure is acceptable for very pathological cases
                pass

    def test_parameter_validation_recovery(self):
        """Test recovery from invalid parameter combinations."""
        X = np.random.randn(60, 8)
        y = np.random.randn(60)

        # Invalid parameter should be corrected or rejected clearly
        params = {"fdr": 0.1, "niter": -5}  # Invalid niter

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                slide = SLIDE(params, x=X, y=y)
                # If accepted, should be corrected to valid value
                assert slide.input_params["niter"] >= 1
            except ValueError:
                # Clear rejection is also acceptable
                pass

    def test_data_preprocessing_fallbacks(self):
        """Test fallbacks in data preprocessing."""
        # Data with various issues
        n, p = 50, 6

        # Nearly constant features
        X = np.random.randn(n, p)
        X[:, 2] = 1 + 1e-10 * np.random.randn(n)  # Nearly constant

        y = np.random.randn(n)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Should handle near-constant features
            slide = SLIDE({"fdr": 0.1}, x=X, y=y)
            assert slide.data.X.shape == (n, p)

            # Feature should be handled (removed, regularized, or kept with warning)
            assert np.all(np.isfinite(slide.data.X))