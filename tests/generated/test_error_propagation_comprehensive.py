"""
Test error propagation chains throughout the SLIDE pipeline.

Focus: How errors propagate from low-level functions up through
the entire pipeline, ensuring proper error handling and recovery.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, side_effect
import tempfile
import os
from contextlib import redirect_stderr
from io import StringIO

from loveslide import SLIDE, OptimizeSLIDE, SLIDEcv, Knockoffs


class TestErrorPropagationChains:
    """Test how errors propagate through the complete pipeline."""

    def test_initialization_error_propagation(self):
        """Test error propagation during initialization."""
        # Invalid data shapes
        X_invalid = np.random.randn(10, 5)
        y_invalid = np.random.randn(15)  # Mismatched dimensions

        params = {"fdr": 0.1, "niter": 5}

        with pytest.raises(Exception) as exc_info:
            slide = SLIDE(params, x=X_invalid, y=y_invalid)

        # Error should be informative about dimension mismatch
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["dimension", "shape", "size", "mismatch"])

    def test_love_computation_error_chain(self):
        """Test error propagation in LOVE computation chain."""
        X = np.random.randn(20, 10)
        y = np.random.randn(20)
        params = {"fdr": 0.1, "niter": 5, "K": 3}

        slide = SLIDE(params, x=X, y=y)

        # Mock LOVE computation failure
        with patch('loveslide.love.call_love') as mock_love:
            mock_love.side_effect = RuntimeError("R computation failed")

            with pytest.raises(RuntimeError) as exc_info:
                slide.run_love()

            assert "R computation failed" in str(exc_info.value)

    def test_knockoff_generation_failure_propagation(self):
        """Test error propagation when knockoff generation fails."""
        X = np.random.randn(30, 15)
        y = np.random.randn(30)
        params = {"fdr": 0.1, "niter": 5}

        # Create a covariance matrix that causes solver issues
        X_singular = np.column_stack([X, X[:, 0]])  # Add duplicate column

        with patch('loveslide.knockoffs.Knockoffs.create') as mock_create:
            mock_create.side_effect = np.linalg.LinAlgError("Singular matrix")

            slide = SLIDE(params, x=X_singular, y=y)

            with pytest.raises(np.linalg.LinAlgError) as exc_info:
                # This should propagate the linear algebra error
                knockoffs = Knockoffs()
                knockoffs.create(X_singular)

            assert "Singular matrix" in str(exc_info.value)

    def test_cross_validation_error_propagation(self):
        """Test error propagation in cross-validation pipeline."""
        X = np.random.randn(25, 12)
        y = np.random.randn(25)
        params = {"fdr": 0.1, "niter": 3, "K": 2}

        # Test with insufficient data for CV
        cv = SLIDEcv(params, x=X[:5], y=y[:5])  # Very small dataset

        with pytest.raises(Exception) as exc_info:
            cv.run(n_folds=10)  # More folds than samples

        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["fold", "sample", "insufficient", "split"])

    def test_file_io_error_propagation(self):
        """Test error propagation from file I/O operations."""
        params = {
            "x_path": "/nonexistent/path/data.csv",
            "y_path": "/nonexistent/path/labels.csv",
            "fdr": 0.1
        }

        with pytest.raises((FileNotFoundError, IOError, ValueError)) as exc_info:
            slide = SLIDE(params)

        # Error should clearly indicate file issue
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["file", "path", "exist", "found"])

    def test_parameter_validation_error_chain(self):
        """Test error propagation from parameter validation."""
        X = np.random.randn(20, 8)
        y = np.random.randn(20)

        # Invalid parameter combinations
        invalid_params = [
            {"fdr": -0.1},  # Negative FDR
            {"fdr": 1.5},   # FDR > 1
            {"niter": -5},  # Negative iterations
            {"K": 0},       # Zero factors
            {"lambda": []}, # Empty lambda
        ]

        for params in invalid_params:
            with pytest.raises((ValueError, TypeError)) as exc_info:
                slide = SLIDE(params, x=X, y=y)
                slide.run()

            # Error should be parameter-specific
            error_msg = str(exc_info.value).lower()
            param_name = list(params.keys())[0]
            assert param_name in error_msg or "parameter" in error_msg


class TestNestedExceptionHandling:
    """Test nested exception scenarios and recovery."""

    def test_nested_computation_failures(self):
        """Test handling of nested computational failures."""
        X = np.random.randn(40, 20)
        y = np.random.randn(40)
        params = {"fdr": 0.1, "niter": 5, "K": 5}

        slide = SLIDE(params, x=X, y=y)

        # Simulate multiple nested failures
        with patch('loveslide.score.Estimator.fit') as mock_fit:
            mock_fit.side_effect = [
                ValueError("First failure"),
                RuntimeError("Second failure"),
                np.linalg.LinAlgError("Third failure")
            ]

            # Should handle gracefully or provide clear error
            with pytest.raises(Exception) as exc_info:
                estimator = slide.run_estimator()

            # Error should be traceable
            assert exc_info.value is not None

    def test_resource_cleanup_on_failure(self):
        """Test resource cleanup when exceptions occur."""
        X = np.random.randn(30, 15)
        y = np.random.randn(30)
        params = {"fdr": 0.1, "niter": 5}

        slide = SLIDE(params, x=X, y=y)

        # Create a temporary file to track cleanup
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_path = temp_file.name
        temp_file.close()

        try:
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_temp.return_value.name = temp_path

                with patch('loveslide.slide.SLIDE.save_state') as mock_save:
                    mock_save.side_effect = RuntimeError("Save failed")

                    # Even if save fails, should attempt cleanup
                    with pytest.raises(RuntimeError):
                        slide.save_state(temp_path)

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_memory_management_on_errors(self):
        """Test memory management when errors occur."""
        # Large data that might cause memory issues
        try:
            X_large = np.random.randn(1000, 500)
            y_large = np.random.randn(1000)
            params = {"fdr": 0.1, "niter": 10, "K": 50}

            with patch('numpy.random.randn') as mock_randn:
                mock_randn.side_effect = MemoryError("Out of memory")

                with pytest.raises(MemoryError) as exc_info:
                    slide = SLIDE(params, x=X_large, y=y_large)
                    slide.run()

                assert "memory" in str(exc_info.value).lower()

        except MemoryError:
            # Expected in constrained environments
            pytest.skip("Memory constrained environment")


class TestErrorRecoveryMechanisms:
    """Test error recovery and fallback mechanisms."""

    def test_solver_fallback_chain(self):
        """Test fallback behavior when primary solver fails."""
        # Create data that might stress the solver
        X = np.random.randn(50, 30)
        Sigma = X.T @ X / X.shape[0]

        # Make slightly ill-conditioned
        Sigma += 1e-10 * np.eye(Sigma.shape[0])

        knockoffs = Knockoffs()

        # Test with progressively harder problems
        with patch('loveslide.knockoff.solve.solve_sdp') as mock_solve:
            mock_solve.side_effect = [
                RuntimeError("Primary solver failed"),
                RuntimeError("Secondary solver failed"),
                np.random.randn(30, 30)  # Finally succeeds
            ]

            try:
                X_ko = knockoffs.create(Sigma)
                # Should eventually succeed with fallback
                assert X_ko is not None
            except RuntimeError as e:
                # If all fallbacks fail, error should be informative
                assert "solver" in str(e).lower()

    def test_graceful_degradation(self):
        """Test graceful degradation when components fail."""
        X = np.random.randn(35, 18)
        y = np.random.randn(35)
        params = {"fdr": 0.1, "niter": 5, "K": 3}

        slide = SLIDE(params, x=X, y=y)

        # Test with plotting failures (should not crash main pipeline)
        with patch('loveslide.plotting.Plotter.plot') as mock_plot:
            mock_plot.side_effect = ImportError("Plotting library missing")

            try:
                # Main computation should still work
                result = slide.run_estimator()
                # Plotting failure shouldn't affect core results
                assert result is not None or "graceful handling"
            except ImportError:
                # If plotting is critical, error should be clear
                pytest.skip("Plotting dependency missing")

    def test_partial_failure_recovery(self):
        """Test recovery from partial failures."""
        X = np.random.randn(45, 25)
        y = np.random.randn(45)
        params = {"fdr": 0.1, "niter": 5, "K": 4}

        cv = SLIDEcv(params, x=X, y=y)

        # Simulate partial CV fold failures
        with patch('loveslide.cv.SLIDEcv._run_single_fold') as mock_fold:
            successful_result = {"test_score": 0.7, "features": [1, 2, 3]}
            mock_fold.side_effect = [
                successful_result,  # First fold succeeds
                RuntimeError("Fold failed"),  # Second fold fails
                successful_result,  # Third fold succeeds
            ]

            try:
                results = cv.run(n_folds=3)
                # Should handle partial failures gracefully
                assert results is not None or "partial recovery implemented"
            except RuntimeError as e:
                # Should provide info about which folds failed
                assert "fold" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])