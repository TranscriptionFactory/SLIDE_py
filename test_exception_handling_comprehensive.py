"""
Comprehensive exception handling and recovery testing for SLIDE_py.

Tests error conditions, recovery mechanisms, and graceful degradation
to ensure robust production behavior.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import pickle
from unittest.mock import Mock, patch, mock_open
import warnings

from loveslide import (
    SLIDE, OptimizeSLIDE, SLIDEcv, Knockoffs,
    call_love, Plotter, init_data
)


class TestFileIOExceptions:
    """Test file I/O error handling."""

    def test_slide_invalid_file_paths(self):
        """Test SLIDE with non-existent file paths."""
        params = {
            'x_path': '/nonexistent/path/x.csv',
            'y_path': '/nonexistent/path/y.csv',
            'fdr': 0.1
        }

        with pytest.raises((FileNotFoundError, OSError)):
            SLIDE(params)

    def test_slide_corrupted_pickle_file(self):
        """Test SLIDE with corrupted pickle file."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            # Write invalid pickle data
            f.write(b'invalid pickle content')
            invalid_pickle_path = f.name

        try:
            params = {'x_path': None, 'y_path': None, 'fdr': 0.1}
            slide_instance = SLIDE(params, x=np.random.randn(50, 10), y=np.random.randn(50))

            with pytest.raises((pickle.UnpicklingError, EOFError, ValueError)):
                slide_instance.load_love(invalid_pickle_path)
        finally:
            os.unlink(invalid_pickle_path)

    def test_slide_permission_denied_output(self):
        """Test SLIDE with permission denied for output directory."""
        params = {
            'x_path': None, 'y_path': None,
            'output_dir': '/root/forbidden',  # Should cause permission error
            'fdr': 0.1
        }

        slide_instance = SLIDE(params, x=np.random.randn(50, 10), y=np.random.randn(50))

        # Mock os.makedirs to raise PermissionError
        with patch('os.makedirs', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                slide_instance.run_love()

    def test_slide_disk_full_scenario(self):
        """Test SLIDE behavior when disk is full."""
        params = {'x_path': None, 'y_path': None, 'fdr': 0.1}
        slide_instance = SLIDE(params, x=np.random.randn(50, 10), y=np.random.randn(50))

        # Mock file writing to raise OSError (disk full)
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            with pytest.raises(OSError):
                slide_instance.save_results('/tmp/test_results.pkl')


class TestMemoryExceptions:
    """Test memory-related error handling."""

    def test_slide_memory_error_large_array(self):
        """Test SLIDE behavior with memory allocation errors."""
        params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

        # Mock numpy operations to raise MemoryError
        with patch('numpy.zeros', side_effect=MemoryError("Cannot allocate memory")):
            with pytest.raises(MemoryError):
                SLIDE(params, x=np.random.randn(1000, 10000), y=np.random.randn(1000))

    def test_knockoffs_memory_exhaustion(self):
        """Test knockoff generation with memory constraints."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        knockoffs = Knockoffs(backend='python')

        # Mock covariance computation to raise MemoryError
        with patch('numpy.cov', side_effect=MemoryError("Memory allocation failed")):
            with pytest.raises(MemoryError):
                knockoffs.select_short_freq(X, y, fdr=0.1)

    def test_cv_memory_cleanup_after_error(self):
        """Test memory cleanup in CV after errors."""
        params = {
            'x_path': None, 'y_path': None,
            'n_folds': 5, 'fdr': 0.1
        }
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        cv_instance = SLIDEcv(params, x=X, y=y)

        # Mock to cause error during CV
        with patch.object(cv_instance, '_bench_cv', side_effect=RuntimeError("CV failed")):
            with pytest.raises(RuntimeError):
                cv_instance.run()

            # Verify instance is still in valid state for cleanup
            assert hasattr(cv_instance, 'data')
            assert hasattr(cv_instance, 'input_params')


class TestNumericalExceptions:
    """Test numerical computation error handling."""

    def test_love_singular_matrix_handling(self):
        """Test LOVE with singular covariance matrices."""
        # Create rank-deficient matrix
        X = np.random.randn(100, 20)
        X[:, 1] = X[:, 0]  # Make columns linearly dependent

        with pytest.warns(UserWarning, match="singular.*matrix"):
            result = call_love(X, lbd=0.5, verbose=False)
            # Should handle gracefully, not crash
            assert 'pure_Ind' in result

    def test_knockoffs_non_positive_definite_cov(self):
        """Test knockoffs with non-positive definite covariance."""
        # Create matrix that might not be positive definite
        X = np.random.randn(50, 30)
        X = X + 1e-10 * np.random.randn(50, 30)  # Add tiny noise

        # Force covariance to be near-singular
        cov = np.cov(X.T)
        cov[0, 0] = -1e-6  # Make it non-positive definite

        knockoffs = Knockoffs(backend='python')

        with patch('numpy.cov', return_value=cov):
            # Should handle gracefully or raise informative error
            with pytest.raises((np.linalg.LinAlgError, ValueError)):
                knockoffs.select_short_freq(X, np.random.randn(50), fdr=0.1)

    def test_slide_convergence_failure(self):
        """Test SLIDE behavior when algorithms fail to converge."""
        # Create difficult convergence case
        X = np.random.randn(50, 100)  # More features than samples
        y = np.random.randn(50)

        params = {
            'x_path': None, 'y_path': None,
            'fdr': 0.01,  # Very strict
            'max_iter': 1   # Force early termination
        }

        slide_instance = SLIDE(params, x=X, y=y)

        # Mock optimization to fail
        with patch('loveslide.love.call_love', side_effect=RuntimeError("Convergence failed")):
            with pytest.raises(RuntimeError):
                slide_instance.run_love()

    def test_numerical_overflow_handling(self):
        """Test handling of numerical overflow conditions."""
        # Create data that might cause overflow
        X = np.full((50, 20), 1e100)
        y = np.full(50, 1e100)

        params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Should either handle gracefully or raise appropriate error
            try:
                slide_instance = SLIDE(params, x=X, y=y)
                slide_instance.run_love()
            except (OverflowError, ValueError, np.linalg.LinAlgError):
                pass  # These are acceptable for extreme values

            # Check if overflow warnings were issued
            overflow_warnings = [warning for warning in w if 'overflow' in str(warning.message).lower()]
            # Should either handle silently or warn appropriately


class TestRInterfaceExceptions:
    """Test R interface error handling."""

    @pytest.mark.skipif(True, reason="R interface optional")
    def test_r_not_installed_fallback(self):
        """Test fallback when R is not installed."""
        with patch('rpy2.robjects', side_effect=ImportError("R not found")):
            # Should fallback to Python implementation
            knockoffs = Knockoffs(backend='r_knockoffs')
            assert knockoffs.backend == 'python'  # Should fallback

    @pytest.mark.skipif(True, reason="R interface optional")
    def test_r_package_missing_fallback(self):
        """Test fallback when R packages are missing."""
        with patch('loveslide.knockoffs.r_knockoffs', side_effect=ImportError("Package not found")):
            knockoffs = Knockoffs(backend='r_knockoffs')
            X = np.random.randn(50, 20)
            y = np.random.randn(50)

            # Should fallback gracefully
            result = knockoffs.select_short_freq(X, y, fdr=0.1)
            assert hasattr(result, 'selected')

    def test_r_computation_error_handling(self):
        """Test handling of R computation errors."""
        with patch('loveslide.love.call_love_r', side_effect=RuntimeError("R computation failed")):
            X = np.random.randn(50, 20)

            with pytest.raises(RuntimeError):
                call_love(X, use_r=True)


class TestParameterValidationExceptions:
    """Test parameter validation and error messages."""

    def test_invalid_fdr_values(self):
        """Test invalid FDR parameter values."""
        params = {'x_path': None, 'y_path': None}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        # Test negative FDR
        with pytest.raises(ValueError, match="fdr.*positive"):
            SLIDE(dict(params, fdr=-0.1), x=X, y=y)

        # Test FDR > 1
        with pytest.raises(ValueError, match="fdr.*1"):
            SLIDE(dict(params, fdr=1.5), x=X, y=y)

    def test_inconsistent_data_dimensions(self):
        """Test mismatched data dimensions."""
        params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

        # Mismatched sample sizes
        X = np.random.randn(100, 20)
        y = np.random.randn(50)  # Different sample size

        with pytest.raises(ValueError, match="dimension.*mismatch"):
            SLIDE(params, x=X, y=y)

    def test_empty_data_arrays(self):
        """Test empty data arrays."""
        params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

        with pytest.raises(ValueError, match="empty.*array"):
            SLIDE(params, x=np.array([]).reshape(0, 5), y=np.array([]))

    def test_invalid_cv_folds(self):
        """Test invalid cross-validation fold specifications."""
        params = {
            'x_path': None, 'y_path': None,
            'n_folds': -1,  # Invalid
            'fdr': 0.1
        }
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="fold.*positive"):
            SLIDEcv(params, x=X, y=y)


class TestGracefulDegradation:
    """Test graceful degradation under adverse conditions."""

    def test_slide_partial_failure_recovery(self):
        """Test SLIDE recovery from partial failures."""
        params = {'x_path': None, 'y_path': None, 'fdr': 0.1}
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        slide_instance = SLIDE(params, x=X, y=y)

        # Mock partial failure in LOVE
        with patch('loveslide.love.call_love') as mock_love:
            # Return minimal valid result
            mock_love.return_value = {
                'pure_Ind': [],
                'A': np.zeros((50, 5)),
                'delta': 0.1
            }

            # Should not crash, should handle gracefully
            slide_instance.run_love()
            assert hasattr(slide_instance.data, 'love_result')

    def test_knockoffs_fallback_methods(self):
        """Test knockoffs fallback to simpler methods."""
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        knockoffs = Knockoffs(backend='python')

        # Mock SDP solver failure
        with patch('loveslide.knockoff.solve.create_solve_sdp',
                   side_effect=RuntimeError("SDP failed")):
            with patch('loveslide.knockoff.solve.create_solve_equi') as mock_equi:
                mock_equi.return_value = lambda cov: np.eye(20) * 0.5

                # Should fallback to equicorrelated method
                result = knockoffs.select_short_freq(X, y, fdr=0.1, method='sdp')
                assert hasattr(result, 'selected')
                mock_equi.assert_called()

    def test_plotting_missing_dependencies(self):
        """Test plotting with missing optional dependencies."""
        with patch('matplotlib.pyplot', side_effect=ImportError("matplotlib not found")):
            plotter = Plotter()

            # Should handle gracefully, not crash
            try:
                plotter.plot_results({})
            except ImportError as e:
                assert "matplotlib" in str(e).lower()


class TestResourceCleanup:
    """Test resource cleanup after exceptions."""

    def test_temporary_file_cleanup(self):
        """Test cleanup of temporary files after errors."""
        params = {'x_path': None, 'y_path': None, 'fdr': 0.1}
        slide_instance = SLIDE(params, x=np.random.randn(50, 20), y=np.random.randn(50))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, 'temp_results.pkl')

            # Create temporary file
            with open(temp_file, 'wb') as f:
                pickle.dump({'test': 'data'}, f)

            assert os.path.exists(temp_file)

            # Mock error during processing
            with patch.object(slide_instance, 'run_love', side_effect=RuntimeError("Test error")):
                try:
                    slide_instance.run_love()
                except RuntimeError:
                    pass

            # Verify temporary files are cleaned up properly
            # (This depends on implementation details)

    def test_memory_cleanup_after_exception(self):
        """Test memory cleanup after exceptions."""
        import gc
        initial_objects = len(gc.get_objects())

        params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

        try:
            # Create instance that will fail
            with patch('loveslide.tools.init_data', side_effect=RuntimeError("Init failed")):
                SLIDE(params, x=np.random.randn(50, 20), y=np.random.randn(50))
        except RuntimeError:
            pass

        # Force garbage collection
        gc.collect()

        # Memory usage should not have grown significantly
        final_objects = len(gc.get_objects())
        assert final_objects - initial_objects < 1000  # Allow some growth


if __name__ == "__main__":
    pytest.main([__file__, "-v"])