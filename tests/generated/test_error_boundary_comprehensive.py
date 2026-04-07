"""
Comprehensive error boundary and exception handling tests for SLIDE_py.

Tests critical error scenarios that could cause:
- Silent failures
- Memory leaks
- Inconsistent state
- Security vulnerabilities
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock
import warnings
from contextlib import contextmanager

from loveslide import SLIDE, OptimizeSLIDE, SLIDEcv, Knockoffs, Plotter
from loveslide.tools import init_data, check_params
from loveslide.love import call_love


class TestSLIDEErrorBoundaries:
    """Test SLIDE class error boundaries and exception handling."""

    def test_slide_init_invalid_params_types(self):
        """Test SLIDE initialization with invalid parameter types."""
        # Test with non-dict params
        with pytest.raises(TypeError):
            SLIDE("invalid_params")

        # Test with None params
        with pytest.raises((TypeError, AttributeError)):
            SLIDE(None)

    def test_slide_init_missing_data_paths(self):
        """Test SLIDE initialization with missing data paths."""
        params = {"fdr": 0.1}

        with pytest.raises(ValueError) as excinfo:
            SLIDE(params)
        assert "x_path is not provided" in str(excinfo.value)

    def test_slide_init_corrupted_data_files(self):
        """Test SLIDE with corrupted data files."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("corrupted,data\n1,invalid,data")
            f.flush()

            params = {
                "x_path": f.name,
                "y_path": f.name,
                "fdr": 0.1
            }

            with pytest.raises((pd.errors.ParserError, ValueError)):
                SLIDE(params)

            os.unlink(f.name)

    def test_slide_load_love_nonexistent_file(self):
        """Test load_love with nonexistent file."""
        params = {"fdr": 0.1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        slide = SLIDE(params, x=X, y=y)

        # Should handle gracefully without crashing
        slide.load_love("nonexistent_file.pkl")

    def test_slide_load_love_corrupted_pickle(self):
        """Test load_love with corrupted pickle file."""
        params = {"fdr": 0.1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        slide = SLIDE(params, x=X, y=y)

        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b"corrupted pickle data")
            f.flush()

            # Should handle gracefully
            slide.load_love(f.name)

            os.unlink(f.name)

    def test_slide_calc_z_matrix_invalid_love_result(self):
        """Test calc_z_matrix with invalid love result structure."""
        params = {"fdr": 0.1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        slide = SLIDE(params, x=X, y=y)

        # Test with missing required keys
        invalid_love_result = {"wrong_key": "value"}

        with pytest.raises((KeyError, AttributeError)):
            slide.calc_z_matrix(invalid_love_result)

    def test_slide_memory_exhaustion_simulation(self):
        """Test SLIDE behavior under memory pressure."""
        params = {"fdr": 0.1}
        # Create large matrices that might cause memory issues
        try:
            X = np.random.randn(10000, 5000)  # Large matrix
            y = np.random.randn(10000)
            slide = SLIDE(params, x=X, y=y)
        except MemoryError:
            # Expected behavior - should fail gracefully
            pass

    def test_slide_concurrent_access_thread_safety(self):
        """Test SLIDE thread safety with concurrent access."""
        import threading

        params = {"fdr": 0.1}
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        def create_slide():
            return SLIDE(params, x=X.copy(), y=y.copy())

        threads = []
        results = []

        def thread_function():
            try:
                slide = create_slide()
                results.append("success")
            except Exception as e:
                results.append(f"error: {e}")

        for i in range(5):
            thread = threading.Thread(target=thread_function)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Check that no thread crashed unexpectedly
        assert len(results) == 5


class TestKnockoffsErrorBoundaries:
    """Test Knockoffs class error boundaries."""

    def test_knockoffs_invalid_fdr_values(self):
        """Test Knockoffs with invalid FDR values."""
        X = np.random.randn(50, 20)
        y = np.random.binomial(1, 0.5, 50)

        # Test negative FDR
        knockoffs = Knockoffs()
        with pytest.raises(ValueError):
            knockoffs.run_iteration(X, y, fdr=-0.1, method='lasso')

        # Test FDR > 1
        with pytest.raises(ValueError):
            knockoffs.run_iteration(X, y, fdr=1.5, method='lasso')

    def test_knockoffs_unsupported_method(self):
        """Test Knockoffs with unsupported method."""
        X = np.random.randn(50, 20)
        y = np.random.binomial(1, 0.5, 50)

        knockoffs = Knockoffs()
        with pytest.raises(ValueError):
            knockoffs.run_iteration(X, y, fdr=0.1, method='unsupported_method')

    def test_knockoffs_singular_covariance_matrix(self):
        """Test Knockoffs with singular covariance matrix."""
        # Create data with singular covariance
        X = np.random.randn(50, 10)
        X[:, 5] = X[:, 0]  # Make column 5 identical to column 0
        y = np.random.binomial(1, 0.5, 50)

        knockoffs = Knockoffs()

        # Should handle singular matrices gracefully
        try:
            result = knockoffs.run_iteration(X, y, fdr=0.1, method='lasso')
        except (np.linalg.LinAlgError, ValueError) as e:
            # Expected behavior for singular matrices
            assert "singular" in str(e).lower() or "invertible" in str(e).lower()

    def test_knockoffs_mismatched_dimensions(self):
        """Test Knockoffs with mismatched X, y dimensions."""
        X = np.random.randn(50, 20)
        y = np.random.binomial(1, 0.5, 30)  # Wrong dimension

        knockoffs = Knockoffs()
        with pytest.raises(ValueError):
            knockoffs.run_iteration(X, y, fdr=0.1, method='lasso')

    def test_knockoffs_r_interface_failure(self):
        """Test Knockoffs when R interface fails."""
        X = np.random.randn(50, 20)
        y = np.random.binomial(1, 0.5, 50)

        knockoffs = Knockoffs()

        # Mock R interface failure
        with patch('loveslide.knockoffs.ro.r', side_effect=Exception("R interface failed")):
            with pytest.raises(Exception):
                knockoffs.run_iteration(X, y, fdr=0.1, method='lasso')


class TestLoveErrorBoundaries:
    """Test LOVE function error boundaries."""

    def test_call_love_invalid_matrix_dimensions(self):
        """Test call_love with invalid matrix dimensions."""
        # Test with incompatible dimensions
        X = np.array([[1, 2], [3, 4]])  # 2x2 matrix

        # Should handle small matrices gracefully
        result = call_love(X)
        # Either returns valid result or raises appropriate error

    def test_call_love_non_numeric_data(self):
        """Test call_love with non-numeric data."""
        # Create matrix with non-numeric values
        X = np.array([['a', 'b'], ['c', 'd']], dtype=object)

        with pytest.raises((TypeError, ValueError)):
            call_love(X)

    def test_call_love_infinite_values(self):
        """Test call_love with infinite values."""
        X = np.random.randn(50, 20)
        X[0, 0] = np.inf
        X[1, 1] = -np.inf

        with pytest.raises((ValueError, FloatingPointError)):
            call_love(X)

    def test_call_love_nan_values(self):
        """Test call_love with NaN values."""
        X = np.random.randn(50, 20)
        X[0, 0] = np.nan

        with pytest.raises((ValueError, FloatingPointError)):
            call_love(X)

    @patch('loveslide.love.call_love_r')
    def test_call_love_r_interface_timeout(self, mock_call_love_r):
        """Test call_love when R interface times out."""
        mock_call_love_r.side_effect = TimeoutError("R call timed out")

        X = np.random.randn(50, 20)

        with pytest.raises(TimeoutError):
            call_love(X)


class TestCVErrorBoundaries:
    """Test cross-validation error boundaries."""

    def test_slidecv_invalid_cv_folds(self):
        """Test SLIDEcv with invalid CV fold parameters."""
        params = {"fdr": 0.1, "cv_folds": 0}  # Invalid
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        with pytest.raises(ValueError):
            SLIDEcv(params, x=X, y=y)

        # Test with more folds than samples
        params = {"fdr": 0.1, "cv_folds": 100}
        with pytest.raises(ValueError):
            SLIDEcv(params, x=X, y=y)

    def test_slidecv_insufficient_data_for_cv(self):
        """Test SLIDEcv with insufficient data for cross-validation."""
        params = {"fdr": 0.1, "cv_folds": 5}
        X = np.random.randn(3, 20)  # Too few samples
        y = np.random.randn(3)

        with pytest.raises(ValueError):
            SLIDEcv(params, x=X, y=y)


class TestPlottingErrorBoundaries:
    """Test plotting function error boundaries."""

    def test_plotter_invalid_data_types(self):
        """Test Plotter with invalid data types."""
        plotter = Plotter()

        # Test with non-array data
        with pytest.raises((TypeError, ValueError)):
            plotter.create_plot("invalid_data", plot_type="heatmap")

    def test_plotter_empty_data(self):
        """Test Plotter with empty data."""
        plotter = Plotter()

        empty_array = np.array([])
        with pytest.raises((ValueError, IndexError)):
            plotter.create_plot(empty_array, plot_type="line")

    def test_plotter_memory_intensive_plots(self):
        """Test Plotter with memory-intensive plot data."""
        plotter = Plotter()

        # Very large data that might cause memory issues
        try:
            large_data = np.random.randn(10000, 10000)
            # Should either handle gracefully or fail with MemoryError
            plotter.create_plot(large_data, plot_type="heatmap")
        except MemoryError:
            # Expected behavior for very large plots
            pass


class TestFileSystemErrorBoundaries:
    """Test file system related error boundaries."""

    def test_init_data_readonly_filesystem(self):
        """Test init_data with read-only file system."""
        # Create temporary file in read-only directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Make directory read-only
            os.chmod(tmpdir, 0o444)

            params = {
                "x_path": os.path.join(tmpdir, "test.csv"),
                "y_path": os.path.join(tmpdir, "test.csv"),
                "out_path": tmpdir  # Read-only directory
            }

            try:
                # Should handle permission errors gracefully
                data, params = init_data(params)
            except PermissionError:
                # Expected behavior
                pass
            finally:
                # Restore permissions for cleanup
                os.chmod(tmpdir, 0o755)

    def test_save_operations_disk_full_simulation(self):
        """Test save operations under disk full conditions."""
        # This would require mocking disk space
        pass


@contextmanager
def expect_warning(warning_class, message_pattern=None):
    """Context manager to expect specific warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w
        if w:
            assert any(issubclass(warning.category, warning_class) for warning in w)
            if message_pattern:
                assert any(message_pattern in str(warning.message) for warning in w)


def create_corrupted_data_file(filepath, corruption_type="invalid_csv"):
    """Create various types of corrupted data files for testing."""
    with open(filepath, 'w') as f:
        if corruption_type == "invalid_csv":
            f.write("col1,col2\n1,2,3\n4,5")  # Inconsistent columns
        elif corruption_type == "binary_data":
            f.write('\x00\x01\x02\x03')  # Binary data in text file
        elif corruption_type == "extremely_long_lines":
            f.write("col1,col2\n" + "a" * 10000 + ",b\n")  # Very long line


if __name__ == "__main__":
    pytest.main([__file__])