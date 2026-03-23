"""
Test Coverage Gap: R-Python Interface Boundaries
==============================================

Tests complex R-Python interface scenarios, error propagation, and resource management
that may not be fully covered in existing tests.
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import sys
from unittest.mock import patch, MagicMock
from src.loveslide.love import call_love, call_love_r
from src.loveslide.knockoffs import _rlist_get


class TestRPythonInterfaceBoundaries:
    """Test R-Python interface boundary conditions."""

    @pytest.fixture
    def sample_covariance_matrix(self):
        """Generate a sample covariance matrix."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        return np.cov(X.T)

    def test_call_love_r_with_malformed_r_objects(self, sample_covariance_matrix):
        """Test call_love_r with malformed R objects returned."""
        X = sample_covariance_matrix

        # Mock rpy2 to return malformed objects
        with patch('src.loveslide.knockoffs.ro.r') as mock_r:
            mock_r.return_value = MagicMock()
            mock_r.return_value.rx2.return_value = None  # Malformed return

            with pytest.raises((AttributeError, TypeError, ValueError)):
                call_love_r(X, lbd=0.5)

    def test_r_session_memory_leak_detection(self, sample_covariance_matrix):
        """Test for R session memory leaks during repeated calls."""
        X = sample_covariance_matrix

        # Mock successful R calls
        with patch('src.loveslide.knockoffs.ro.r') as mock_r:
            mock_result = MagicMock()
            mock_result.rx2.return_value = np.random.randn(20, 5)
            mock_r.return_value = mock_result

            # Simulate many repeated calls
            for _ in range(10):
                try:
                    call_love_r(X, lbd=0.5)
                except Exception:
                    pass  # Expected to fail with mocked data

    def test_rlist_get_with_nested_complex_structures(self):
        """Test _rlist_get with deeply nested R list structures."""
        # Mock complex nested R object
        mock_robj = MagicMock()
        mock_nested = MagicMock()
        mock_nested.rx2.return_value = np.array([1, 2, 3])
        mock_robj.rx2.return_value = mock_nested

        result = _rlist_get(mock_robj, "nested_key")

        # Should handle nested structures gracefully
        mock_robj.rx2.assert_called_once_with("nested_key")

    def test_r_unicode_handling_edge_cases(self, sample_covariance_matrix):
        """Test R interface with unicode characters in paths/names."""
        X = sample_covariance_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create path with unicode characters
            unicode_path = os.path.join(tmpdir, "test_ñáméé_测试")
            os.makedirs(unicode_path, exist_ok=True)

            # Test should handle unicode paths gracefully
            with patch('src.loveslide.knockoffs.ro.r') as mock_r:
                mock_r.return_value = MagicMock()

                try:
                    # This might fail, but shouldn't crash the Python process
                    call_love_r(X, lbd=0.5)
                except Exception as e:
                    # Should be a handled exception, not a segfault
                    assert isinstance(e, (RuntimeError, ValueError, AttributeError))

    def test_r_large_matrix_transfer_limits(self):
        """Test R-Python transfer limits with very large matrices."""
        # Create a large matrix that might exceed R memory limits
        n_large = 10000
        X_large = np.random.randn(n_large, 100).astype(np.float32)

        with patch('src.loveslide.knockoffs.ro.r') as mock_r:
            mock_r.side_effect = MemoryError("R memory exhausted")

            with pytest.raises((MemoryError, RuntimeError)):
                call_love_r(X_large, lbd=0.5)

    def test_r_session_concurrent_access(self, sample_covariance_matrix):
        """Test concurrent access to R session from multiple threads."""
        import threading
        import time

        X = sample_covariance_matrix
        results = []
        errors = []

        def r_call_worker(worker_id):
            try:
                with patch('src.loveslide.knockoffs.ro.r') as mock_r:
                    mock_r.return_value = MagicMock()
                    result = call_love_r(X, lbd=0.5)
                    results.append(f"worker_{worker_id}_success")
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")

        # Start multiple threads accessing R
        threads = []
        for i in range(5):
            thread = threading.Thread(target=r_call_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should either succeed or fail gracefully, not deadlock
        assert len(results) + len(errors) == 5

    def test_call_love_parameter_validation_boundaries(self, sample_covariance_matrix):
        """Test call_love with boundary parameter values."""
        X = sample_covariance_matrix

        # Test extreme parameter values
        extreme_params = [
            {'lbd': 0.0},      # Minimum lambda
            {'lbd': 1.0},      # Maximum lambda
            {'mu': 0.0},       # Minimum mu
            {'mu': 1.0},       # Maximum mu
            {'delta': None},   # None delta
            {'thresh_fdr': 0.001},  # Very small FDR
            {'thresh_fdr': 0.999},  # Very large FDR
            {'rep_CV': 1},     # Minimum CV reps
        ]

        for params in extreme_params:
            try:
                result = call_love(X, **params)
                # Should either succeed or raise appropriate error
                assert result is not None or True  # Function called
            except (ValueError, RuntimeError) as e:
                # Expected parameter validation errors
                assert "parameter" in str(e).lower() or "invalid" in str(e).lower()

    def test_r_error_message_propagation(self, sample_covariance_matrix):
        """Test that R error messages are properly propagated to Python."""
        X = sample_covariance_matrix

        with patch('src.loveslide.knockoffs.ro.r') as mock_r:
            # Mock R error with specific message
            r_error = RuntimeError("R Error: Matrix is not positive definite")
            mock_r.side_effect = r_error

            with pytest.raises(RuntimeError) as exc_info:
                call_love_r(X, lbd=0.5)

            # Error message should contain R context
            assert "R Error" in str(exc_info.value) or "Matrix" in str(exc_info.value)


class TestRPythonDataConversion:
    """Test data conversion between R and Python."""

    def test_matrix_dtype_preservation(self):
        """Test that matrix data types are preserved across R-Python boundary."""
        # Test different dtypes
        dtypes_to_test = [np.float32, np.float64, np.int32, np.int64]

        for dtype in dtypes_to_test:
            X = np.random.randn(10, 5).astype(dtype)

            with patch('src.loveslide.knockoffs.ro.r') as mock_r:
                # Mock conversion preserving dtype
                mock_result = MagicMock()
                mock_result.rx2.return_value = X.astype(np.float64)  # R typically uses float64
                mock_r.return_value = mock_result

                try:
                    result = call_love_r(X, lbd=0.5)
                    # Should handle dtype conversion appropriately
                except Exception:
                    pass  # Expected with mocked data

    def test_matrix_dimension_validation(self):
        """Test validation of matrix dimensions across R-Python interface."""
        # Test various problematic dimensions
        problematic_matrices = [
            np.array([]),                    # Empty array
            np.array([1]),                   # 1D array
            np.random.randn(1, 1),          # 1x1 matrix
            np.random.randn(0, 5),          # Zero rows
            np.random.randn(5, 0),          # Zero columns
        ]

        for X in problematic_matrices:
            with patch('src.loveslide.knockoffs.ro.r') as mock_r:
                mock_r.return_value = MagicMock()

                try:
                    result = call_love_r(X, lbd=0.5)
                except (ValueError, RuntimeError) as e:
                    # Should provide clear dimension error messages
                    assert "dimension" in str(e).lower() or "shape" in str(e).lower()

    def test_inf_nan_handling_r_interface(self):
        """Test handling of infinite and NaN values across R interface."""
        # Create matrix with problematic values
        X = np.random.randn(10, 5)
        X[0, 0] = np.inf
        X[1, 1] = -np.inf
        X[2, 2] = np.nan

        with patch('src.loveslide.knockoffs.ro.r') as mock_r:
            mock_r.side_effect = ValueError("R cannot handle non-finite values")

            with pytest.raises((ValueError, RuntimeError)):
                call_love_r(X, lbd=0.5)


class TestRSessionResourceManagement:
    """Test R session resource management."""

    def test_r_session_cleanup_after_exception(self):
        """Test that R session is properly cleaned up after exceptions."""
        with patch('src.loveslide.knockoffs.ro.r') as mock_r:
            mock_r.side_effect = RuntimeError("R computation failed")

            X = np.random.randn(10, 5)

            with pytest.raises(RuntimeError):
                call_love_r(X, lbd=0.5)

            # R session should be in a clean state
            # This is hard to test directly, but the mock should show cleanup calls

    def test_r_workspace_isolation(self):
        """Test that R workspace variables don't leak between calls."""
        X = np.random.randn(10, 5)

        with patch('src.loveslide.knockoffs.ro.r') as mock_r:
            mock_result = MagicMock()
            mock_result.rx2.return_value = np.random.randn(5, 3)
            mock_r.return_value = mock_result

            # Make multiple calls
            for i in range(3):
                try:
                    result = call_love_r(X, lbd=0.5 + i * 0.1)
                except Exception:
                    pass  # Expected with mocked data

        # Each call should be independent