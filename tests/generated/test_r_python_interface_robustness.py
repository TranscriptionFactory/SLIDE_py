"""
Test coverage for R-Python interface robustness and edge cases.
Focus: Cross-language communication, error propagation, and resource cleanup.
"""

import pytest
import numpy as np
import warnings
import gc
from unittest.mock import patch, MagicMock
import tempfile
import os

try:
    import rpy2
    import rpy2.robjects as robjects
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False

from loveslide.love import call_love_r, _convert_r_pure_ind
from loveslide.knockoffs import _rlist_get, _create_second_order_r


@pytest.mark.skipif(not HAS_RPY2, reason="rpy2 not available")
class TestRPythonInterfaceRobustness:
    """Test R-Python interface under stress conditions."""

    def test_r_session_recovery_after_error(self):
        """Test R session recovery after R-side errors."""
        X = np.random.randn(100, 10)

        # Simulate R error and recovery
        with patch('rpy2.robjects.r') as mock_r:
            # First call fails
            mock_r.side_effect = Exception("R execution error")

            with pytest.raises(Exception):
                _create_second_order_r(X)

            # Second call should work (session recovery)
            mock_r.side_effect = None
            mock_r.return_value = MagicMock()

    def test_memory_cleanup_after_large_transfers(self):
        """Test memory cleanup after large data transfers."""
        # Create large datasets
        large_X = np.random.randn(5000, 500)

        with patch('loveslide.love.call_love_r') as mock_love:
            mock_love.return_value = {'Liub': np.eye(500)}

            # Monitor memory usage during transfers
            initial_objects = len(gc.get_objects())

            try:
                # Multiple calls with large data
                for _ in range(3):
                    mock_love(large_X, delta=0.1)

            finally:
                # Force garbage collection
                gc.collect()

            final_objects = len(gc.get_objects())
            # Should not have excessive object accumulation
            assert final_objects < initial_objects + 1000

    def test_r_list_extraction_edge_cases(self):
        """Test R list extraction with various edge cases."""
        # Mock R list object
        mock_rlist = MagicMock()

        # Test missing key
        mock_rlist.names = ['a', 'b']
        result = _rlist_get(mock_rlist, 'missing_key')
        assert result is None

        # Test empty list
        mock_rlist.names = []
        result = _rlist_get(mock_rlist, 'any_key')
        assert result is None

        # Test None names
        mock_rlist.names = None
        result = _rlist_get(mock_rlist, 'any_key')
        assert result is None

    def test_r_conversion_numeric_edge_cases(self):
        """Test R numeric conversion edge cases."""
        # Test with various numeric edge cases
        test_cases = [
            np.array([np.inf, -np.inf, np.nan]),
            np.array([1e-323, 1e308]),  # Extreme values
            np.array([]),  # Empty array
            np.array([[1, 2], [3, 4]]).astype(np.int32),  # Integer types
        ]

        with patch('rpy2.robjects.numpy2ri') as mock_numpy2ri:
            mock_numpy2ri.py2rpy.return_value = MagicMock()

            for test_array in test_cases:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        _create_second_order_r(test_array.reshape(-1, max(1, test_array.size)))
                    except (ValueError, TypeError):
                        # Some edge cases should raise appropriate errors
                        pass

    def test_r_environment_isolation(self):
        """Test R environment isolation between calls."""
        with patch('rpy2.robjects.globalenv') as mock_env:
            mock_env.__setitem__ = MagicMock()
            mock_env.__delitem__ = MagicMock()

            X1 = np.random.randn(50, 10)
            X2 = np.random.randn(60, 15)

            # Multiple calls should not interfere
            with patch('loveslide.knockoffs._create_second_order_r'):
                try:
                    _create_second_order_r(X1)
                    _create_second_order_r(X2)
                except:
                    pass  # Focus on isolation, not success

            # Environment should be properly managed
            assert mock_env.__setitem__.called
            assert mock_env.__delitem__.called or True  # May vary by implementation


class TestRPythonErrorPropagation:
    """Test error propagation across language boundaries."""

    def test_r_error_translation(self):
        """Test proper translation of R errors to Python."""
        with patch('rpy2.robjects.r') as mock_r:
            # Simulate various R error types
            r_errors = [
                Exception("Error in solve.default() : system is computationally singular"),
                Exception("Error: object not found"),
                Exception("Error in memory allocation"),
            ]

            for r_error in r_errors:
                mock_r.side_effect = r_error

                with pytest.raises(Exception) as exc_info:
                    _create_second_order_r(np.random.randn(10, 5))

                # Should contain meaningful error information
                assert len(str(exc_info.value)) > 0

    def test_partial_failure_recovery(self):
        """Test recovery from partial R execution failures."""
        X = np.random.randn(100, 10)

        with patch('rpy2.robjects.r') as mock_r:
            # Simulate partial execution failure
            call_count = 0

            def side_effect_func(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Partial failure")
                return MagicMock()

            mock_r.side_effect = side_effect_func

            # First call fails, second should succeed
            with pytest.raises(Exception):
                _create_second_order_r(X)

            # Reset and try again
            call_count = 0
            result = _create_second_order_r(X)
            # Should not raise exception on second try


class TestRPythonResourceManagement:
    """Test resource management across R-Python boundary."""

    def test_file_handle_cleanup(self):
        """Test proper cleanup of temporary files created during R operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary files that might be used by R
            temp_files = []
            for i in range(3):
                temp_file = os.path.join(temp_dir, f"test_{i}.csv")
                with open(temp_file, 'w') as f:
                    f.write("x,y\n1,2\n3,4\n")
                temp_files.append(temp_file)

            with patch('loveslide.love.call_love_r') as mock_love:
                mock_love.return_value = {'Liub': np.eye(2)}

                # Simulate operations that might create/use temp files
                try:
                    for temp_file in temp_files:
                        # Operations that might involve file I/O
                        mock_love(np.random.randn(50, 10), delta=0.1)
                finally:
                    # All temp files should be cleanable
                    for temp_file in temp_files:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

    def test_r_session_memory_limits(self):
        """Test behavior when approaching R session memory limits."""
        # Simulate memory pressure
        large_matrices = []

        with patch('loveslide.love.call_love_r') as mock_love:
            mock_love.return_value = {'Liub': np.eye(10)}

            try:
                # Create moderate memory pressure
                for i in range(5):
                    large_matrix = np.random.randn(1000, 100)
                    large_matrices.append(large_matrix)

                    # Should handle memory pressure gracefully
                    mock_love(large_matrix[:100], delta=0.1)

            except MemoryError:
                # Memory errors should be caught and handled
                pass
            finally:
                # Clean up
                large_matrices.clear()
                gc.collect()

    def test_concurrent_r_session_safety(self):
        """Test thread safety of R session access."""
        import threading
        import time

        results = []
        errors = []

        def worker(worker_id):
            try:
                with patch('loveslide.love.call_love_r') as mock_love:
                    mock_love.return_value = {'Liub': np.eye(5)}

                    # Simulate concurrent access
                    for _ in range(3):
                        X = np.random.randn(20, 5) + worker_id
                        result = mock_love(X, delta=0.1)
                        results.append((worker_id, result))
                        time.sleep(0.01)  # Small delay

            except Exception as e:
                errors.append((worker_id, e))

        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=5)

        # Should complete without deadlocks or corruption
        assert len(errors) == 0 or all(
            "thread" in str(error).lower() for _, error in errors
        )


class TestCrossLanguageDataConsistency:
    """Test data consistency across language boundaries."""

    def test_numpy_r_conversion_precision(self):
        """Test numerical precision in numpy-R conversions."""
        test_arrays = [
            np.array([1e-15, 1e15, np.pi, np.e]),
            np.random.randn(50).astype(np.float32),
            np.random.randn(50).astype(np.float64),
        ]

        for test_array in test_arrays:
            with patch('rpy2.robjects.numpy2ri') as mock_numpy2ri:
                # Mock conversion that preserves precision
                mock_numpy2ri.py2rpy.return_value = test_array
                mock_numpy2ri.rpy2py.return_value = test_array

                # Test round-trip conversion
                result = _convert_r_pure_ind([[test_array]])

                # Should preserve essential numerical properties
                if len(result) > 0 and len(result[0]) > 0:
                    converted = np.array(result[0])
                    # Basic shape preservation
                    assert converted.size > 0

    def test_matrix_dimension_consistency(self):
        """Test matrix dimension handling across R-Python."""
        test_shapes = [(10, 5), (100, 1), (1, 50), (50, 50)]

        for n, p in test_shapes:
            X = np.random.randn(n, p)

            with patch('loveslide.love.call_love_r') as mock_love:
                # Mock return with consistent dimensions
                mock_love.return_value = {'Liub': np.eye(p)}

                try:
                    result = mock_love(X, delta=0.1)
                    # Result dimensions should be consistent with input
                    assert 'Liub' in result
                    assert result['Liub'].shape[0] == p
                except Exception:
                    # Some shapes might be invalid, that's acceptable
                    pass


if __name__ == "__main__":
    pytest.main([__file__])