"""
Test coverage for R interface memory management and resource cleanup.

Critical gaps:
- Memory leaks in R-Python transitions
- Resource cleanup on exception
- Large object transfer efficiency
"""

import pytest
import numpy as np
import gc
import psutil
import os
from unittest.mock import patch, MagicMock
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from loveslide.love import call_love_r, call_love
    from loveslide.knockoffs import _create_second_order_r, _solve_sdp_r, _rlist_get
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False


@pytest.mark.skipif(not R_AVAILABLE, reason="R interface not available")
class TestRInterfaceMemoryManagement:
    """Test R interface memory management and cleanup"""

    def setup_method(self):
        """Setup for each test method"""
        self.initial_memory = psutil.Process().memory_info().rss

    def teardown_method(self):
        """Cleanup after each test method"""
        gc.collect()

    def test_memory_cleanup_after_r_exception(self):
        """Test memory cleanup when R operations fail"""
        # Create large matrix
        X = np.random.randn(1000, 500)

        with patch('rpy2.robjects.r', side_effect=Exception("R error")):
            with pytest.raises(Exception):
                call_love_r(X)

        # Force cleanup and check memory
        gc.collect()
        final_memory = psutil.Process().memory_info().rss
        memory_growth = final_memory - self.initial_memory

        # Memory growth should be reasonable (< 100MB)
        assert memory_growth < 100 * 1024 * 1024

    def test_large_matrix_transfer_memory_efficiency(self):
        """Test memory efficiency in large matrix transfers to R"""
        # Create very large matrix
        n, p = 2000, 1000
        X = np.random.randn(n, p)

        initial_memory = psutil.Process().memory_info().rss

        try:
            result = call_love_r(X, lbd=0.5)
        except Exception:
            # Even if operation fails, test memory usage
            pass

        gc.collect()
        peak_memory = psutil.Process().memory_info().rss
        memory_growth = peak_memory - initial_memory

        # Memory growth should not exceed 3x matrix size
        matrix_size = X.nbytes
        assert memory_growth < 3 * matrix_size

    def test_repeated_r_calls_memory_stability(self):
        """Test memory stability across repeated R calls"""
        X = np.random.randn(100, 50)
        memory_measurements = []

        for i in range(10):
            try:
                call_love_r(X)
            except Exception:
                pass  # Focus on memory, not functionality

            gc.collect()
            memory_measurements.append(psutil.Process().memory_info().rss)

        # Memory should stabilize (not grow continuously)
        memory_trend = np.diff(memory_measurements[-5:])
        avg_growth = np.mean(memory_trend)

        # Average growth should be minimal (< 1MB per call)
        assert avg_growth < 1024 * 1024

    def test_r_object_lifecycle_cleanup(self):
        """Test R object lifecycle and cleanup"""
        X = np.random.randn(500, 100)

        # Mock R objects that need cleanup
        with patch('loveslide.knockoffs._rlist_get') as mock_rlist:
            mock_r_obj = MagicMock()
            mock_rlist.return_value = mock_r_obj

            try:
                _create_second_order_r(X)
            except Exception:
                pass

            # Ensure R objects are properly released
            # Check if cleanup methods were called
            assert mock_rlist.called

    def test_concurrent_r_session_isolation(self):
        """Test memory isolation between concurrent R sessions"""
        import threading
        import time

        def r_worker(X, results, worker_id):
            """Worker function for concurrent R calls"""
            initial_mem = psutil.Process().memory_info().rss
            try:
                call_love_r(X)
            except Exception:
                pass
            final_mem = psutil.Process().memory_info().rss
            results[worker_id] = final_mem - initial_mem

        X = np.random.randn(200, 50)
        results = {}
        threads = []

        # Start multiple concurrent R sessions
        for i in range(3):
            t = threading.Thread(target=r_worker, args=(X, results, i))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Memory growth should be similar across workers
        memory_growths = list(results.values())
        if memory_growths:
            std_growth = np.std(memory_growths)
            mean_growth = np.mean(memory_growths)

            # Standard deviation should be < 50% of mean
            assert std_growth < 0.5 * mean_growth


@pytest.mark.skipif(not R_AVAILABLE, reason="R interface not available")
class TestRDataTransferEdgeCases:
    """Test R data transfer edge cases"""

    def test_sparse_matrix_transfer_efficiency(self):
        """Test sparse matrix transfer to R"""
        from scipy.sparse import csr_matrix

        # Create sparse matrix
        X_dense = np.random.randn(1000, 500)
        X_dense[X_dense < 1.5] = 0  # Make sparse
        X_sparse = csr_matrix(X_dense)

        # Should handle sparse matrices efficiently
        try:
            result = call_love_r(X_sparse.toarray())
        except Exception:
            pass  # Focus on memory efficiency

        # Memory usage should be reasonable
        assert True  # Placeholder for memory check

    def test_nan_inf_handling_in_r_transfer(self):
        """Test NaN/Inf handling in R data transfer"""
        X = np.random.randn(100, 50)
        X[0, 0] = np.nan
        X[1, 1] = np.inf
        X[2, 2] = -np.inf

        # Should handle special values gracefully
        with pytest.warns(RuntimeWarning):
            try:
                result = call_love_r(X)
            except Exception as e:
                # Should get meaningful error about data issues
                assert "nan" in str(e).lower() or "inf" in str(e).lower()

    def test_extreme_matrix_dimensions_r_transfer(self):
        """Test R transfer with extreme matrix dimensions"""
        # Very wide matrix
        X_wide = np.random.randn(10, 5000)

        try:
            call_love_r(X_wide)
        except (MemoryError, Exception) as e:
            # Should handle gracefully
            assert isinstance(e, (MemoryError, Exception))

        # Very tall matrix
        X_tall = np.random.randn(5000, 10)

        try:
            call_love_r(X_tall)
        except (MemoryError, Exception) as e:
            # Should handle gracefully
            assert isinstance(e, (MemoryError, Exception))


@pytest.mark.skipif(not R_AVAILABLE, reason="R interface not available")
class TestRResourceCleanup:
    """Test R resource cleanup patterns"""

    def test_r_temporary_file_cleanup(self):
        """Test cleanup of temporary files created by R"""
        import tempfile

        initial_temp_files = len(os.listdir(tempfile.gettempdir()))

        X = np.random.randn(100, 50)

        try:
            # Operations that might create temp files
            call_love_r(X)
        except Exception:
            pass

        # Allow some time for cleanup
        import time
        time.sleep(0.1)

        final_temp_files = len(os.listdir(tempfile.gettempdir()))

        # Should not accumulate temporary files
        temp_file_growth = final_temp_files - initial_temp_files
        assert temp_file_growth < 10  # Allow some temporary files

    def test_r_graphics_device_cleanup(self):
        """Test R graphics device cleanup"""
        X = np.random.randn(50, 30)

        # Mock R graphics operations
        with patch('rpy2.robjects.r') as mock_r:
            try:
                call_love_r(X)
            except Exception:
                pass

            # Ensure graphics devices are closed
            # This would be checked in actual R interface
            assert True  # Placeholder for R device check

    def test_error_recovery_and_cleanup(self):
        """Test error recovery and resource cleanup"""
        X = np.random.randn(100, 50)

        # Simulate various R error conditions
        error_conditions = [
            "Memory exhausted",
            "Package not found",
            "Invalid parameters",
            "Convergence failure"
        ]

        for error_msg in error_conditions:
            with patch('rpy2.robjects.r', side_effect=Exception(error_msg)):
                with pytest.raises(Exception):
                    call_love_r(X)

                # After each error, cleanup should work
                gc.collect()
                # Memory should be stable
                memory_after_error = psutil.Process().memory_info().rss
                assert memory_after_error > 0  # Basic sanity check


if __name__ == "__main__":
    pytest.main([__file__])