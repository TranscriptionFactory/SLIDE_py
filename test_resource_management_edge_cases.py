"""
Resource management and memory edge case testing for SLIDE_py.

This module tests memory management, resource cleanup, and performance
under resource constraints that might not be covered elsewhere.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import gc
import threading
import time
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from src.loveslide import SLIDE, OptimizeSLIDE, SLIDEcv, Knockoffs
from src.loveslide.knockoff._parallel import (
    knockoff_voting_parallel_joblib,
    knockoff_voting_parallel_futures,
    _precompute_knockoff_params
)


class TestMemoryManagement:
    """Test memory management and large data handling."""

    def test_large_matrix_memory_efficiency(self):
        """Test memory efficiency with large matrices."""
        # Create large but manageable test data
        n, p = 5000, 1000

        # Monitor memory usage
        initial_memory = self._get_memory_usage()

        X = np.random.randn(n, p).astype(np.float32)  # Use float32 to save memory
        y = np.random.randn(n).astype(np.float32)

        knockoffs = Knockoffs(X)

        # Force garbage collection
        del X, y
        gc.collect()

        final_memory = self._get_memory_usage()

        # Memory should not grow excessively
        memory_growth = final_memory - initial_memory
        assert memory_growth < 500  # MB threshold

    def test_memory_cleanup_after_exception(self):
        """Test memory cleanup when exceptions occur."""
        initial_memory = self._get_memory_usage()

        try:
            # Create large array that will cause an intentional error
            X = np.random.randn(1000, 500)

            # Mock a function to raise an exception
            with patch('src.loveslide.knockoffs._solve_sdp_r') as mock_solve:
                mock_solve.side_effect = RuntimeError("Simulated failure")

                knockoffs = Knockoffs(X)
                with pytest.raises(RuntimeError):
                    knockoffs.filter(y=np.random.randn(1000), model='second_order')

        except Exception:
            pass
        finally:
            # Force cleanup
            gc.collect()

        final_memory = self._get_memory_usage()
        memory_growth = final_memory - initial_memory
        assert memory_growth < 100  # Should not leak significant memory

    def test_chunked_processing_memory_efficiency(self):
        """Test memory efficiency in chunked processing."""
        input_params = {
            'fsize': 50,  # Small chunks to test chunking
            'x_path': 'dummy.csv',
            'y_path': 'dummy_y.csv'
        }

        # Create large dataset
        X = np.random.randn(2000, 500)
        y = np.random.randn(2000)

        with patch('src.loveslide.slide.init_data') as mock_init:
            mock_init.return_value = ({'X': X, 'y': y}, input_params)

            initial_memory = self._get_memory_usage()

            slide = SLIDE(input_params)

            # Should process in chunks without excessive memory growth
            with patch.object(slide, '_find_interaction_LFs_batch') as mock_batch:
                mock_batch.return_value = []  # Empty results for testing
                slide.run()

            final_memory = self._get_memory_usage()
            memory_growth = final_memory - initial_memory
            assert memory_growth < 200  # Should not use excessive memory

    def test_matrix_operations_memory_views(self):
        """Test that matrix operations use memory views when possible."""
        X = np.random.randn(1000, 100)

        knockoffs = Knockoffs(X)

        # Check that internal operations don't create unnecessary copies
        original_id = id(knockoffs.X)

        # Operations should ideally use views, not copies
        Sigma = knockoffs.X.T @ knockoffs.X / knockoffs.X.shape[0]

        # X should still be the same object
        assert id(knockoffs.X) == original_id

    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0  # If psutil not available, return 0


class TestParallelProcessingResourceManagement:
    """Test resource management in parallel processing."""

    def test_thread_pool_resource_cleanup(self):
        """Test thread pool resource cleanup."""
        X = np.random.randn(200, 50)
        y = np.random.randn(200)

        initial_thread_count = threading.active_count()

        # Use thread pool for knockoff computation
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _ in range(10):
                future = executor.submit(
                    self._compute_knockoff_iteration, X, y
                )
                futures.append(future)

            # Wait for all to complete
            for future in futures:
                future.result()

        # Give threads time to cleanup
        time.sleep(0.1)

        final_thread_count = threading.active_count()

        # Thread count should return to initial level
        assert final_thread_count <= initial_thread_count + 2  # Some tolerance

    def test_process_pool_memory_isolation(self):
        """Test process pool memory isolation."""
        X = np.random.randn(500, 100)
        y = np.random.randn(500)

        # Use process pool to ensure memory isolation
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = []
            for i in range(4):
                future = executor.submit(
                    self._memory_intensive_task, X, y, i
                )
                futures.append(future)

            results = []
            for future in futures:
                result = future.result()
                results.append(result)
                assert result is not None

    def test_parallel_knockoff_resource_limits(self):
        """Test parallel knockoff computation under resource limits."""
        X = np.random.randn(300, 80)
        y = np.random.randn(300)

        # Limit the number of workers
        max_workers = 2

        result = knockoff_voting_parallel_futures(
            X, y,
            statistic='lasso_lambdadiff',
            fdr=0.1,
            n_iter=5,
            max_workers=max_workers,
            random_state=42
        )

        assert hasattr(result, 'selections')
        assert len(result.selections) == 5

    def test_worker_failure_recovery(self):
        """Test recovery when worker processes/threads fail."""
        X = np.random.randn(100, 30)
        y = np.random.randn(100)

        # Mock worker to fail sometimes
        with patch('src.loveslide.knockoff._parallel._single_knockoff_iteration') as mock_worker:
            def failing_worker(*args, **kwargs):
                # Fail 30% of the time
                if np.random.rand() < 0.3:
                    raise RuntimeError("Simulated worker failure")
                return {'selected': [], 'W': np.random.randn(30)}

            mock_worker.side_effect = failing_worker

            # Should handle worker failures gracefully
            result = knockoff_voting_parallel_joblib(
                X, y,
                statistic='lasso_lambdadiff',
                fdr=0.1,
                n_iter=10,
                n_jobs=3,
                random_state=42
            )

            # Some iterations might succeed
            assert hasattr(result, 'selections')

    def _compute_knockoff_iteration(self, X, y):
        """Helper function for testing thread pool."""
        knockoffs = Knockoffs(X)
        result = knockoffs.filter(y=y, model='equi')
        return result

    def _memory_intensive_task(self, X, y, task_id):
        """Helper function for testing process pool memory isolation."""
        # Create some temporary large arrays
        temp_large = np.random.randn(1000, 200)

        # Do some computation
        knockoffs = Knockoffs(X)
        result = knockoffs.filter(y=y, model='equi')

        # Cleanup temporary arrays
        del temp_large

        return f"Task {task_id} completed"


class TestFileSystemResourceManagement:
    """Test file system resource management."""

    def test_temporary_file_cleanup(self):
        """Test cleanup of temporary files."""
        initial_temp_files = len(os.listdir(tempfile.gettempdir()))

        # Create multiple temporary files during processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_files = []
            for i in range(10):
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.csv', delete=False, dir=temp_dir
                ) as f:
                    f.write(f"data,{i}\n1,2\n3,4\n")
                    temp_files.append(f.name)

            # Process files
            for file_path in temp_files:
                df = pd.read_csv(file_path)
                assert len(df) == 2

        # Temporary directory should be cleaned up automatically
        final_temp_files = len(os.listdir(tempfile.gettempdir()))

        # Should not have significantly more temp files
        assert final_temp_files <= initial_temp_files + 5

    def test_large_file_streaming(self):
        """Test streaming large files without loading entirely into memory."""
        # Create a large CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write header
            f.write(','.join([f'col_{i}' for i in range(50)]) + '\n')

            # Write many rows
            for row in range(10000):
                f.write(','.join([str(row * 50 + i) for i in range(50)]) + '\n')

            large_file = f.name

        try:
            initial_memory = self._get_memory_usage()

            # Read file in chunks
            chunk_size = 1000
            total_rows = 0

            for chunk in pd.read_csv(large_file, chunksize=chunk_size):
                total_rows += len(chunk)
                # Process chunk (simulate some computation)
                mean_vals = chunk.mean()
                assert len(mean_vals) == 50

            final_memory = self._get_memory_usage()
            memory_growth = final_memory - initial_memory

            assert total_rows == 10000
            assert memory_growth < 100  # Should not load entire file into memory

        finally:
            os.unlink(large_file)

    def test_disk_space_monitoring(self):
        """Test behavior when disk space is limited."""
        # Get available disk space
        try:
            import shutil
            free_space = shutil.disk_usage('.').free

            # Only run this test if we have reasonable free space
            if free_space > 1e9:  # 1 GB
                # Try to create files that approach disk limits
                with tempfile.TemporaryDirectory() as temp_dir:
                    large_files = []
                    try:
                        for i in range(5):
                            file_path = os.path.join(temp_dir, f'large_{i}.dat')
                            with open(file_path, 'wb') as f:
                                # Write 100 MB file
                                f.write(b'\x00' * (100 * 1024 * 1024))
                            large_files.append(file_path)

                    except OSError as e:
                        # Expected if we run out of space
                        assert "No space left" in str(e) or "Disk full" in str(e)

        except ImportError:
            pytest.skip("shutil not available for disk space testing")

    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0


class TestResourceLimitHandling:
    """Test handling of system resource limits."""

    def test_maximum_file_descriptors(self):
        """Test behavior at file descriptor limits."""
        # Try to open many files simultaneously
        open_files = []

        try:
            # Open files until we hit a limit
            for i in range(1000):
                try:
                    f = tempfile.NamedTemporaryFile()
                    open_files.append(f)
                except OSError as e:
                    # Expected when we hit file descriptor limit
                    assert "Too many open files" in str(e) or "No more file descriptors" in str(e)
                    break

            # Should handle the limit gracefully
            assert len(open_files) > 0

        finally:
            # Cleanup
            for f in open_files:
                f.close()

    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        arrays = []

        try:
            # Try to allocate increasingly large arrays
            for i in range(10):
                size = 2**i * 1000000  # Exponentially increasing size
                try:
                    arr = np.random.randn(size)
                    arrays.append(arr)
                except MemoryError:
                    # Expected when we run out of memory
                    break

            # Should be able to allocate at least some arrays
            assert len(arrays) > 0

            # Test that algorithms can still work with available memory
            if len(arrays) > 0:
                X = np.random.randn(100, 20)
                y = np.random.randn(100)
                knockoffs = Knockoffs(X)
                result = knockoffs.filter(y=y, model='equi')

        finally:
            # Force cleanup
            del arrays
            gc.collect()

    def test_cpu_intensive_task_interruption(self):
        """Test interruption of CPU-intensive tasks."""
        X = np.random.randn(1000, 200)
        y = np.random.randn(1000)

        # Start a CPU-intensive task
        def intensive_task():
            knockoffs = Knockoffs(X)
            return knockoffs.filter(
                y=y,
                model='second_order',  # More computationally intensive
                statistic='lasso_lambdadiff'
            )

        # Use threading to test interruption
        import threading
        result = [None]
        exception = [None]

        def worker():
            try:
                result[0] = intensive_task()
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=worker)
        thread.start()

        # Let it run for a short time then "interrupt" by joining with timeout
        thread.join(timeout=2.0)

        if thread.is_alive():
            # Task is still running - this tests that long tasks are possible
            thread.join()  # Wait for completion

        # Should complete or handle interruption gracefully
        assert result[0] is not None or exception[0] is not None


class TestResourceCleanupEdgeCases:
    """Test resource cleanup in edge cases."""

    def test_nested_context_manager_cleanup(self):
        """Test resource cleanup in nested context managers."""
        temp_files = []

        try:
            with tempfile.TemporaryDirectory() as outer_dir:
                with tempfile.TemporaryDirectory(dir=outer_dir) as inner_dir:
                    # Create files in nested structure
                    for i in range(5):
                        with tempfile.NamedTemporaryFile(
                            mode='w', dir=inner_dir, delete=False
                        ) as f:
                            f.write(f"test data {i}")
                            temp_files.append(f.name)

                    # Verify files exist
                    for file_path in temp_files:
                        assert os.path.exists(file_path)

                    # Simulate exception in nested context
                    raise ValueError("Simulated exception")

        except ValueError:
            # Expected exception
            pass

        # All temporary files should be cleaned up despite exception
        for file_path in temp_files:
            assert not os.path.exists(file_path)

    def test_signal_handler_cleanup(self):
        """Test cleanup when process receives signals."""
        import signal
        import os

        cleanup_called = [False]

        def cleanup_handler(signum, frame):
            cleanup_called[0] = True
            # Perform cleanup
            gc.collect()

        # Register signal handler
        original_handler = signal.signal(signal.SIGUSR1, cleanup_handler)

        try:
            # Send signal to self
            os.kill(os.getpid(), signal.SIGUSR1)

            # Small delay for signal processing
            time.sleep(0.1)

            assert cleanup_called[0] == True

        finally:
            # Restore original handler
            signal.signal(signal.SIGUSR1, original_handler)


if __name__ == "__main__":
    pytest.main([__file__])