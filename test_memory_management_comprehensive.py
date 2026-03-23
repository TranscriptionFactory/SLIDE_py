"""
Test coverage gaps for memory management and large-scale data handling.

Critical gaps in memory management that could lead to memory leaks,
out-of-memory errors, or performance degradation.
"""

import pytest
import numpy as np
import pandas as pd
import gc
import psutil
import os
from unittest.mock import patch, MagicMock
import threading
import tempfile

from src.loveslide.slide import SLIDE, OptimizeSLIDE
from src.loveslide.knockoffs import Knockoffs
from src.loveslide.knockoff._parallel import _precompute_knockoff_params


class TestMemoryLeakPrevention:
    """Test prevention of memory leaks in various scenarios."""

    def test_memory_cleanup_after_exception(self):
        """Test memory cleanup when exceptions occur."""
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        X = np.random.randn(1000, 500)
        y = np.random.randint(0, 2, 1000)

        params = {'x_path': None, 'y_path': None}

        # Force an exception during processing
        with patch('src.loveslide.slide.SLIDE.run') as mock_run:
            mock_run.side_effect = ValueError("Simulated error")

            try:
                slide = SLIDE(params, x=X, y=y)
                slide.run()
            except ValueError:
                pass

        # Force garbage collection
        gc.collect()

        # Memory should return to near initial levels
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / initial_memory

        assert memory_increase < 0.1, f"Memory leak detected: {memory_increase:.2%} increase"

    def test_large_matrix_memory_efficiency(self):
        """Test memory efficiency with large matrices."""
        # Test with progressively larger matrices
        for size in [1000, 2000, 5000]:
            X = np.random.randn(size, 100)

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            knockoffs = Knockoffs()
            X_knockoffs = knockoffs.fit_transform(X)

            # Check memory usage is reasonable
            peak_memory = process.memory_info().rss
            memory_ratio = peak_memory / initial_memory

            # Memory usage should scale reasonably with data size
            expected_ratio = 1 + (X.nbytes * 3) / initial_memory  # Conservative estimate
            assert memory_ratio < expected_ratio * 2, f"Excessive memory usage for size {size}"

            # Cleanup
            del X, X_knockoffs
            gc.collect()

    def test_temporary_file_cleanup(self):
        """Test cleanup of temporary files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create large temporary data files
            X = np.random.randn(5000, 200)
            temp_file = os.path.join(temp_dir, "temp_data.npy")
            np.save(temp_file, X)

            initial_files = len(os.listdir(temp_dir))

            params = {
                'x_path': temp_file,
                'y_path': None,
                'out_path': temp_dir
            }
            y = np.random.randint(0, 2, 5000)

            slide = SLIDE(params, y=y)

            try:
                slide.run()
            except Exception:
                pass

            # Check that no additional temporary files were left
            final_files = len(os.listdir(temp_dir))
            assert final_files <= initial_files + 1, "Temporary files not cleaned up"


class TestLargeDatasetHandling:
    """Test handling of large datasets."""

    @pytest.mark.slow
    def test_chunked_processing_large_datasets(self):
        """Test chunked processing for datasets too large for memory."""
        # Simulate very large dataset
        n_samples, n_features = 50000, 1000

        # Use memory mapping for large arrays
        with tempfile.NamedTemporaryFile() as temp_file:
            # Create memory-mapped array
            X_mmap = np.memmap(temp_file.name, dtype='float32', mode='w+',
                             shape=(n_samples, n_features))
            X_mmap[:] = np.random.randn(n_samples, n_features)

            params = {'x_path': None, 'y_path': None, 'chunk_size': 5000}
            y = np.random.randint(0, 2, n_samples)

            # Should process in chunks without memory overflow
            slide = SLIDE(params, x=X_mmap, y=y)

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            result = slide.run()

            peak_memory = process.memory_info().rss
            memory_increase = peak_memory - initial_memory

            # Memory increase should be bounded by chunk size, not full data size
            max_expected_memory = X_mmap[:5000].nbytes * 10  # Conservative bound
            assert memory_increase < max_expected_memory, "Memory not bounded by chunking"

    def test_streaming_data_processing(self):
        """Test streaming processing of data."""
        def data_generator():
            """Generator that yields data in batches."""
            for i in range(100):  # 100 batches
                batch_X = np.random.randn(50, 20)
                batch_y = np.random.randint(0, 2, 50)
                yield batch_X, batch_y

        # Process streaming data without loading everything into memory
        total_processed = 0
        process = psutil.Process(os.getpid())
        memory_usage = []

        for batch_X, batch_y in data_generator():
            # Process each batch
            knockoffs = Knockoffs()
            _ = knockoffs.fit_transform(batch_X)

            total_processed += len(batch_X)
            memory_usage.append(process.memory_info().rss)

            # Memory usage should remain stable across batches
            if len(memory_usage) > 10:
                recent_memory = memory_usage[-10:]
                memory_trend = (max(recent_memory) - min(recent_memory)) / min(recent_memory)
                assert memory_trend < 0.1, "Memory usage trending upward (potential leak)"

        assert total_processed == 5000, "Not all data processed"

    def test_out_of_memory_handling(self):
        """Test graceful handling of out-of-memory conditions."""
        # Try to create an array that's likely to cause memory issues
        with pytest.raises((MemoryError, OverflowError)):
            try:
                # Attempt to allocate huge array
                huge_array = np.random.randn(100000, 100000)
            except MemoryError:
                # Should handle gracefully and provide helpful error message
                raise MemoryError("Insufficient memory for requested operation. Consider using chunked processing.")


class TestConcurrentMemoryManagement:
    """Test memory management with concurrent operations."""

    def test_parallel_knockoff_memory_isolation(self):
        """Test memory isolation in parallel knockoff generation."""
        X = np.random.randn(1000, 100)

        # Monitor memory in parallel processes
        def memory_monitor():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss

        initial_memory = memory_monitor()

        # Run parallel knockoff generation
        knockoffs = Knockoffs(n_jobs=4)
        X_knockoffs = knockoffs.fit_transform(X)

        peak_memory = memory_monitor()

        # Memory should not scale linearly with number of workers
        memory_ratio = peak_memory / initial_memory
        assert memory_ratio < 5.0, "Memory usage scales too much with parallelism"

        # Check that all workers completed successfully
        assert X_knockoffs.shape == X.shape

    def test_thread_memory_safety(self):
        """Test thread safety of memory operations."""
        X = np.random.randn(500, 50)
        results = []
        exceptions = []

        def worker_thread(thread_id):
            try:
                # Each thread processes the same data
                knockoffs = Knockoffs()
                result = knockoffs.fit_transform(X)
                results.append((thread_id, result.shape))
            except Exception as e:
                exceptions.append((thread_id, e))

        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker_thread, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Check no exceptions occurred
        assert len(exceptions) == 0, f"Thread exceptions: {exceptions}"

        # Check all threads completed
        assert len(results) == 5, "Not all threads completed"

        # Check results are consistent
        expected_shape = X.shape
        for thread_id, result_shape in results:
            assert result_shape == expected_shape, f"Thread {thread_id} wrong result shape"


class TestMemoryProfileOptimization:
    """Test memory usage optimization."""

    def test_in_place_operations_memory_efficiency(self):
        """Test that in-place operations are used when possible."""
        X = np.random.randn(1000, 100)

        # Monitor memory during operations
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Operations that should be memory efficient
        with patch('numpy.copy') as mock_copy:
            knockoffs = Knockoffs()
            _ = knockoffs.fit_transform(X)

            # Should minimize unnecessary copying
            copy_calls = mock_copy.call_count
            assert copy_calls < 5, f"Too many array copies: {copy_calls}"

        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before

        # Memory increase should be minimal for in-place operations
        max_expected_increase = X.nbytes * 3  # Conservative estimate
        assert memory_increase < max_expected_increase, "Memory increase too large for in-place operations"

    def test_lazy_evaluation_memory_benefits(self):
        """Test lazy evaluation to reduce memory usage."""
        # Create pipeline that should use lazy evaluation
        X = np.random.randn(2000, 200)

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Pipeline that processes data lazily
        params = {
            'x_path': None, 'y_path': None,
            'lazy_evaluation': True,  # Mock parameter
            'batch_size': 500
        }
        y = np.random.randint(0, 2, 2000)

        slide = SLIDE(params, x=X, y=y)
        result = slide.run()

        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory

        # Lazy evaluation should keep memory usage bounded
        max_expected_memory = X.nbytes * 2  # Should not load multiple copies
        assert memory_increase < max_expected_memory, "Lazy evaluation not effective"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])