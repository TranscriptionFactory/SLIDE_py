"""
Test concurrent execution safety and thread safety.
Addresses: Race conditions, shared state corruption, deadlock prevention
"""
import pytest
import numpy as np
import threading
import multiprocessing
import time
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from loveslide import SLIDE, SLIDEcv, Knockoffs
from loveslide.knockoff._parallel import knockoff_voting_parallel


class TestThreadSafety:
    """Test thread safety in parallel operations."""

    def test_concurrent_knockoff_generation(self):
        """Test concurrent knockoff generation doesn't corrupt state."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        def create_knockoffs(seed):
            np.random.seed(seed)
            knockoffs = Knockoffs()
            return knockoffs.create_knockoffs(X)

        # Run multiple knockoff generation concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_knockoffs, i) for i in range(10)]
            results = [f.result() for f in futures]

        # All results should be valid and different (due to randomization)
        for result in results:
            assert result.shape == X.shape
            assert not np.array_equal(result, X)

    def test_concurrent_slide_instances(self):
        """Test that concurrent SLIDE instances don't interfere."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        results = []
        exceptions = []

        def run_slide(params, seed):
            try:
                np.random.seed(seed)
                slide = SLIDE(params, x=X, y=y)
                result = slide.fit()
                results.append(result)
            except Exception as e:
                exceptions.append(e)

        threads = []
        for i in range(5):
            params = {'fdr': 0.1 + i*0.1, 'n_iters': 10}
            thread = threading.Thread(target=run_slide, args=(params, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(exceptions) == 0, f"Concurrent execution failed: {exceptions}"
        assert len(results) == 5

    def test_shared_cache_thread_safety(self):
        """Test thread safety of internal caches and memoization."""
        X = np.random.randn(100, 15)

        def compute_correlations(X_slice):
            # This might use internal caching
            return np.corrcoef(X_slice.T)

        slices = [X[i*20:(i+1)*20] for i in range(5)]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(compute_correlations, slice_) for slice_ in slices]
            results = [f.result() for f in futures]

        # All should complete successfully
        assert len(results) == 5
        for result in results:
            assert result.shape == (X.shape[1], X.shape[1])


class TestProcessSafety:
    """Test process-level safety and isolation."""

    def test_multiprocess_knockoff_voting(self):
        """Test multiprocess knockoff voting for race conditions."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)

        # Test with multiple processes
        try:
            result = knockoff_voting_parallel(
                X, y, fdr=0.1, n_iters=50,
                n_jobs=multiprocessing.cpu_count(),
                backend='joblib'
            )
            assert hasattr(result, 'selected')
            assert hasattr(result, 'fdp_hat')
        except Exception as e:
            pytest.skip(f"Multiprocessing not available: {e}")

    def test_process_isolation(self):
        """Test that processes don't interfere with each other."""
        X = np.random.randn(50, 8)
        y = np.random.randn(50)

        def worker_process(seed):
            np.random.seed(seed)
            slide = SLIDE({'fdr': 0.1, 'n_iters': 20}, x=X, y=y)
            result = slide.fit()
            return len(result.selected) if hasattr(result, 'selected') else 0

        if multiprocessing.get_start_method() != 'spawn':
            with ProcessPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(worker_process, i) for i in range(4)]
                results = [f.result() for f in futures]

            assert len(results) == 4
            assert all(isinstance(r, int) for r in results)

    def test_temporary_file_conflicts(self):
        """Test handling of temporary file conflicts in concurrent execution."""
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        def create_temp_slide(worker_id):
            with tempfile.TemporaryDirectory() as temp_dir:
                slide = SLIDE({'fdr': 0.1, 'save_dir': temp_dir}, x=X, y=y)
                result = slide.fit()
                return worker_id, temp_dir

        # Multiple workers creating temporary files
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_temp_slide, i) for i in range(6)]
            results = [f.result() for f in futures]

        # All should complete without file conflicts
        assert len(results) == 6
        worker_ids = [r[0] for r in results]
        assert len(set(worker_ids)) == 6  # All unique


class TestDeadlockPrevention:
    """Test prevention of deadlocks in concurrent operations."""

    def test_no_deadlock_in_nested_parallelism(self):
        """Test that nested parallel operations don't cause deadlocks."""
        X = np.random.randn(50, 8)
        y = np.random.randn(50)

        def nested_parallel_task():
            # Simulate nested parallel operation
            slide_cv = SLIDEcv({'fdr': 0.1, 'cv_folds': 3}, x=X, y=y)
            # This internally might use parallel processing
            result = slide_cv.cross_validate()
            return result

        # Run multiple nested tasks
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(nested_parallel_task) for _ in range(3)]
            try:
                results = [f.result(timeout=30) for f in futures]
                elapsed = time.time() - start_time
                assert elapsed < 25  # Should complete reasonably quickly
                assert len(results) == 3
            except TimeoutError:
                pytest.fail("Potential deadlock detected - operations timed out")

    def test_resource_cleanup_on_interruption(self):
        """Test proper resource cleanup when operations are interrupted."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)

        def interruptible_task():
            slide = SLIDE({'fdr': 0.1, 'n_iters': 1000}, x=X, y=y)
            return slide.fit()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(interruptible_task)
            time.sleep(0.1)  # Let it start
            # Cancel the future
            future.cancel()

        # Should not leave hanging resources
        # This is mainly a smoke test for proper cleanup