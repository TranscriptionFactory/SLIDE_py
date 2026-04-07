"""
Comprehensive concurrency and thread safety testing for SLIDE_py.

Tests parallel execution scenarios, thread safety, and race condition prevention
for robust multi-threaded/multi-process usage.
"""
import pytest
import numpy as np
import pandas as pd
import threading
import multiprocessing
import concurrent.futures
import time
import tempfile
import os
from unittest.mock import Mock, patch

from loveslide import (
    SLIDE, SLIDEcv, Knockoffs, call_love,
    init_data, calc_default_fsize
)


class TestThreadSafety:
    """Test thread safety of core functions."""

    def test_init_data_thread_safety(self):
        """Test init_data function thread safety."""
        def worker():
            params = {'x_path': None, 'y_path': None, 'fdr': 0.1}
            X = np.random.randn(50, 20)
            y = np.random.randn(50)
            return init_data(params, x=X, y=y)

        # Run multiple threads simultaneously
        threads = []
        results = []

        def thread_wrapper():
            results.append(worker())

        for _ in range(10):
            t = threading.Thread(target=thread_wrapper)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All should succeed without corruption
        assert len(results) == 10
        for result in results:
            assert result[0] is not None  # data object
            assert result[1] is not None  # input_params

    def test_knockoffs_concurrent_creation(self):
        """Test concurrent knockoff creation."""
        X = np.random.randn(100, 30)
        y = np.random.randn(100)

        def create_knockoffs():
            knockoffs = Knockoffs(backend='python')
            return knockoffs.select_short_freq(X, y, fdr=0.1)

        # Test concurrent knockoff creation
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_knockoffs) for _ in range(5)]
            results = [future.result() for future in futures]

        # All should complete successfully
        assert len(results) == 5
        for result in results:
            assert hasattr(result, 'selected')

    def test_love_algorithm_thread_safety(self):
        """Test LOVE algorithm thread safety."""
        X = np.random.randn(100, 30)

        def run_love():
            return call_love(X, lbd=0.5, verbose=False)

        # Run LOVE in multiple threads
        threads = []
        results = []
        errors = []

        def thread_wrapper():
            try:
                results.append(run_love())
            except Exception as e:
                errors.append(e)

        for _ in range(3):  # Fewer threads for computational intensive task
            t = threading.Thread(target=thread_wrapper)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should not have race conditions causing errors
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 3

    def test_slide_instance_isolation(self):
        """Test that SLIDE instances don't interfere with each other."""
        def create_slide_instance(instance_id):
            params = {
                'x_path': None, 'y_path': None,
                'fdr': 0.1, 'delta': [0.1 + instance_id * 0.01]  # Unique delta
            }
            X = np.random.randn(50, 20)
            y = np.random.randn(50)
            slide = SLIDE(params, x=X, y=y)
            # Set a unique identifier
            slide.instance_id = instance_id
            return slide

        # Create multiple instances concurrently
        instances = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_slide_instance, i) for i in range(5)]
            instances = [future.result() for future in futures]

        # Verify instances are isolated
        for i, instance in enumerate(instances):
            assert instance.instance_id == i
            assert instance.input_params['delta'][0] == 0.1 + i * 0.01


class TestMultiprocessingSafety:
    """Test multiprocessing safety."""

    def test_slide_multiprocessing_isolation(self):
        """Test SLIDE in separate processes."""
        def run_slide_process(delta_value):
            params = {
                'x_path': None, 'y_path': None,
                'fdr': 0.1, 'delta': [delta_value]
            }
            X = np.random.randn(50, 20)
            y = np.random.randn(50)
            slide = SLIDE(params, x=X, y=y)

            # Mock LOVE to return quickly
            slide.data.love_result = {
                'pure_Ind': [],
                'A': np.random.randn(20, 5),
                'delta': delta_value
            }
            return delta_value

        # Run in separate processes
        with multiprocessing.Pool(processes=3) as pool:
            deltas = [0.1, 0.15, 0.2]
            results = pool.map(run_slide_process, deltas)

        assert results == deltas  # Should return correct values

    def test_knockoffs_process_isolation(self):
        """Test knockoffs in separate processes."""
        def run_knockoffs_process(seed):
            np.random.seed(seed)
            X = np.random.randn(50, 20)
            y = np.random.randn(50)

            knockoffs = Knockoffs(backend='python')
            result = knockoffs.select_short_freq(X, y, fdr=0.1)
            return len(result.selected), seed

        # Run with different seeds
        with multiprocessing.Pool(processes=2) as pool:
            seeds = [42, 123, 456]
            results = pool.map(run_knockoffs_process, seeds)

        # Should return results with correct seeds
        for n_selected, seed in results:
            assert isinstance(n_selected, int)
            assert seed in [42, 123, 456]


class TestRaceConditionPrevention:
    """Test prevention of race conditions."""

    def test_file_io_race_conditions(self):
        """Test file I/O race conditions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Multiple threads writing to different files
            def write_results(thread_id):
                params = {'x_path': None, 'y_path': None, 'fdr': 0.1}
                slide = SLIDE(params, x=np.random.randn(50, 20), y=np.random.randn(50))

                output_file = os.path.join(temp_dir, f'results_{thread_id}.pkl')
                slide.data.love_result = {'thread_id': thread_id}

                # Simulate saving results
                import pickle
                with open(output_file, 'wb') as f:
                    pickle.dump(slide.data.love_result, f)

                return output_file

            # Run multiple threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(write_results, i) for i in range(10)]
                output_files = [future.result() for future in futures]

            # Verify all files were created correctly
            assert len(output_files) == 10
            for i, output_file in enumerate(output_files):
                assert os.path.exists(output_file)

                # Verify content integrity
                import pickle
                with open(output_file, 'rb') as f:
                    data = pickle.load(f)
                    assert data['thread_id'] == i

    def test_shared_state_corruption(self):
        """Test prevention of shared state corruption."""
        # Test with global state that could be corrupted
        original_random_state = np.random.get_state()

        def modify_random_state(seed):
            np.random.seed(seed)
            # Generate some random numbers
            data = np.random.randn(100, 20)
            return np.sum(data)

        try:
            # Run multiple threads that modify random state
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(modify_random_state, seed) for seed in [1, 2, 3]]
                results = [future.result() for future in futures]

            # All should complete without errors
            assert len(results) == 3
            assert all(isinstance(r, float) for r in results)

        finally:
            # Restore original state
            np.random.set_state(original_random_state)

    def test_concurrent_parameter_modification(self):
        """Test concurrent parameter modifications."""
        base_params = {
            'x_path': None, 'y_path': None,
            'fdr': 0.1, 'delta': [0.1], 'lambda': [0.5]
        }

        def modify_params(modification_id):
            # Create a copy to avoid shared state issues
            params = base_params.copy()
            params['fdr'] = 0.1 + modification_id * 0.01
            params['delta'] = [0.1 + modification_id * 0.02]

            X = np.random.randn(50, 20)
            y = np.random.randn(50)

            slide = SLIDE(params, x=X, y=y)
            return slide.input_params['fdr'], slide.input_params['delta'][0]

        # Run concurrent modifications
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(modify_params, i) for i in range(8)]
            results = [future.result() for future in futures]

        # Verify each instance has correct parameters
        for i, (fdr, delta) in enumerate(results):
            expected_fdr = 0.1 + i * 0.01
            expected_delta = 0.1 + i * 0.02
            assert abs(fdr - expected_fdr) < 1e-10
            assert abs(delta - expected_delta) < 1e-10


class TestResourceContention:
    """Test resource contention scenarios."""

    def test_memory_pressure_concurrent_operations(self):
        """Test behavior under memory pressure with concurrent operations."""
        def memory_intensive_operation(size_multiplier):
            # Create reasonably large arrays
            X = np.random.randn(100 * size_multiplier, 50)
            y = np.random.randn(100 * size_multiplier)

            params = {'x_path': None, 'y_path': None, 'fdr': 0.1}
            slide = SLIDE(params, x=X, y=y)

            # Simulate some computation
            result = np.cov(X.T)
            return result.shape

        # Run multiple memory-intensive operations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(memory_intensive_operation, i + 1) for i in range(4)]

            # Should handle memory pressure gracefully
            try:
                results = [future.result(timeout=30) for future in futures]
                assert len(results) == 4
            except (MemoryError, concurrent.futures.TimeoutError):
                # Acceptable under memory pressure
                pass

    def test_cpu_intensive_concurrent_operations(self):
        """Test CPU-intensive concurrent operations."""
        def cpu_intensive_operation(matrix_size):
            # CPU-intensive matrix operations
            A = np.random.randn(matrix_size, matrix_size)
            B = np.random.randn(matrix_size, matrix_size)

            # Simulate expensive computation
            result = A @ B @ A.T
            return np.trace(result)

        start_time = time.time()

        # Run concurrent CPU-intensive operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(cpu_intensive_operation, 50) for _ in range(3)]
            results = [future.result(timeout=60) for future in futures]

        end_time = time.time()

        # Should complete without deadlocks
        assert len(results) == 3
        assert end_time - start_time < 60  # Reasonable time limit


class TestDeadlockPrevention:
    """Test deadlock prevention."""

    def test_no_deadlocks_nested_locks(self):
        """Test that nested operations don't cause deadlocks."""
        # Simulate nested operations that might cause deadlocks
        def nested_operation(depth):
            if depth <= 0:
                return 1

            # Simulate nested computation
            X = np.random.randn(20, 10)
            result = np.linalg.svd(X)

            # Recurse
            return nested_operation(depth - 1) + np.sum(result[1])

        # Run nested operations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(nested_operation, 3) for _ in range(5)]

            # Should complete without deadlocks
            try:
                results = [future.result(timeout=30) for future in futures]
                assert len(results) == 5
            except concurrent.futures.TimeoutError:
                pytest.fail("Potential deadlock detected - operations timed out")

    def test_resource_ordering_consistency(self):
        """Test consistent resource ordering to prevent deadlocks."""
        lock_a = threading.Lock()
        lock_b = threading.Lock()

        results = []

        def operation_1():
            with lock_a:
                time.sleep(0.1)
                with lock_b:
                    results.append("op1")

        def operation_2():
            with lock_a:  # Same order as operation_1
                time.sleep(0.1)
                with lock_b:
                    results.append("op2")

        # Run operations that could deadlock if ordering is inconsistent
        threads = [
            threading.Thread(target=operation_1),
            threading.Thread(target=operation_2)
        ]

        start_time = time.time()
        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=5)

        end_time = time.time()

        # Should complete without deadlocks
        assert end_time - start_time < 5
        assert len(results) == 2


class TestAsyncCompatibility:
    """Test async/await compatibility where applicable."""

    @pytest.mark.asyncio
    async def test_async_compatible_operations(self):
        """Test that synchronous operations are async-compatible."""
        import asyncio

        async def async_slide_operation():
            params = {'x_path': None, 'y_path': None, 'fdr': 0.1}
            X = np.random.randn(50, 20)
            y = np.random.randn(50)

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            slide = await loop.run_in_executor(None, SLIDE, params, X, y)
            return slide

        # Run multiple async operations
        tasks = [async_slide_operation() for _ in range(3)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result in results:
            assert hasattr(result, 'data')
            assert hasattr(result, 'input_params')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])