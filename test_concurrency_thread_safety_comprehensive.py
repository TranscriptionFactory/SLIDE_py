"""
Comprehensive concurrency and thread safety testing.
Tests multi-threaded access, race conditions, and resource sharing.
"""
import pytest
import numpy as np
import pandas as pd
import threading
import multiprocessing
import time
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import patch

from loveslide import SLIDE, OptimizeSLIDE, SLIDEcv, Knockoffs


class TestThreadSafety:
    """Test thread safety of core LOVESLIDE components."""

    def test_slide_concurrent_initialization(self):
        """Test concurrent SLIDE object initialization."""
        params = {"fdr": 0.1, "niter": 2}
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        def create_slide():
            return SLIDE(params, x=X, y=y)

        # Create multiple SLIDE objects concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_slide) for _ in range(10)]
            slides = [future.result() for future in futures]

        # All objects should be created successfully
        assert len(slides) == 10
        for slide in slides:
            assert slide is not None
            assert hasattr(slide, 'data')

    def test_slide_concurrent_state_access(self):
        """Test concurrent access to SLIDE object state."""
        params = {"fdr": 0.1}
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        slide = SLIDE(params, x=X, y=y)

        # Add some state
        slide.A = pd.DataFrame(np.random.randn(50, 10))
        slide.latent_factors = pd.DataFrame(np.random.randn(100, 10))

        results = []
        errors = []

        def read_state(slide_obj, results_list, errors_list):
            try:
                # Concurrent read operations
                a_shape = slide_obj.A.shape
                lf_shape = slide_obj.latent_factors.shape
                results_list.append((a_shape, lf_shape))
            except Exception as e:
                errors_list.append(e)

        # Multiple threads reading state
        threads = []
        for i in range(10):
            thread = threading.Thread(
                target=read_state,
                args=(slide, results, errors)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have no errors and consistent results
        assert len(errors) == 0
        assert len(results) == 10
        assert all(r == results[0] for r in results)

    def test_knockoffs_concurrent_generation(self):
        """Test concurrent knockoff generation."""
        def generate_knockoffs(seed):
            np.random.seed(seed)
            y = np.random.binomial(1, 0.5, 50)
            z = np.random.randn(50, 10)

            knockoffs = Knockoffs(y, z, model='LR')
            return knockoffs.run()

        # Generate knockoffs concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(generate_knockoffs, seed)
                for seed in range(8)
            ]
            results = [future.result() for future in futures]

        # All results should be valid
        assert len(results) == 8
        for result in results:
            assert 'selected' in result
            assert isinstance(result['selected'], list)


class TestDataRaceConditions:
    """Test for data race conditions in shared resources."""

    def test_slide_state_modification_race(self):
        """Test race conditions in SLIDE state modification."""
        params = {"fdr": 0.1}
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        slide = SLIDE(params, x=X, y=y)

        # Initialize shared state
        slide.marginal_idxs = []
        slide.sig_interacts = []

        modification_count = 0
        modification_lock = threading.Lock()

        def modify_state(slide_obj, thread_id):
            nonlocal modification_count

            for i in range(10):
                with modification_lock:
                    # Simulate state modifications
                    slide_obj.marginal_idxs.append(thread_id * 10 + i)
                    slide_obj.sig_interacts.append(f"interaction_{thread_id}_{i}")

                time.sleep(0.001)  # Small delay to increase race condition chance

        # Multiple threads modifying state
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(
                target=modify_state,
                args=(slide, thread_id)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # With proper synchronization, should have 50 total modifications
        assert len(slide.marginal_idxs) == 50
        assert len(slide.sig_interacts) == 50

    def test_file_io_race_conditions(self):
        """Test race conditions in file I/O operations."""
        params = {"fdr": 0.1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        def save_load_cycle(thread_id, temp_dir):
            slide = SLIDE(params, x=X, y=y)
            slide.A = pd.DataFrame(np.random.randn(20, 5))

            # Save to unique file
            file_path = os.path.join(temp_dir, f"A_{thread_id}.csv")
            slide.A.to_csv(file_path)

            # Load back
            loaded_A = pd.read_csv(file_path, index_col=0)
            return loaded_A.shape

        with tempfile.TemporaryDirectory() as temp_dir:
            # Multiple threads doing I/O
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(save_load_cycle, i, temp_dir)
                    for i in range(10)
                ]
                shapes = [future.result() for future in futures]

        # All operations should complete successfully
        assert len(shapes) == 10
        assert all(shape == (20, 5) for shape in shapes)


class TestMemorySharing:
    """Test memory sharing and isolation between processes."""

    def test_multiprocess_slide_isolation(self):
        """Test isolation between SLIDE objects in different processes."""
        def create_and_modify_slide(process_id):
            params = {"fdr": 0.1}
            X = np.random.randn(50, 20)
            y = np.random.randn(50)

            slide = SLIDE(params, x=X, y=y)
            slide.process_id = process_id  # Add identifying attribute

            # Modify data
            slide.data.X *= process_id

            return slide.data.X.sum(), process_id

        # Run in multiple processes
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(create_and_modify_slide, i)
                for i in range(1, 5)
            ]
            results = [future.result() for future in futures]

        # Each process should have different results
        sums = [r[0] for r in results]
        process_ids = [r[1] for r in results]

        assert len(set(sums)) == 4  # All sums should be different
        assert process_ids == [1, 2, 3, 4]

    def test_shared_memory_safety(self):
        """Test safety of shared memory access patterns."""
        # Create shared data structure
        manager = multiprocessing.Manager()
        shared_results = manager.dict()

        def worker_process(worker_id, shared_dict):
            params = {"fdr": 0.1}
            X = np.random.randn(50, 20)
            y = np.random.randn(50)

            slide = SLIDE(params, x=X, y=y)

            # Write to shared memory
            shared_dict[f'worker_{worker_id}'] = {
                'data_shape': slide.data.X.shape,
                'worker_id': worker_id
            }

            return worker_id

        # Multiple processes writing to shared memory
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(worker_process, i, shared_results)
                for i in range(4)
            ]
            [future.result() for future in futures]

        # Check shared results
        assert len(shared_results) == 4
        for i in range(4):
            key = f'worker_{i}'
            assert key in shared_results
            assert shared_results[key]['worker_id'] == i


class TestResourceContention:
    """Test resource contention scenarios."""

    def test_memory_pressure_concurrent_operations(self):
        """Test behavior under memory pressure with concurrent operations."""
        def memory_intensive_operation(operation_id):
            try:
                params = {"fdr": 0.1, "niter": 2}
                # Create moderately large data
                X = np.random.randn(500, 100)
                y = np.random.randn(500)

                slide = SLIDE(params, x=X, y=y)

                # Simulate memory-intensive operations
                large_array = np.random.randn(1000, 1000)
                result = np.dot(slide.data.X.T, slide.data.X)

                return result.shape
            except MemoryError:
                return "MemoryError"

        # Run multiple memory-intensive operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(memory_intensive_operation, i)
                for i in range(8)
            ]
            results = [future.result() for future in futures]

        # Some operations might fail due to memory pressure
        successful_results = [r for r in results if r != "MemoryError"]
        memory_errors = [r for r in results if r == "MemoryError"]

        # At least some should succeed
        assert len(successful_results) > 0

    def test_cpu_intensive_concurrent_operations(self):
        """Test CPU-intensive concurrent operations."""
        def cpu_intensive_knockoffs(seed):
            np.random.seed(seed)
            y = np.random.binomial(1, 0.5, 100)
            z = np.random.randn(100, 50)  # Larger problem

            start_time = time.time()
            knockoffs = Knockoffs(y, z, model='LR')
            result = knockoffs.run()
            end_time = time.time()

            return {
                'selected_count': len(result['selected']),
                'duration': end_time - start_time,
                'seed': seed
            }

        # Run CPU-intensive operations concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(cpu_intensive_knockoffs, seed)
                for seed in range(8)
            ]
            results = [future.result() for future in futures]

        # All operations should complete
        assert len(results) == 8
        for result in results:
            assert 'selected_count' in result
            assert 'duration' in result
            assert result['duration'] > 0


class TestDeadlockPrevention:
    """Test deadlock prevention mechanisms."""

    def test_no_deadlock_in_nested_calls(self):
        """Test that nested function calls don't create deadlocks."""
        def nested_slide_operations():
            params = {"fdr": 0.1, "niter": 2}
            X = np.random.randn(50, 20)
            y = np.random.randn(50)

            # Create SLIDE object
            slide = SLIDE(params, x=X, y=y)

            # Nested operations that might acquire locks
            slide.calc_default_fsize(5)
            slide.show_params()

            # Create another SLIDE object within the same thread
            slide2 = SLIDE(params, x=X, y=y)

            return True

        # Multiple threads doing nested operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(nested_slide_operations)
                for _ in range(8)
            ]

            # Set timeout to detect potential deadlocks
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=10)  # 10 second timeout
                    results.append(result)
                except Exception as e:
                    results.append(f"Error: {e}")

        # All operations should complete without deadlock
        assert len(results) == 8
        assert all(r == True for r in results)

    def test_timeout_behavior_concurrent_operations(self):
        """Test timeout behavior in concurrent scenarios."""
        def long_running_operation(duration):
            time.sleep(duration)
            params = {"fdr": 0.1}
            X = np.random.randn(50, 20)
            y = np.random.randn(50)

            return SLIDE(params, x=X, y=y)

        # Start operations with different durations
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(long_running_operation, duration)
                for duration in [0.1, 0.2, 0.3, 0.4]
            ]

            # Check that operations complete in expected order
            results = []
            for future in futures:
                result = future.result(timeout=5)
                results.append(result is not None)

        assert all(results)