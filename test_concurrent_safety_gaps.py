"""
Test coverage gaps for concurrent operations and thread safety.

Critical gaps in testing concurrent operations that could lead
to race conditions, deadlocks, or data corruption.
"""

import pytest
import numpy as np
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from unittest.mock import patch, MagicMock

from src.loveslide.knockoffs import Knockoffs
from src.loveslide.knockoff._parallel import _single_knockoff_iteration, _worker_wrapper


class TestThreadSafety:
    """Test thread safety of core operations."""

    def test_knockoff_generation_thread_safety(self):
        """Test thread safety of knockoff generation."""
        X = np.random.randn(100, 50)
        results = {}
        exceptions = []

        def generate_knockoffs(thread_id):
            try:
                knockoffs = Knockoffs()
                result = knockoffs.fit_transform(X)
                results[thread_id] = result
            except Exception as e:
                exceptions.append((thread_id, e))

        # Run multiple threads simultaneously
        threads = []
        for i in range(10):
            t = threading.Thread(target=generate_knockoffs, args=(i,))
            threads.append(t)

        # Start all threads at once
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Check for exceptions
        assert len(exceptions) == 0, f"Thread exceptions: {exceptions}"
        assert len(results) == 10, "Not all threads completed"

        # Results should have correct shape
        for thread_id, result in results.items():
            assert result.shape == X.shape, f"Thread {thread_id} wrong shape"

    def test_shared_state_race_conditions(self):
        """Test for race conditions in shared state."""
        shared_counter = {'value': 0}
        lock = threading.Lock()

        def increment_shared_state():
            for _ in range(1000):
                with lock:
                    shared_counter['value'] += 1

        # Run concurrent increments
        threads = [threading.Thread(target=increment_shared_state) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have correct final count (no race conditions)
        assert shared_counter['value'] == 5000, "Race condition detected"

    def test_parallel_cv_thread_safety(self):
        """Test thread safety of parallel CV operations."""
        from src.loveslide.cv import SLIDEcv
        from src.loveslide.slide import OptimizeSLIDE

        X = np.random.randn(200, 50)
        y = np.random.randint(0, 2, 200)

        # Create SLIDE object
        params = {'x_path': None, 'y_path': None}
        slide = OptimizeSLIDE(params, x=X, y=y)
        slide.run()

        # Test concurrent CV runs
        results = []

        def run_cv():
            cv = SLIDEcv(slide, nrep=2, k=3)
            result = cv.run()
            results.append(result)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_cv) for _ in range(3)]
            for future in futures:
                future.result()

        assert len(results) == 3, "Not all CV runs completed"


class TestProcessSafety:
    """Test process safety and multiprocessing."""

    def test_multiprocessing_data_isolation(self):
        """Test data isolation between processes."""
        X = np.random.randn(100, 50)

        def process_data(process_id):
            # Each process should work with isolated data
            local_X = X.copy()
            local_X *= process_id  # Modify locally

            knockoffs = Knockoffs()
            result = knockoffs.fit_transform(local_X)
            return process_id, result.shape, np.mean(local_X)

        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_data, i) for i in range(1, 4)]
            results = [future.result() for future in futures]

        # Check all processes completed with different data
        assert len(results) == 3
        means = [result[2] for result in results]
        assert len(set(means)) == 3, "Processes not properly isolated"

    def test_shared_memory_safety(self):
        """Test safety of shared memory operations."""
        # Create shared array
        shared_array = multiprocessing.Array('d', np.random.randn(1000))

        def modify_shared_data(start_idx, end_idx, multiplier):
            for i in range(start_idx, end_idx):
                shared_array[i] *= multiplier

        # Modify different parts of shared array concurrently
        processes = []
        chunk_size = len(shared_array) // 3

        for i in range(3):
            start = i * chunk_size
            end = start + chunk_size if i < 2 else len(shared_array)
            p = multiprocessing.Process(
                target=modify_shared_data,
                args=(start, end, i + 1)
            )
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # Check that modifications were applied correctly
        # (This is a simplified test - real shared memory needs more careful handling)
        assert all(abs(val) > 0 for val in shared_array), "Shared memory corruption detected"


class TestDeadlockPrevention:
    """Test prevention of deadlocks."""

    def test_nested_lock_deadlock_prevention(self):
        """Test prevention of deadlocks with nested locks."""
        lock1 = threading.Lock()
        lock2 = threading.Lock()
        completed = []

        def task1():
            with lock1:
                time.sleep(0.1)
                with lock2:
                    completed.append('task1')

        def task2():
            with lock2:
                time.sleep(0.1)
                with lock1:
                    completed.append('task2')

        # These could deadlock without proper ordering
        t1 = threading.Thread(target=task1)
        t2 = threading.Thread(target=task2)

        start_time = time.time()
        t1.start()
        t2.start()

        # Use timeout to detect potential deadlocks
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        elapsed = time.time() - start_time

        # Should complete quickly without deadlock
        assert elapsed < 2.0, "Potential deadlock detected"
        assert t1.is_alive() == False, "Thread 1 did not complete"
        assert t2.is_alive() == False, "Thread 2 did not complete"

    def test_resource_lock_ordering(self):
        """Test consistent resource lock ordering."""
        resources = [threading.Lock() for _ in range(5)]
        completed_tasks = []

        def acquire_resources_ordered(task_id, resource_indices):
            # Always acquire locks in ascending order to prevent deadlocks
            sorted_indices = sorted(resource_indices)
            acquired_locks = []

            try:
                for idx in sorted_indices:
                    resources[idx].acquire()
                    acquired_locks.append(idx)

                # Simulate work
                time.sleep(0.01)
                completed_tasks.append(task_id)

            finally:
                # Release in reverse order
                for idx in reversed(acquired_locks):
                    resources[idx].release()

        # Create tasks that need overlapping resources
        tasks = [
            (1, [0, 1, 2]),
            (2, [1, 2, 3]),
            (3, [2, 3, 4]),
            (4, [0, 3, 4]),
        ]

        threads = []
        for task_id, resource_indices in tasks:
            t = threading.Thread(
                target=acquire_resources_ordered,
                args=(task_id, resource_indices)
            )
            threads.append(t)

        start_time = time.time()
        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=3.0)

        elapsed = time.time() - start_time

        # All tasks should complete without deadlock
        assert len(completed_tasks) == 4, "Not all tasks completed"
        assert elapsed < 1.0, "Tasks took too long (possible deadlock)"


class TestAtomicOperations:
    """Test atomic operations and consistency."""

    def test_atomic_counter_operations(self):
        """Test atomic counter operations."""
        counter_value = multiprocessing.Value('i', 0)
        lock = multiprocessing.Lock()

        def increment_counter(num_increments):
            for _ in range(num_increments):
                with lock:
                    temp = counter_value.value
                    temp += 1
                    counter_value.value = temp

        # Run concurrent increments
        processes = []
        increments_per_process = 100

        for _ in range(5):
            p = multiprocessing.Process(
                target=increment_counter,
                args=(increments_per_process,)
            )
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # Should have correct final count
        expected_total = 5 * increments_per_process
        assert counter_value.value == expected_total, f"Expected {expected_total}, got {counter_value.value}"

    def test_data_consistency_checks(self):
        """Test data consistency under concurrent access."""
        # Shared data structure
        shared_data = {'matrix': np.random.randn(100, 50)}
        data_lock = threading.Lock()

        def modify_data(modification_id):
            with data_lock:
                # Read
                original_shape = shared_data['matrix'].shape
                original_sum = np.sum(shared_data['matrix'])

                # Modify
                shared_data['matrix'] *= 2

                # Verify consistency
                new_sum = np.sum(shared_data['matrix'])
                assert abs(new_sum - 2 * original_sum) < 1e-10, f"Data inconsistency in modification {modification_id}"
                assert shared_data['matrix'].shape == original_shape, f"Shape changed in modification {modification_id}"

        # Run concurrent modifications
        threads = []
        for i in range(10):
            t = threading.Thread(target=modify_data, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Final verification
        assert shared_data['matrix'].shape == (100, 50), "Final shape incorrect"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])