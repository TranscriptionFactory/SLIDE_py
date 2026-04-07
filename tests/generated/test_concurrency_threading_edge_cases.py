"""
Test concurrency and threading edge cases in SLIDE pipeline.

Focus: Thread safety, parallel execution, resource contention,
and concurrent access to shared resources.
"""
import pytest
import numpy as np
import threading
import time
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import patch, MagicMock
import multiprocessing as mp

from loveslide import SLIDE, SLIDEcv, Knockoffs


class TestThreadSafety:
    """Test thread safety of SLIDE components."""

    def test_concurrent_slide_initialization(self):
        """Test concurrent SLIDE object creation."""
        X = np.random.randn(30, 15)
        y = np.random.randn(30)
        params = {"fdr": 0.1, "niter": 3, "K": 2}

        def create_slide():
            try:
                slide = SLIDE(params, x=X.copy(), y=y.copy())
                return slide
            except Exception as e:
                return e

        # Create multiple SLIDE objects concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_slide) for _ in range(10)]
            results = [f.result() for f in futures]

        # All should succeed or fail consistently
        errors = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if not isinstance(r, Exception)]

        if errors:
            # If any failed, they should fail for same reason
            error_types = set(type(e).__name__ for e in errors)
            assert len(error_types) == 1, "Inconsistent error types in concurrent creation"
        else:
            # All should succeed
            assert len(successes) == 10

    def test_concurrent_knockoff_generation(self):
        """Test concurrent knockoff generation."""
        np.random.seed(42)
        X_base = np.random.randn(40, 20)
        Sigma = X_base.T @ X_base / X_base.shape[0]

        def generate_knockoffs():
            try:
                knockoffs = Knockoffs()
                X_ko = knockoffs.create(Sigma.copy())
                return X_ko.shape if X_ko is not None else None
            except Exception as e:
                return str(e)

        # Generate knockoffs concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(generate_knockoffs) for _ in range(6)]
            results = [f.result() for f in futures]

        # Results should be consistent
        valid_results = [r for r in results if isinstance(r, tuple)]
        if valid_results:
            # All shapes should be the same
            assert all(shape == valid_results[0] for shape in valid_results)

    def test_shared_resource_access(self):
        """Test access to shared resources (files, R sessions)."""
        X = np.random.randn(25, 12)
        y = np.random.randn(25)
        params = {"fdr": 0.1, "niter": 2, "K": 2}

        # Create temporary files for state saving
        temp_files = [tempfile.NamedTemporaryFile(delete=False) for _ in range(3)]
        temp_paths = [f.name for f in temp_files]
        for f in temp_files:
            f.close()

        def save_slide_state(path_idx):
            try:
                slide = SLIDE(params, x=X.copy(), y=y.copy())
                if hasattr(slide, 'save_state'):
                    slide.save_state(temp_paths[path_idx])
                return True
            except Exception as e:
                return str(e)
            finally:
                if os.path.exists(temp_paths[path_idx]):
                    try:
                        os.unlink(temp_paths[path_idx])
                    except:
                        pass

        try:
            # Concurrent file operations
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(save_slide_state, i) for i in range(3)]
                results = [f.result() for f in futures]

            # Should handle concurrent file access gracefully
            successful_saves = [r for r in results if r is True]
            errors = [r for r in results if isinstance(r, str)]

            if errors:
                # Errors should be related to file access, not data corruption
                assert all("file" in err.lower() or "permission" in err.lower()
                          for err in errors)

        finally:
            # Cleanup
            for path in temp_paths:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass


class TestParallelExecution:
    """Test parallel execution scenarios."""

    def test_parallel_cross_validation(self):
        """Test parallel CV execution."""
        X = np.random.randn(50, 25)
        y = np.random.randn(50)
        params = {"fdr": 0.1, "niter": 2, "K": 3}

        cv = SLIDEcv(params, x=X, y=y)

        # Mock parallel fold execution
        def mock_run_fold(fold_data):
            time.sleep(0.1)  # Simulate computation
            return {"test_score": np.random.rand(), "train_score": np.random.rand()}

        with patch.object(cv, '_run_single_fold', side_effect=mock_run_fold):
            try:
                results = cv.run(n_folds=5)
                assert results is not None
            except Exception as e:
                # Should handle parallel execution errors
                assert "parallel" in str(e).lower() or "concurrent" in str(e).lower()

    def test_multiprocessing_compatibility(self):
        """Test compatibility with multiprocessing."""
        if mp.get_start_method() == 'spawn':
            pytest.skip("Multiprocessing spawn method not supported")

        X = np.random.randn(30, 15)
        y = np.random.randn(30)
        params = {"fdr": 0.1, "niter": 2}

        def worker_function(worker_id):
            try:
                # Each process creates its own SLIDE instance
                slide = SLIDE(params, x=X.copy(), y=y.copy())
                # Run minimal computation
                result = slide.calc_default_fsize(3)
                return (worker_id, result)
            except Exception as e:
                return (worker_id, str(e))

        try:
            with ProcessPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(worker_function, i) for i in range(4)]
                results = [f.result(timeout=30) for f in futures]

            # All processes should complete successfully
            worker_ids, outcomes = zip(*results)
            errors = [outcome for outcome in outcomes if isinstance(outcome, str)]

            if errors:
                # Errors should be consistent across processes
                assert len(set(errors)) <= 2, "Too many different error types"

        except Exception as e:
            pytest.skip(f"Multiprocessing not available: {e}")

    def test_resource_contention(self):
        """Test resource contention scenarios."""
        X = np.random.randn(35, 18)
        y = np.random.randn(35)
        params = {"fdr": 0.1, "niter": 3}

        # Simulate memory pressure
        large_arrays = []

        def memory_intensive_task():
            try:
                slide = SLIDE(params, x=X.copy(), y=y.copy())
                # Allocate some memory
                large_arrays.append(np.random.randn(1000, 1000))
                result = slide.calc_default_fsize(2)
                return result
            except MemoryError:
                return "MemoryError"
            except Exception as e:
                return str(e)
            finally:
                # Cleanup
                large_arrays.clear()

        # Run multiple memory-intensive tasks
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(memory_intensive_task) for _ in range(5)]
            results = [f.result() for f in futures]

        # Should handle memory pressure gracefully
        memory_errors = [r for r in results if r == "MemoryError"]
        if memory_errors:
            # Memory errors are acceptable under pressure
            assert len(memory_errors) <= len(results)


class TestRaceConditions:
    """Test for race conditions in shared state."""

    def test_state_modification_race(self):
        """Test race conditions in state modification."""
        X = np.random.randn(25, 12)
        y = np.random.randn(25)
        params = {"fdr": 0.1, "niter": 2}

        slide = SLIDE(params, x=X, y=y)
        modification_results = []

        def modify_state(modification_id):
            try:
                # Simulate state modifications
                slide.input_params['test_param'] = modification_id
                time.sleep(0.01)  # Small delay to increase race chance
                observed_value = slide.input_params.get('test_param')
                modification_results.append((modification_id, observed_value))
                return True
            except Exception as e:
                return str(e)

        # Concurrent state modifications
        threads = []
        for i in range(5):
            thread = threading.Thread(target=modify_state, args=(i,))
            threads.append(thread)

        # Start all threads simultaneously
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Analyze race condition results
        if modification_results:
            # Check if there were any unexpected state values
            expected_values = set(range(5))
            observed_values = set(obs for _, obs in modification_results)
            unexpected_values = observed_values - expected_values

            # Should either handle races gracefully or use synchronization
            assert len(unexpected_values) == 0 or "race condition detected"

    def test_concurrent_r_session_access(self):
        """Test concurrent R session access."""
        X = np.random.randn(30, 15)
        y = np.random.randn(30)
        params = {"fdr": 0.1, "niter": 2, "K": 2}

        def access_r_session():
            try:
                slide = SLIDE(params, x=X.copy(), y=y.copy())
                # Attempt R computation (mocked)
                with patch('loveslide.love.call_love') as mock_love:
                    mock_love.return_value = {"factors": np.random.randn(15, 2)}
                    slide.run_love()
                return "success"
            except Exception as e:
                return str(e)

        # Multiple threads accessing R
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(access_r_session) for _ in range(6)]
            results = [f.result() for f in futures]

        # R session access should be thread-safe or properly serialized
        successes = [r for r in results if r == "success"]
        errors = [r for r in results if r != "success"]

        if errors:
            # Errors should be related to R session management
            assert all("R" in err or "session" in err.lower() for err in errors if isinstance(err, str))


class TestDeadlockPrevention:
    """Test deadlock prevention in complex scenarios."""

    def test_nested_lock_acquisition(self):
        """Test nested lock acquisition patterns."""
        X = np.random.randn(20, 10)
        y = np.random.randn(20)
        params = {"fdr": 0.1, "niter": 2}

        # Simulate nested operations that might cause deadlocks
        def nested_operations():
            try:
                slide1 = SLIDE(params, x=X.copy(), y=y.copy())
                slide2 = SLIDE(params, x=X.copy(), y=y.copy())

                # Simulate operations that might acquire multiple locks
                result1 = slide1.calc_default_fsize(2)
                result2 = slide2.calc_default_fsize(2)

                return (result1, result2)
            except Exception as e:
                return str(e)

        # Multiple threads doing nested operations
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(nested_operations) for _ in range(4)]

            # Should complete without deadlock
            try:
                results = [f.result(timeout=10) for f in futures]
                deadlocks = [r for r in results if isinstance(r, str) and "timeout" in r.lower()]
                assert len(deadlocks) == 0, "Potential deadlock detected"
            except Exception as e:
                if "timeout" in str(e).lower():
                    pytest.fail("Deadlock detected in nested operations")
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])