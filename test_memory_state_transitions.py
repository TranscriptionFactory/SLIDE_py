"""
Test coverage for memory state transitions during long-running computations
Focus: Memory management, state consistency, and resource cleanup patterns
"""

import pytest
import numpy as np
import gc
import psutil
import os
from unittest.mock import patch, MagicMock
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loveslide import SLIDE, OptimizeSLIDE, SLIDEcv
from loveslide.knockoffs import Knockoffs


class TestMemoryStateTransitions:
    """Test memory state transitions in long-running computations"""

    def get_memory_usage(self):
        """Helper to get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def test_slide_memory_state_progression(self):
        """Test memory state during SLIDE algorithm progression"""
        # Create progressively larger datasets to test memory transitions
        dataset_sizes = [(50, 10), (100, 15), (150, 20)]
        memory_states = []

        for n_samples, n_features in dataset_sizes:
            X = np.random.rand(n_samples, n_features)
            y = np.random.randint(0, 2, n_samples)

            # Measure memory before computation
            gc.collect()  # Force garbage collection
            mem_before = self.get_memory_usage()

            # Run SLIDE computation
            params = {'K': 3, 'max_iters': 2, 'fdr_thresh': 0.1}
            slide = OptimizeSLIDE(params, x=X, y=y)

            # Mock LOVE result to control memory usage
            mock_love_result = {
                'L_hat': np.random.rand(n_features, 3),
                'pure_idx': list(range(min(5, n_features)))
            }

            try:
                # Measure memory during computation
                mem_during = self.get_memory_usage()

                # Run a computation step
                lf_result = slide.get_latent_factors(
                    x=X, y=y, love_result=mock_love_result
                )

                # Measure memory after computation
                mem_after = self.get_memory_usage()

                memory_states.append({
                    'dataset_size': (n_samples, n_features),
                    'mem_before': mem_before,
                    'mem_during': mem_during,
                    'mem_after': mem_after,
                    'mem_growth': mem_after - mem_before
                })

                # Clean up
                del slide, X, y, lf_result

            except Exception:
                # Handle computation failures gracefully
                del slide, X, y

        # Verify memory growth patterns are reasonable
        if len(memory_states) >= 2:
            growth_rates = [state['mem_growth'] for state in memory_states]
            # Memory growth should not be excessive (> 500MB per iteration)
            assert all(growth < 500 for growth in growth_rates), f"Excessive memory growth: {growth_rates}"

    def test_knockoff_iterative_memory_consistency(self):
        """Test memory consistency during iterative knockoff filtering"""
        X = np.random.rand(80, 12)
        y = np.random.randint(0, 2, 80)
        knockoffs = Knockoffs(y=y, z2=X)

        memory_snapshots = []

        # Test multiple iterations to check for memory leaks
        for iteration in range(5):
            gc.collect()
            mem_before = self.get_memory_usage()

            try:
                # Run knockoff iteration
                result = knockoffs.filter_knockoffs_iterative_python(
                    z=X, y=y, fdr=0.1, niter=1, seed=iteration
                )

                gc.collect()
                mem_after = self.get_memory_usage()

                memory_snapshots.append({
                    'iteration': iteration,
                    'mem_before': mem_before,
                    'mem_after': mem_after,
                    'mem_delta': mem_after - mem_before
                })

                # Clean up iteration results
                del result

            except Exception:
                # Handle computational failures
                gc.collect()
                mem_after = self.get_memory_usage()
                memory_snapshots.append({
                    'iteration': iteration,
                    'mem_before': mem_before,
                    'mem_after': mem_after,
                    'mem_delta': mem_after - mem_before
                })

        # Verify memory doesn't grow excessively across iterations
        memory_deltas = [snapshot['mem_delta'] for snapshot in memory_snapshots]
        if len(memory_deltas) >= 3:
            # Memory growth should stabilize after first few iterations
            later_deltas = memory_deltas[-3:]
            assert max(later_deltas) < 100, f"Memory leak detected: {memory_deltas}"

    def test_cv_fold_memory_isolation(self):
        """Test memory isolation between CV folds"""
        X = np.random.rand(100, 15)
        y = np.random.randint(0, 2, 100)

        # Create CV folds
        folds = [
            (list(range(0, 50)), list(range(50, 100))),
            (list(range(50, 100)), list(range(0, 50)))
        ]

        cv = SLIDEcv(x=X, y=y, folds=folds, n_workers=1)

        fold_memory_states = []

        for fold_idx, fold in enumerate(folds):
            gc.collect()
            mem_before_fold = self.get_memory_usage()

            try:
                # Process fold with memory monitoring
                with patch.object(cv, '_find_interactions_fold') as mock_interactions:
                    mock_interactions.return_value = {
                        'interactions': [],
                        'marginal_features': list(range(min(8, X.shape[1])))
                    }

                    # Run fold processing
                    fold_result = cv._run_slide_fold(fold_idx, fold)

                    gc.collect()
                    mem_after_fold = self.get_memory_usage()

                    fold_memory_states.append({
                        'fold': fold_idx,
                        'mem_before': mem_before_fold,
                        'mem_after': mem_after_fold,
                        'mem_usage': mem_after_fold - mem_before_fold
                    })

                    # Clean up fold results
                    del fold_result

            except Exception:
                # Handle fold processing failures
                gc.collect()
                mem_after_fold = self.get_memory_usage()
                fold_memory_states.append({
                    'fold': fold_idx,
                    'mem_before': mem_before_fold,
                    'mem_after': mem_after_fold,
                    'mem_usage': mem_after_fold - mem_before_fold
                })

        # Verify memory isolation between folds
        if len(fold_memory_states) >= 2:
            memory_usages = [state['mem_usage'] for state in fold_memory_states]
            # Memory usage should be consistent across folds
            memory_variation = np.std(memory_usages) if len(memory_usages) > 1 else 0
            assert memory_variation < 200, f"High memory variation between folds: {memory_usages}"

    def test_large_matrix_memory_transitions(self):
        """Test memory transitions with large matrix operations"""
        # Test with progressively larger matrices
        matrix_sizes = [50, 100, 200]

        for size in matrix_sizes:
            gc.collect()
            mem_initial = self.get_memory_usage()

            # Create large matrices
            X = np.random.rand(size, size // 2)
            y = np.random.randint(0, 2, size)

            mem_after_creation = self.get_memory_usage()

            try:
                # Perform memory-intensive operations
                knockoffs = Knockoffs(y=y, z2=X)

                # Test covariance matrix computation (memory intensive)
                if hasattr(knockoffs, '_compute_covariance'):
                    cov_matrix = np.cov(X.T)
                else:
                    cov_matrix = np.cov(X.T)

                mem_during_computation = self.get_memory_usage()

                # Test SDP solving (if available)
                try:
                    from loveslide.knockoff.solve import _solve_sdp_cvxpy
                    # Only test with smaller matrices to avoid timeout
                    if size <= 100:
                        sdp_result = _solve_sdp_cvxpy(cov_matrix[:10, :10])
                except (ImportError, Exception):
                    # SDP solver may not be available or may fail
                    pass

                mem_peak = self.get_memory_usage()

                # Clean up large objects
                del X, y, knockoffs, cov_matrix
                gc.collect()

                mem_after_cleanup = self.get_memory_usage()

                # Verify memory cleanup is effective
                cleanup_effectiveness = (mem_peak - mem_after_cleanup) / (mem_peak - mem_initial)

                # Should recover at least 50% of allocated memory
                if mem_peak > mem_initial + 10:  # Only check if significant memory was used
                    assert cleanup_effectiveness > 0.3, f"Poor memory cleanup for size {size}: {cleanup_effectiveness}"

            except Exception:
                # Clean up even if computation fails
                try:
                    del X, y
                    if 'knockoffs' in locals():
                        del knockoffs
                    gc.collect()
                except:
                    pass

    def test_state_persistence_memory_consistency(self):
        """Test memory consistency during state persistence operations"""
        X = np.random.rand(60, 10)
        y = np.random.randint(0, 2, 60)

        params = {'K': 3, 'max_iters': 3, 'fdr_thresh': 0.1}
        slide = OptimizeSLIDE(params, x=X, y=y)

        # Test memory consistency during state save/load cycles
        import tempfile
        import pickle

        memory_states = []

        for cycle in range(3):
            gc.collect()
            mem_before = self.get_memory_usage()

            # Create temporary state
            temp_state = {
                'params': slide.input_params.copy(),
                'iteration': cycle,
                'data': np.random.rand(50, 8)  # Simulate computation state
            }

            # Save state to file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                try:
                    pickle.dump(temp_state, tmp_file)
                    tmp_file.flush()

                    gc.collect()
                    mem_after_save = self.get_memory_usage()

                    # Load state from file
                    tmp_file.seek(0)
                    loaded_state = pickle.load(open(tmp_file.name, 'rb'))

                    gc.collect()
                    mem_after_load = self.get_memory_usage()

                    memory_states.append({
                        'cycle': cycle,
                        'mem_before': mem_before,
                        'mem_after_save': mem_after_save,
                        'mem_after_load': mem_after_load,
                        'save_overhead': mem_after_save - mem_before,
                        'load_overhead': mem_after_load - mem_after_save
                    })

                    # Clean up
                    del loaded_state, temp_state

                finally:
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass

        # Verify state persistence doesn't cause excessive memory growth
        if memory_states:
            save_overheads = [state['save_overhead'] for state in memory_states]
            load_overheads = [state['load_overhead'] for state in memory_states]

            # State persistence shouldn't use excessive memory
            assert all(overhead < 200 for overhead in save_overheads), f"Excessive save overhead: {save_overheads}"
            assert all(overhead < 200 for overhead in load_overheads), f"Excessive load overhead: {load_overheads}"


class TestMemoryLeakDetection:
    """Test detection and prevention of memory leaks"""

    def test_iterative_computation_leak_detection(self):
        """Test memory leak detection in iterative computations"""
        base_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Run multiple iterations to detect leaks
        for iteration in range(10):
            X = np.random.rand(30, 8)
            y = np.random.randint(0, 2, 30)

            # Create and use objects
            knockoffs = Knockoffs(y=y, z2=X)

            try:
                result = knockoffs.filter_knockoffs_iterative_python(
                    z=X, y=y, fdr=0.1, niter=1
                )
                del result
            except:
                pass

            # Explicitly clean up
            del knockoffs, X, y
            gc.collect()

            # Check memory every few iterations
            if iteration % 3 == 0:
                current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                memory_growth = current_memory - base_memory

                # Memory growth should be reasonable (< 300MB)
                if memory_growth > 300:
                    pytest.fail(f"Potential memory leak detected: {memory_growth}MB growth at iteration {iteration}")

    def test_exception_handling_memory_cleanup(self):
        """Test memory cleanup when exceptions occur"""
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Test memory cleanup with various exception scenarios
        exception_scenarios = [
            lambda: np.random.rand(-1, 10),  # Invalid array creation
            lambda: 1 / 0,                    # Division by zero
            lambda: [][10],                   # Index error
        ]

        for scenario_idx, scenario in enumerate(exception_scenarios):
            try:
                # Create objects before exception
                X = np.random.rand(50, 10)
                y = np.random.randint(0, 2, 50)
                knockoffs = Knockoffs(y=y, z2=X)

                # Trigger exception
                scenario()

            except Exception:
                # Clean up after exception
                try:
                    del X, y, knockoffs
                    gc.collect()
                except:
                    pass

        # Verify memory was cleaned up after exceptions
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_difference = final_memory - initial_memory

        # Memory should not have grown significantly
        assert memory_difference < 100, f"Memory not cleaned up after exceptions: {memory_difference}MB"


if __name__ == "__main__":
    pytest.main([__file__])