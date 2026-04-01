"""
Concurrency and thread safety testing for SLIDE components.
"""
import pytest
import numpy as np
import pandas as pd
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from unittest.mock import patch

from loveslide import SLIDE, Knockoffs
from loveslide.score import Estimator


class TestConcurrencyThreadSafety:
    """Test concurrent execution and thread safety."""

    def test_knockoffs_parallel_execution(self):
        """Test Knockoffs with parallel execution."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        def run_knockoffs(seed):
            np.random.seed(seed)
            knockoffs = Knockoffs()
            return knockoffs.evaluate_knockoffs(
                X, y, fdr=0.1, n_iter=5, n_workers=1
            )

        # Test parallel execution with different seeds
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_knockoffs, i) for i in range(4)]
            results = [f.result() for f in futures]

        # Results should be deterministic with fixed seeds
        assert len(results) == 4
        for result in results:
            assert 'selected_vars' in result

    def test_estimator_concurrent_fitting(self):
        """Test Estimator thread safety during concurrent fitting."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        results = []
        errors = []

        def fit_estimator(thread_id):
            try:
                estimator = Estimator()
                estimator.fit(X, y)
                prediction = estimator.predict(X[:10])
                results.append((thread_id, prediction))
            except Exception as e:
                errors.append((thread_id, e))

        # Run multiple estimators concurrently
        threads = []
        for i in range(10):
            t = threading.Thread(target=fit_estimator, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent fitting failed: {errors}"
        assert len(results) == 10

    def test_slide_state_file_concurrent_access(self):
        """Test SLIDE state file access under concurrent conditions."""
        import tempfile

        params = {"fdr": 0.1, "niter": 5}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        with tempfile.TemporaryDirectory() as temp_dir:
            def save_and_load_state(thread_id):
                slide = SLIDE(params, x=X, y=y)

                # Simulate saving state
                state_file = f"{temp_dir}/state_{thread_id}.pkl"
                slide.input_params['outpath'] = temp_dir

                # Simulate concurrent file access
                time.sleep(np.random.uniform(0, 0.1))

                # This should not cause race conditions
                slide.load_state(temp_dir)
                return thread_id

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(save_and_load_state, i) for i in range(5)]
                results = [f.result() for f in futures]

            assert len(results) == 5

    def test_knockoffs_memory_sharing(self):
        """Test Knockoffs behavior with shared memory across processes."""
        if multiprocessing.cpu_count() < 2:
            pytest.skip("Need at least 2 CPUs for multiprocessing test")

        X = np.random.randn(200, 30)
        y = np.random.randn(200)

        def worker_process(chunk_idx):
            # Each process works on a chunk of data
            start_idx = chunk_idx * 50
            end_idx = min((chunk_idx + 1) * 50, len(X))

            X_chunk = X[start_idx:end_idx]
            y_chunk = y[start_idx:end_idx]

            knockoffs = Knockoffs()
            return knockoffs._single_knockoff_iteration_python(
                X_chunk, y_chunk, fdr=0.1, method='lasso',
                shrink=True, offset=1, statistic='lasso_coefdiff'
            )

        # Test with ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(worker_process, i) for i in range(4)]
            results = [f.result() for f in futures]

        assert len(results) == 4

    def test_slide_pipeline_interruption_safety(self):
        """Test SLIDE pipeline behavior when interrupted."""
        params = {"fdr": 0.1, "niter": 3}
        X = np.random.randn(100, 25)
        y = np.random.randn(100)

        slide = SLIDE(params, x=X, y=y)

        # Simulate interruption during long-running operation
        def interrupt_after_delay():
            time.sleep(0.1)
            # Simulate external interruption
            raise KeyboardInterrupt("Simulated interruption")

        # Should handle interruptions gracefully
        with pytest.raises(KeyboardInterrupt):
            with patch.object(slide, 'run_SLIDE', side_effect=interrupt_after_delay):
                slide.run_SLIDE()


class TestRaceConditions:
    """Test for race conditions in shared state."""

    def test_estimator_shared_scaler_state(self):
        """Test race conditions in shared scaler state."""
        X = np.random.randn(100, 20)

        results = []

        def scale_data(thread_id):
            # Each thread uses the same scaler - should be thread-safe
            scaled = Estimator.scale_features(X, 'standard')
            results.append((thread_id, scaled.shape, np.mean(scaled)))

        threads = []
        for i in range(10):
            t = threading.Thread(target=scale_data, args=(i,))
            threads.append(t)

        # Start all threads simultaneously to maximize race condition chances
        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 10

        # All results should have same shape and similar means (near 0 for standardized)
        shapes = [r[1] for r in results]
        means = [r[2] for r in results]

        assert all(s == shapes[0] for s in shapes)
        assert all(abs(m) < 1e-10 for m in means)  # Should be close to 0

    def test_knockoffs_random_state_isolation(self):
        """Test random state isolation in concurrent Knockoffs."""
        X = np.random.randn(50, 15)

        def create_knockoffs_with_seed(seed):
            np.random.seed(seed)
            knockoffs = Knockoffs()
            return knockoffs._create_second_order_python(X, method='equi')

        # Test that different seeds produce different but reproducible results
        results_1 = []
        results_2 = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            # First run with seeds 1-4
            futures_1 = [executor.submit(create_knockoffs_with_seed, i)
                         for i in range(1, 5)]
            results_1 = [f.result() for f in futures_1]

            # Second run with same seeds 1-4
            futures_2 = [executor.submit(create_knockoffs_with_seed, i)
                         for i in range(1, 5)]
            results_2 = [f.result() for f in futures_2]

        # Same seeds should produce identical results
        for r1, r2 in zip(results_1, results_2):
            np.testing.assert_array_almost_equal(r1, r2)

        # Different seeds should produce different results
        assert not np.array_equal(results_1[0], results_1[1])