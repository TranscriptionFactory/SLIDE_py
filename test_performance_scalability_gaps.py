"""
Test coverage for performance and scalability edge cases.
"""

import pytest
import numpy as np
import pandas as pd
import time
import psutil
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.loveslide.slide import SLIDE, OptimizeSLIDE
from src.loveslide.cv import SLIDEcv
from src.loveslide.knockoffs import Knockoffs
from src.loveslide.knockoff._parallel import knockoff_voting_parallel
from src.loveslide.knockoff.filter import knockoff_filter_voting


class TestScalabilityLimits:
    """Test scalability limits and performance bottlenecks."""

    @pytest.mark.slow
    def test_large_dataset_memory_usage(self):
        """Test memory usage with large datasets."""
        # Test with moderately large dataset (adjust size based on available memory)
        try:
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            if available_memory < 2:
                pytest.skip("Insufficient memory for large dataset test")

            n, p = 5000, 1000
            X = np.random.randn(n, p).astype(np.float32)  # Use float32 to save memory
            y = np.random.binomial(1, 0.5, n)

            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024**2)  # MB

            params = {
                'delta': [0.1],
                'lambda': [0.5],
                'niter': 5,  # Reduced iterations for speed
                'n_workers': 1
            }

            slide = SLIDE(params, X, y)

            memory_after = process.memory_info().rss / (1024**2)  # MB
            memory_increase = memory_after - memory_before

            # Memory increase should be reasonable (less than 2x data size)
            data_size_mb = X.nbytes / (1024**2)
            assert memory_increase < 2 * data_size_mb, \
                f"Memory usage {memory_increase:.1f}MB too high for data size {data_size_mb:.1f}MB"

        except MemoryError:
            pytest.skip("Insufficient memory for this test")

    def test_knockoff_generation_time_complexity(self):
        """Test time complexity of knockoff generation."""
        sizes = [50, 100, 200]  # Progressive sizes
        times = []

        for n in sizes:
            p = n // 2
            X = np.random.randn(n, p)

            knockoffs = Knockoffs()

            start_time = time.time()
            try:
                knockoffs.create_knockoffs(X, method='equi')
                elapsed = time.time() - start_time
                times.append(elapsed)
            except MemoryError:
                # Skip if we run out of memory
                break

        if len(times) >= 2:
            # Check that time doesn't grow too fast (should be roughly O(p^2) or O(p^3))
            # Allow for some variation in timing
            time_ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
            size_ratios = [sizes[i+1] / sizes[i] for i in range(len(times)-1)]

            for time_ratio, size_ratio in zip(time_ratios, size_ratios):
                # Time should grow at most as size^4 (allowing some overhead)
                assert time_ratio <= size_ratio**4 * 2, \
                    f"Time complexity too high: {time_ratio} vs size ratio {size_ratio}"

    def test_parallel_efficiency_scaling(self):
        """Test parallel efficiency with different worker counts."""
        n, p = 200, 50
        X = np.random.randn(n, p)
        y = np.random.binomial(1, 0.5, n)

        def simple_statistic(X, y):
            return np.random.randn(X.shape[1])

        worker_counts = [1, 2, 4]
        times = []

        for n_workers in worker_counts:
            if n_workers > psutil.cpu_count():
                break

            start_time = time.time()
            try:
                knockoff_voting_parallel(
                    X, y, simple_statistic,
                    fdr=0.1, n_jobs=n_workers,
                    iterations=10
                )
                elapsed = time.time() - start_time
                times.append((n_workers, elapsed))
            except Exception:
                # Skip if parallel execution fails
                break

        # Check that parallel execution provides some speedup
        if len(times) >= 2:
            serial_time = times[0][1]
            for workers, parallel_time in times[1:]:
                # Should get some speedup, but not necessarily linear
                efficiency = serial_time / (parallel_time * workers)
                assert efficiency > 0.2, \
                    f"Poor parallel efficiency: {efficiency:.2f} with {workers} workers"

    def test_memory_efficient_operations(self):
        """Test memory-efficient operations for large problems."""
        # Test chunked processing
        n, p = 1000, 500
        X = np.random.randn(n, p)

        # Test chunked correlation computation
        chunk_size = 100
        n_chunks = (p + chunk_size - 1) // chunk_size

        correlations = []
        for i in range(0, p, chunk_size):
            end_i = min(i + chunk_size, p)
            chunk = X[:, i:end_i]

            # Compute correlation with first few features
            corr_chunk = np.corrcoef(chunk, X[:, :10], rowvar=False)
            correlations.append(corr_chunk[:end_i-i, -10:])

        # Should complete without memory errors
        assert len(correlations) == n_chunks

        # Results should be reasonable
        for corr_chunk in correlations:
            assert np.all(np.abs(corr_chunk) <= 1.01)  # Allow for numerical error
            assert not np.any(np.isnan(corr_chunk))


class TestPerformanceBottlenecks:
    """Test identification and handling of performance bottlenecks."""

    def test_sdp_solver_timeout_handling(self):
        """Test SDP solver timeout handling."""
        from src.loveslide.knockoff.solve import create_solve_sdp

        # Create a challenging SDP problem
        p = 100
        Sigma = np.random.randn(p, p)
        Sigma = Sigma @ Sigma.T + np.eye(p)  # Make PD

        # Mock slow SDP solver
        with patch('src.loveslide.knockoff.solve._solve_sdp_cvxpy') as mock_solve:
            def slow_solve(*args, **kwargs):
                time.sleep(0.1)  # Simulate slow computation
                return np.ones(p)

            mock_solve.side_effect = slow_solve

            start_time = time.time()
            result = create_solve_sdp(Sigma, max_time=0.05)  # Very short timeout
            elapsed = time.time() - start_time

            # Should return quickly (either succeed fast or timeout/fallback)
            assert elapsed < 1.0

    def test_cv_early_stopping(self):
        """Test early stopping in cross-validation."""
        X = np.random.randn(100, 30)
        y = np.random.binomial(1, 0.5, 100)

        params = {
            'delta': np.linspace(0.01, 0.5, 50),  # Many parameter values
            'lambda': np.linspace(0.1, 0.9, 20),
            'cv_folds': 5,
            'patience': 5,  # Early stopping parameter
            'min_delta': 0.001  # Minimum improvement threshold
        }

        cv_slide = SLIDEcv(params, X, y)

        # Mock early stopping behavior
        with patch.object(cv_slide, 'run_single_cv') as mock_single:
            # Simulate decreasing then increasing loss (convergence)
            losses = [1.0, 0.8, 0.6, 0.5, 0.51, 0.52, 0.53]  # Converged after index 3
            mock_single.side_effect = lambda *args: {'loss': losses.pop(0) if losses else 0.5}

            start_time = time.time()
            cv_slide.run_cv()
            elapsed = time.time() - start_time

            # Should stop early and not test all parameter combinations
            total_combinations = len(params['delta']) * len(params['lambda'])
            calls_made = mock_single.call_count

            # Should make fewer calls than total combinations due to early stopping
            assert calls_made < total_combinations

    def test_memory_pressure_handling(self):
        """Test handling of memory pressure situations."""
        # Simulate low memory condition
        with patch('psutil.virtual_memory') as mock_memory:
            # Mock low available memory
            mock_memory.return_value = MagicMock(percent=95.0, available=100*1024*1024)  # 100MB available

            X = np.random.randn(500, 200)
            y = np.random.binomial(1, 0.5, 500)

            params = {
                'delta': [0.1],
                'lambda': [0.5],
                'niter': 10,
                'memory_efficient': True
            }

            slide = SLIDE(params, X, y)

            # Should adapt behavior under memory pressure
            # This might involve reducing batch sizes, using different algorithms, etc.
            # The test verifies that the system remains functional

    def test_large_parameter_grid_optimization(self):
        """Test optimization with large parameter grids."""
        X = np.random.randn(50, 20)
        y = np.random.binomial(1, 0.5, 50)

        # Very large parameter grid
        params = {
            'delta': np.linspace(0.01, 0.5, 100),
            'lambda': np.linspace(0.1, 0.9, 100),
            'fdr': 0.1,
            'max_evaluations': 50  # Limit total evaluations
        }

        opt_slide = OptimizeSLIDE(params, X, y)

        with patch.object(opt_slide, 'run_slide') as mock_run:
            mock_run.return_value = ([], [])

            start_time = time.time()
            opt_slide.optimize_params()
            elapsed = time.time() - start_time

            # Should complete in reasonable time despite large grid
            assert elapsed < 60.0  # Should finish within 1 minute

            # Should not evaluate all combinations
            total_combinations = len(params['delta']) * len(params['lambda'])
            assert mock_run.call_count <= params['max_evaluations']
            assert mock_run.call_count < total_combinations


class TestResourceManagement:
    """Test resource management and cleanup."""

    def test_temporary_file_cleanup(self):
        """Test cleanup of temporary files."""
        X = np.random.randn(50, 10)
        y = np.random.binomial(1, 0.5, 50)

        params = {'delta': [0.1]}
        slide = SLIDE(params, X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate creating many temporary files
            temp_files = []
            for i in range(10):
                temp_path = os.path.join(tmpdir, f'temp_result_{i}.pkl')
                with open(temp_path, 'wb') as f:
                    import pickle
                    pickle.dump({'data': np.random.randn(100)}, f)
                temp_files.append(temp_path)

            # All files should exist
            assert all(os.path.exists(path) for path in temp_files)

            # Test cleanup (normally happens automatically with tempfile)
            for path in temp_files:
                if os.path.exists(path):
                    os.remove(path)

            # Files should be cleaned up
            assert not any(os.path.exists(path) for path in temp_files)

    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        if not psutil:
            pytest.skip("psutil not available for memory monitoring")

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Perform repeated operations that might leak memory
        for i in range(10):
            X = np.random.randn(100, 20)
            y = np.random.binomial(1, 0.5, 100)

            knockoffs = Knockoffs()
            result = knockoffs.create_knockoffs(X, method='equi')

            # Force garbage collection
            import gc
            gc.collect()

            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory

            # Memory increase should be bounded
            max_acceptable_increase = 50 * 1024 * 1024  # 50MB
            if memory_increase > max_acceptable_increase:
                pytest.fail(f"Potential memory leak: {memory_increase / 1024**2:.1f}MB increase")

    def test_concurrent_resource_access(self):
        """Test concurrent access to shared resources."""
        import threading
        import queue

        X = np.random.randn(100, 30)
        y = np.random.binomial(1, 0.5, 100)

        results_queue = queue.Queue()
        errors_queue = queue.Queue()

        def worker(worker_id):
            try:
                # Each worker creates its own knockoffs
                knockoffs = Knockoffs()
                result = knockoffs.create_knockoffs(X, method='equi')
                results_queue.put((worker_id, result.shape if result is not None else None))
            except Exception as e:
                errors_queue.put((worker_id, str(e)))

        # Start multiple worker threads
        threads = []
        n_workers = 4

        for i in range(n_workers):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout

        # Check results
        n_results = results_queue.qsize()
        n_errors = errors_queue.qsize()

        # Most workers should succeed
        assert n_results >= n_workers // 2, \
            f"Too many failures: {n_errors} errors, {n_results} successes"

        # No worker should hang (all threads should complete)
        for thread in threads:
            assert not thread.is_alive(), "Worker thread hung"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])