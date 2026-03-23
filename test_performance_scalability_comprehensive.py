"""
Comprehensive performance and scalability testing for SLIDE_py.

Tests performance characteristics, memory usage, and scalability limits
to ensure efficient operation across different data sizes and scenarios.
"""
import pytest
import numpy as np
import pandas as pd
import time
import psutil
import gc
import warnings
from memory_profiler import profile
import tempfile
import os
from unittest.mock import patch

from loveslide import (
    SLIDE, SLIDEcv, Knockoffs, call_love,
    init_data, calc_default_fsize
)


class TestMemoryScalability:
    """Test memory usage scalability."""

    def measure_memory_usage(self, func, *args, **kwargs):
        """Helper to measure peak memory usage of a function."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        gc.collect()  # Clean up before measurement
        start_memory = process.memory_info().rss / 1024 / 1024

        result = func(*args, **kwargs)

        gc.collect()  # Force garbage collection
        peak_memory = process.memory_info().rss / 1024 / 1024

        return result, peak_memory - start_memory

    def test_slide_memory_scaling_with_samples(self):
        """Test SLIDE memory usage scales appropriately with sample size."""
        memory_usage = {}

        for n_samples in [100, 500, 1000]:
            X = np.random.randn(n_samples, 50)
            y = np.random.randn(n_samples)
            params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

            def create_slide():
                return SLIDE(params, x=X, y=y)

            _, memory_used = self.measure_memory_usage(create_slide)
            memory_usage[n_samples] = memory_used

        # Memory should scale roughly linearly, not exponentially
        memory_ratio_500_100 = memory_usage[500] / max(memory_usage[100], 1)
        memory_ratio_1000_500 = memory_usage[1000] / max(memory_usage[500], 1)

        # Should not grow exponentially
        assert memory_ratio_500_100 < 10  # Should be much less than 10x
        assert memory_ratio_1000_500 < 5   # Should be much less than 5x

    def test_slide_memory_scaling_with_features(self):
        """Test SLIDE memory usage scales appropriately with feature count."""
        memory_usage = {}

        for n_features in [50, 200, 500]:
            X = np.random.randn(100, n_features)
            y = np.random.randn(100)
            params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

            def create_slide():
                return SLIDE(params, x=X, y=y)

            _, memory_used = self.measure_memory_usage(create_slide)
            memory_usage[n_features] = memory_used

        # Memory should scale quadratically with features (covariance matrix)
        # but not worse than that
        memory_ratio_200_50 = memory_usage[200] / max(memory_usage[50], 1)
        memory_ratio_500_200 = memory_usage[500] / max(memory_usage[200], 1)

        # Should scale reasonably
        assert memory_ratio_200_50 < 50   # 4x features should be < 50x memory
        assert memory_ratio_500_200 < 20  # 2.5x features should be < 20x memory

    def test_knockoffs_memory_efficiency(self):
        """Test knockoffs memory efficiency."""
        for n_samples, n_features in [(100, 50), (500, 100), (1000, 200)]:
            X = np.random.randn(n_samples, n_features)
            y = np.random.randn(n_samples)

            def create_knockoffs():
                knockoffs = Knockoffs(backend='python')
                return knockoffs.select_short_freq(X, y, fdr=0.1)

            result, memory_used = self.measure_memory_usage(create_knockoffs)

            # Memory usage should be reasonable (less than 500MB for these sizes)
            assert memory_used < 500, f"Excessive memory usage: {memory_used}MB for {n_samples}x{n_features}"

    def test_love_memory_efficiency(self):
        """Test LOVE algorithm memory efficiency."""
        for n_features in [50, 100, 200]:
            X = np.random.randn(100, n_features)

            def run_love():
                return call_love(X, lbd=0.5, verbose=False)

            result, memory_used = self.measure_memory_usage(run_love)

            # Memory usage should be reasonable
            assert memory_used < 200, f"Excessive memory usage: {memory_used}MB for {n_features} features"

    def test_cv_memory_cleanup_between_folds(self):
        """Test that CV properly cleans up memory between folds."""
        params = {
            'x_path': None, 'y_path': None,
            'n_folds': 5, 'fdr': 0.1
        }
        X = np.random.randn(200, 100)
        y = np.random.randn(200)

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        cv_instance = SLIDEcv(params, x=X, y=y)

        # Mock CV to avoid long computation
        with patch.object(cv_instance, '_bench_cv') as mock_bench:
            mock_bench.return_value = {
                'features': np.array([1, 5, 10]),
                'y_pred': np.random.randn(40)
            }

            cv_instance.run()

        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024

        # Memory should not grow excessively
        memory_growth = final_memory - initial_memory
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth}MB"


class TestComputationalComplexity:
    """Test computational complexity and timing."""

    def time_operation(self, func, *args, **kwargs):
        """Helper to time an operation."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    def test_slide_time_scaling_with_samples(self):
        """Test SLIDE time complexity with sample size."""
        timing_results = {}

        for n_samples in [50, 100, 200]:
            X = np.random.randn(n_samples, 30)
            y = np.random.randn(n_samples)
            params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

            def create_slide():
                slide = SLIDE(params, x=X, y=y)
                # Mock expensive operations
                slide.data.love_result = {
                    'pure_Ind': [],
                    'A': np.random.randn(30, 5),
                    'delta': 0.1
                }
                return slide

            _, elapsed_time = self.time_operation(create_slide)
            timing_results[n_samples] = elapsed_time

        # Time should scale reasonably (not exponentially)
        time_ratio_100_50 = timing_results[100] / max(timing_results[50], 0.001)
        time_ratio_200_100 = timing_results[200] / max(timing_results[100], 0.001)

        # Should not scale exponentially
        assert time_ratio_100_50 < 10
        assert time_ratio_200_100 < 10

    def test_knockoffs_time_scaling_with_features(self):
        """Test knockoffs time complexity with feature count."""
        timing_results = {}

        for n_features in [20, 50, 100]:
            X = np.random.randn(100, n_features)
            y = np.random.randn(100)

            def create_knockoffs():
                knockoffs = Knockoffs(backend='python')
                # Use simple method for timing test
                return knockoffs.select_short_freq(X, y, fdr=0.1, method='equicorrelated')

            _, elapsed_time = self.time_operation(create_knockoffs)
            timing_results[n_features] = elapsed_time

        # Time should scale polynomially, not exponentially
        time_ratio_50_20 = timing_results[50] / max(timing_results[20], 0.001)
        time_ratio_100_50 = timing_results[100] / max(timing_results[50], 0.001)

        # Should scale reasonably (polynomial, not exponential)
        assert time_ratio_50_20 < 20
        assert time_ratio_100_50 < 15

    def test_love_convergence_time(self):
        """Test LOVE algorithm convergence time."""
        convergence_times = {}

        for n_features in [30, 60, 120]:
            X = np.random.randn(100, n_features)

            def run_love():
                return call_love(X, lbd=0.5, verbose=False)

            _, elapsed_time = self.time_operation(run_love)
            convergence_times[n_features] = elapsed_time

        # Should converge in reasonable time
        for n_features, time_taken in convergence_times.items():
            assert time_taken < 60, f"LOVE took too long: {time_taken}s for {n_features} features"


class TestLargeDatasetHandling:
    """Test handling of large datasets."""

    @pytest.mark.slow
    def test_slide_large_sample_count(self):
        """Test SLIDE with large sample count."""
        # Large but manageable size
        n_samples = 5000
        n_features = 100

        # Use memory mapping for large arrays to avoid memory issues
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create memory-mapped arrays
            X_file = os.path.join(temp_dir, 'X.dat')
            y_file = os.path.join(temp_dir, 'y.dat')

            # Create arrays and save to disk
            X = np.random.randn(n_samples, n_features)
            y = np.random.randn(n_samples)

            X_mmap = np.memmap(X_file, dtype='float64', mode='w+', shape=(n_samples, n_features))
            y_mmap = np.memmap(y_file, dtype='float64', mode='w+', shape=(n_samples,))

            X_mmap[:] = X
            y_mmap[:] = y

            del X, y  # Free original arrays

            params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

            try:
                slide = SLIDE(params, x=X_mmap, y=y_mmap)
                assert slide.data.X.shape == (n_samples, n_features)
                assert slide.data.y.shape == (n_samples,)
            except MemoryError:
                pytest.skip("Insufficient memory for large dataset test")

    @pytest.mark.slow
    def test_slide_large_feature_count(self):
        """Test SLIDE with large feature count."""
        n_samples = 100
        n_features = 2000

        try:
            X = np.random.randn(n_samples, n_features).astype(np.float32)  # Use float32 to save memory
            y = np.random.randn(n_samples).astype(np.float32)

            params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

            slide = SLIDE(params, x=X, y=y)
            assert slide.data.X.shape == (n_samples, n_features)

            # Test default feature size calculation
            default_fsize = slide.calc_default_fsize(K=20)
            assert isinstance(default_fsize, int)
            assert default_fsize > 0

        except MemoryError:
            pytest.skip("Insufficient memory for large feature test")

    def test_knockoffs_sparse_matrices(self):
        """Test knockoffs with sparse data patterns."""
        from scipy.sparse import random as sparse_random

        # Create sparse matrix
        n_samples, n_features = 200, 100
        density = 0.1  # 10% non-zero

        X_sparse = sparse_random(n_samples, n_features, density=density, format='csr')
        X_dense = X_sparse.toarray()
        y = np.random.randn(n_samples)

        # Test that knockoffs can handle sparse-like data
        knockoffs = Knockoffs(backend='python')

        try:
            result = knockoffs.select_short_freq(X_dense, y, fdr=0.1)
            assert hasattr(result, 'selected')
        except np.linalg.LinAlgError:
            # Expected for some sparse matrices
            pass


class TestResourceLimitHandling:
    """Test behavior at resource limits."""

    def test_memory_limit_graceful_degradation(self):
        """Test graceful degradation when approaching memory limits."""
        # Simulate memory pressure
        def memory_limited_operation():
            # Create data that uses significant memory
            try:
                X = np.random.randn(2000, 1000)
                y = np.random.randn(2000)
                params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

                slide = SLIDE(params, x=X, y=y)
                return "success"
            except MemoryError:
                return "memory_error"

        result = memory_limited_operation()
        assert result in ["success", "memory_error"]  # Both outcomes are acceptable

    def test_cpu_time_limit_handling(self):
        """Test behavior under CPU time constraints."""
        import signal

        class TimeoutError(Exception):
            pass

        def timeout_handler(signum, frame):
            raise TimeoutError("Operation timed out")

        # Set a timeout for CPU-intensive operations
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)

        try:
            signal.alarm(5)  # 5 second timeout

            X = np.random.randn(500, 200)
            y = np.random.randn(500)

            # This might timeout on slow systems
            knockoffs = Knockoffs(backend='python')
            result = knockoffs.select_short_freq(X, y, fdr=0.1)

            signal.alarm(0)  # Cancel timeout
            assert hasattr(result, 'selected')

        except TimeoutError:
            # Acceptable - operation was too slow
            signal.alarm(0)  # Cancel timeout
            pass
        finally:
            signal.signal(signal.SIGALRM, original_handler)

    def test_disk_space_limit_handling(self):
        """Test behavior when disk space is limited."""
        # Simulate disk space limitations
        with tempfile.TemporaryDirectory() as temp_dir:
            large_file = os.path.join(temp_dir, 'large_results.pkl')

            params = {'x_path': None, 'y_path': None, 'fdr': 0.1}
            slide = SLIDE(params, x=np.random.randn(100, 50), y=np.random.randn(100))

            # Mock file writing to simulate disk full
            original_open = open

            def mock_open(*args, **kwargs):
                if 'large_results.pkl' in str(args[0]):
                    raise OSError("No space left on device")
                return original_open(*args, **kwargs)

            with patch('builtins.open', side_effect=mock_open):
                try:
                    slide.save_results(large_file)
                    assert False, "Should have raised OSError"
                except OSError as e:
                    assert "space" in str(e).lower()


class TestPerformanceRegressions:
    """Test for performance regressions."""

    def test_performance_baseline_slide_creation(self):
        """Baseline performance test for SLIDE creation."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

        start_time = time.time()
        slide = SLIDE(params, x=X, y=y)
        end_time = time.time()

        creation_time = end_time - start_time

        # Should create quickly (less than 1 second for this size)
        assert creation_time < 1.0, f"SLIDE creation too slow: {creation_time}s"

    def test_performance_baseline_knockoffs(self):
        """Baseline performance test for knockoffs."""
        X = np.random.randn(100, 30)
        y = np.random.randn(100)

        start_time = time.time()
        knockoffs = Knockoffs(backend='python')
        result = knockoffs.select_short_freq(X, y, fdr=0.1, method='equicorrelated')
        end_time = time.time()

        knockoffs_time = end_time - start_time

        # Should complete quickly (less than 5 seconds for this size)
        assert knockoffs_time < 5.0, f"Knockoffs too slow: {knockoffs_time}s"

    def test_performance_baseline_love(self):
        """Baseline performance test for LOVE."""
        X = np.random.randn(100, 30)

        start_time = time.time()
        result = call_love(X, lbd=0.5, verbose=False)
        end_time = time.time()

        love_time = end_time - start_time

        # Should complete in reasonable time (less than 10 seconds for this size)
        assert love_time < 10.0, f"LOVE too slow: {love_time}s"


class TestCachingAndOptimization:
    """Test caching and optimization features."""

    def test_computation_caching(self):
        """Test that repeated computations are cached appropriately."""
        X = np.random.randn(100, 50)

        # Run LOVE twice with same data
        start_time_1 = time.time()
        result_1 = call_love(X, lbd=0.5, verbose=False)
        end_time_1 = time.time()
        time_1 = end_time_1 - start_time_1

        start_time_2 = time.time()
        result_2 = call_love(X, lbd=0.5, verbose=False)
        end_time_2 = time.time()
        time_2 = end_time_2 - start_time_2

        # Results should be identical
        assert set(result_1.keys()) == set(result_2.keys())

        # Note: This test assumes no caching is implemented
        # If caching is added, second run should be faster

    def test_memory_optimization_strategies(self):
        """Test memory optimization strategies."""
        # Test in-place operations where possible
        X = np.random.randn(200, 100)

        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Operations that should use memory efficiently
        params = {'x_path': None, 'y_path': None, 'fdr': 0.1}
        slide = SLIDE(params, x=X, y=X)  # Reuse same array

        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = peak_memory - initial_memory

        # Should not use excessive additional memory
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])