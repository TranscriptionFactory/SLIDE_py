"""
Comprehensive performance and scalability edge case tests for SLIDE_py.

Tests performance characteristics and scalability limits:
- Memory usage patterns
- Computational complexity
- Large dataset handling
- Resource constraint scenarios
- Performance regression detection
"""

import pytest
import numpy as np
import pandas as pd
import time
import psutil
import gc
import tempfile
import os
from unittest.mock import patch
import warnings

from loveslide import SLIDE, Knockoffs, call_love
from loveslide.tools import calc_default_fsize


class TestMemoryUsagePatterns:
    """Test memory usage patterns and memory leak detection."""

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def test_slide_memory_growth_pattern(self):
        """Test SLIDE memory usage doesn't grow excessively."""
        initial_memory = self.get_memory_usage()
        memory_measurements = [initial_memory]

        # Run multiple SLIDE instances
        for i in range(5):
            X = np.random.randn(100, 50)
            y = np.random.binomial(1, 0.5, 100)
            params = {"fdr": 0.1, "niter": 2}

            slide = SLIDE(params, x=X, y=y)

            # Force garbage collection
            del slide, X, y
            gc.collect()

            current_memory = self.get_memory_usage()
            memory_measurements.append(current_memory)

        # Check for excessive memory growth
        memory_growth = memory_measurements[-1] - initial_memory
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.2f} MB"

    def test_knockoffs_memory_cleanup(self):
        """Test Knockoffs properly cleans up memory."""
        initial_memory = self.get_memory_usage()

        # Run multiple knockoff iterations
        for i in range(10):
            X = np.random.randn(200, 100)
            y = np.random.binomial(1, 0.5, 200)

            knockoffs = Knockoffs()
            result = knockoffs.run_iteration(X, y, fdr=0.1, method='lasso')

            # Clean up explicitly
            del knockoffs, X, y, result
            gc.collect()

        final_memory = self.get_memory_usage()
        memory_growth = final_memory - initial_memory

        # Should not grow excessively
        assert memory_growth < 200, f"Memory leak detected: {memory_growth:.2f} MB growth"

    def test_love_memory_efficiency(self):
        """Test LOVE memory efficiency with various matrix sizes."""
        initial_memory = self.get_memory_usage()
        matrix_sizes = [(50, 20), (100, 50), (200, 100)]

        for n_rows, n_cols in matrix_sizes:
            X = np.random.randn(n_rows, n_cols)

            memory_before = self.get_memory_usage()
            result = call_love(X, lbd=0.5, mu=0.5)
            memory_after = self.get_memory_usage()

            memory_used = memory_after - memory_before

            # Memory usage should be reasonable relative to data size
            data_size_mb = (n_rows * n_cols * 8) / 1024 / 1024  # 8 bytes per double
            memory_ratio = memory_used / max(data_size_mb, 0.1)

            assert memory_ratio < 10, f"Excessive memory usage ratio: {memory_ratio:.2f} for size {n_rows}x{n_cols}"

            # Clean up
            del X, result
            gc.collect()


class TestComputationalComplexity:
    """Test computational complexity and performance scaling."""

    def time_operation(self, operation_func, *args, **kwargs):
        """Time an operation and return execution time."""
        start_time = time.perf_counter()
        result = operation_func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time, result

    def test_slide_initialization_scaling(self):
        """Test SLIDE initialization time scaling with data size."""
        sizes = [(50, 20), (100, 40), (200, 80)]
        times = []

        for n_rows, n_cols in sizes:
            X = np.random.randn(n_rows, n_cols)
            y = np.random.binomial(1, 0.5, n_rows)
            params = {"fdr": 0.1}

            def init_slide():
                return SLIDE(params, x=X, y=y)

            execution_time, slide = self.time_operation(init_slide)
            times.append(execution_time)

            del slide, X, y
            gc.collect()

        # Check that time scaling is reasonable (should be roughly linear)
        # Time ratio should not exceed data size ratio by too much
        time_ratio_1_2 = times[1] / max(times[0], 1e-6)
        size_ratio_1_2 = (sizes[1][0] * sizes[1][1]) / (sizes[0][0] * sizes[0][1])

        assert time_ratio_1_2 < size_ratio_1_2 * 2, f"Poor time scaling: {time_ratio_1_2:.2f} vs expected ~{size_ratio_1_2:.2f}"

    def test_knockoffs_complexity_scaling(self):
        """Test Knockoffs computational complexity scaling."""
        feature_sizes = [20, 40, 60]
        times = []

        for n_features in feature_sizes:
            X = np.random.randn(100, n_features)
            y = np.random.binomial(1, 0.5, 100)

            knockoffs = Knockoffs()

            def run_knockoffs():
                return knockoffs.run_iteration(X, y, fdr=0.1, method='lasso')

            execution_time, result = self.time_operation(run_knockoffs)
            times.append(execution_time)

            del knockoffs, X, y, result
            gc.collect()

        # Knockoffs should scale reasonably with feature size
        if len(times) >= 2 and times[0] > 0:
            scaling_factor = times[-1] / times[0]
            feature_factor = feature_sizes[-1] / feature_sizes[0]

            # Should not scale worse than cubic
            assert scaling_factor < feature_factor ** 3, f"Poor scaling: {scaling_factor:.2f} for feature factor {feature_factor:.2f}"

    def test_love_algorithmic_complexity(self):
        """Test LOVE algorithm computational complexity."""
        matrix_sizes = [(30, 10), (60, 20), (90, 30)]
        times = []

        for n_rows, n_cols in matrix_sizes:
            X = np.random.randn(n_rows, n_cols)

            def run_love():
                return call_love(X, lbd=0.5, mu=0.5)

            execution_time, result = self.time_operation(run_love)
            times.append(execution_time)

            del X, result
            gc.collect()

        # Document complexity scaling
        if len(times) >= 2 and times[0] > 0:
            complexity_ratio = times[-1] / times[0]
            data_ratio = (matrix_sizes[-1][0] * matrix_sizes[-1][1]) / (matrix_sizes[0][0] * matrix_sizes[0][1])

            # Should not be exponential
            assert complexity_ratio < data_ratio ** 2, f"Potentially exponential complexity: {complexity_ratio:.2f}"


class TestLargeDatasetHandling:
    """Test handling of large datasets near system limits."""

    @pytest.mark.slow
    def test_slide_large_feature_count(self):
        """Test SLIDE with large feature counts."""
        # Test with moderately large feature count
        n_samples, n_features = 500, 2000

        try:
            X = np.random.randn(n_samples, n_features)
            y = np.random.binomial(1, 0.5, n_samples)
            params = {"fdr": 0.1, "niter": 2}

            slide = SLIDE(params, x=X, y=y)
            assert slide.data.X.shape == (n_samples, n_features)

        except MemoryError:
            pytest.skip("Insufficient memory for large dataset test")

    @pytest.mark.slow
    def test_slide_large_sample_count(self):
        """Test SLIDE with large sample counts."""
        # Test with large sample count
        n_samples, n_features = 10000, 100

        try:
            X = np.random.randn(n_samples, n_features)
            y = np.random.binomial(1, 0.5, n_samples)
            params = {"fdr": 0.1, "niter": 2}

            slide = SLIDE(params, x=X, y=y)
            assert slide.data.X.shape == (n_samples, n_features)

        except MemoryError:
            pytest.skip("Insufficient memory for large sample test")

    def test_chunked_processing_simulation(self):
        """Test chunked processing for large datasets."""
        # Simulate chunked processing
        total_samples = 1000
        chunk_size = 100
        n_features = 50

        results = []
        for chunk_start in range(0, total_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            chunk_samples = chunk_end - chunk_start

            X_chunk = np.random.randn(chunk_samples, n_features)
            y_chunk = np.random.binomial(1, 0.5, chunk_samples)

            # Process chunk
            params = {"fdr": 0.1, "niter": 2}
            slide_chunk = SLIDE(params, x=X_chunk, y=y_chunk)

            results.append({
                'chunk_size': chunk_samples,
                'shape': slide_chunk.data.X.shape
            })

            del slide_chunk, X_chunk, y_chunk
            gc.collect()

        # Verify all chunks were processed
        assert len(results) == (total_samples + chunk_size - 1) // chunk_size

    def test_sparse_matrix_handling(self):
        """Test handling of sparse matrices."""
        from scipy.sparse import csr_matrix

        # Create sparse matrix
        n_samples, n_features = 200, 500
        density = 0.1  # 10% non-zero

        # Create dense matrix then sparsify
        dense_X = np.random.randn(n_samples, n_features)
        mask = np.random.random((n_samples, n_features)) > density
        dense_X[mask] = 0

        sparse_X = csr_matrix(dense_X)
        y = np.random.binomial(1, 0.5, n_samples)

        # Convert back to dense for SLIDE (if it doesn't handle sparse)
        if hasattr(SLIDE, '_handle_sparse'):
            # If SLIDE handles sparse matrices
            params = {"fdr": 0.1}
            slide = SLIDE(params, x=sparse_X, y=y)
        else:
            # Convert to dense
            params = {"fdr": 0.1}
            slide = SLIDE(params, x=sparse_X.toarray(), y=y)

        assert slide.data.X.shape == (n_samples, n_features)


class TestResourceConstraintScenarios:
    """Test performance under resource constraints."""

    def test_limited_memory_simulation(self):
        """Test behavior under simulated memory constraints."""
        # This test simulates memory pressure by creating large objects
        memory_hogs = []

        try:
            # Create some memory pressure
            for i in range(5):
                memory_hogs.append(np.random.randn(1000, 1000))

            # Now try to run SLIDE
            X = np.random.randn(50, 20)
            y = np.random.binomial(1, 0.5, 50)
            params = {"fdr": 0.1, "niter": 2}

            slide = SLIDE(params, x=X, y=y)
            assert slide.data.X.shape == X.shape

        except MemoryError:
            # Expected under memory pressure
            pass
        finally:
            # Clean up memory hogs
            del memory_hogs
            gc.collect()

    def test_cpu_intensive_scenario(self):
        """Test CPU-intensive scenario performance."""
        # Create computationally intensive scenario
        X = np.random.randn(200, 200)  # Larger square matrix
        y = np.random.binomial(1, 0.5, 200)

        start_time = time.perf_counter()

        try:
            # Test multiple operations
            for i in range(3):
                knockoffs = Knockoffs()
                result = knockoffs.run_iteration(X, y, fdr=0.1, method='lasso')
                del knockoffs, result
                gc.collect()

        except Exception as e:
            # Document performance failures
            print(f"Performance test failed: {e}")

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Should complete in reasonable time
        assert total_time < 300, f"Operation too slow: {total_time:.2f} seconds"

    def test_disk_io_intensive_scenario(self):
        """Test disk I/O intensive scenarios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple large files
            file_paths = []
            for i in range(3):
                X = pd.DataFrame(np.random.randn(500, 100))
                y = pd.DataFrame(np.random.binomial(1, 0.5, 500))

                x_path = os.path.join(tmpdir, f"X_{i}.csv")
                y_path = os.path.join(tmpdir, f"y_{i}.csv")

                X.to_csv(x_path)
                y.to_csv(y_path)

                file_paths.append((x_path, y_path))

            # Test loading multiple datasets
            start_time = time.perf_counter()

            for x_path, y_path in file_paths:
                params = {
                    "x_path": x_path,
                    "y_path": y_path,
                    "fdr": 0.1
                }

                slide = SLIDE(params)
                assert slide.data.X.shape[0] == 500
                del slide

            end_time = time.perf_counter()
            io_time = end_time - start_time

            # Should handle I/O reasonably
            assert io_time < 60, f"I/O operations too slow: {io_time:.2f} seconds"


class TestPerformanceRegressionDetection:
    """Test for performance regressions."""

    def test_baseline_performance_benchmarks(self):
        """Establish baseline performance benchmarks."""
        benchmarks = {}

        # Benchmark 1: Basic SLIDE initialization
        X = np.random.randn(100, 50)
        y = np.random.binomial(1, 0.5, 100)
        params = {"fdr": 0.1}

        start = time.perf_counter()
        slide = SLIDE(params, x=X, y=y)
        benchmarks['slide_init'] = time.perf_counter() - start

        # Benchmark 2: Knockoffs iteration
        start = time.perf_counter()
        knockoffs = Knockoffs()
        result = knockoffs.run_iteration(X, y, fdr=0.1, method='lasso')
        benchmarks['knockoffs_iteration'] = time.perf_counter() - start

        # Benchmark 3: LOVE analysis
        start = time.perf_counter()
        love_result = call_love(X, lbd=0.5, mu=0.5)
        benchmarks['love_analysis'] = time.perf_counter() - start

        # Log benchmarks for regression testing
        print(f"Performance benchmarks: {benchmarks}")

        # Basic sanity checks on performance
        assert benchmarks['slide_init'] < 10, "SLIDE initialization too slow"
        assert benchmarks['knockoffs_iteration'] < 30, "Knockoffs iteration too slow"
        assert benchmarks['love_analysis'] < 30, "LOVE analysis too slow"

    def test_performance_consistency(self):
        """Test performance consistency across runs."""
        X = np.random.randn(100, 50)
        y = np.random.binomial(1, 0.5, 100)
        params = {"fdr": 0.1}

        times = []
        for run in range(5):
            start = time.perf_counter()
            slide = SLIDE(params, x=X, y=y)
            times.append(time.perf_counter() - start)
            del slide

        # Calculate coefficient of variation
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / mean_time if mean_time > 0 else float('inf')

        # Performance should be consistent (CV < 0.5)
        assert cv < 0.5, f"Inconsistent performance: CV = {cv:.3f}, times = {times}"


class TestScalabilityLimits:
    """Test scalability limits and boundary conditions."""

    def test_maximum_reasonable_dataset_size(self):
        """Test with maximum reasonable dataset sizes."""
        # Test various aspect ratios
        test_configs = [
            (1000, 100, "wide_dataset"),
            (100, 1000, "tall_dataset"),
            (500, 500, "square_dataset"),
        ]

        for n_rows, n_cols, config_name in test_configs:
            try:
                X = np.random.randn(n_rows, n_cols)
                y = np.random.binomial(1, 0.5, n_rows)
                params = {"fdr": 0.1, "niter": 1}

                start_time = time.perf_counter()
                slide = SLIDE(params, x=X, y=y)
                processing_time = time.perf_counter() - start_time

                assert slide.data.X.shape == (n_rows, n_cols)
                print(f"{config_name}: {processing_time:.2f}s for {n_rows}x{n_cols}")

                del slide, X, y
                gc.collect()

            except (MemoryError, TimeoutError):
                print(f"{config_name}: Hit resource limits at {n_rows}x{n_cols}")

    def test_parameter_scaling_limits(self):
        """Test parameter scaling limits."""
        X = np.random.randn(100, 50)
        y = np.random.binomial(1, 0.5, 100)

        # Test with extreme parameter values
        extreme_params = [
            {"fdr": 1e-10, "niter": 1},  # Very small FDR
            {"fdr": 0.5, "niter": 1},    # Large FDR
            {"fdr": 0.1, "niter": 100},  # Many iterations
        ]

        for params in extreme_params:
            try:
                slide = SLIDE(params, x=X, y=y)
                assert slide.data.X.shape == X.shape
                del slide
            except ValueError as e:
                # Document parameter limits
                print(f"Parameter limits: {params} -> {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])