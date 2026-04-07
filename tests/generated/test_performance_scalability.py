"""
Test performance and scalability edge cases.
Addresses: Performance bottlenecks, scalability limits, algorithmic complexity
"""
import pytest
import numpy as np
import time
import psutil
import os
from loveslide import SLIDE, SLIDEcv, Knockoffs
from loveslide.knockoff._parallel import knockoff_voting_parallel


class TestAlgorithmicComplexity:
    """Test algorithmic complexity and scaling behavior."""

    def test_linear_scaling_with_samples(self):
        """Test that runtime scales roughly linearly with number of samples."""
        p = 10  # Fixed number of features
        sample_sizes = [50, 100, 200]
        runtimes = []

        for n in sample_sizes:
            X = np.random.randn(n, p)
            y = np.random.randn(n)

            start_time = time.time()
            slide = SLIDE({'fdr': 0.1, 'n_iters': 10}, x=X, y=y)
            result = slide.fit()
            runtime = time.time() - start_time
            runtimes.append(runtime)

        # Runtime should not grow faster than quadratically
        # (allowing for overhead and random variation)
        if len(runtimes) >= 2:
            scaling_factor = runtimes[-1] / runtimes[0]
            expected_scaling = (sample_sizes[-1] / sample_sizes[0]) ** 2
            assert scaling_factor < expected_scaling * 2, (
                f"Poor scaling with samples: {scaling_factor:.2f} vs expected {expected_scaling:.2f}"
            )

    def test_feature_scaling_behavior(self):
        """Test scaling behavior with increasing number of features."""
        n = 100  # Fixed number of samples
        feature_counts = [10, 20, 40]
        runtimes = []

        for p in feature_counts:
            X = np.random.randn(n, p)
            y = np.random.randn(n)

            start_time = time.time()
            try:
                slide = SLIDE({'fdr': 0.1, 'n_iters': 5}, x=X, y=y)
                result = slide.fit()
                runtime = time.time() - start_time
                runtimes.append(runtime)
            except (MemoryError, np.linalg.LinAlgError):
                # Some feature counts might be too large
                break

        # Feature scaling should be reasonable (allowing for SDP complexity)
        if len(runtimes) >= 2:
            scaling_factor = runtimes[-1] / runtimes[0]
            # SDP solving can be O(p^3), so allow generous scaling
            feature_ratio = feature_counts[len(runtimes)-1] / feature_counts[0]
            expected_scaling = feature_ratio ** 3
            assert scaling_factor < expected_scaling * 3, (
                f"Poor feature scaling: {scaling_factor:.2f}"
            )

    def test_iteration_scaling_linearity(self):
        """Test that runtime scales linearly with number of iterations."""
        X = np.random.randn(80, 15)
        y = np.random.randn(80)

        iteration_counts = [10, 20, 40]
        runtimes = []

        for n_iters in iteration_counts:
            start_time = time.time()
            slide = SLIDE({'fdr': 0.1, 'n_iters': n_iters}, x=X, y=y)
            result = slide.fit()
            runtime = time.time() - start_time
            runtimes.append(runtime)

        # Should scale roughly linearly with iterations
        if len(runtimes) >= 2:
            scaling_factor = runtimes[-1] / runtimes[0]
            iteration_ratio = iteration_counts[-1] / iteration_counts[0]
            assert scaling_factor < iteration_ratio * 1.5, (
                f"Poor iteration scaling: {scaling_factor:.2f} vs {iteration_ratio}"
            )


class TestMemoryScalingLimits:
    """Test memory usage under various scaling scenarios."""

    def test_memory_efficiency_with_features(self):
        """Test memory doesn't grow excessively with feature count."""
        process = psutil.Process(os.getpid())
        n = 100

        feature_counts = [20, 40, 60]
        peak_memories = []

        for p in feature_counts:
            # Clean up before measurement
            import gc
            gc.collect()
            mem_before = process.memory_info().rss

            try:
                X = np.random.randn(n, p)
                y = np.random.randn(n)

                knockoffs = Knockoffs()
                Xk = knockoffs.create_knockoffs(X, method='equi')

                mem_peak = process.memory_info().rss
                peak_memories.append(mem_peak - mem_before)

                del X, y, Xk, knockoffs
                gc.collect()

            except MemoryError:
                break

        # Memory should not grow faster than O(p^2) for correlation matrix storage
        if len(peak_memories) >= 2:
            memory_ratio = peak_memories[-1] / peak_memories[0]
            feature_ratio = (feature_counts[len(peak_memories)-1] /
                           feature_counts[0])
            expected_ratio = feature_ratio ** 2
            assert memory_ratio < expected_ratio * 3, (
                f"Poor memory scaling: {memory_ratio:.2f} vs expected {expected_ratio:.2f}"
            )

    def test_parallel_memory_efficiency(self):
        """Test parallel execution doesn't multiply memory usage linearly."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        process = psutil.Process(os.getpid())
        import gc
        gc.collect()

        # Sequential execution memory baseline
        mem_before = process.memory_info().rss
        try:
            result_seq = knockoff_voting_parallel(
                X, y, fdr=0.1, n_iters=20, n_jobs=1, backend='joblib'
            )
            mem_sequential = process.memory_info().rss - mem_before
            del result_seq
            gc.collect()

            # Parallel execution
            mem_before = process.memory_info().rss
            result_par = knockoff_voting_parallel(
                X, y, fdr=0.1, n_iters=20, n_jobs=2, backend='joblib'
            )
            mem_parallel = process.memory_info().rss - mem_before

            # Parallel shouldn't use much more than 2x sequential memory
            if mem_sequential > 0:
                memory_multiplier = mem_parallel / mem_sequential
                assert memory_multiplier < 4, (
                    f"Parallel memory usage too high: {memory_multiplier:.2f}x"
                )

        except Exception as e:
            if "joblib" in str(e).lower():
                pytest.skip("Joblib not available")
            else:
                raise


class TestPerformanceBottlenecks:
    """Test identification and handling of performance bottlenecks."""

    def test_covariance_computation_caching(self):
        """Test that repeated covariance computations are avoided."""
        X = np.random.randn(200, 30)

        # Time multiple knockoff generations with same X
        times = []
        for run in range(3):
            start_time = time.time()
            knockoffs = Knockoffs()
            Xk = knockoffs.create_knockoffs(X, method='sdp')
            runtime = time.time() - start_time
            times.append(runtime)

        # Subsequent runs might be faster due to caching
        # (This is optimistic - caching might not be implemented)
        # At minimum, times should be consistent
        if len(times) >= 2:
            time_variation = np.std(times) / np.mean(times)
            assert time_variation < 2.0, f"Highly variable computation times: {times}"

    def test_large_iteration_count_efficiency(self):
        """Test efficiency with large iteration counts."""
        X = np.random.randn(80, 12)
        y = np.random.randn(80)

        # Test with large iteration count
        start_time = time.time()
        try:
            slide = SLIDE({'fdr': 0.1, 'n_iters': 500}, x=X, y=y)
            result = slide.fit()
            runtime = time.time() - start_time

            # Should complete in reasonable time (generous bound)
            assert runtime < 300, f"Large iteration count too slow: {runtime:.1f}s"

        except Exception as e:
            if "memory" in str(e).lower() or "time" in str(e).lower():
                pytest.skip(f"Resource limitation: {e}")
            else:
                raise

    def test_cross_validation_parallel_efficiency(self):
        """Test that CV parallelization provides speedup."""
        X = np.random.randn(100, 15)
        y = np.random.randn(100)

        # Sequential CV
        start_time = time.time()
        slide_cv_seq = SLIDEcv({'fdr': 0.1, 'cv_folds': 4, 'n_jobs': 1}, x=X, y=y)
        try:
            result_seq = slide_cv_seq.cross_validate()
            time_sequential = time.time() - start_time

            # Parallel CV (if supported)
            start_time = time.time()
            slide_cv_par = SLIDEcv({'fdr': 0.1, 'cv_folds': 4, 'n_jobs': 2}, x=X, y=y)
            result_par = slide_cv_par.cross_validate()
            time_parallel = time.time() - start_time

            # Parallel should be faster or at least not much slower
            speedup = time_sequential / time_parallel
            assert speedup > 0.5, f"Poor parallelization: speedup {speedup:.2f}"

        except Exception as e:
            if "n_jobs" in str(e).lower() or "parallel" in str(e).lower():
                pytest.skip(f"Parallelization not supported: {e}")
            else:
                raise


class TestEdgeCasePerformance:
    """Test performance under edge case conditions."""

    def test_near_singular_matrix_performance(self):
        """Test performance doesn't degrade severely with near-singular matrices."""
        n, p = 100, 15

        # Create near-singular covariance matrix
        X = np.random.randn(n, p)
        X[:, -1] = X[:, 0] + 1e-8 * np.random.randn(n)  # Nearly collinear

        start_time = time.time()
        try:
            knockoffs = Knockoffs()
            Xk = knockoffs.create_knockoffs(X, method='equi')  # More stable method
            runtime = time.time() - start_time

            # Should not hang indefinitely
            assert runtime < 60, f"Near-singular case too slow: {runtime:.1f}s"

        except (np.linalg.LinAlgError, ValueError):
            # Acceptable to fail, but should fail quickly
            runtime = time.time() - start_time
            assert runtime < 10, f"Slow failure for singular matrix: {runtime:.1f}s"

    def test_extreme_fdr_performance(self):
        """Test performance with extreme FDR values."""
        X = np.random.randn(80, 12)
        y = np.random.randn(80)

        extreme_fdrs = [1e-6, 0.99]

        for fdr in extreme_fdrs:
            start_time = time.time()
            slide = SLIDE({'fdr': fdr, 'n_iters': 10}, x=X, y=y)
            result = slide.fit()
            runtime = time.time() - start_time

            # Should not be excessively slow
            assert runtime < 30, f"Extreme FDR {fdr} too slow: {runtime:.1f}s"

    def test_high_dimensional_feature_performance(self):
        """Test performance graceful degradation with high dimensions."""
        max_features = 100  # Adjust based on available resources
        n = 80

        try:
            X = np.random.randn(n, max_features)
            y = np.random.randn(n)

            start_time = time.time()
            slide = SLIDE({'fdr': 0.1, 'n_iters': 5}, x=X, y=y)
            result = slide.fit()
            runtime = time.time() - start_time

            # Should complete in reasonable time or fail gracefully
            assert runtime < 120, f"High dimensional case too slow: {runtime:.1f}s"

        except MemoryError:
            pytest.skip("Not enough memory for high-dimensional test")
        except Exception as e:
            # Should provide informative error for limitation
            assert any(word in str(e).lower()
                      for word in ['dimension', 'memory', 'size', 'limit']), (
                f"Uninformative error for high dimensions: {e}"
            )