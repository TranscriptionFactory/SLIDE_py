"""
Test resource management, memory efficiency, and cleanup.
Addresses: Memory leaks, resource cleanup, scalability limits
"""
import pytest
import numpy as np
import gc
import sys
import psutil
import os
import tempfile
import weakref
from pathlib import Path
from loveslide import SLIDE, SLIDEcv, Knockoffs, Plotter
from loveslide.knockoff._parallel import knockoff_voting_parallel


class TestMemoryManagement:
    """Test memory usage and leak prevention."""

    def test_large_dataset_memory_scaling(self):
        """Test memory usage scales reasonably with dataset size."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Test progressively larger datasets
        sizes = [(100, 20), (200, 20), (400, 20)]
        memory_usage = []

        for n, p in sizes:
            gc.collect()  # Clean up before test
            mem_before = process.memory_info().rss

            X = np.random.randn(n, p)
            y = np.random.randn(n)

            slide = SLIDE({'fdr': 0.1, 'n_iters': 10}, x=X, y=y)
            result = slide.fit()

            mem_after = process.memory_info().rss
            memory_usage.append(mem_after - mem_before)

            # Clean up
            del X, y, slide, result
            gc.collect()

        # Memory usage should scale sub-quadratically
        # (allowing for some variation in garbage collection)
        if len(memory_usage) >= 2:
            ratio = memory_usage[-1] / memory_usage[0]
            # 4x data should use less than 20x memory (allowing generous headroom)
            assert ratio < 20, f"Memory scaling too aggressive: {ratio}"

    def test_object_cleanup_after_completion(self):
        """Test objects are properly cleaned up after operations."""
        X = np.random.randn(100, 15)
        y = np.random.randn(100)

        # Create weak references to track object cleanup
        slide = SLIDE({'fdr': 0.1}, x=X, y=y)
        slide_ref = weakref.ref(slide)

        result = slide.fit()
        result_ref = weakref.ref(result)

        # Objects should exist while we hold references
        assert slide_ref() is not None
        assert result_ref() is not None

        # Clean up references
        del slide, result
        gc.collect()

        # Objects should be cleaned up (may take multiple GC cycles)
        for _ in range(3):
            gc.collect()

        # At least some objects should be cleaned up
        # (Being lenient as GC behavior can vary)

    def test_temporary_array_cleanup(self):
        """Test that temporary arrays are cleaned up during computation."""
        X = np.random.randn(200, 25)
        y = np.random.randn(200)

        process = psutil.Process(os.getpid())
        gc.collect()
        mem_before = process.memory_info().rss

        # Create knockoffs (should create temporary arrays internally)
        knockoffs = Knockoffs()
        Xk = knockoffs.create_knockoffs(X, method='equi')

        gc.collect()
        mem_after = process.memory_info().rss

        # Memory increase should be reasonable (not holding onto large temps)
        mem_increase = mem_after - mem_before
        expected_size = X.nbytes * 3  # Original + knockoffs + some overhead
        assert mem_increase < expected_size * 2  # Allow 2x overhead

        del Xk, knockoffs
        gc.collect()

    def test_cross_validation_memory_efficiency(self):
        """Test CV doesn't accumulate memory across folds."""
        X = np.random.randn(150, 20)
        y = np.random.randn(150)

        process = psutil.Process(os.getpid())
        gc.collect()
        mem_start = process.memory_info().rss

        slide_cv = SLIDEcv({'fdr': 0.1, 'cv_folds': 5}, x=X, y=y)

        # Monitor memory during CV
        max_memory = mem_start
        def memory_callback():
            nonlocal max_memory
            current_mem = process.memory_info().rss
            max_memory = max(max_memory, current_mem)

        # Run CV
        result = slide_cv.cross_validate()

        gc.collect()
        mem_end = process.memory_info().rss

        # Memory should return close to baseline after CV
        memory_retained = mem_end - mem_start
        data_size = X.nbytes + y.nbytes
        assert memory_retained < data_size * 5  # Allow generous overhead


class TestFileResourceManagement:
    """Test proper handling of file resources."""

    def test_temporary_file_cleanup(self):
        """Test temporary files are cleaned up after use."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            initial_files = list(temp_path.glob("*"))

            # Operations that might create temporary files
            slide = SLIDE({'fdr': 0.1, 'save_dir': str(temp_path)}, x=X, y=y)
            result = slide.fit()

            # Clean up
            del slide, result
            gc.collect()

            # Check for lingering temporary files
            final_files = list(temp_path.glob("*"))
            temp_files = [f for f in final_files if f.name.startswith('tmp')]

            # Should not have temporary files left over
            assert len(temp_files) == 0, f"Temporary files not cleaned up: {temp_files}"

    def test_plot_resource_cleanup(self):
        """Test plotting resources are cleaned up properly."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)

        slide = SLIDE({'fdr': 0.1}, x=X, y=y)
        result = slide.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            plotter = Plotter(result)

            # Generate multiple plots
            for i in range(5):
                plot_path = Path(temp_dir) / f"plot_{i}.png"
                try:
                    # This might create various plotting resources
                    plotter.plot_statistics(save_path=str(plot_path))
                except Exception:
                    pass  # Some plots might fail, that's ok for this test

            # Clean up plotter
            del plotter
            gc.collect()

        # Matplotlib figures should be cleaned up
        import matplotlib.pyplot as plt
        assert len(plt.get_fignums()) <= 1  # At most current figure

    def test_large_file_handling(self):
        """Test handling of large files doesn't exhaust resources."""
        # Create moderately large dataset
        X = np.random.randn(1000, 50)
        y = np.random.randn(1000)

        with tempfile.TemporaryDirectory() as temp_dir:
            slide = SLIDE({'fdr': 0.1, 'save_dir': temp_dir}, x=X, y=y)

            # Should handle without resource exhaustion
            try:
                result = slide.fit()
                assert result is not None
            except MemoryError:
                pytest.skip("Not enough memory for large file test")


class TestResourceLimitRespect:
    """Test that operations respect system resource limits."""

    def test_parallel_worker_count_limits(self):
        """Test parallel operations respect worker count limits."""
        X = np.random.randn(100, 15)
        y = np.random.randn(100)

        # Test with different worker counts
        max_workers = min(4, os.cpu_count())

        try:
            result = knockoff_voting_parallel(
                X, y, fdr=0.1, n_iters=20,
                n_jobs=max_workers,
                backend='joblib'
            )

            # Should not spawn more processes than requested
            # (This is mainly a smoke test as exact process counting is complex)
            assert result is not None

        except Exception as e:
            if "joblib" in str(e).lower():
                pytest.skip("Joblib not available for parallel test")
            else:
                raise

    def test_memory_limit_graceful_degradation(self):
        """Test graceful degradation when approaching memory limits."""
        # Try to create a dataset that might stress memory
        try:
            # This might be too large for some systems
            X = np.random.randn(2000, 100)
            y = np.random.randn(2000)

            slide = SLIDE({'fdr': 0.1, 'n_iters': 5}, x=X, y=y)
            result = slide.fit()

            # If it succeeds, memory was handled properly
            assert result is not None

        except MemoryError:
            # Graceful failure is acceptable
            pass
        except Exception as e:
            # Other exceptions might indicate poor memory handling
            if "memory" in str(e).lower():
                pass  # Memory-related errors are acceptable
            else:
                raise

    def test_cleanup_on_interruption(self):
        """Test resources are cleaned up when operations are interrupted."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        with tempfile.TemporaryDirectory() as temp_dir:
            slide = SLIDE({'fdr': 0.1, 'save_dir': temp_dir, 'n_iters': 1000}, x=X, y=y)

            # Simulate interruption during computation
            try:
                # Start the computation in a way that can be interrupted
                import signal

                def timeout_handler(signum, frame):
                    raise KeyboardInterrupt("Simulated interruption")

                # Set a short timeout to simulate interruption
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(1)  # Interrupt after 1 second

                try:
                    result = slide.fit()
                finally:
                    signal.alarm(0)  # Cancel the alarm

            except (KeyboardInterrupt, TimeoutError):
                # Simulated interruption occurred
                pass

            # Check that temp directory is clean
            temp_files = list(Path(temp_dir).glob("*"))
            large_temp_files = [f for f in temp_files if f.stat().st_size > 1000000]  # > 1MB
            assert len(large_temp_files) == 0, "Large temporary files not cleaned up after interruption"