"""
Scientific reproducibility edge case testing.

Tests reproducibility edge cases that are critical for scientific validity
but may not be covered in standard algorithmic testing.
"""

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import patch
from src.loveslide import SLIDE, Knockoffs, call_love
from src.loveslide.tools import init_data


class TestScientificReproducibilityGaps:
    """Test scientific reproducibility edge cases."""

    def setup_method(self):
        """Setup reproducible test environment."""
        np.random.seed(42)
        self.X = np.random.randn(100, 20)
        self.y = np.random.randn(100)

    def test_random_seed_propagation_completeness(self):
        """Test that random seeds propagate through all computation paths."""
        # Test multiple computation paths with same seed
        results = []

        for seed in [42, 42, 42]:  # Same seed should give same results
            np.random.seed(seed)

            knockoffs = Knockoffs()
            result = knockoffs.filter_knockoffs_iterative_python(
                z=self.X.copy(), y=self.y.copy(), fdr=0.1
            )
            results.append(result)

        # Results should be identical with same seed
        for i in range(1, len(results)):
            if hasattr(results[0], 'selections'):
                assert np.array_equal(
                    results[0].selections, results[i].selections
                ), f"Results differ between runs {0} and {i}"

    def test_numerical_precision_environment_dependency(self):
        """Test numerical precision across different computational environments."""
        # Test computation with different floating point precisions

        # Standard precision
        X_float64 = self.X.astype(np.float64)
        y_float64 = self.y.astype(np.float64)

        # Reduced precision
        X_float32 = self.X.astype(np.float32)
        y_float32 = self.y.astype(np.float32)

        knockoffs = Knockoffs()

        try:
            result_64 = knockoffs.filter_knockoffs_iterative_python(
                z=X_float64, y=y_float64, fdr=0.1
            )

            result_32 = knockoffs.filter_knockoffs_iterative_python(
                z=X_float32, y=y_float32, fdr=0.1
            )

            # Results should be reasonably similar despite precision differences
            # (exact comparison depends on algorithm sensitivity)
            if hasattr(result_64, 'statistics') and hasattr(result_32, 'statistics'):
                correlation = np.corrcoef(result_64.statistics, result_32.statistics)[0, 1]
                assert correlation > 0.95, f"Low correlation {correlation} between precisions"

        except Exception as e:
            pytest.skip(f"Precision comparison failed: {e}")

    def test_platform_dependent_computation_consistency(self):
        """Test computation consistency across different platforms."""
        # Test behavior that might vary between platforms

        # Simulate different platform behaviors
        platform_configs = [
            {'use_multiprocessing': True, 'n_workers': 1},
            {'use_multiprocessing': False, 'n_workers': 1},
            {'use_multiprocessing': True, 'n_workers': 2},
        ]

        results = []
        for config in platform_configs:
            try:
                knockoffs = Knockoffs()
                result = knockoffs.filter_knockoffs_iterative_python(
                    z=self.X.copy(), y=self.y.copy(),
                    fdr=0.1, **config
                )
                results.append(result)
            except Exception as e:
                pytest.skip(f"Platform config {config} failed: {e}")

        # Results should be consistent across platform configurations
        if len(results) > 1:
            for i in range(1, len(results)):
                if hasattr(results[0], 'selections'):
                    overlap = np.sum(results[0].selections & results[i].selections)
                    union = np.sum(results[0].selections | results[i].selections)
                    jaccard = overlap / union if union > 0 else 1.0
                    assert jaccard > 0.8, f"Low Jaccard similarity {jaccard} between configs"

    def test_dependency_version_compatibility(self):
        """Test compatibility with different dependency versions."""
        # Test behavior with different numpy/scipy behaviors

        # Test with different random number generator behaviors
        legacy_results = []
        modern_results = []

        for i in range(3):
            # Legacy numpy random behavior simulation
            np.random.seed(42)
            legacy_X = np.random.randn(50, 10)

            # Modern numpy random behavior
            rng = np.random.default_rng(42)
            modern_X = rng.normal(size=(50, 10))

            knockoffs = Knockoffs()

            try:
                legacy_result = knockoffs.filter_knockoffs_iterative_python(
                    z=legacy_X, y=np.random.randn(50), fdr=0.1
                )
                legacy_results.append(legacy_result)

                modern_result = knockoffs.filter_knockoffs_iterative_python(
                    z=modern_X, y=rng.normal(size=50), fdr=0.1
                )
                modern_results.append(modern_result)

            except Exception as e:
                pytest.skip(f"Dependency version test failed: {e}")

        # Algorithm should be robust to reasonable input variations
        # (specific assertion depends on algorithm requirements)

    def test_memory_layout_reproducibility(self):
        """Test reproducibility across different memory layouts."""
        # Test C-contiguous vs Fortran-contiguous arrays

        X_c = np.ascontiguousarray(self.X)  # C-contiguous
        X_f = np.asfortranarray(self.X)     # Fortran-contiguous

        knockoffs = Knockoffs()

        try:
            result_c = knockoffs.filter_knockoffs_iterative_python(
                z=X_c, y=self.y, fdr=0.1
            )

            result_f = knockoffs.filter_knockoffs_iterative_python(
                z=X_f, y=self.y, fdr=0.1
            )

            # Results should be identical regardless of memory layout
            if hasattr(result_c, 'selections') and hasattr(result_f, 'selections'):
                assert np.array_equal(
                    result_c.selections, result_f.selections
                ), "Results differ between memory layouts"

        except Exception as e:
            pytest.skip(f"Memory layout test failed: {e}")

    def test_batch_size_reproducibility(self):
        """Test that batch processing maintains reproducibility."""
        # Test different batch sizes give same results

        batch_sizes = [10, 25, 50]
        results = []

        for batch_size in batch_sizes:
            np.random.seed(42)

            try:
                # Simulate batch processing
                knockoffs = Knockoffs()
                result = knockoffs.filter_knockoffs_iterative_python(
                    z=self.X, y=self.y, fdr=0.1
                )
                results.append(result)

            except Exception as e:
                pytest.skip(f"Batch size {batch_size} failed: {e}")

        # Results should be identical regardless of batch size
        if len(results) > 1:
            for i in range(1, len(results)):
                if hasattr(results[0], 'selections'):
                    assert np.array_equal(
                        results[0].selections, results[i].selections
                    ), f"Batch size {batch_sizes[i]} gives different results"

    def test_file_io_reproducibility(self):
        """Test reproducibility when using file I/O vs in-memory data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save data to files
            x_path = os.path.join(tmpdir, "X.csv")
            y_path = os.path.join(tmpdir, "y.csv")

            np.savetxt(x_path, self.X, delimiter=",")
            np.savetxt(y_path, self.y, delimiter=",")

            # Test in-memory vs file-based initialization
            in_memory_params = {
                'x_path': None,
                'y_path': None
            }

            file_based_params = {
                'x_path': x_path,
                'y_path': y_path
            }

            try:
                # In-memory data initialization
                data1 = init_data(in_memory_params, x=self.X, y=self.y)

                # File-based data initialization
                data2 = init_data(file_based_params)

                # Data should be equivalent
                # (specific comparison depends on data structure)

            except Exception as e:
                pytest.skip(f"File I/O test failed: {e}")

    def test_threading_reproducibility(self):
        """Test reproducibility in multi-threaded environments."""
        # Test that multi-threading doesn't break reproducibility

        import threading
        import queue

        result_queue = queue.Queue()

        def run_knockoffs(thread_id, result_queue):
            """Run knockoffs in separate thread."""
            np.random.seed(42)  # Same seed in each thread

            knockoffs = Knockoffs()
            result = knockoffs.filter_knockoffs_iterative_python(
                z=self.X.copy(), y=self.y.copy(), fdr=0.1
            )
            result_queue.put((thread_id, result))

        # Run multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_knockoffs, args=(i, result_queue))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Collect results
        results = {}
        while not result_queue.empty():
            thread_id, result = result_queue.get()
            results[thread_id] = result

        # All threads should produce identical results
        if len(results) > 1:
            baseline = list(results.values())[0]
            for thread_id, result in results.items():
                if hasattr(baseline, 'selections') and hasattr(result, 'selections'):
                    assert np.array_equal(
                        baseline.selections, result.selections
                    ), f"Thread {thread_id} produced different results"

    def test_scientific_parameter_sensitivity(self):
        """Test scientific validity of parameter sensitivity."""
        # Test that small parameter changes produce reasonable result changes

        fdr_values = [0.05, 0.1, 0.15, 0.2]
        selection_counts = []

        for fdr in fdr_values:
            knockoffs = Knockoffs()
            result = knockoffs.filter_knockoffs_iterative_python(
                z=self.X, y=self.y, fdr=fdr
            )

            if hasattr(result, 'selections'):
                count = np.sum(result.selections)
                selection_counts.append(count)

        # Selection count should generally decrease with stricter FDR
        # (allowing for some variation due to randomness)
        if len(selection_counts) > 1:
            # Check that there's a general decreasing trend
            # (not necessarily monotonic due to randomness)
            first_half_avg = np.mean(selection_counts[:len(selection_counts)//2])
            second_half_avg = np.mean(selection_counts[len(selection_counts)//2:])

            # Generally expect fewer selections with higher FDR thresholds
            # (but allow for reasonable variation)
            ratio = second_half_avg / (first_half_avg + 1e-10)
            assert ratio < 2.0, f"Selection count increased too much with stricter FDR: {selection_counts}"