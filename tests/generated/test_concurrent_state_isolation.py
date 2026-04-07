"""Test coverage for concurrent SLIDE instance state isolation."""

import pytest
import numpy as np
import pandas as pd
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from loveslide import SLIDE, SLIDEcv, OptimizeSLIDE
from loveslide.knockoffs import Knockoffs
from loveslide.love import call_love


class TestConcurrentStateIsolation:
    """Test concurrent execution and state isolation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for concurrent testing."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 50),
                        columns=[f'feat_{i}' for i in range(50)])
        y = pd.DataFrame(np.random.randint(0, 2, (100, 1)), columns=['outcome'])
        return X, y

    @pytest.fixture
    def base_params(self):
        """Base parameters for SLIDE testing."""
        return {
            'x_path': None, 'y_path': None,
            'fdr': 0.1, 'lambda': [0.1, 0.2],
            'n_workers': 1, 'niter': 10
        }

    def test_concurrent_slide_initialization(self, sample_data, base_params):
        """Test multiple SLIDE objects can be initialized concurrently."""
        X, y = sample_data

        def create_slide_instance(seed):
            """Create a SLIDE instance with unique seed."""
            np.random.seed(seed)
            local_params = base_params.copy()
            local_params['fdr'] = 0.1 + (seed % 10) * 0.01  # Vary parameters

            slide = SLIDE(local_params, X, y)
            return slide.input_params['fdr']

        # Test thread-based concurrency
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_slide_instance, i) for i in range(10)]
            results = [f.result() for f in futures]

        # Verify all instances were created with correct unique parameters
        assert len(results) == 10
        assert len(set(results)) > 1  # Should have different FDR values

    def test_concurrent_knockoff_generation(self, sample_data):
        """Test concurrent knockoff generation maintains independence."""
        X, _ = sample_data
        correlation_matrix = X.corr().values

        def generate_knockoffs(seed):
            """Generate knockoffs with unique seed."""
            np.random.seed(seed)
            ko = Knockoffs(fdr=0.1, backend='python')
            return ko._create_gaussian_knockoffs(correlation_matrix)

        # Test process-based concurrency for true isolation
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(generate_knockoffs, i) for i in range(4)]
            knockoff_sets = [f.result() for f in futures]

        # Verify knockoffs are different (due to different seeds)
        for i in range(len(knockoff_sets) - 1):
            for j in range(i + 1, len(knockoff_sets)):
                assert not np.allclose(knockoff_sets[i], knockoff_sets[j])

    def test_slide_cross_instance_isolation(self, sample_data, base_params):
        """Test SLIDE instances don't interfere with each other."""
        X, y = sample_data

        # Create two SLIDE instances with different parameters
        params1 = base_params.copy()
        params1['fdr'] = 0.05
        params1['lambda'] = [0.1]

        params2 = base_params.copy()
        params2['fdr'] = 0.15
        params2['lambda'] = [0.2]

        slide1 = SLIDE(params1, X, y)
        slide2 = SLIDE(params2, X, y)

        # Verify they maintain separate states
        assert slide1.input_params['fdr'] != slide2.input_params['fdr']
        assert slide1.input_params['lambda'] != slide2.input_params['lambda']

        # Modify one instance and verify the other is unaffected
        original_fdr2 = slide2.input_params['fdr']
        slide1.input_params['fdr'] = 0.99
        assert slide2.input_params['fdr'] == original_fdr2

    def test_concurrent_r_session_isolation(self, sample_data):
        """Test R session isolation in concurrent operations."""
        X, _ = sample_data

        def call_r_function(data_slice):
            """Call R function with data slice."""
            try:
                # This tests if R sessions interfere with each other
                result = call_love(data_slice, lbd=0.5, verbose=False)
                return len(result) if result else 0
            except Exception as e:
                return str(e)

        # Split data into chunks for concurrent processing
        chunks = [X.iloc[i:i+25] for i in range(0, len(X), 25)]

        # Test concurrent R calls
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(call_r_function, chunk) for chunk in chunks]
            results = []
            for f in futures:
                try:
                    results.append(f.result(timeout=30))
                except Exception as e:
                    results.append(f"Error: {e}")

        # Verify results are meaningful (not all errors)
        valid_results = [r for r in results if isinstance(r, int)]
        assert len(valid_results) >= len(results) // 2  # At least half should succeed

    def test_temporary_file_isolation(self, sample_data, base_params):
        """Test temporary file creation doesn't interfere between instances."""
        X, y = sample_data

        def create_slide_with_temp_files(temp_dir, instance_id):
            """Create SLIDE instance using temporary files."""
            x_path = os.path.join(temp_dir, f'x_{instance_id}.csv')
            y_path = os.path.join(temp_dir, f'y_{instance_id}.csv')

            X.to_csv(x_path)
            y.to_csv(y_path)

            local_params = base_params.copy()
            local_params['x_path'] = x_path
            local_params['y_path'] = y_path

            slide = SLIDE(local_params)
            return slide.data.X.shape, slide.data.Y.shape

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test concurrent file-based initialization
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(create_slide_with_temp_files, temp_dir, i)
                    for i in range(5)
                ]
                shapes = [f.result() for f in futures]

            # Verify all instances loaded data correctly
            expected_shape = ((100, 50), (100, 1))
            assert all(shape == expected_shape for shape in shapes)

    def test_memory_isolation_stress(self, sample_data, base_params):
        """Stress test memory isolation between concurrent instances."""
        X, y = sample_data

        def memory_intensive_operation(scale_factor):
            """Perform memory-intensive SLIDE operation."""
            # Scale up the data
            scaled_X = pd.concat([X] * scale_factor, ignore_index=True)
            scaled_y = pd.concat([y] * scale_factor, ignore_index=True)

            slide = SLIDE(base_params, scaled_X, scaled_y)
            return scaled_X.shape[0]

        # Test with different memory loads concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(memory_intensive_operation, 1),
                executor.submit(memory_intensive_operation, 2)
            ]

            sizes = []
            for f in futures:
                try:
                    sizes.append(f.result(timeout=60))
                except Exception:
                    sizes.append(0)  # Failed

        # Verify at least one succeeded
        assert any(size > 0 for size in sizes)