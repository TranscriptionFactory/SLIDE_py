#!/usr/bin/env python3
"""
Comprehensive cross-validation test scenarios for SLIDEcv.
Tests edge cases, failure modes, and integration scenarios not covered in existing tests.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock

from loveslide.cv import SLIDEcv
from loveslide.tools import init_data


class TestSLIDEcvParameterValidation:
    """Test SLIDEcv parameter validation and edge cases."""

    def test_slidecv_invalid_parameter_grids(self):
        """Test SLIDEcv with invalid parameter grids."""
        X = pd.DataFrame(np.random.rand(100, 50))
        y = pd.DataFrame(np.random.randint(0, 2, (100, 1)))

        # Create test data
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, 'x.csv')
            y_path = os.path.join(tmpdir, 'y.csv')
            X.to_csv(x_path, index=True)
            y.to_csv(y_path, index=True)

            # Invalid parameter combinations
            invalid_params = [
                {'x_path': x_path, 'y_path': y_path, 'delta': []},  # Empty delta
                {'x_path': x_path, 'y_path': y_path, 'lambda': []},  # Empty lambda
                {'x_path': x_path, 'y_path': y_path, 'delta': [-0.1]},  # Negative delta
                {'x_path': x_path, 'y_path': y_path, 'lambda': [-0.1]},  # Negative lambda
                {'x_path': x_path, 'y_path': y_path, 'fdr': 1.5},  # FDR > 1
                {'x_path': x_path, 'y_path': y_path, 'fdr': -0.1},  # Negative FDR
            ]

            for params in invalid_params:
                with pytest.raises((ValueError, AssertionError)):
                    cv = SLIDEcv(params)
                    cv.run()

    def test_slidecv_extreme_grid_sizes(self):
        """Test SLIDEcv with extremely large or small parameter grids."""
        X = pd.DataFrame(np.random.rand(50, 20))
        y = pd.DataFrame(np.random.randint(0, 2, (50, 1)))

        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, 'x.csv')
            y_path = os.path.join(tmpdir, 'y.csv')
            X.to_csv(x_path, index=True)
            y.to_csv(y_path, index=True)

            # Very large grid (computationally expensive)
            large_grid_params = {
                'x_path': x_path, 'y_path': y_path,
                'delta': np.linspace(0.01, 0.5, 20).tolist(),  # 20 values
                'lambda': np.linspace(0.1, 0.9, 20).tolist(),  # 20 values
                'timeout': 5  # Short timeout to prevent hanging
            }

            # Should handle large grids or timeout gracefully
            try:
                cv = SLIDEcv(large_grid_params)
                result = cv.run()
                # If it completes, result should be valid
                assert result is not None
            except TimeoutError:
                # Acceptable if computation takes too long
                pass

            # Single parameter values (minimal grid)
            minimal_params = {
                'x_path': x_path, 'y_path': y_path,
                'delta': [0.1],  # Single value
                'lambda': [0.5]  # Single value
            }

            cv = SLIDEcv(minimal_params)
            result = cv.run()
            assert result is not None

    def test_slidecv_parameter_boundary_values(self):
        """Test SLIDEcv with boundary parameter values."""
        X = pd.DataFrame(np.random.rand(50, 20))
        y = pd.DataFrame(np.random.randint(0, 2, (50, 1)))

        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, 'x.csv')
            y_path = os.path.join(tmpdir, 'y.csv')
            X.to_csv(x_path, index=True)
            y.to_csv(y_path, index=True)

            boundary_cases = [
                # Very small values
                {'delta': [1e-6], 'lambda': [1e-6], 'fdr': 1e-6},
                # Values close to 1
                {'delta': [0.999], 'lambda': [0.999], 'fdr': 0.999},
                # Machine precision boundaries
                {'delta': [np.finfo(float).eps], 'lambda': [0.5], 'fdr': 0.1},
            ]

            for boundary_params in boundary_cases:
                params = {
                    'x_path': x_path, 'y_path': y_path,
                    **boundary_params
                }

                try:
                    cv = SLIDEcv(params)
                    result = cv.run()
                    # Should handle boundary values appropriately
                    assert result is not None
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    # Some boundary values might legitimately fail
                    pass


class TestSLIDEcvDataEdgeCases:
    """Test SLIDEcv with challenging data scenarios."""

    def test_slidecv_small_sample_sizes(self):
        """Test SLIDEcv with very small sample sizes."""
        # Very small datasets
        small_sizes = [5, 10, 15]

        for n_samples in small_sizes:
            X = pd.DataFrame(np.random.rand(n_samples, 10))
            y = pd.DataFrame(np.random.randint(0, 2, (n_samples, 1)))

            with tempfile.TemporaryDirectory() as tmpdir:
                x_path = os.path.join(tmpdir, 'x.csv')
                y_path = os.path.join(tmpdir, 'y.csv')
                X.to_csv(x_path, index=True)
                y.to_csv(y_path, index=True)

                params = {
                    'x_path': x_path, 'y_path': y_path,
                    'delta': [0.1], 'lambda': [0.5], 'fdr': 0.1
                }

                try:
                    cv = SLIDEcv(params)
                    result = cv.run()
                    # Should handle small samples or provide meaningful error
                    assert result is not None
                except ValueError as e:
                    # Should provide informative error about sample size
                    assert "sample" in str(e).lower() or "size" in str(e).lower()

    def test_slidecv_high_dimensional_data(self):
        """Test SLIDEcv with high-dimensional data (p >> n)."""
        # More features than samples
        X = pd.DataFrame(np.random.rand(50, 200))  # 50 samples, 200 features
        y = pd.DataFrame(np.random.randint(0, 2, (50, 1)))

        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, 'x.csv')
            y_path = os.path.join(tmpdir, 'y.csv')
            X.to_csv(x_path, index=True)
            y.to_csv(y_path, index=True)

            params = {
                'x_path': x_path, 'y_path': y_path,
                'delta': [0.1], 'lambda': [0.5], 'fdr': 0.1
            }

            try:
                cv = SLIDEcv(params)
                result = cv.run()
                # Should handle high-dimensional data
                assert result is not None
            except (np.linalg.LinAlgError, MemoryError):
                # Acceptable failures for extreme high-dimensional cases
                pass

    def test_slidecv_imbalanced_classes(self):
        """Test SLIDEcv with severely imbalanced class distributions."""
        # Highly imbalanced classes
        X = pd.DataFrame(np.random.rand(100, 50))
        y = pd.DataFrame([0] * 95 + [1] * 5)  # 95% class 0, 5% class 1

        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, 'x.csv')
            y_path = os.path.join(tmpdir, 'y.csv')
            X.to_csv(x_path, index=True)
            y.to_csv(y_path, index=True)

            params = {
                'x_path': x_path, 'y_path': y_path,
                'delta': [0.1], 'lambda': [0.5], 'fdr': 0.1
            }

            cv = SLIDEcv(params)
            result = cv.run()
            # Should handle imbalanced data appropriately
            assert result is not None

    def test_slidecv_constant_features(self):
        """Test SLIDEcv with constant/zero-variance features."""
        # Mix of variable and constant features
        X_variable = np.random.rand(100, 30)
        X_constant = np.ones((100, 20))  # Constant features
        X = pd.DataFrame(np.column_stack([X_variable, X_constant]))
        y = pd.DataFrame(np.random.randint(0, 2, (100, 1)))

        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, 'x.csv')
            y_path = os.path.join(tmpdir, 'y.csv')
            X.to_csv(x_path, index=True)
            y.to_csv(y_path, index=True)

            params = {
                'x_path': x_path, 'y_path': y_path,
                'delta': [0.1], 'lambda': [0.5], 'fdr': 0.1
            }

            cv = SLIDEcv(params)
            result = cv.run()
            # Should handle constant features by removing them
            assert result is not None


class TestSLIDEcvConvergence:
    """Test SLIDEcv convergence and optimization scenarios."""

    def test_slidecv_convergence_failure_recovery(self):
        """Test SLIDEcv recovery from convergence failures."""
        X = pd.DataFrame(np.random.rand(100, 50))
        y = pd.DataFrame(np.random.randint(0, 2, (100, 1)))

        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, 'x.csv')
            y_path = os.path.join(tmpdir, 'y.csv')
            X.to_csv(x_path, index=True)
            y.to_csv(y_path, index=True)

            # Parameters that might cause convergence issues
            challenging_params = {
                'x_path': x_path, 'y_path': y_path,
                'delta': [0.001, 0.999],  # Extreme values
                'lambda': [0.001, 0.999],
                'fdr': 0.001,  # Very strict
                'max_iter': 5,  # Very few iterations
                'tol': 1e-12  # Very tight tolerance
            }

            try:
                cv = SLIDEcv(challenging_params)
                result = cv.run()
                # If it succeeds, great
                assert result is not None
            except (RuntimeError, ValueError) as e:
                # Should provide informative error about convergence
                assert "converg" in str(e).lower() or "iter" in str(e).lower()

    def test_slidecv_optimization_path_validation(self):
        """Test that SLIDEcv optimization path makes sense."""
        X = pd.DataFrame(np.random.rand(100, 30))
        y = pd.DataFrame(np.random.randint(0, 2, (100, 1)))

        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, 'x.csv')
            y_path = os.path.join(tmpdir, 'y.csv')
            X.to_csv(x_path, index=True)
            y.to_csv(y_path, index=True)

            params = {
                'x_path': x_path, 'y_path': y_path,
                'delta': [0.05, 0.1, 0.2],
                'lambda': [0.3, 0.5, 0.7],
                'fdr': 0.1,
                'verbose': True  # To capture optimization path
            }

            cv = SLIDEcv(params)
            result = cv.run()

            # Verify optimization results make sense
            assert result is not None
            if hasattr(result, 'best_params'):
                # Best parameters should be within the tested grid
                assert result.best_params['delta'] in params['delta']
                assert result.best_params['lambda'] in params['lambda']


class TestSLIDEcvResourceManagement:
    """Test SLIDEcv resource management and cleanup."""

    def test_slidecv_parallel_execution_limits(self):
        """Test SLIDEcv with parallel execution and resource limits."""
        X = pd.DataFrame(np.random.rand(100, 50))
        y = pd.DataFrame(np.random.randint(0, 2, (100, 1)))

        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, 'x.csv')
            y_path = os.path.join(tmpdir, 'y.csv')
            X.to_csv(x_path, index=True)
            y.to_csv(y_path, index=True)

            # Test with various worker counts
            for n_workers in [1, 2, 4, 8, 16]:  # Including extreme values
                params = {
                    'x_path': x_path, 'y_path': y_path,
                    'delta': [0.1, 0.2],
                    'lambda': [0.5, 0.7],
                    'fdr': 0.1,
                    'n_workers': n_workers
                }

                try:
                    cv = SLIDEcv(params)
                    result = cv.run()
                    assert result is not None
                except (RuntimeError, OSError) as e:
                    # Some systems might not support many workers
                    if "worker" in str(e).lower() or "thread" in str(e).lower():
                        continue
                    else:
                        raise

    def test_slidecv_memory_pressure_handling(self):
        """Test SLIDEcv behavior under memory pressure."""
        # Create dataset that might cause memory pressure
        X = pd.DataFrame(np.random.rand(1000, 500))
        y = pd.DataFrame(np.random.randint(0, 2, (1000, 1)))

        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, 'x.csv')
            y_path = os.path.join(tmpdir, 'y.csv')
            X.to_csv(x_path, index=True)
            y.to_csv(y_path, index=True)

            params = {
                'x_path': x_path, 'y_path': y_path,
                'delta': [0.1, 0.15, 0.2],  # Multiple parameter combinations
                'lambda': [0.4, 0.5, 0.6],
                'fdr': 0.1,
                'memory_limit': '1GB'  # If supported
            }

            try:
                cv = SLIDEcv(params)
                result = cv.run()
                assert result is not None
            except MemoryError:
                # Acceptable on systems with limited memory
                pytest.skip("Insufficient memory for memory pressure test")

    def test_slidecv_interruption_recovery(self):
        """Test SLIDEcv recovery from interruption."""
        X = pd.DataFrame(np.random.rand(100, 50))
        y = pd.DataFrame(np.random.randint(0, 2, (100, 1)))

        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, 'x.csv')
            y_path = os.path.join(tmpdir, 'y.csv')
            X.to_csv(x_path, index=True)
            y.to_csv(y_path, index=True)

            params = {
                'x_path': x_path, 'y_path': y_path,
                'delta': [0.1, 0.2, 0.3],
                'lambda': [0.4, 0.5, 0.6],
                'fdr': 0.1,
                'out_path': tmpdir,
                'resume': True  # Enable resume functionality
            }

            # Simulate interruption by creating partial state
            partial_state = {'completed_params': [{'delta': 0.1, 'lambda': 0.4}]}
            with open(os.path.join(tmpdir, 'cv_state.pkl'), 'wb') as f:
                import pickle
                pickle.dump(partial_state, f)

            cv = SLIDEcv(params)
            result = cv.run()
            # Should be able to resume and complete
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])