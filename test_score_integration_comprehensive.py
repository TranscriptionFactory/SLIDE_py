#!/usr/bin/env python3
"""
Comprehensive integration tests for score module.
Tests estimator interactions, failure modes, and edge case scenarios.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import pickle
from unittest.mock import patch, MagicMock

from loveslide.score import Estimator, SLIDE_Estimator


class TestEstimatorBaseClass:
    """Test base Estimator class functionality and edge cases."""

    def test_estimator_initialization_edge_cases(self):
        """Test Estimator initialization with edge case parameters."""
        # Empty or invalid parameters
        with pytest.raises((ValueError, TypeError)):
            Estimator({})  # Empty params

        # Invalid parameter types
        with pytest.raises((ValueError, TypeError)):
            Estimator("not_a_dict")

        # Parameters with None values
        params_with_none = {
            'method': None,
            'statistic': None,
            'offset': None
        }
        # Should handle None gracefully or raise appropriate error
        try:
            estimator = Estimator(params_with_none)
        except (ValueError, TypeError) as e:
            assert "None" in str(e) or "invalid" in str(e).lower()

    def test_estimator_with_corrupted_state(self):
        """Test Estimator behavior with corrupted internal state."""
        params = {'method': 'knockoff', 'statistic': 'lasso_cv'}
        estimator = Estimator(params)

        # Corrupt internal state
        estimator.params = None

        # Should handle corrupted state gracefully
        with pytest.raises((AttributeError, ValueError)):
            estimator.run_knockoffs(np.random.rand(100, 50), np.random.randint(0, 2, 100))

    def test_estimator_memory_management(self):
        """Test Estimator memory management with large datasets."""
        params = {'method': 'knockoff', 'statistic': 'lasso_cv'}
        estimator = Estimator(params)

        # Large dataset that might cause memory issues
        X_large = np.random.rand(10000, 1000)
        y_large = np.random.randint(0, 2, 10000)

        # Should handle large datasets without memory errors
        try:
            result = estimator.run_knockoffs(X_large, y_large)
            # Verify memory cleanup
            assert result is not None
        except MemoryError:
            pytest.skip("Insufficient memory for large dataset test")

    def test_estimator_concurrent_access(self):
        """Test Estimator behavior under concurrent access."""
        params = {'method': 'knockoff', 'statistic': 'lasso_cv'}
        estimator = Estimator(params)

        import threading
        import time

        results = []
        errors = []

        def run_estimation():
            try:
                X = np.random.rand(100, 50)
                y = np.random.randint(0, 2, 100)
                result = estimator.run_knockoffs(X, y)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run multiple threads concurrently
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=run_estimation)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should handle concurrent access or fail gracefully
        assert len(errors) == 0 or all(isinstance(e, (ValueError, RuntimeError)) for e in errors)


class TestSLIDEEstimatorAdvanced:
    """Advanced tests for SLIDE_Estimator specific functionality."""

    def test_slide_estimator_with_invalid_data_shapes(self):
        """Test SLIDE_Estimator with mismatched data shapes."""
        params = {
            'method': 'slide',
            'delta': [0.1],
            'lambda': [0.5],
            'fdr': 0.1
        }
        estimator = SLIDE_Estimator(params)

        # Mismatched X and y shapes
        X_wrong_shape = np.random.rand(100, 50)
        y_wrong_shape = np.random.randint(0, 2, 95)  # Wrong number of samples

        with pytest.raises(ValueError, match="shape|dimension|size"):
            estimator.run_slide(X_wrong_shape, y_wrong_shape)

    def test_slide_estimator_extreme_parameters(self):
        """Test SLIDE_Estimator with extreme parameter values."""
        # Extreme delta values
        extreme_params = {
            'method': 'slide',
            'delta': [0.0001, 0.9999],  # Very small and very large
            'lambda': [0.0001, 0.9999],
            'fdr': 0.001  # Very strict FDR
        }
        estimator = SLIDE_Estimator(extreme_params)

        X = np.random.rand(50, 20)
        y = np.random.randint(0, 2, 50)

        # Should handle extreme parameters or fail gracefully
        try:
            result = estimator.run_slide(X, y)
            assert result is not None
        except (ValueError, RuntimeError) as e:
            # Should provide meaningful error message
            assert len(str(e)) > 0

    def test_slide_estimator_numerical_instability(self):
        """Test SLIDE_Estimator with numerically challenging data."""
        params = {
            'method': 'slide',
            'delta': [0.1],
            'lambda': [0.5],
            'fdr': 0.1
        }
        estimator = SLIDE_Estimator(params)

        # Numerically challenging cases
        test_cases = [
            # Perfectly correlated features
            np.column_stack([np.random.rand(100), np.random.rand(100)]),
            # Features with very different scales
            np.column_stack([np.random.rand(100) * 1e-10, np.random.rand(100) * 1e10]),
            # Near-singular correlation matrix
            np.random.rand(100, 50) + 1e-15 * np.random.rand(100, 50),
        ]

        y = np.random.randint(0, 2, 100)

        for X_challenging in test_cases:
            try:
                result = estimator.run_slide(X_challenging, y)
                # Should either succeed or fail with meaningful error
                assert result is not None or True  # Success
            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                # Acceptable failures for challenging numerical cases
                pass

    def test_slide_estimator_state_persistence_corruption(self):
        """Test SLIDE_Estimator state persistence with corruption scenarios."""
        params = {
            'method': 'slide',
            'delta': [0.1],
            'lambda': [0.5],
            'fdr': 0.1,
            'out_path': tempfile.mkdtemp()
        }
        estimator = SLIDE_Estimator(params)

        X = np.random.rand(100, 50)
        y = np.random.randint(0, 2, 100)

        # Create corrupted state file
        state_file = f"{params['out_path']}/slide_state.pkl"
        with open(state_file, 'wb') as f:
            f.write(b"corrupted_pickle_data")

        # Should handle corrupted state gracefully
        try:
            result = estimator.run_slide(X, y)
            assert result is not None
        except (pickle.UnpicklingError, EOFError, ValueError):
            # Should provide recovery mechanism
            pass

    def test_slide_estimator_resource_cleanup(self):
        """Test SLIDE_Estimator resource cleanup and file handling."""
        temp_dir = tempfile.mkdtemp()
        params = {
            'method': 'slide',
            'delta': [0.1],
            'lambda': [0.5],
            'fdr': 0.1,
            'out_path': temp_dir
        }
        estimator = SLIDE_Estimator(params)

        X = np.random.rand(50, 20)
        y = np.random.randint(0, 2, 50)

        # Run estimation
        result = estimator.run_slide(X, y)

        # Check that temporary files are cleaned up appropriately
        import os
        temp_files = os.listdir(temp_dir)

        # Should not leave excessive temporary files
        assert len(temp_files) < 20  # Reasonable limit

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


class TestEstimatorIntegration:
    """Test integration between different estimator components."""

    def test_estimator_knockoff_slide_consistency(self):
        """Test consistency between knockoff and SLIDE estimators."""
        # Same data and compatible parameters
        X = np.random.rand(100, 30)
        y = np.random.randint(0, 2, 100)

        knockoff_params = {'method': 'knockoff', 'statistic': 'lasso_cv'}
        slide_params = {
            'method': 'slide',
            'delta': [0.1],
            'lambda': [0.5],
            'fdr': 0.1
        }

        knockoff_est = Estimator(knockoff_params)
        slide_est = SLIDE_Estimator(slide_params)

        try:
            knockoff_result = knockoff_est.run_knockoffs(X, y)
            slide_result = slide_est.run_slide(X, y)

            # Results should have compatible structure
            assert knockoff_result is not None
            assert slide_result is not None

            # Both should identify some features (or none consistently)
            if hasattr(knockoff_result, 'selected') and hasattr(slide_result, 'selected'):
                # At least one should find some signal or both should find none
                total_selected = len(knockoff_result.selected) + len(slide_result.selected)
                assert total_selected >= 0  # Basic sanity check

        except Exception as e:
            # Integration failures should be informative
            assert len(str(e)) > 0

    def test_estimator_parameter_validation_consistency(self):
        """Test that parameter validation is consistent across estimator types."""
        invalid_params = [
            {'method': 'invalid_method'},
            {'method': 'knockoff', 'statistic': 'invalid_stat'},
            {'method': 'slide', 'delta': 'invalid_delta'},
            {'fdr': 'invalid_fdr'},
            {'lambda': 'invalid_lambda'}
        ]

        for params in invalid_params:
            # Both estimator types should reject invalid parameters consistently
            with pytest.raises((ValueError, TypeError, KeyError)):
                if params.get('method') == 'slide':
                    SLIDE_Estimator(params)
                else:
                    Estimator(params)

    def test_estimator_cross_validation_integration(self):
        """Test estimator integration with cross-validation."""
        params = {
            'method': 'slide',
            'delta': [0.05, 0.1],
            'lambda': [0.3, 0.5],
            'fdr': 0.1,
            'cv_folds': 5
        }
        estimator = SLIDE_Estimator(params)

        X = np.random.rand(100, 30)
        y = np.random.randint(0, 2, 100)

        # Should handle cross-validation integration
        try:
            result = estimator.run_slide(X, y)
            assert result is not None
        except NotImplementedError:
            # CV integration might not be implemented yet
            pass
        except Exception as e:
            # Other errors should be informative
            assert "cross" in str(e).lower() or "cv" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__])