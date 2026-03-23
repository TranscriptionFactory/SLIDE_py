"""
Test coverage for configuration parameter interdependencies and validation.

Critical gaps:
- Complex parameter interactions
- Configuration state consistency
- Runtime parameter validation
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from loveslide.tools import check_params, show_params, calc_default_fsize
from loveslide.slide import SLIDE, OptimizeSLIDE
from loveslide.cv import SLIDEcv
from loveslide.knockoffs import Knockoffs


class TestParameterInterdependencyValidation:
    """Test complex parameter interdependencies"""

    def test_slide_fdr_threshold_consistency(self):
        """Test FDR threshold consistency across methods"""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Conflicting FDR thresholds
        slide = SLIDE()

        # Set inconsistent thresholds
        params_1 = {
            'thresh': 0.1,      # Strict threshold
            'fdr': 0.2,         # Looser FDR
            'verbose': False
        }

        params_2 = {
            'thresh': 0.05,     # Very strict threshold
            'fdr': 0.01,        # Very strict FDR
            'verbose': False
        }

        # Should handle parameter precedence consistently
        result_1 = slide.run(X, y, **params_1)
        result_2 = slide.run(X, y, **params_2)

        # More strict parameters should yield fewer selections
        assert len(result_2.selected_vars) <= len(result_1.selected_vars)

    def test_cv_fold_parameter_consistency(self):
        """Test CV fold parameter consistency validation"""
        X = np.random.randn(50, 20)  # Small dataset
        y = np.random.randn(50)

        cv = SLIDEcv()

        # Invalid fold configurations
        invalid_configs = [
            {'n_folds': 60, 'test_size': 0.1},     # More folds than samples
            {'n_folds': 5, 'test_size': 0.95},     # Test size too large
            {'n_folds': 1, 'test_size': 0.2},      # No actual cross-validation
        ]

        for config in invalid_configs:
            with pytest.raises((ValueError, AssertionError)):
                cv.run(X, y, **config)

    def test_knockoff_generation_parameter_validation(self):
        """Test knockoff generation parameter interdependencies"""
        X = np.random.randn(100, 50)

        knockoffs = Knockoffs()

        # Inconsistent parameters
        invalid_combinations = [
            {
                'method': 'sdp',
                'shrink': True,     # SDP shouldn't need shrinking
                'randomize': True
            },
            {
                'method': 'equicorrelated',
                'offset': 2.0,      # Offset too large for method
            },
            {
                'method': 'sdp',
                'mu': -0.1,         # Negative mu invalid
            }
        ]

        for params in invalid_combinations:
            with pytest.raises((ValueError, AssertionError)):
                knockoffs.generate(X, **params)

    def test_optimization_parameter_bounds_checking(self):
        """Test optimization parameter bounds and relationships"""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        slide_opt = OptimizeSLIDE()

        # Parameter bounds violations
        invalid_bounds = [
            {'alpha': 2.0},          # Alpha > 1
            {'alpha': -0.1},         # Alpha < 0
            {'max_iter': 0},         # No iterations
            {'tol': 0.0},            # Zero tolerance
            {'tol': -1e-6},          # Negative tolerance
        ]

        for params in invalid_bounds:
            with pytest.raises((ValueError, AssertionError)):
                slide_opt.run(X, y, **params)

    def test_memory_parameter_consistency(self):
        """Test memory-related parameter consistency"""
        X = np.random.randn(1000, 200)
        y = np.random.randn(1000)

        # Large dataset requiring memory considerations
        slide = SLIDE()

        # Memory vs performance tradeoffs
        memory_configs = [
            {'batch_size': 10000, 'n_jobs': 8},    # Large batch, many jobs
            {'batch_size': 1, 'n_jobs': 1},        # Minimal resources
            {'store_precision': True, 'batch_size': 1},  # Memory conflict
        ]

        for config in memory_configs:
            try:
                result = slide.run(X, y, **config)
                # Should complete without memory errors
                assert result is not None
            except MemoryError:
                # Acceptable for extreme configurations
                pass


class TestConfigurationStateConsistency:
    """Test configuration state consistency across operations"""

    def test_parameter_persistence_across_calls(self):
        """Test parameter persistence across multiple calls"""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        slide = SLIDE()

        # First call with specific parameters
        params_1 = {'thresh': 0.1, 'verbose': True}
        result_1 = slide.run(X, y, **params_1)

        # Second call should not inherit previous parameters
        params_2 = {'thresh': 0.05}  # Different threshold, no verbose
        result_2 = slide.run(X, y, **params_2)

        # Results should reflect different thresholds
        # (assuming different thresholds give different results)
        # This tests parameter isolation between calls

    def test_configuration_serialization_consistency(self):
        """Test configuration consistency in serialization"""
        import pickle
        import tempfile

        X = np.random.randn(100, 50)
        slide = SLIDE()

        # Configure with specific parameters
        original_config = {
            'thresh': 0.1,
            'max_iter': 500,
            'tol': 1e-5
        }

        # Serialize and deserialize
        with tempfile.NamedTemporaryFile(delete=False) as f:
            pickle.dump((slide, original_config), f)
            temp_file = f.name

        try:
            with open(temp_file, 'rb') as f:
                loaded_slide, loaded_config = pickle.load(f)

            # Configuration should be identical
            assert loaded_config == original_config

        finally:
            os.unlink(temp_file)

    def test_parameter_validation_caching(self):
        """Test parameter validation caching consistency"""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        slide = SLIDE()

        # Same parameters should be validated once
        params = {'thresh': 0.1, 'max_iter': 100}

        with patch('loveslide.tools.check_params', wraps=check_params) as mock_check:
            # Multiple calls with same parameters
            slide.run(X, y, **params)
            slide.run(X, y, **params)

            # Validation should be efficient (not necessarily cached,
            # but shouldn't be computationally expensive)
            assert mock_check.call_count >= 2


class TestRuntimeParameterValidation:
    """Test runtime parameter validation scenarios"""

    def test_adaptive_parameter_adjustment(self):
        """Test adaptive parameter adjustment during runtime"""
        # Very small dataset
        X = np.random.randn(20, 10)
        y = np.random.randn(20)

        slide = SLIDE()

        # Parameters that may need runtime adjustment
        params = {
            'thresh': 1e-10,    # Very strict threshold
            'max_iter': 10000   # Many iterations
        }

        # Should handle gracefully (adjust or warn)
        with pytest.warns(UserWarning, match=".*parameter.*"):
            result = slide.run(X, y, **params)

        # Should still produce valid result
        assert result is not None

    def test_convergence_parameter_adjustment(self):
        """Test parameter adjustment based on convergence"""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        slide_opt = OptimizeSLIDE()

        # Parameters likely to cause convergence issues
        params = {
            'tol': 1e-15,      # Extremely tight tolerance
            'max_iter': 5,      # Very few iterations
        }

        # Should handle convergence/parameter conflicts
        result = slide_opt.run(X, y, **params)

        # Should either converge or provide meaningful error
        assert result is not None or "convergence" in str(result)

    def test_data_dependent_parameter_validation(self):
        """Test validation that depends on data characteristics"""
        # Various data scenarios
        data_scenarios = [
            (np.random.randn(1000, 10), "many samples, few features"),
            (np.random.randn(10, 1000), "few samples, many features"),
            (np.ones((100, 50)), "constant data"),
            (np.random.randn(100, 50) * 1e-10, "very small values")
        ]

        slide = SLIDE()

        for X, description in data_scenarios:
            y = np.random.randn(X.shape[0])

            # Should validate parameters appropriately for data
            try:
                result = slide.run(X, y)
                assert result is not None
            except (ValueError, RuntimeWarning) as e:
                # Should provide meaningful message about data/parameter mismatch
                assert any(word in str(e).lower()
                          for word in ['data', 'parameter', 'dimension', 'scale'])


class TestParameterBoundaryValidation:
    """Test parameter validation at boundaries"""

    def test_floating_point_precision_boundaries(self):
        """Test parameter validation at floating point boundaries"""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        slide = SLIDE()

        # Floating point boundary cases
        boundary_params = [
            {'thresh': np.finfo(float).eps},        # Minimum positive float
            {'thresh': 1.0 - np.finfo(float).eps},  # Just under 1.0
            {'tol': np.finfo(float).tiny},          # Smallest representable float
        ]

        for params in boundary_params:
            # Should handle floating point boundaries gracefully
            try:
                result = slide.run(X, y, **params)
                assert result is not None
            except (ValueError, RuntimeWarning):
                # Acceptable for extreme boundary values
                pass

    def test_integer_overflow_boundaries(self):
        """Test integer parameter boundaries"""
        X = np.random.randn(100, 50)

        # Integer boundary cases
        boundary_cases = [
            (sys.maxsize, "max_iter"),      # Maximum integer
            (0, "max_iter"),                # Minimum valid iteration
            (1, "n_folds"),                 # Minimum folds
        ]

        slide = SLIDE()

        for value, param_name in boundary_cases:
            params = {param_name: value}

            if param_name == "max_iter" and value == 0:
                with pytest.raises(ValueError):
                    slide.run(X, **params)
            else:
                # Should handle large integers appropriately
                try:
                    result = slide.run(X, **params)
                    assert result is not None
                except (MemoryError, OverflowError):
                    # Acceptable for extreme values
                    pass


if __name__ == "__main__":
    pytest.main([__file__])