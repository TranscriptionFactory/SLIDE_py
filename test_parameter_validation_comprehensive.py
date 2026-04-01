"""
Comprehensive parameter validation test coverage.
Tests edge cases, malformed inputs, and boundary conditions.
"""
import pytest
import numpy as np
import pandas as pd
from loveslide.tools import init_data, check_params, calc_default_fsize


class TestParameterValidationEdgeCases:
    """Test parameter validation edge cases and boundary conditions."""

    def test_init_data_none_parameters(self):
        """Test init_data with None values for critical parameters."""
        params = {}
        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        with pytest.raises(ValueError, match="x_path is not provided"):
            init_data(params)

    def test_init_data_negative_dimensions(self):
        """Test init_data with negative dimension parameters."""
        params = {"delta": [-0.1], "lambda": [-0.5], "fdr": -0.1}
        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        # Should handle negative values gracefully
        data, processed_params = init_data(params, x=X, y=y)
        assert processed_params["delta"] != [-0.1]  # Should be corrected or raise error

    def test_init_data_extreme_values(self):
        """Test init_data with extreme parameter values."""
        params = {
            "delta": [1e10, 1e-10],
            "lambda": [0, 1],
            "fdr": 1.0,  # At boundary
            "thresh_fdr": 2.0  # Above boundary
        }
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Should handle extreme values appropriately
        data, processed_params = init_data(params, x=X, y=y)
        assert 0 <= processed_params["fdr"] <= 1

    def test_init_data_conflicting_parameters(self):
        """Test init_data with conflicting parameter combinations."""
        params = {
            "fdr": 0.1,
            "thresh_fdr": 0.05,  # thresh_fdr < fdr conflict
            "pure_homo": True,
            "spec": -0.1  # Invalid spec
        }
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        # Should detect and resolve conflicts
        data, processed_params = init_data(params, x=X, y=y)
        # Add assertions based on expected conflict resolution

    def test_calc_default_fsize_edge_cases(self):
        """Test calc_default_fsize with edge cases."""
        # Very small datasets
        assert calc_default_fsize(10, 2) > 0

        # Very large datasets
        large_fsize = calc_default_fsize(100000, 100)
        assert large_fsize > 0

        # Zero or negative inputs
        with pytest.raises(ValueError):
            calc_default_fsize(0, 5)

        with pytest.raises(ValueError):
            calc_default_fsize(-1, 5)


class TestDataLoadingEdgeCases:
    """Test data loading with various edge cases."""

    def test_mismatched_dimensions(self):
        """Test behavior with mismatched X and y dimensions."""
        params = {}
        X = np.random.randn(100, 50)
        y = np.random.randn(90)  # Wrong size

        with pytest.raises(ValueError, match="dimension mismatch"):
            init_data(params, x=X, y=y)

    def test_empty_data(self):
        """Test handling of empty data arrays."""
        params = {}
        X = np.array([]).reshape(0, 5)
        y = np.array([])

        with pytest.raises(ValueError, match="empty data"):
            init_data(params, x=X, y=y)

    def test_single_sample(self):
        """Test handling of single sample data."""
        params = {}
        X = np.random.randn(1, 10)
        y = np.random.randn(1)

        # Should handle gracefully or raise appropriate warning
        data, processed_params = init_data(params, x=X, y=y)
        assert data.X.shape[0] == 1

    def test_nan_inf_data(self):
        """Test handling of NaN and infinity values."""
        params = {}
        X = np.random.randn(50, 10)
        X[0, 0] = np.nan
        X[1, 1] = np.inf
        y = np.random.randn(50)
        y[0] = np.nan

        # Should detect and handle NaN/inf appropriately
        with pytest.raises(ValueError, match="contains NaN or infinite values"):
            init_data(params, x=X, y=y)


class TestParameterTypeValidation:
    """Test parameter type validation and conversion."""

    def test_string_numeric_conversion(self):
        """Test conversion of string parameters to numeric."""
        params = {
            "delta": ["0.1", "0.2"],  # String list
            "lambda": "0.5",  # String scalar
            "fdr": "0.1"
        }
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        data, processed_params = init_data(params, x=X, y=y)
        assert isinstance(processed_params["delta"], list)
        assert all(isinstance(d, float) for d in processed_params["delta"])

    def test_invalid_type_parameters(self):
        """Test behavior with invalid parameter types."""
        params = {
            "delta": {"invalid": "dict"},
            "lambda": set([0.1, 0.2]),
            "fdr": lambda x: x  # Function instead of float
        }
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        with pytest.raises(TypeError):
            init_data(params, x=X, y=y)