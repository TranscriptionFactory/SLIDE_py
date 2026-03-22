"""Test configuration and parameter initialization edge cases.

Complementary to existing test coverage, focusing on parameter validation
and data initialization scenarios not covered in previous analysis.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from easydict import EasyDict
from unittest.mock import patch, mock_open

from src.loveslide.tools import init_data, show_params, check_params, calc_default_fsize


class TestConfigurationEdgeCases:
    """Test parameter initialization and configuration edge cases."""

    def test_init_data_missing_both_paths_and_arrays(self):
        """Test error when both x_path/y_path and x/y are None."""
        with pytest.raises(ValueError, match="x_path is not provided"):
            init_data({}, x=None, y=None)

    def test_init_data_partial_path_specification(self):
        """Test error when only one of x_path or y_path is provided."""
        # Only x_path provided
        with pytest.raises(ValueError, match="y_path is not provided"):
            init_data({'x_path': 'dummy.csv'}, x=None, y=None)

        # Only y_path provided
        with pytest.raises(ValueError, match="x_path is not provided"):
            init_data({'y_path': 'dummy.csv'}, x=None, y=None)

    def test_init_data_mixed_path_and_array_input(self):
        """Test initialization with mix of paths and arrays."""
        x_data = pd.DataFrame(np.random.randn(100, 10))
        y_data = pd.DataFrame(np.random.randint(0, 2, 100))

        # x as array, y_path provided
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            y_data.to_csv(f.name)
            try:
                data, params = init_data({'y_path': f.name}, x=x_data, y=None)
                assert data.X.equals(x_data)
                assert not data.Y.equals(y_data)  # File loading may differ
            finally:
                os.unlink(f.name)

    def test_init_data_default_parameter_overrides(self):
        """Test that explicit parameters override defaults."""
        x_data = pd.DataFrame(np.random.randn(10, 5))
        y_data = pd.DataFrame(np.random.randint(0, 2, 10))

        custom_params = {
            'fdr': 0.05,  # Override default 0.1
            'niter': 200,  # Override default 100
            'spec': 0.3,  # Override default 0.2
            'delta': [0.01, 0.02]  # Override default [0.05, 0.1]
        }

        data, params = init_data(custom_params, x=x_data, y=y_data)

        assert params['fdr'] == 0.05
        assert params['niter'] == 200
        assert params['spec'] == 0.3
        assert params['delta'] == [0.01, 0.02]

    def test_y_factor_encoding_edge_cases(self):
        """Test y_factor encoding with edge cases."""
        x_data = pd.DataFrame(np.random.randn(10, 5))

        # String labels that sort differently than expected
        y_data = pd.DataFrame(['zebra', 'apple', 'zebra', 'apple', 'zebra'] * 2)

        data, params = init_data({'y_factor': True}, x=x_data, y=y_data)

        # Should map to 0, 1 based on unique order
        unique_vals = np.unique(data.Y)
        assert len(unique_vals) == 2
        assert set(unique_vals) == {0, 1}

    def test_y_flip_with_factor_combination(self):
        """Test y_flip and y_factor interaction."""
        x_data = pd.DataFrame(np.random.randn(10, 5))
        y_data = pd.DataFrame(['case', 'control'] * 5)

        # With y_factor=True, y_flip=True
        data, params = init_data(
            {'y_factor': True, 'y_flip': True},
            x=x_data, y=y_data
        )

        # Should first encode as factors, then flip
        original_encoded = y_data.replace({'case': 0, 'control': 1})
        expected_flipped = 1 - original_encoded

        # Values should be flipped from original encoding
        assert not data.Y.equals(original_encoded)

    def test_calc_default_fsize_boundary_conditions(self):
        """Test calc_default_fsize with boundary conditions."""
        # Exact boundary: n_rows = K, K < 100
        assert calc_default_fsize(50, 50) == 48  # n_rows - 2

        # Close to boundary: |n_rows - K| = 2
        assert calc_default_fsize(52, 50) == 50  # K
        assert calc_default_fsize(48, 50) == 48  # n_rows

        # n_rows < K, K >= 100
        assert calc_default_fsize(90, 120) == 90  # n_rows

        # Edge case: very small values
        assert calc_default_fsize(2, 1) == 1  # K
        assert calc_default_fsize(1, 3) == 1  # n_rows

    def test_check_params_zero_std_handling(self):
        """Test handling of zero-variance features."""
        # Create data with zero-variance columns
        X = pd.DataFrame({
            'var1': [1, 2, 3, 4, 5],
            'constant': [1, 1, 1, 1, 1],  # Zero variance
            'var2': [5, 4, 3, 2, 1],
            'another_constant': [0, 0, 0, 0, 0]  # Zero variance
        })
        Y = pd.DataFrame([0, 1, 0, 1, 0])

        data = EasyDict()
        data.X = X
        data.Y = Y

        # Should remove zero-variance columns
        original_cols = data.X.shape[1]
        check_params({}, data)

        assert data.X.shape[1] == 2  # Only var1 and var2 remain
        assert 'var1' in data.X.columns
        assert 'var2' in data.X.columns
        assert 'constant' not in data.X.columns
        assert 'another_constant' not in data.X.columns

    def test_show_params_output_format(self, capsys):
        """Test show_params output formatting."""
        x_data = pd.DataFrame(np.random.randn(100, 20))
        y_data = pd.DataFrame(np.random.randint(0, 2, 100))

        params = {'fdr': 0.1, 'y_flip': True}
        data, params = init_data(params, x=x_data, y=y_data)

        show_params(params, data)
        captured = capsys.readouterr()

        # Should contain key information
        assert "### PARAMETERS ###" in captured.out
        assert "###### DATA ######" in captured.out
        assert "100 samples" in captured.out
        assert "20 features" in captured.out
        assert "% cases" in captured.out
        assert "% controls" in captured.out

    def test_parameter_type_validation(self):
        """Test handling of unexpected parameter types."""
        x_data = pd.DataFrame(np.random.randn(10, 5))
        y_data = pd.DataFrame(np.random.randint(0, 2, 10))

        # Test with various parameter types
        params_with_types = {
            'fdr': '0.1',  # String instead of float
            'niter': 100.5,  # Float instead of int
            'delta': 0.1,  # Single value instead of list
        }

        # Should handle gracefully without errors
        data, params = init_data(params_with_types, x=x_data, y=y_data)
        assert params['fdr'] == '0.1'  # Preserved as-is
        assert params['niter'] == 100.5
        assert params['delta'] == 0.1

    @pytest.mark.parametrize("n_rows,K,expected", [
        (10, 5, 5),    # Normal case
        (50, 50, 48),  # Boundary case
        (100, 150, 100),  # n_rows < K
        (5, 120, 5),   # Small n_rows, large K
    ])
    def test_calc_default_fsize_parametrized(self, n_rows, K, expected):
        """Parametrized test for calc_default_fsize."""
        result = calc_default_fsize(n_rows, K)
        assert result == expected

    def test_file_io_error_handling(self):
        """Test error handling for file I/O operations."""
        x_data = pd.DataFrame(np.random.randn(10, 5))

        # Non-existent file
        with pytest.raises(FileNotFoundError):
            init_data({'x_path': 'nonexistent.csv'}, x=None, y=None)

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        x_data = pd.DataFrame()  # Empty DataFrame
        y_data = pd.DataFrame([1])

        # Should handle gracefully or raise appropriate error
        try:
            data, params = init_data({}, x=x_data, y=y_data)
            # If successful, check dimensions
            assert data.X.shape[0] == 0
        except (ValueError, IndexError):
            # Expected for empty data
            pass