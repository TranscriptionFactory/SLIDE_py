"""
Comprehensive test coverage for utility functions and integration scenarios.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import pickle

from loveslide.tools import init_data, show_params, check_params, calc_default_fsize


class TestInitData:
    """Test init_data function for data initialization and validation."""

    def test_init_data_numpy_arrays(self):
        """Test init_data with numpy arrays."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)
        params = {"fdr": 0.1, "niter": 5}

        data, processed_params = init_data(params, x=X, y=y)

        assert hasattr(data, 'X')
        assert hasattr(data, 'y')
        np.testing.assert_array_equal(data.X, X)
        np.testing.assert_array_equal(data.y, y)
        assert processed_params["fdr"] == 0.1

    def test_init_data_pandas_dataframes(self):
        """Test init_data with pandas DataFrames."""
        X_df = pd.DataFrame(np.random.randn(100, 20),
                          columns=[f'feature_{i}' for i in range(20)])
        y_series = pd.Series(np.random.randn(100), name='target')
        params = {"fdr": 0.1}

        data, processed_params = init_data(params, x=X_df, y=y_series)

        # Should convert to numpy arrays
        assert isinstance(data.X, np.ndarray)
        assert isinstance(data.y, np.ndarray)
        assert data.X.shape == (100, 20)
        assert data.y.shape == (100,)

    def test_init_data_mismatched_shapes(self):
        """Test init_data with mismatched X and y shapes."""
        X = np.random.randn(100, 20)
        y = np.random.randn(50)  # Wrong shape
        params = {}

        with pytest.raises(ValueError, match="shape"):
            init_data(params, x=X, y=y)

    def test_init_data_from_file_paths(self):
        """Test init_data loading from file paths."""
        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save test data
            X = np.random.randn(100, 20)
            y = np.random.randn(100)

            X_path = tmpdir / "X.npy"
            y_path = tmpdir / "y.npy"

            np.save(X_path, X)
            np.save(y_path, y)

            params = {
                "X_path": str(X_path),
                "y_path": str(y_path)
            }

            # Should load from files
            data, processed_params = init_data(params)

            np.testing.assert_array_equal(data.X, X)
            np.testing.assert_array_equal(data.y, y)

    def test_init_data_csv_files(self):
        """Test init_data loading from CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create CSV files
            X_df = pd.DataFrame(np.random.randn(50, 10))
            y_df = pd.DataFrame(np.random.randn(50), columns=['target'])

            X_path = tmpdir / "features.csv"
            y_path = tmpdir / "targets.csv"

            X_df.to_csv(X_path, index=False)
            y_df.to_csv(y_path, index=False)

            params = {
                "X_path": str(X_path),
                "y_path": str(y_path)
            }

            data, processed_params = init_data(params)

            assert data.X.shape == (50, 10)
            assert data.y.shape == (50,)

    def test_init_data_invalid_file_paths(self):
        """Test init_data with invalid file paths."""
        params = {
            "X_path": "/nonexistent/file.npy",
            "y_path": "/nonexistent/target.npy"
        }

        with pytest.raises(FileNotFoundError):
            init_data(params)

    def test_init_data_missing_data(self):
        """Test init_data with no data provided."""
        params = {"fdr": 0.1}

        with pytest.raises(ValueError, match="No data provided"):
            init_data(params)

    def test_init_data_nan_values(self):
        """Test init_data with NaN values."""
        X = np.random.randn(100, 20)
        X[0, 0] = np.nan
        y = np.random.randn(100)
        params = {}

        with pytest.raises(ValueError, match="NaN"):
            init_data(params, x=X, y=y)

    def test_init_data_infinite_values(self):
        """Test init_data with infinite values."""
        X = np.random.randn(100, 20)
        X[0, 0] = np.inf
        y = np.random.randn(100)
        params = {}

        with pytest.raises(ValueError, match="infinite"):
            init_data(params, x=X, y=y)

    def test_init_data_parameter_defaults(self):
        """Test init_data sets appropriate parameter defaults."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)
        params = {}

        data, processed_params = init_data(params, x=X, y=y)

        # Should set default parameters
        assert "fdr" in processed_params
        assert "niter" in processed_params
        assert isinstance(processed_params["fdr"], (int, float))

    def test_init_data_parameter_validation(self):
        """Test init_data validates parameter types and ranges."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        # Invalid FDR
        with pytest.raises(ValueError):
            init_data({"fdr": -0.1}, x=X, y=y)

        with pytest.raises(ValueError):
            init_data({"fdr": 1.5}, x=X, y=y)

        # Invalid niter
        with pytest.raises(ValueError):
            init_data({"niter": 0}, x=X, y=y)

        with pytest.raises(ValueError):
            init_data({"niter": -5}, x=X, y=y)


class TestShowParams:
    """Test show_params function for parameter display."""

    def test_show_params_basic(self):
        """Test basic show_params functionality."""
        params = {"fdr": 0.1, "niter": 5, "f_size": 100}

        # Create mock data object
        data = Mock()
        data.X = np.random.randn(100, 20)
        data.y = np.random.randn(100)

        # Should not raise any errors
        show_params(params, data)

    def test_show_params_none_data(self):
        """Test show_params with None data."""
        params = {"fdr": 0.1}

        # Should handle None data gracefully
        show_params(params, None)

    def test_show_params_empty_params(self):
        """Test show_params with empty parameters."""
        data = Mock()
        data.X = np.random.randn(100, 20)
        data.y = np.random.randn(100)

        # Should handle empty parameters
        show_params({}, data)

    @patch('builtins.print')
    def test_show_params_output_format(self, mock_print):
        """Test that show_params produces expected output format."""
        params = {"fdr": 0.1, "niter": 5}
        data = Mock()
        data.X = np.random.randn(100, 20)
        data.y = np.random.randn(100)

        show_params(params, data)

        # Should have printed something
        assert mock_print.called

        # Check that parameters were printed
        printed_output = ' '.join(str(call[0][0]) for call in mock_print.call_args_list)
        assert "fdr" in printed_output.lower()
        assert "0.1" in printed_output


class TestCheckParams:
    """Test check_params function for parameter validation."""

    def test_check_params_valid(self):
        """Test check_params with valid parameters."""
        params = {
            "fdr": 0.1,
            "niter": 5,
            "f_size": 100,
            "lbd": 0.5,
            "delta": 0.2
        }

        data = Mock()
        data.X = np.random.randn(100, 20)
        data.y = np.random.randn(100)

        # Should not raise any errors
        check_params(params, data)

    def test_check_params_invalid_fdr(self):
        """Test check_params with invalid FDR values."""
        data = Mock()
        data.X = np.random.randn(100, 20)
        data.y = np.random.randn(100)

        # FDR out of range
        with pytest.raises(ValueError, match="fdr"):
            check_params({"fdr": -0.1}, data)

        with pytest.raises(ValueError, match="fdr"):
            check_params({"fdr": 1.5}, data)

    def test_check_params_invalid_niter(self):
        """Test check_params with invalid iteration count."""
        data = Mock()
        data.X = np.random.randn(100, 20)
        data.y = np.random.randn(100)

        with pytest.raises(ValueError, match="niter"):
            check_params({"niter": 0}, data)

        with pytest.raises(ValueError, match="niter"):
            check_params({"niter": -1}, data)

    def test_check_params_invalid_f_size(self):
        """Test check_params with invalid feature size."""
        data = Mock()
        data.X = np.random.randn(100, 20)
        data.y = np.random.randn(100)

        with pytest.raises(ValueError, match="f_size"):
            check_params({"f_size": 0}, data)

        # f_size larger than number of features
        with pytest.raises(ValueError):
            check_params({"f_size": 25}, data)  # X has 20 features

    def test_check_params_invalid_lambda_delta(self):
        """Test check_params with invalid lambda and delta values."""
        data = Mock()
        data.X = np.random.randn(100, 20)
        data.y = np.random.randn(100)

        # Invalid lambda
        with pytest.raises(ValueError):
            check_params({"lbd": -0.1}, data)

        with pytest.raises(ValueError):
            check_params({"lbd": 1.5}, data)

        # Invalid delta
        with pytest.raises(ValueError):
            check_params({"delta": -0.1}, data)

        with pytest.raises(ValueError):
            check_params({"delta": 1.5}, data)

    def test_check_params_missing_required(self):
        """Test check_params with missing required parameters."""
        data = Mock()
        data.X = np.random.randn(100, 20)
        data.y = np.random.randn(100)

        # Test with minimal parameters (should set defaults)
        check_params({}, data)

    def test_check_params_type_validation(self):
        """Test check_params validates parameter types."""
        data = Mock()
        data.X = np.random.randn(100, 20)
        data.y = np.random.randn(100)

        # String instead of number
        with pytest.raises(TypeError):
            check_params({"fdr": "0.1"}, data)

        # List instead of scalar
        with pytest.raises(TypeError):
            check_params({"niter": [5]}, data)


class TestCalcDefaultFsize:
    """Test calc_default_fsize function."""

    def test_calc_default_fsize_basic(self):
        """Test basic default feature size calculation."""
        n_rows = 1000
        K = 10

        f_size = calc_default_fsize(n_rows, K)

        assert isinstance(f_size, int)
        assert f_size > 0
        assert f_size <= n_rows  # Shouldn't exceed sample size

    def test_calc_default_fsize_small_n(self):
        """Test default feature size with small sample size."""
        n_rows = 50
        K = 5

        f_size = calc_default_fsize(n_rows, K)

        assert f_size > 0
        assert f_size <= n_rows

    def test_calc_default_fsize_large_k(self):
        """Test default feature size with large K."""
        n_rows = 1000
        K = 100  # Large number of factors

        f_size = calc_default_fsize(n_rows, K)

        assert f_size > 0
        # Should scale appropriately with K

    def test_calc_default_fsize_edge_cases(self):
        """Test edge cases for default feature size calculation."""
        # Very small inputs
        f_size_small = calc_default_fsize(10, 2)
        assert f_size_small > 0

        # K = 1
        f_size_k1 = calc_default_fsize(100, 1)
        assert f_size_k1 > 0

    def test_calc_default_fsize_invalid_inputs(self):
        """Test default feature size with invalid inputs."""
        # Zero or negative n_rows
        with pytest.raises(ValueError):
            calc_default_fsize(0, 5)

        with pytest.raises(ValueError):
            calc_default_fsize(-10, 5)

        # Zero or negative K
        with pytest.raises(ValueError):
            calc_default_fsize(100, 0)

        with pytest.raises(ValueError):
            calc_default_fsize(100, -5)

    def test_calc_default_fsize_consistency(self):
        """Test that default feature size calculation is consistent."""
        n_rows = 500
        K = 10

        # Multiple calls should return same result
        f_size1 = calc_default_fsize(n_rows, K)
        f_size2 = calc_default_fsize(n_rows, K)

        assert f_size1 == f_size2

    def test_calc_default_fsize_scaling(self):
        """Test that default feature size scales appropriately."""
        K = 10

        # Larger n_rows should generally allow larger f_size
        f_size_small = calc_default_fsize(100, K)
        f_size_large = calc_default_fsize(1000, K)

        assert f_size_large >= f_size_small


class TestUtilityIntegration:
    """Test integration scenarios between utility functions."""

    def test_init_data_check_params_integration(self):
        """Test integration between init_data and check_params."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)
        params = {"fdr": 0.1, "niter": 5}

        # Initialize data
        data, processed_params = init_data(params, x=X, y=y)

        # Check parameters should pass
        check_params(processed_params, data)

    def test_full_pipeline_integration(self):
        """Test full utility pipeline integration."""
        X = np.random.randn(200, 30)
        y = np.random.randn(200)
        params = {"fdr": 0.2, "niter": 10}

        # Full pipeline
        data, processed_params = init_data(params, x=X, y=y)
        check_params(processed_params, data)
        show_params(processed_params, data)

        # Calculate default feature size
        K = 5
        f_size = calc_default_fsize(data.X.shape[0], K)

        assert f_size > 0
        assert f_size <= data.X.shape[1]  # Shouldn't exceed features

    def test_parameter_persistence(self):
        """Test that parameters are consistently handled across functions."""
        X = np.random.randn(150, 25)
        y = np.random.randn(150)

        original_params = {
            "fdr": 0.15,
            "niter": 8,
            "custom_param": "test_value"
        }

        data, processed_params = init_data(original_params, x=X, y=y)

        # Original custom parameters should be preserved
        assert "custom_param" in processed_params
        assert processed_params["custom_param"] == "test_value"
        assert processed_params["fdr"] == 0.15
        assert processed_params["niter"] == 8

    def test_data_format_consistency(self):
        """Test data format consistency across different inputs."""
        # Test with different input formats
        X_numpy = np.random.randn(100, 20)
        y_numpy = np.random.randn(100)

        X_pandas = pd.DataFrame(X_numpy)
        y_pandas = pd.Series(y_numpy)

        params = {"fdr": 0.1}

        # Both should produce identical data objects
        data_numpy, _ = init_data(params, x=X_numpy, y=y_numpy)
        data_pandas, _ = init_data(params, x=X_pandas, y=y_pandas)

        np.testing.assert_array_equal(data_numpy.X, data_pandas.X)
        np.testing.assert_array_equal(data_numpy.y, data_pandas.y)

    def test_error_propagation(self):
        """Test that errors propagate correctly through utility functions."""
        # Create invalid data that should fail in check_params
        X = np.random.randn(100, 20)
        y = np.random.randn(100)
        invalid_params = {"fdr": 1.5}  # Invalid FDR

        # Should fail in init_data parameter validation
        with pytest.raises(ValueError):
            data, processed_params = init_data(invalid_params, x=X, y=y)

    def test_memory_efficiency_utilities(self):
        """Test memory efficiency of utility functions."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Process moderately large data
        X = np.random.randn(1000, 50)
        y = np.random.randn(1000)
        params = {"fdr": 0.1, "niter": 5}

        data, processed_params = init_data(params, x=X, y=y)
        check_params(processed_params, data)

        memory_after = process.memory_info().rss
        memory_increase = (memory_after - memory_before) / (1024 * 1024)  # MB

        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB for this size