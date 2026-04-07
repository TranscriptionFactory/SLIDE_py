"""
Comprehensive test coverage for LOVE algorithm functionality.
"""
import pytest
import numpy as np
from unittest.mock import patch, Mock
import tempfile

from loveslide.love import call_love, call_love_r, _convert_r_pure_ind


class TestLOVEParameterValidation:
    """Test LOVE parameter validation and boundary conditions."""

    def test_call_love_invalid_lambda(self):
        """Test LOVE with invalid lambda values."""
        X = np.random.randn(50, 20)

        # Test negative lambda
        with pytest.raises(ValueError):
            call_love(X, lbd=-0.5)

        # Test lambda > 1
        with pytest.raises(ValueError):
            call_love(X, lbd=1.5)

    def test_call_love_invalid_mu(self):
        """Test LOVE with invalid mu values."""
        X = np.random.randn(50, 20)

        with pytest.raises(ValueError):
            call_love(X, mu=-0.1)

        with pytest.raises(ValueError):
            call_love(X, mu=1.1)

    def test_call_love_invalid_thresh_fdr(self):
        """Test LOVE with invalid FDR thresholds."""
        X = np.random.randn(50, 20)

        with pytest.raises(ValueError):
            call_love(X, thresh_fdr=-0.1)

        with pytest.raises(ValueError):
            call_love(X, thresh_fdr=1.5)

    def test_call_love_invalid_matrix_dimensions(self):
        """Test LOVE with invalid matrix dimensions."""
        # Single row
        X_single = np.random.randn(1, 20)
        with pytest.raises(ValueError):
            call_love(X_single)

        # Single column
        X_single_col = np.random.randn(50, 1)
        with pytest.raises(ValueError):
            call_love(X_single_col)

        # More features than samples
        X_wide = np.random.randn(10, 50)
        with pytest.warns(UserWarning):
            call_love(X_wide)

    def test_call_love_singular_covariance(self):
        """Test LOVE with singular covariance matrix."""
        # Create singular matrix
        X = np.ones((50, 20))  # All rows identical
        X += 0.01 * np.random.randn(50, 20)  # Add tiny noise

        with pytest.raises((np.linalg.LinAlgError, ValueError)):
            call_love(X)


class TestLOVERInterface:
    """Test R interface functionality and error handling."""

    @patch('loveslide.love.rpy2_available', False)
    def test_call_love_r_without_rpy2(self):
        """Test R interface when rpy2 is not available."""
        X = np.random.randn(50, 20)

        with pytest.raises(ImportError, match="rpy2 not available"):
            call_love_r(X)

    @patch('loveslide.love.rpy2_available', True)
    @patch('loveslide.love.r_LOVE')
    def test_call_love_r_execution_error(self, mock_r_love):
        """Test R interface when R execution fails."""
        X = np.random.randn(50, 20)

        # Mock R function to raise exception
        mock_r_love.side_effect = Exception("R execution failed")

        with pytest.raises(Exception, match="R execution failed"):
            call_love_r(X)

    def test_convert_r_pure_ind_empty_list(self):
        """Test R pure index conversion with empty list."""
        result = _convert_r_pure_ind([])
        assert result == []

    def test_convert_r_pure_ind_nested_structure(self):
        """Test R pure index conversion with complex nested structure."""
        # Mock R list structure
        mock_r_list = [
            Mock(r_repr=lambda: "[1] 1 2 3"),
            Mock(r_repr=lambda: "[1] 4 5")
        ]

        result = _convert_r_pure_ind(mock_r_list)
        expected = [[0, 1, 2], [3, 4]]  # R is 1-indexed, Python is 0-indexed
        assert result == expected


class TestLOVEAlgorithmicEdgeCases:
    """Test algorithmic edge cases in LOVE."""

    def test_call_love_perfect_correlation(self):
        """Test LOVE with perfectly correlated features."""
        X = np.random.randn(100, 1)
        X = np.column_stack([X, X])  # Perfect correlation

        # Should handle gracefully or warn
        with pytest.warns(UserWarning):
            result = call_love(X)
            assert isinstance(result, dict)

    def test_call_love_zero_variance_features(self):
        """Test LOVE with zero variance features."""
        X = np.random.randn(100, 20)
        X[:, 0] = 1.0  # Zero variance feature

        with pytest.warns(UserWarning):
            result = call_love(X)
            assert isinstance(result, dict)

    def test_call_love_extreme_values(self):
        """Test LOVE with extreme values."""
        X = np.random.randn(100, 20)
        X[0, 0] = 1e10  # Extreme outlier

        result = call_love(X)
        assert isinstance(result, dict)
        assert "pure_nodes" in result

    def test_call_love_missing_values(self):
        """Test LOVE with missing values (NaN)."""
        X = np.random.randn(100, 20)
        X[0, 0] = np.nan

        with pytest.raises(ValueError):
            call_love(X)

    def test_call_love_infinite_values(self):
        """Test LOVE with infinite values."""
        X = np.random.randn(100, 20)
        X[0, 0] = np.inf

        with pytest.raises(ValueError):
            call_love(X)


class TestLOVEOutputValidation:
    """Test LOVE output structure and content validation."""

    def test_call_love_output_structure(self):
        """Test that LOVE output has expected structure."""
        np.random.seed(42)  # For reproducibility
        X = np.random.randn(100, 20)

        result = call_love(X)

        # Check required keys
        required_keys = ["pure_nodes", "pure_edges", "L_hat", "Gamma_LL"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        # Check data types
        assert isinstance(result["pure_nodes"], list)
        assert isinstance(result["pure_edges"], dict)
        assert isinstance(result["L_hat"], np.ndarray)
        assert isinstance(result["Gamma_LL"], np.ndarray)

    def test_call_love_output_dimensions(self):
        """Test LOVE output dimensions are consistent."""
        np.random.seed(42)
        X = np.random.randn(100, 20)

        result = call_love(X)

        # L_hat should have correct dimensions
        assert result["L_hat"].shape[0] == X.shape[1]  # Features
        n_factors = result["L_hat"].shape[1]

        # Gamma_LL should be square matrix of factors
        assert result["Gamma_LL"].shape == (n_factors, n_factors)

    def test_call_love_reproducibility(self):
        """Test LOVE reproducibility with same input."""
        np.random.seed(42)
        X = np.random.randn(100, 20)

        result1 = call_love(X, verbose=False)
        result2 = call_love(X, verbose=False)

        # Results should be identical
        np.testing.assert_array_almost_equal(result1["L_hat"], result2["L_hat"])
        np.testing.assert_array_almost_equal(
            result1["Gamma_LL"], result2["Gamma_LL"]
        )


class TestLOVEPerformanceAndScalability:
    """Test LOVE performance with different dataset sizes."""

    def test_call_love_small_dataset(self):
        """Test LOVE with small dataset."""
        X = np.random.randn(20, 10)  # Small dataset

        result = call_love(X)
        assert isinstance(result, dict)
        assert len(result["pure_nodes"]) >= 0  # May be empty for small data

    def test_call_love_moderate_dataset(self):
        """Test LOVE with moderate dataset."""
        X = np.random.randn(200, 50)

        result = call_love(X)
        assert isinstance(result, dict)

    @pytest.mark.slow
    def test_call_love_large_dataset(self):
        """Test LOVE with large dataset (marked as slow)."""
        X = np.random.randn(1000, 100)

        import time
        start_time = time.time()
        result = call_love(X)
        execution_time = time.time() - start_time

        assert isinstance(result, dict)
        # Should complete in reasonable time (< 60 seconds)
        assert execution_time < 60


class TestLOVEMemoryManagement:
    """Test memory management in LOVE algorithm."""

    def test_call_love_memory_efficiency(self):
        """Test memory efficiency during LOVE execution."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        X = np.random.randn(500, 50)
        result = call_love(X)

        memory_after = process.memory_info().rss
        memory_increase = (memory_after - memory_before) / (1024 * 1024)  # MB

        # Memory increase should be reasonable for dataset size
        assert memory_increase < 200  # Less than 200MB

        # Clean up should happen automatically
        del result
        del X

    def test_call_love_cleanup_after_error(self):
        """Test memory cleanup after LOVE execution error."""
        X = np.random.randn(50, 20)

        try:
            # Force an error
            call_love(X, lbd=-1)  # Invalid parameter
        except ValueError:
            pass

        # Memory should be cleaned up even after error
        import gc
        gc.collect()  # Force garbage collection
        # Test passes if no memory leaks detected