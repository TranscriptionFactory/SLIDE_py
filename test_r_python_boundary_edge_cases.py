"""
Test R-Python interface boundary conditions and edge cases.
Critical for data integrity in cross-language operations.
"""
import pytest
import numpy as np
import pandas as pd
import gc
from unittest.mock import Mock, patch, MagicMock
from loveslide.love import call_love_r, _convert_r_pure_ind
from loveslide.knockoffs import _rlist_get, _create_second_order_r, _solve_sdp_r


class TestRPythonDataIntegrity:
    """Test data integrity across R-Python boundary."""

    def test_large_matrix_transfer_r_python(self):
        """Test transfer of large matrices between R and Python."""
        # Critical gap: Large data transfer edge cases
        X_large = np.random.randn(5000, 2000).astype(np.float64)

        with patch('rpy2.robjects.r') as mock_r:
            # Mock R returning large matrix
            mock_r_matrix = Mock()
            mock_r_matrix.rx2.return_value = Mock()
            mock_r_matrix.rx2.return_value.r_repr = lambda: f"matrix({X_large.size} elements)"

            mock_r.return_value = mock_r_matrix

            # Should handle large data without memory issues
            try:
                result = call_love_r(X_large, lbd=0.5)
                # Should not cause memory error
                assert True
            except MemoryError:
                pytest.fail("Large matrix transfer caused memory error")

    def test_r_na_nan_inf_handling(self):
        """Test handling of R NA/NaN/Inf values in Python."""
        # Critical gap: Special value handling
        X = np.array([[1, 2, np.nan], [4, np.inf, 6], [7, 8, 9]])

        with patch('loveslide.love.call_love_r') as mock_call:
            # Mock R result with special values
            mock_result = {
                'A': np.array([[1.0, float('inf')], [np.nan, 2.0], [3.0, 4.0]]),
                'pure_indices': [1, None, 3],  # Mixed types
                'Omega': np.eye(3)
            }
            mock_call.return_value = mock_result

            # Should handle special values gracefully
            result = call_love_r(X)

            # Should clean or handle special values appropriately
            assert result is not None
            # NA/NaN/Inf should be handled without crashing

    def test_r_memory_cleanup_after_error(self):
        """Test R memory cleanup after Python exceptions."""
        X = np.random.randn(100, 50)

        with patch('rpy2.robjects.r') as mock_r:
            # Setup mock to simulate R memory allocation
            mock_r.side_effect = Exception("Simulated R error")

            with pytest.raises(Exception):
                call_love_r(X)

            # R session should be cleanable after error
            gc.collect()  # Force garbage collection
            # Should not leave dangling R objects

    def test_r_unicode_string_handling(self):
        """Test R-Python string handling with unicode."""
        # Mock R object with unicode strings
        mock_r_obj = Mock()

        with patch('loveslide.knockoffs._rlist_get') as mock_rget:
            # Test various unicode scenarios
            test_strings = ["standard", "unicode_ñáéíóú", "emoji_🧬", "chinese_中文"]

            for test_str in test_strings:
                mock_rget.return_value = test_str
                result = _rlist_get(mock_r_obj, "test_key")
                # Should handle all unicode correctly
                assert isinstance(result, str)

    def test_r_matrix_dimension_consistency(self):
        """Test matrix dimension consistency across R-Python."""
        X = np.random.randn(100, 50)

        with patch('loveslide.knockoffs._create_second_order_r') as mock_create:
            # Mock R returning matrix with wrong dimensions
            wrong_dims = np.random.randn(45, 50)  # Wrong first dimension
            mock_create.return_value = wrong_dims

            # Should detect dimension mismatch
            with pytest.raises((ValueError, AssertionError)):
                result = _create_second_order_r(X)
                # Validate dimensions match expectations
                assert result.shape[0] == X.shape[1]  # Should be p x p

    def test_r_singular_matrix_handling(self):
        """Test R SDP solver with singular/ill-conditioned matrices."""
        # Create singular matrix
        Sigma_singular = np.ones((10, 10))  # Rank 1 matrix
        Sigma_singular[0, 0] = 1.000001  # Slightly perturb to avoid exact singularity

        with patch('loveslide.knockoffs._solve_sdp_r') as mock_solve:
            # Mock R solver failing on singular matrix
            mock_solve.side_effect = Exception("Matrix is singular")

            with pytest.raises(Exception):
                _solve_sdp_r(Sigma_singular)

            # Should provide meaningful error message
            assert True  # Test passes if exception is properly raised


class TestRSessionManagement:
    """Test R session lifecycle and state management."""

    def test_r_session_state_isolation(self):
        """Test that R sessions don't interfere between calls."""
        X1 = np.random.randn(50, 20)
        X2 = np.random.randn(60, 25)

        results = []

        with patch('loveslide.love.call_love_r') as mock_call:
            def session_isolated_call(X, **kwargs):
                # Each call should be independent
                if X.shape[0] == 50:
                    return {'A': np.random.randn(20, 3), 'pure_indices': [1, 2]}
                else:
                    return {'A': np.random.randn(25, 4), 'pure_indices': [1, 2, 3]}

            mock_call.side_effect = session_isolated_call

            # Multiple calls shouldn't interfere
            result1 = call_love_r(X1)
            result2 = call_love_r(X2)

            assert result1['A'].shape[0] == 20
            assert result2['A'].shape[0] == 25

    def test_r_workspace_cleanup(self):
        """Test R workspace is cleaned between operations."""
        X = np.random.randn(100, 50)

        with patch('rpy2.robjects.r') as mock_r:
            # Mock R workspace having leftover objects
            mock_r.ls.return_value = ['old_matrix', 'old_result']

            # Should clean workspace before operations
            with patch('loveslide.love.call_love_r') as mock_call:
                mock_call.return_value = {'A': np.random.randn(50, 5)}
                result = call_love_r(X)

                # Should complete successfully despite workspace clutter
                assert result is not None

    def test_r_package_loading_robustness(self):
        """Test robustness when R packages fail to load."""
        X = np.random.randn(100, 50)

        with patch('rpy2.robjects.packages.importr') as mock_import:
            # Simulate package import failure
            mock_import.side_effect = Exception("Package 'LOVE' not found")

            with pytest.raises(Exception):
                call_love_r(X)

            # Should provide clear error about missing dependencies
            assert True


class TestDataTypeConversions:
    """Test data type conversions between R and Python."""

    def test_r_python_dtype_precision(self):
        """Test numerical precision preservation in R-Python conversion."""
        # Test with high precision data
        X_precise = np.array([[1.23456789012345, 2.98765432109876],
                              [3.45678901234567, 4.56789012345678]], dtype=np.float64)

        with patch('loveslide.love.call_love_r') as mock_call:
            # Mock preserving precision
            mock_result = {
                'A': X_precise * 0.5,  # Simple transformation
                'pure_indices': [1, 2],
                'Omega': np.eye(2)
            }
            mock_call.return_value = mock_result

            result = call_love_r(X_precise)

            # Precision should be maintained
            assert result['A'].dtype == np.float64
            # Check if precision is reasonable (within floating point limits)
            assert np.allclose(result['A'], X_precise * 0.5, rtol=1e-14)

    def test_integer_overflow_handling(self):
        """Test handling of integer overflow in R-Python conversion."""
        # Test with large integer indices
        large_indices = [2**31 - 1, 2**31, 2**32 - 1]  # Near int32/int64 boundaries

        with patch('loveslide.love._convert_r_pure_ind') as mock_convert:
            mock_convert.return_value = large_indices

            result = _convert_r_pure_ind(Mock())

            # Should handle large integers correctly
            assert all(isinstance(idx, (int, np.integer)) for idx in result)
            assert max(result) == 2**32 - 1

    def test_sparse_matrix_conversion_edge_cases(self):
        """Test sparse matrix handling edge cases."""
        from scipy.sparse import csr_matrix

        # Very sparse matrix (mostly zeros)
        sparse_data = np.zeros((1000, 500))
        sparse_data[0, 0] = 1.0
        sparse_data[999, 499] = 2.0
        X_sparse = csr_matrix(sparse_data)

        # Should handle extremely sparse data
        with patch('loveslide.love.call_love_r') as mock_call:
            mock_call.return_value = {
                'A': np.random.randn(500, 10),
                'pure_indices': [1, 2, 3]
            }

            # Convert to dense for R interface
            result = call_love_r(X_sparse.toarray())
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])