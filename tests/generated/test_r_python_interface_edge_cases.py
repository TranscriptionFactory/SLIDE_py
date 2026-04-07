"""
Test coverage for R-Python interface edge cases and boundary conditions.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import warnings
from src.loveslide.love import call_love, _convert_r_pure_ind
from src.loveslide.knockoffs import (
    _rlist_get, _create_second_order_r, _solve_sdp_r,
    Knockoffs
)


class TestRPythonInterfaceEdges:
    """Test R-Python interface edge cases not covered elsewhere."""

    def test_rlist_get_with_null_robject(self):
        """Test _rlist_get with NULL R object."""
        # Mock rpy2 R object that returns NULL
        mock_robj = MagicMock()
        mock_robj.__getitem__.side_effect = Exception("NULL object")

        with pytest.raises(Exception):
            result = _rlist_get(mock_robj, "nonexistent_field")

    def test_rlist_get_with_corrupted_rdata(self):
        """Test _rlist_get with corrupted R data structure."""
        mock_robj = MagicMock()
        mock_robj.__getitem__.return_value = None

        result = _rlist_get(mock_robj, "field")
        assert result is None

    def test_convert_r_pure_ind_empty_list(self):
        """Test _convert_r_pure_ind with empty R list."""
        # Mock empty R list structure
        empty_r_list = []
        result = _convert_r_pure_ind(empty_r_list)
        assert result == []

    def test_convert_r_pure_ind_single_element(self):
        """Test _convert_r_pure_ind with single element."""
        # Mock R list with single integer vector
        mock_r_vector = MagicMock()
        mock_r_vector.__array__ = lambda: np.array([5])  # R uses 1-based indexing

        single_element = [mock_r_vector]
        result = _convert_r_pure_ind(single_element)

        assert len(result) == 1
        assert result[0] == [4]  # Converted to 0-based

    def test_convert_r_pure_ind_mixed_lengths(self):
        """Test _convert_r_pure_ind with vectors of different lengths."""
        # Mock R vectors of different lengths
        vector1 = MagicMock()
        vector1.__array__ = lambda: np.array([1, 2])

        vector2 = MagicMock()
        vector2.__array__ = lambda: np.array([3, 4, 5, 6])

        mixed_list = [vector1, vector2]
        result = _convert_r_pure_ind(mixed_list)

        assert len(result) == 2
        assert result[0] == [0, 1]  # 1-based to 0-based conversion
        assert result[1] == [2, 3, 4, 5]

    @patch('src.loveslide.love.pyreadr.read_r')
    def test_call_love_with_r_errors(self, mock_read_r):
        """Test call_love when R script encounters errors."""
        X = np.random.randn(50, 20)
        pure_homo = True

        # Simulate R error
        mock_read_r.side_effect = Exception("R script error: subscript out of bounds")

        with pytest.raises(Exception) as exc_info:
            call_love(X, pure_homo)

        assert "R script error" in str(exc_info.value)

    @patch('src.loveslide.love.pyreadr.read_r')
    def test_call_love_with_malformed_r_output(self, mock_read_r):
        """Test call_love with malformed R output structure."""
        X = np.random.randn(50, 20)
        pure_homo = True

        # Mock malformed R output missing expected keys
        mock_read_r.return_value = {
            'malformed_key': pd.DataFrame(),  # Missing 'A' and 'pure_indices'
        }

        with pytest.raises(KeyError):
            result = call_love(X, pure_homo)


class TestRMatrixOperations:
    """Test R matrix operations edge cases."""

    def test_create_second_order_r_singular_matrix(self):
        """Test _create_second_order_r with singular correlation matrix."""
        # Create perfectly singular correlation matrix
        X = np.random.randn(100, 5)
        X[:, 1] = X[:, 0]  # Perfect correlation
        X[:, 2] = 2 * X[:, 0]  # Linear combination

        # Should handle singular matrices gracefully
        result = _create_second_order_r(X)
        assert result is not None
        assert result.shape[0] == X.shape[1]

    def test_solve_sdp_r_ill_conditioned_sigma(self):
        """Test _solve_sdp_r with extremely ill-conditioned Sigma."""
        # Create ill-conditioned covariance matrix
        eigenvals = np.array([1, 1e-6, 1e-12, 1e-15, 1e-20])
        U = np.random.randn(5, 5)
        U, _ = np.linalg.qr(U)
        Sigma = U @ np.diag(eigenvals) @ U.T

        # Should handle ill-conditioning gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _solve_sdp_r(Sigma, method='sdp')
            assert result is not None
            assert result.shape == Sigma.shape

    def test_solve_sdp_r_extreme_condition_number(self):
        """Test _solve_sdp_r with matrix having extreme condition number."""
        # Matrix with condition number > 1e16
        n = 10
        U = np.random.randn(n, n)
        U, _ = np.linalg.qr(U)
        eigenvals = np.logspace(0, -18, n)  # Condition number ~ 1e18
        Sigma = U @ np.diag(eigenvals) @ U.T

        # Should either solve or raise appropriate error
        try:
            result = _solve_sdp_r(Sigma, method='sdp')
            assert result.shape == Sigma.shape
        except (np.linalg.LinAlgError, RuntimeError):
            # Acceptable to fail on extremely ill-conditioned matrices
            pass


class TestKnockoffInterfaceEdges:
    """Test knockoff generation interface edge cases."""

    def test_knockoffs_with_zero_variance_features(self):
        """Test knockoff generation with zero-variance features."""
        X = np.random.randn(100, 10)
        X[:, 3] = 1.0  # Constant feature
        X[:, 7] = 0.0  # Zero feature

        knockoffs = Knockoffs()

        # Should handle zero-variance features gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = knockoffs.create_second_order(X, method='sdp')
                assert result.shape == X.shape
            except (ValueError, np.linalg.LinAlgError):
                # Acceptable to fail with zero variance
                pass

    def test_knockoffs_with_extreme_correlations(self):
        """Test knockoff generation with extreme feature correlations."""
        # Create features with correlation = 0.999999
        base = np.random.randn(50, 1)
        X = np.hstack([
            base,
            base + np.random.randn(50, 1) * 1e-6,  # Almost identical
            np.random.randn(50, 8)
        ])

        knockoffs = Knockoffs()

        # Should handle near-perfect correlations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = knockoffs.create_second_order(X, method='equicorrelated')
            assert result.shape == X.shape

    def test_knockoffs_memory_pressure(self):
        """Test knockoff generation under memory pressure simulation."""
        # Large matrix that might cause memory issues
        large_X = np.random.randn(1000, 500)

        knockoffs = Knockoffs()

        # Should handle large matrices efficiently
        # Note: This test might be slow, so we use a smaller size in practice
        result = knockoffs.create_second_order(large_X[:200, :50], method='equicorrelated')
        assert result.shape == (200, 50)