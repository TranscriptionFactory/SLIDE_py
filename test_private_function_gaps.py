"""
Test coverage for private functions that are not directly tested.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.loveslide.knockoff.utils import (
    diag_pre_multiply, diag_post_multiply, is_posdef, canonical_svd,
    normc, cov2cor, rnorm_matrix, random_problem, with_seed
)
from src.loveslide.knockoff.filter import (
    _run_single_knockoff, _prepare_knockoff_cache, _sample_knockoffs_from_cache,
    _cached_iteration
)
from src.loveslide.knockoff.create import _decompose, _create_equicorrelated, _create_sdp
from src.loveslide.knockoffs import _rlist_get, _create_second_order_r, _solve_sdp_r
from src.loveslide.love import _convert_r_pure_ind


class TestPrivateUtilityFunctions:
    """Test private utility functions from knockoff.utils module."""

    def test_diag_pre_multiply_edge_cases(self):
        """Test diagonal pre-multiplication with edge cases."""
        # Test with zero diagonal
        d = np.array([0, 1, 0])
        X = np.random.randn(3, 4)
        result = diag_pre_multiply(d, X)
        assert np.allclose(result[0, :], 0)
        assert np.allclose(result[2, :], 0)
        assert np.allclose(result[1, :], X[1, :])

        # Test with negative values
        d = np.array([-1, 2, -3])
        result = diag_pre_multiply(d, X)
        expected = np.diag(d) @ X
        assert np.allclose(result, expected)

        # Test with single element
        d = np.array([5])
        X = np.array([[1, 2, 3]])
        result = diag_pre_multiply(d, X)
        assert np.allclose(result, d * X)

    def test_diag_post_multiply_edge_cases(self):
        """Test diagonal post-multiplication with edge cases."""
        # Test with mismatched dimensions
        X = np.random.randn(3, 4)
        d = np.array([1, 0, -2, 3])
        result = diag_post_multiply(X, d)
        assert result.shape == X.shape
        assert np.allclose(result[:, 1], 0)  # Zero diagonal element

        # Test broadcasting behavior
        X = np.array([[1, 2], [3, 4]])
        d = np.array([0.5, 2])
        result = diag_post_multiply(X, d)
        expected = X @ np.diag(d)
        assert np.allclose(result, expected)

    def test_is_posdef_numerical_edge_cases(self):
        """Test positive definiteness checking with numerical edge cases."""
        # Test with nearly singular matrix
        A = np.array([[1, 0.999999], [0.999999, 1]])
        assert is_posdef(A, tol=1e-8)
        assert not is_posdef(A, tol=1e-5)

        # Test with ill-conditioned matrix
        n = 5
        A = np.random.randn(n, n)
        A = A @ A.T + 1e-12 * np.eye(n)  # Nearly singular
        result = is_posdef(A)
        assert isinstance(result, bool)

        # Test large matrix fallback
        with patch('src.loveslide.knockoff.utils.eigsh') as mock_eigsh:
            mock_eigsh.side_effect = Exception("Sparse solver failed")
            large_A = np.eye(600) + 0.1 * np.random.randn(600, 600)
            large_A = large_A @ large_A.T  # Ensure positive definite
            result = is_posdef(large_A)
            assert result is True

    def test_canonical_svd_sign_consistency(self):
        """Test canonical SVD sign choice consistency."""
        X = np.array([[1, -2], [3, 4], [-5, 6]])
        U, d, V = canonical_svd(X)

        # Check that largest absolute value in each U column is positive
        for j in range(U.shape[1]):
            max_idx = np.argmax(np.abs(U[:, j]))
            assert U[max_idx, j] > 0

        # Test with matrix that would naturally have negative leading elements
        X_neg = -np.abs(np.random.randn(4, 3))
        U, d, V = canonical_svd(X_neg)
        for j in range(U.shape[1]):
            max_idx = np.argmax(np.abs(U[:, j]))
            assert U[max_idx, j] > 0

    def test_canonical_svd_failure_handling(self):
        """Test SVD failure handling."""
        with patch('src.loveslide.knockoff.utils.linalg.svd') as mock_svd:
            mock_svd.side_effect = np.linalg.LinAlgError("SVD did not converge")

            X = np.random.randn(3, 2)
            with pytest.raises(RuntimeError, match="SVD failed"):
                canonical_svd(X)

    def test_normc_zero_norm_columns(self):
        """Test column normalization with zero-norm columns."""
        X = np.array([[1, 0, 2], [2, 0, 4], [3, 0, 6]])
        result = normc(X, center=False)

        # Zero-norm column should remain unchanged (division by 1.0)
        assert np.allclose(result[:, 1], [0, 0, 0])

        # Other columns should have unit norm
        assert np.allclose(np.linalg.norm(result[:, 0]), 1)
        assert np.allclose(np.linalg.norm(result[:, 2]), 1)

    def test_cov2cor_degenerate_cases(self):
        """Test covariance to correlation with degenerate cases."""
        # Test with zero variance variables
        Sigma = np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 0]])
        R = cov2cor(Sigma)

        assert np.allclose(np.diag(R), 1)  # Unit diagonal
        assert np.allclose(R, R.T)        # Symmetry
        assert np.allclose(R[2, :], [0, 0, 1])  # Zero variance row/col

    def test_random_problem_edge_cases(self):
        """Test random problem generation with edge cases."""
        # Test with k > p
        with pytest.raises(ValueError):
            random_problem(n=10, p=5, k=10)

        # Test with n < p scenario
        prob = random_problem(n=5, p=10, k=2, seed=42)
        assert prob['X'].shape == (5, 10)
        assert len(prob['nonzero']) == 2
        assert np.sum(prob['beta'] != 0) == 2

        # Test deterministic behavior with seed
        prob1 = random_problem(n=10, p=5, k=2, seed=123)
        prob2 = random_problem(n=10, p=5, k=2, seed=123)
        assert np.allclose(prob1['X'], prob2['X'])
        assert np.array_equal(prob1['nonzero'], prob2['nonzero'])

    def test_with_seed_state_restoration(self):
        """Test random state restoration in with_seed."""
        # Set a specific state
        np.random.seed(42)
        state_before = np.random.get_state()

        # Use with_seed to generate some numbers
        def random_func():
            return np.random.randn(3)

        result = with_seed(123, random_func)
        state_after = np.random.get_state()

        # State should be restored
        assert np.array_equal(state_before[1], state_after[1])
        assert state_before[0] == state_after[0]

        # Function should have run with the specified seed
        np.random.seed(123)
        expected = np.random.randn(3)
        np.random.seed(123)  # Reset to test
        actual = with_seed(123, random_func)
        assert np.allclose(actual, expected)


class TestPrivateKnockoffFunctions:
    """Test private functions from knockoff modules."""

    def test_decompose_edge_cases(self):
        """Test matrix decomposition with edge cases."""
        # Test with rank-deficient matrix
        X = np.array([[1, 2], [2, 4], [3, 6]])  # Rank 1
        result = _decompose(X, randomize=False)

        assert 'U' in result
        assert 'D' in result
        assert 'V' in result
        assert result['U'].shape[1] <= min(X.shape)

        # Test with square matrix
        X_square = np.eye(3) + 0.1 * np.random.randn(3, 3)
        result = _decompose(X_square, randomize=True)
        assert result['U'].shape == (3, 3)

    def test_rlist_get_error_handling(self):
        """Test R object attribute access error handling."""
        # Mock R object without the requested attribute
        mock_robj = MagicMock()
        mock_robj.names = ['attr1', 'attr2']

        # Test with missing attribute
        result = _rlist_get(mock_robj, 'missing_attr')
        assert result is None

        # Test with existing attribute
        mock_robj.__getitem__ = lambda x: f"value_{x}"
        result = _rlist_get(mock_robj, 'attr1')
        assert result == 'value_attr1'

    @patch('src.loveslide.knockoffs.ro.r')
    def test_create_second_order_r_failure(self, mock_r):
        """Test R-based second order knockoff creation failure."""
        mock_r.side_effect = Exception("R interface failed")

        X = np.random.randn(10, 5)
        with pytest.raises(Exception):
            _create_second_order_r(X)

    def test_convert_r_pure_ind_edge_cases(self):
        """Test R pure indices conversion with edge cases."""
        # Test with empty list
        result = _convert_r_pure_ind([])
        assert result == []

        # Test with single group
        mock_robj = MagicMock()
        mock_robj.__len__ = lambda: 3
        mock_robj.__getitem__ = lambda i: np.array([1, 2, 3]) + i

        r_list = [mock_robj]
        result = _convert_r_pure_ind(r_list)
        assert len(result) == 1
        assert isinstance(result[0], list)


class TestPrivateFilterFunctions:
    """Test private functions from knockoff filter module."""

    def test_prepare_knockoff_cache_memory_optimization(self):
        """Test knockoff cache preparation with memory constraints."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # Test with very small cache size
        with patch('src.loveslide.knockoff.filter.get_memory_usage') as mock_mem:
            mock_mem.return_value = 0.9  # 90% memory usage

            cache = _prepare_knockoff_cache(
                X, y, statistic=lambda x, y: np.random.randn(x.shape[1]),
                fdr=0.1, max_size=10
            )
            assert len(cache['knockoffs']) <= 10

    def test_cached_iteration_consistency(self):
        """Test cached iteration consistency across runs."""
        np.random.seed(42)
        X = np.random.randn(20, 10)
        y = np.random.binomial(1, 0.5, 20)

        cache = {
            'knockoffs': [np.random.randn(20, 10) for _ in range(5)],
            'statistics': [np.random.randn(10) for _ in range(5)]
        }

        def dummy_statistic(x, y):
            return np.random.randn(x.shape[1])

        # Run multiple times with same seed
        results1 = _cached_iteration(X, y, cache, dummy_statistic, 0.1, 1, 42)
        results2 = _cached_iteration(X, y, cache, dummy_statistic, 0.1, 1, 42)

        assert np.allclose(results1['W'], results2['W'])
        assert results1['selected'] == results2['selected']


if __name__ == "__main__":
    pytest.main([__file__])