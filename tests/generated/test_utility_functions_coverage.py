"""
Test coverage for utility functions in knockoff modules.
Addresses: Mathematical utilities, matrix operations, validation functions
"""
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import warnings

from loveslide.knockoff.utils import (
    diag_pre_multiply, diag_post_multiply, is_posdef, canonical_svd,
    normc, cov2cor, rnorm_matrix, random_problem, with_seed
)


class TestMatrixUtilities:
    """Test mathematical utility functions."""

    def test_diag_pre_multiply_basic(self):
        """Test diagonal pre-multiplication."""
        d = np.array([1, 2, 3])
        X = np.array([[1, 2], [3, 4], [5, 6]])

        result = diag_pre_multiply(d, X)
        expected = np.diag(d) @ X
        assert_array_almost_equal(result, expected)

    def test_diag_pre_multiply_edge_cases(self):
        """Test diagonal pre-multiplication edge cases."""
        # Zero diagonal
        d = np.zeros(3)
        X = np.random.randn(3, 2)
        result = diag_pre_multiply(d, X)
        assert_array_equal(result, np.zeros_like(X))

        # Single element
        d = np.array([5.0])
        X = np.array([[2]])
        result = diag_pre_multiply(d, X)
        assert result == 10.0

    def test_diag_post_multiply_basic(self):
        """Test diagonal post-multiplication."""
        X = np.array([[1, 2], [3, 4]])
        d = np.array([1, 2])

        result = diag_post_multiply(X, d)
        expected = X @ np.diag(d)
        assert_array_almost_equal(result, expected)

    def test_diag_multiply_dimension_mismatch(self):
        """Test diagonal multiplication with mismatched dimensions."""
        d = np.array([1, 2])
        X = np.array([[1, 2, 3], [4, 5, 6]])

        with pytest.raises((ValueError, IndexError)):
            diag_pre_multiply(d, X)

    def test_is_posdef_positive_definite(self):
        """Test positive definiteness detection."""
        # Positive definite matrix
        A = np.array([[2, 1], [1, 2]])
        assert is_posdef(A) == True

        # Identity matrix
        I = np.eye(5)
        assert is_posdef(I) == True

    def test_is_posdef_negative_cases(self):
        """Test non-positive definite matrices."""
        # Singular matrix
        A = np.array([[1, 1], [1, 1]])
        assert is_posdef(A) == False

        # Negative definite
        A = np.array([[-2, 0], [0, -1]])
        assert is_posdef(A) == False

        # Indefinite
        A = np.array([[1, 0], [0, -1]])
        assert is_posdef(A) == False

    def test_is_posdef_tolerance(self):
        """Test positive definiteness with different tolerance levels."""
        # Nearly singular matrix
        A = np.array([[1, 0.999999], [0.999999, 1]])

        assert is_posdef(A, tol=1e-3) == False
        assert is_posdef(A, tol=1e-8) == True

    def test_canonical_svd_basic(self):
        """Test canonical SVD decomposition."""
        X = np.random.randn(10, 5)
        U, s, Vt = canonical_svd(X)

        # Verify decomposition
        reconstructed = U @ np.diag(s) @ Vt
        assert_array_almost_equal(X, reconstructed)

        # Verify orthogonality
        assert_array_almost_equal(U.T @ U, np.eye(U.shape[1]), decimal=10)
        assert_array_almost_equal(Vt @ Vt.T, np.eye(Vt.shape[0]), decimal=10)

    def test_canonical_svd_edge_cases(self):
        """Test SVD with edge case matrices."""
        # Rank-deficient matrix
        X = np.array([[1, 2], [2, 4], [3, 6]])
        U, s, Vt = canonical_svd(X)
        assert len(s) == np.linalg.matrix_rank(X)

        # Single row/column
        X = np.array([[1, 2, 3]])
        U, s, Vt = canonical_svd(X)
        assert U.shape[1] == 1

    def test_normc_centering(self):
        """Test column normalization with centering."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

        result = normc(X, center=True)

        # Check centering
        assert_array_almost_equal(np.mean(result, axis=0), 0, decimal=10)

        # Check unit variance
        assert_array_almost_equal(np.var(result, axis=0, ddof=0), 1, decimal=10)

    def test_normc_no_centering(self):
        """Test column normalization without centering."""
        X = np.array([[2, 4], [4, 8]], dtype=float)

        result = normc(X, center=False)

        # Should not be centered at zero
        assert not np.allclose(np.mean(result, axis=0), 0)

    def test_normc_constant_column(self):
        """Test normalization with constant columns."""
        X = np.array([[1, 5], [1, 5], [1, 5]], dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = normc(X)

        # Constant column should become zero after centering
        assert_array_almost_equal(result[:, 0], 0)

    def test_cov2cor_basic(self):
        """Test covariance to correlation conversion."""
        # Known covariance matrix
        Sigma = np.array([[4, 2], [2, 9]])

        result = cov2cor(Sigma)

        # Check diagonal is ones
        assert_array_almost_equal(np.diag(result), 1)

        # Check symmetry
        assert_array_almost_equal(result, result.T)

        # Check correlation values
        expected_corr = 2 / (2 * 3)  # cov(X,Y) / (sd(X) * sd(Y))
        assert_array_almost_equal(result[0, 1], expected_corr)

    def test_cov2cor_edge_cases(self):
        """Test correlation conversion edge cases."""
        # Zero variance
        Sigma = np.array([[0, 0], [0, 1]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cov2cor(Sigma)

        # Should handle division by zero gracefully
        assert np.isfinite(result[1, 1])
        assert result[1, 1] == 1

    def test_rnorm_matrix_shape_and_properties(self):
        """Test random normal matrix generation."""
        n, p = 50, 10
        mean, sd = 2.0, 1.5

        result = rnorm_matrix(n, p, mean=mean, sd=sd)

        # Check shape
        assert result.shape == (n, p)

        # Check approximate mean and std (with tolerance for randomness)
        assert abs(np.mean(result) - mean) < 0.5
        assert abs(np.std(result, ddof=1) - sd) < 0.5

    def test_random_problem_generation(self):
        """Test synthetic problem generation."""
        n, p, k = 100, 20, 5

        result = random_problem(n=n, p=p, k=k, sparsity=0.1, snr=3.0)

        # Check returned structure
        assert 'X' in result
        assert 'y' in result
        assert 'beta' in result

        # Check dimensions
        assert result['X'].shape == (n, p)
        assert len(result['y']) == n
        assert len(result['beta']) == p

        # Check sparsity
        nonzero_count = np.sum(result['beta'] != 0)
        expected_nonzero = int(p * 0.1)
        assert abs(nonzero_count - expected_nonzero) <= 2  # Allow some tolerance

    def test_with_seed_reproducibility(self):
        """Test seeded function execution."""
        def random_func():
            return np.random.randn(10)

        # Same seed should produce same results
        result1 = with_seed(42, random_func)
        result2 = with_seed(42, random_func)
        assert_array_equal(result1, result2)

        # Different seeds should produce different results
        result3 = with_seed(123, random_func)
        assert not np.array_equal(result1, result3)

    def test_with_seed_state_restoration(self):
        """Test that random state is properly restored."""
        np.random.seed(999)
        state_before = np.random.get_state()

        # Execute seeded function
        with_seed(42, lambda: np.random.randn(5))

        # State after should be different but predictable
        value_after = np.random.randn()

        # Reset and check we get same value
        np.random.set_state(state_before)
        value_reset = np.random.randn()

        assert value_after == value_reset


class TestUtilityFunctionIntegration:
    """Test integration between utility functions."""

    def test_posdef_svd_consistency(self):
        """Test that positive definite matrices have consistent SVD properties."""
        # Generate positive definite matrix
        A = np.random.randn(5, 5)
        Sigma = A @ A.T + 0.1 * np.eye(5)

        assert is_posdef(Sigma)

        U, s, Vt = canonical_svd(Sigma)
        assert np.all(s > 0)  # All singular values should be positive

    def test_normalization_correlation_chain(self):
        """Test chain of normalization and correlation operations."""
        X = random_problem(n=100, p=10)['X']

        # Normalize then compute correlation
        X_norm = normc(X)
        cor_matrix = np.corrcoef(X_norm.T)
        cor_from_cov = cov2cor(np.cov(X_norm.T))

        assert_array_almost_equal(cor_matrix, cor_from_cov, decimal=10)

    def test_diagonal_operations_inverse_property(self):
        """Test that diagonal operations are invertible."""
        X = np.random.randn(5, 3)
        d = np.random.randn(5) + 1  # Ensure non-zero

        # Apply then invert
        X_mult = diag_pre_multiply(d, X)
        X_recovered = diag_pre_multiply(1/d, X_mult)

        assert_array_almost_equal(X, X_recovered)