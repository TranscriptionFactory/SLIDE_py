"""
Test skeleton for mathematical utility functions edge cases.

Focus on testing boundary conditions and numerical stability in
the core mathematical utilities used by knockoff methods.
"""
import pytest
import numpy as np
from scipy import linalg
from unittest.mock import patch
import warnings

from loveslide.knockoff.utils import (
    diag_pre_multiply, diag_post_multiply, is_posdef, canonical_svd,
    normc, cov2cor, rnorm_matrix, random_problem, with_seed
)


class TestDiagonalOperations:
    """Test diagonal matrix operations for edge cases."""

    def test_diag_pre_multiply_zero_diagonal(self):
        """Test behavior with zero elements in diagonal."""
        d = np.array([0.0, 1.0, 0.0, 2.0])
        X = np.random.randn(4, 3)

        result = diag_pre_multiply(d, X)

        # Rows 0 and 2 should be zero
        assert np.allclose(result[0, :], 0.0)
        assert np.allclose(result[2, :], 0.0)
        assert not np.allclose(result[1, :], 0.0)
        assert not np.allclose(result[3, :], 0.0)

    def test_diag_pre_multiply_inf_values(self):
        """Test handling of infinite values in diagonal."""
        d = np.array([1.0, np.inf, 2.0])
        X = np.ones((3, 2))

        result = diag_pre_multiply(d, X)

        assert np.isfinite(result[0, :]).all()
        assert np.isinf(result[1, :]).all()
        assert np.isfinite(result[2, :]).all()

    def test_diag_post_multiply_broadcasting_edge_cases(self):
        """Test broadcasting edge cases in post multiplication."""
        X = np.random.randn(5, 1)  # Single column
        d = np.array([2.0])

        result = diag_post_multiply(X, d)

        assert result.shape == (5, 1)
        assert np.allclose(result, X * 2.0)

    def test_diag_operations_empty_arrays(self):
        """Test behavior with empty arrays."""
        d_empty = np.array([])
        X_empty = np.empty((0, 0))

        with pytest.raises((ValueError, IndexError)):
            diag_pre_multiply(d_empty, X_empty)


class TestPositiveDefiniteness:
    """Test positive definiteness checking edge cases."""

    def test_is_posdef_near_singular(self):
        """Test positive definiteness near singularity."""
        # Create nearly singular positive definite matrix
        A = np.eye(3)
        A[2, 2] = 1e-12  # Very small positive eigenvalue

        result_strict = is_posdef(A, tol=1e-10)
        result_loose = is_posdef(A, tol=1e-15)

        assert result_strict == False  # Should fail strict tolerance
        assert result_loose == True   # Should pass loose tolerance

    def test_is_posdef_large_matrix_fallback(self):
        """Test fallback for large matrices when sparse fails."""
        # Mock sparse eigsh to raise exception
        large_A = np.eye(600) + 0.1 * np.random.randn(600, 600)
        large_A = large_A @ large_A.T  # Ensure positive definite

        with patch('scipy.sparse.linalg.eigsh', side_effect=RuntimeError):
            result = is_posdef(large_A)
            assert isinstance(result, bool)

    def test_is_posdef_indefinite_matrix(self):
        """Test with indefinite matrix (mixed eigenvalues)."""
        # Create matrix with negative eigenvalue
        A = np.array([[1, 0], [0, -1]])

        result = is_posdef(A)
        assert result == False

    def test_is_posdef_non_symmetric_matrix(self):
        """Test behavior with non-symmetric matrix."""
        A = np.array([[1, 2], [3, 4]])  # Non-symmetric

        # Should still work but may give unexpected results
        result = is_posdef(A)
        assert isinstance(result, bool)


class TestCanonicalSVD:
    """Test canonical SVD implementation edge cases."""

    def test_canonical_svd_rank_deficient(self):
        """Test SVD with rank-deficient matrix."""
        # Create rank-1 matrix
        X = np.ones((5, 3))  # All rows identical

        U, d, V = canonical_svd(X)

        assert U.shape[1] == min(5, 3)  # Should return reduced SVD
        assert np.sum(d > 1e-10) == 1   # Only one significant singular value

    def test_canonical_svd_sign_convention(self):
        """Test canonical sign choice is applied correctly."""
        X = np.array([[-1, 0], [2, 0], [-0.5, 0]])  # Largest element is positive

        U, d, V = canonical_svd(X)

        # Largest absolute element in first column of U should be positive
        max_idx = np.argmax(np.abs(U[:, 0]))
        assert U[max_idx, 0] > 0

    def test_canonical_svd_singular_matrix(self):
        """Test SVD failure handling."""
        # Create problematic matrix
        X = np.full((3, 3), np.nan)

        with pytest.raises(RuntimeError, match="SVD failed"):
            canonical_svd(X)

    def test_canonical_svd_empty_matrix(self):
        """Test behavior with empty matrix."""
        X = np.empty((0, 0))

        U, d, V = canonical_svd(X)
        assert U.shape == (0, 0)
        assert d.shape == (0,)
        assert V.shape == (0, 0)


class TestNormalization:
    """Test column normalization edge cases."""

    def test_normc_constant_columns(self):
        """Test normalization with constant columns."""
        X = np.array([[1, 0], [1, 0], [1, 0]])  # Second column all zeros

        result = normc(X, center=True)

        # First column should be normalized, second should be zero
        assert np.allclose(np.linalg.norm(result[:, 0]), 1.0)
        assert np.allclose(result[:, 1], 0.0)

    def test_normc_single_row(self):
        """Test normalization with single row."""
        X = np.array([[1, 2, 3]])

        result = normc(X, center=False)

        assert result.shape == (1, 3)
        assert np.allclose(np.linalg.norm(result, axis=0), 1.0)

    def test_normc_all_zeros_column(self):
        """Test handling of all-zeros columns."""
        X = np.array([[1, 0], [2, 0], [3, 0]])

        result = normc(X, center=False)

        # Zero columns should remain normalized to avoid division by zero
        assert np.isfinite(result).all()

    def test_normc_centering_effect(self):
        """Test centering vs non-centering behavior."""
        X = np.array([[1, 4], [2, 5], [3, 6]])

        centered = normc(X, center=True)
        not_centered = normc(X, center=False)

        # Centering should change the result
        assert not np.allclose(centered, not_centered)

        # After centering, means should be near zero
        assert np.allclose(centered.mean(axis=0), 0.0, atol=1e-10)


class TestCovarianceToCorrelation:
    """Test covariance to correlation conversion edge cases."""

    def test_cov2cor_zero_variance_features(self):
        """Test conversion with zero variance features."""
        # Covariance matrix with zero diagonal element
        Sigma = np.array([[1, 0.5], [0.5, 0]])  # Second feature has zero variance

        result = cov2cor(Sigma)

        # Should handle division by zero gracefully
        assert np.isfinite(result).all()
        assert np.allclose(np.diag(result), 1.0)  # Unit diagonal enforced

    def test_cov2cor_symmetry_enforcement(self):
        """Test symmetry enforcement for numerical stability."""
        # Slightly asymmetric matrix
        Sigma = np.array([[1, 0.5], [0.500001, 2]])

        result = cov2cor(Sigma)

        # Should be perfectly symmetric
        assert np.allclose(result, result.T)
        assert np.allclose(np.diag(result), 1.0)

    def test_cov2cor_negative_correlations(self):
        """Test handling of negative correlations."""
        Sigma = np.array([[1, -0.9], [-0.9, 1]])

        result = cov2cor(Sigma)

        assert np.allclose(result[0, 1], -0.9)
        assert np.allclose(result[1, 0], -0.9)
        assert np.allclose(np.diag(result), 1.0)


class TestRandomGeneration:
    """Test random generation utility functions."""

    def test_random_problem_edge_cases(self):
        """Test random problem generation edge cases."""
        # Very small problem
        problem = random_problem(n=3, p=2, k=1, seed=42)

        assert problem['X'].shape == (3, 2)
        assert problem['y'].shape == (3,)
        assert problem['beta'].shape == (2,)
        assert len(problem['nonzero']) == 1

    def test_random_problem_k_larger_than_p(self):
        """Test when k > p (should be handled gracefully)."""
        with pytest.raises(ValueError):
            random_problem(n=10, p=3, k=5, seed=42)  # k=5 > p=3

    def test_with_seed_state_restoration(self):
        """Test that random state is properly restored."""
        # Set initial state
        np.random.seed(123)
        initial_state = np.random.get_state()

        def dummy_func():
            return np.random.randn(5)

        # Use with_seed
        result = with_seed(456, dummy_func)

        # Check state was restored
        current_state = np.random.get_state()
        assert np.array_equal(initial_state[1], current_state[1])

    def test_rnorm_matrix_parameter_validation(self):
        """Test parameter validation for random matrix generation."""
        # Test with negative dimensions
        with pytest.raises(ValueError):
            rnorm_matrix(-1, 5)

        with pytest.raises(ValueError):
            rnorm_matrix(5, -1)

    def test_rnorm_matrix_zero_dimensions(self):
        """Test random matrix generation with zero dimensions."""
        result = rnorm_matrix(0, 5)
        assert result.shape == (0, 5)

        result = rnorm_matrix(5, 0)
        assert result.shape == (5, 0)


class TestNumericalStability:
    """Test numerical stability across utility functions."""

    def test_operations_near_machine_epsilon(self):
        """Test behavior near machine epsilon."""
        eps = np.finfo(float).eps

        # Test matrix with values near machine epsilon
        X = np.full((3, 3), eps)

        # Should handle without overflow/underflow
        result = normc(X)
        assert np.isfinite(result).all()

    def test_overflow_prevention(self):
        """Test prevention of overflow in operations."""
        # Large values that could cause overflow
        large_val = np.sqrt(np.finfo(float).max) / 10
        X = np.full((2, 2), large_val)

        # Operations should remain finite
        result = normc(X)
        assert np.isfinite(result).all()

    def test_underflow_handling(self):
        """Test handling of underflow conditions."""
        tiny_val = np.finfo(float).tiny
        X = np.full((2, 2), tiny_val)

        result = normc(X)
        assert np.isfinite(result).all()


if __name__ == "__main__":
    pytest.main([__file__])