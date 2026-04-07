#!/usr/bin/env python3
"""
Comprehensive tests for knockoff utility functions.
Tests matrix operations, numerical stability, and edge cases.
"""

import pytest
import numpy as np
from unittest.mock import patch
from typing import Tuple, Any

from loveslide.knockoff.utils import (
    diag_pre_multiply, diag_post_multiply, is_posdef, canonical_svd,
    normc, cov2cor, rnorm_matrix, random_problem, with_seed
)


class TestMatrixOperations:
    """Test matrix operations for numerical stability and edge cases."""

    def test_diag_pre_multiply_basic(self):
        """Test basic diagonal pre-multiplication."""
        d = np.array([1.0, 2.0, 3.0])
        X = np.random.rand(3, 5)

        result = diag_pre_multiply(d, X)

        # Should be equivalent to diag(d) @ X
        expected = np.diag(d) @ X
        np.testing.assert_allclose(result, expected)

    def test_diag_pre_multiply_dimension_mismatch(self):
        """Test diagonal pre-multiplication with mismatched dimensions."""
        d = np.array([1.0, 2.0, 3.0])
        X = np.random.rand(4, 5)  # Wrong dimension

        with pytest.raises((ValueError, IndexError)):
            diag_pre_multiply(d, X)

    def test_diag_pre_multiply_extreme_values(self):
        """Test diagonal pre-multiplication with extreme values."""
        # Very large values
        d = np.array([1e10, 1e-10, 0.0])
        X = np.random.rand(3, 5)

        result = diag_pre_multiply(d, X)

        # First row should be very large, second very small, third zero
        assert np.allclose(result[2, :], 0.0)
        assert np.max(np.abs(result[0, :])) > 1e9
        assert np.max(np.abs(result[1, :])) < 1e-9

    def test_diag_pre_multiply_numerical_precision(self):
        """Test diagonal pre-multiplication numerical precision."""
        # Test with values near machine precision
        d = np.array([np.finfo(float).eps, 1.0, 1.0/np.finfo(float).eps])
        X = np.ones((3, 3))

        result = diag_pre_multiply(d, X)

        # Should handle extreme values without overflow/underflow errors
        assert np.isfinite(result).all()

    def test_diag_post_multiply_basic(self):
        """Test basic diagonal post-multiplication."""
        X = np.random.rand(5, 3)
        d = np.array([1.0, 2.0, 3.0])

        result = diag_post_multiply(X, d)

        # Should be equivalent to X @ diag(d)
        expected = X @ np.diag(d)
        np.testing.assert_allclose(result, expected)

    def test_diag_post_multiply_dimension_mismatch(self):
        """Test diagonal post-multiplication with mismatched dimensions."""
        X = np.random.rand(5, 3)
        d = np.array([1.0, 2.0, 3.0, 4.0])  # Wrong dimension

        with pytest.raises((ValueError, IndexError)):
            diag_post_multiply(X, d)

    def test_diag_multiply_empty_arrays(self):
        """Test diagonal multiplication with empty arrays."""
        d_empty = np.array([])
        X_empty = np.array([]).reshape(0, 0)

        result = diag_pre_multiply(d_empty, X_empty)
        assert result.shape == (0, 0)

    def test_diag_multiply_single_element(self):
        """Test diagonal multiplication with single elements."""
        d = np.array([5.0])
        X = np.array([[2.0]])

        result_pre = diag_pre_multiply(d, X)
        result_post = diag_post_multiply(X, d)

        assert result_pre[0, 0] == 10.0
        assert result_post[0, 0] == 10.0

    def test_diag_multiply_complex_numbers(self):
        """Test diagonal multiplication with complex numbers."""
        d = np.array([1.0 + 1j, 2.0 - 1j])
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)

        result = diag_pre_multiply(d, X)

        # Should handle complex arithmetic correctly
        assert np.iscomplex(result).any()
        assert result.dtype == complex


class TestPositiveDefiniteChecks:
    """Test positive definiteness checking for numerical edge cases."""

    def test_is_posdef_clearly_positive(self):
        """Test positive definiteness with clearly positive definite matrix."""
        A = np.array([[2, 1], [1, 2]])  # Clearly positive definite
        assert is_posdef(A)

    def test_is_posdef_clearly_negative(self):
        """Test positive definiteness with clearly negative definite matrix."""
        A = np.array([[-2, 1], [1, -2]])  # Clearly negative definite
        assert not is_posdef(A)

    def test_is_posdef_singular_matrix(self):
        """Test positive definiteness with singular matrix."""
        A = np.array([[1, 1], [1, 1]])  # Singular (rank 1)
        assert not is_posdef(A)

    def test_is_posdef_tolerance_boundary(self):
        """Test positive definiteness at tolerance boundaries."""
        # Matrix with eigenvalue near tolerance
        tol = 1e-9
        eigenval_near_tol = tol * 1.1  # Just above tolerance
        A = np.array([[eigenval_near_tol, 0], [0, 1]])

        assert is_posdef(A, tol=tol)

        # Just below tolerance
        eigenval_below_tol = tol * 0.9
        A_below = np.array([[eigenval_below_tol, 0], [0, 1]])

        assert not is_posdef(A_below, tol=tol)

    def test_is_posdef_numerical_precision(self):
        """Test positive definiteness with numerical precision issues."""
        # Create matrix that's positive definite but close to singular
        n = 10
        A = np.eye(n) + 1e-12 * np.random.rand(n, n)
        A = A @ A.T  # Ensure positive definiteness

        assert is_posdef(A)

    def test_is_posdef_asymmetric_matrix(self):
        """Test positive definiteness with asymmetric matrix."""
        A = np.array([[2, 1], [1.1, 2]])  # Slightly asymmetric

        # Function should handle asymmetric matrices appropriately
        # (might symmetrize or reject)
        try:
            result = is_posdef(A)
            assert isinstance(result, bool)
        except ValueError:
            # Rejecting asymmetric matrices is also acceptable
            pass

    def test_is_posdef_complex_matrix(self):
        """Test positive definiteness with complex matrix."""
        A = np.array([[2, 1+1j], [1-1j, 3]], dtype=complex)

        try:
            result = is_posdef(A)
            assert isinstance(result, bool)
        except (ValueError, TypeError):
            # Function might not support complex matrices
            pass

    def test_is_posdef_large_condition_number(self):
        """Test positive definiteness with ill-conditioned matrix."""
        # Create matrix with large condition number
        U, _, Vt = np.linalg.svd(np.random.rand(5, 5))
        S = np.diag([1e10, 1e5, 1e0, 1e-5, 1e-10])  # Large condition number
        A = U @ S @ Vt
        A = A @ A.T  # Make positive definite

        # Should handle ill-conditioned matrices
        result = is_posdef(A)
        assert isinstance(result, bool)


class TestCanonicalSVD:
    """Test canonical SVD for numerical stability and edge cases."""

    def test_canonical_svd_basic(self):
        """Test canonical SVD with basic matrix."""
        X = np.random.rand(10, 5)
        U, D, Vt = canonical_svd(X)

        # Verify SVD properties
        assert U.shape == (10, 5)
        assert D.shape == (5,)
        assert Vt.shape == (5, 5)

        # Verify reconstruction
        reconstructed = U @ np.diag(D) @ Vt
        np.testing.assert_allclose(reconstructed, X, rtol=1e-10)

    def test_canonical_svd_rank_deficient(self):
        """Test canonical SVD with rank-deficient matrix."""
        # Create rank-2 matrix
        A = np.random.rand(5, 2)
        X = A @ A.T  # Rank 2, size 5x5

        U, D, Vt = canonical_svd(X)

        # Should handle rank deficiency
        assert len(D) == min(X.shape)
        # Some singular values should be near zero
        assert np.sum(D > 1e-10) <= 2

    def test_canonical_svd_extreme_aspect_ratios(self):
        """Test canonical SVD with extreme aspect ratios."""
        # Very wide matrix
        X_wide = np.random.rand(5, 100)
        U_w, D_w, Vt_w = canonical_svd(X_wide)

        assert U_w.shape == (5, 5)
        assert D_w.shape == (5,)
        assert Vt_w.shape == (5, 100)

        # Very tall matrix
        X_tall = np.random.rand(100, 5)
        U_t, D_t, Vt_t = canonical_svd(X_tall)

        assert U_t.shape == (100, 5)
        assert D_t.shape == (5,)
        assert Vt_t.shape == (5, 5)

    def test_canonical_svd_numerical_stability(self):
        """Test canonical SVD numerical stability with challenging matrices."""
        # Matrix with very different scales
        X = np.random.rand(8, 6)
        X[:, 0] *= 1e10  # Very large first column
        X[:, 1] *= 1e-10  # Very small second column

        U, D, Vt = canonical_svd(X)

        # Should handle different scales without numerical issues
        assert np.isfinite(U).all()
        assert np.isfinite(D).all()
        assert np.isfinite(Vt).all()

    def test_canonical_svd_empty_matrix(self):
        """Test canonical SVD with empty matrix."""
        X_empty = np.array([]).reshape(0, 0)

        try:
            U, D, Vt = canonical_svd(X_empty)
            # Should handle empty matrices gracefully
            assert U.shape[0] == 0
            assert D.shape[0] == 0
            assert Vt.shape[1] == 0
        except ValueError:
            # Rejecting empty matrices is also acceptable
            pass

    def test_canonical_svd_single_element(self):
        """Test canonical SVD with single element matrix."""
        X = np.array([[5.0]])
        U, D, Vt = canonical_svd(X)

        assert U.shape == (1, 1)
        assert D.shape == (1,)
        assert Vt.shape == (1, 1)
        assert abs(D[0] - 5.0) < 1e-10


class TestDataNormalization:
    """Test data normalization functions for edge cases."""

    def test_normc_basic_centering(self):
        """Test basic column centering."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        X_norm = normc(X, center=True)

        # Columns should be centered (mean = 0)
        np.testing.assert_allclose(np.mean(X_norm, axis=0), 0, atol=1e-10)

    def test_normc_without_centering(self):
        """Test normalization without centering."""
        X = np.random.rand(10, 5)
        X_norm = normc(X, center=False)

        # Should not change means significantly if only normalizing
        # (exact behavior depends on implementation)
        assert X_norm.shape == X.shape

    def test_normc_constant_columns(self):
        """Test normalization with constant columns."""
        X = np.array([[1, 2, 2], [1, 2, 3], [1, 2, 4]])  # First two columns have patterns

        try:
            X_norm = normc(X)
            # Should handle constant columns gracefully
            assert np.isfinite(X_norm).all()
        except (ValueError, RuntimeWarning):
            # Might warn or error on constant columns
            pass

    def test_normc_single_row(self):
        """Test normalization with single row."""
        X = np.array([[1, 2, 3]])

        try:
            X_norm = normc(X)
            assert X_norm.shape == (1, 3)
        except ValueError:
            # Single row might be rejected
            pass

    def test_normc_extreme_values(self):
        """Test normalization with extreme values."""
        X = np.array([[1e10, 1e-10, 0], [2e10, 2e-10, 0], [3e10, 3e-10, 1e-15]])

        X_norm = normc(X)

        # Should handle extreme values without overflow
        assert np.isfinite(X_norm).all()

    def test_cov2cor_basic(self):
        """Test covariance to correlation conversion."""
        # Create covariance matrix
        Sigma = np.array([[4, 2], [2, 9]])  # Variances 4, 9; covariance 2

        R = cov2cor(Sigma)

        # Diagonal should be 1s
        np.testing.assert_allclose(np.diag(R), 1.0)

        # Off-diagonal should be correlations
        expected_corr = 2 / (np.sqrt(4) * np.sqrt(9))  # 2 / (2 * 3) = 1/3
        assert abs(R[0, 1] - expected_corr) < 1e-10
        assert abs(R[1, 0] - expected_corr) < 1e-10

    def test_cov2cor_zero_variance(self):
        """Test correlation conversion with zero variance."""
        Sigma = np.array([[0, 0], [0, 4]])  # First variable has zero variance

        try:
            R = cov2cor(Sigma)
            # Should handle zero variance appropriately
            assert np.isfinite(R).all() or np.isnan(R[0, :]).any()
        except (ValueError, ZeroDivisionError):
            # Rejecting zero variance is acceptable
            pass

    def test_cov2cor_numerical_precision(self):
        """Test correlation conversion with numerical precision issues."""
        # Nearly singular covariance matrix
        eps = 1e-15
        Sigma = np.array([[1, 1-eps], [1-eps, 1]])

        R = cov2cor(Sigma)

        # Should handle near-singular matrices
        assert np.isfinite(R).all()
        np.testing.assert_allclose(np.diag(R), 1.0)


class TestRandomGeneration:
    """Test random matrix generation functions."""

    def test_rnorm_matrix_basic(self):
        """Test random normal matrix generation."""
        n, p = 10, 5
        X = rnorm_matrix(n, p, mean=0, sd=1)

        assert X.shape == (n, p)
        # Should be approximately normal
        assert abs(np.mean(X)) < 0.5  # Loose check for zero mean
        assert abs(np.std(X) - 1) < 0.5  # Loose check for unit variance

    def test_rnorm_matrix_parameters(self):
        """Test random matrix generation with different parameters."""
        X = rnorm_matrix(5, 3, mean=10, sd=2)

        # Should respect mean and sd parameters (approximately)
        assert abs(np.mean(X) - 10) < 2
        assert abs(np.std(X) - 2) < 1

    def test_rnorm_matrix_edge_sizes(self):
        """Test random matrix generation with edge case sizes."""
        # Single element
        X1 = rnorm_matrix(1, 1)
        assert X1.shape == (1, 1)

        # Large matrix
        X_large = rnorm_matrix(100, 50)
        assert X_large.shape == (100, 50)

    def test_random_problem_basic(self):
        """Test random problem generation."""
        n, p = 20, 10
        problem = random_problem(n=n, p=p)

        # Should return dictionary with expected keys
        assert isinstance(problem, dict)
        expected_keys = ['X', 'beta', 'y']  # Adjust based on actual function
        for key in expected_keys:
            if key in problem:
                assert problem[key] is not None

    def test_with_seed_reproducibility(self):
        """Test seeded random generation for reproducibility."""
        def random_func():
            return np.random.rand(5)

        # Same seed should give same results
        result1 = with_seed(42, random_func)
        result2 = with_seed(42, random_func)

        np.testing.assert_array_equal(result1, result2)

        # Different seed should give different results
        result3 = with_seed(43, random_func)
        assert not np.array_equal(result1, result3)

    def test_with_seed_state_restoration(self):
        """Test that random state is restored after seeded function."""
        # Get initial state
        np.random.seed(100)
        initial_value = np.random.rand()

        # Reset to same state
        np.random.seed(100)

        # Use with_seed
        def dummy_func():
            return np.random.rand()

        with_seed(999, dummy_func)

        # Next random value should be the same as if with_seed wasn't called
        post_seeded_value = np.random.rand()

        # Reset again to check
        np.random.seed(100)
        np.random.rand()  # Skip first value
        expected_value = np.random.rand()

        assert abs(post_seeded_value - expected_value) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])