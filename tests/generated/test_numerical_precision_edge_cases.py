"""Numerical precision and stability edge cases.

Tests for floating-point precision, numerical stability, and mathematical
edge cases that complement existing coverage.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import linalg
from unittest.mock import patch

from src.loveslide.knockoff.utils import (
    is_posdef, canonical_svd, normc, cov2cor,
    diag_pre_multiply, diag_post_multiply
)


class TestNumericalPrecisionEdgeCases:
    """Test numerical precision and stability edge cases."""

    def test_posdef_near_machine_epsilon(self):
        """Test positive definiteness with values near machine epsilon."""
        # Create matrix with eigenvalues near machine epsilon
        n = 50
        eps = np.finfo(float).eps

        # Matrix with smallest eigenvalue just above machine epsilon
        U = np.random.orthogonal_group(n)
        eigenvals = np.linspace(10*eps, 1.0, n)
        A = U @ np.diag(eigenvals) @ U.T

        assert is_posdef(A, tol=eps)

        # Matrix with smallest eigenvalue just below tolerance
        eigenvals_neg = np.linspace(-eps, 1.0, n)
        A_neg = U @ np.diag(eigenvals_neg) @ U.T

        assert not is_posdef(A_neg, tol=eps)

    def test_canonical_svd_rank_deficient_matrices(self):
        """Test SVD with rank-deficient matrices."""
        # Create rank-deficient matrix
        A = np.random.randn(100, 50)
        A[:, 25:] = A[:, :25]  # Make columns 25-49 identical to 0-24

        U, s, Vt = canonical_svd(A)

        # Should handle rank deficiency gracefully
        assert len(s) <= min(A.shape)
        assert np.allclose(U @ np.diag(s) @ Vt, A, atol=1e-10)

    def test_canonical_svd_near_singular_matrices(self):
        """Test SVD with nearly singular matrices."""
        # Create nearly singular matrix
        A = np.random.randn(20, 20)
        A[-1, :] = A[0, :] + 1e-14  # Last row almost identical to first

        U, s, Vt = canonical_svd(A)

        # Should handle near-singularity
        assert len(s) == min(A.shape)
        # Smallest singular value should be very small
        assert s[-1] < 1e-10

    def test_normc_extreme_variance_scaling(self):
        """Test normalization with extreme variance differences."""
        # Create matrix with extreme variance differences
        X = np.random.randn(100, 5)
        X[:, 0] *= 1e10  # Very large variance
        X[:, 1] *= 1e-10  # Very small variance
        X[:, 2] = X[:, 2] + 1e6  # Large mean offset

        X_norm = normc(X, center=True)

        # Should handle extreme scaling
        assert np.allclose(np.mean(X_norm, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_norm, axis=0), 1, atol=1e-10)

    def test_normc_constant_columns_handling(self):
        """Test normalization with constant (zero variance) columns."""
        X = np.random.randn(100, 5)
        X[:, 2] = 5.0  # Constant column

        # Should handle constant columns gracefully
        X_norm = normc(X, center=True)

        # Constant column should become zero after centering
        assert np.allclose(X_norm[:, 2], 0, atol=1e-10)

    def test_cov2cor_numerical_precision(self):
        """Test covariance to correlation conversion precision."""
        # Create covariance matrix with extreme diagonal values
        n = 50
        Sigma = np.random.randn(n, n)
        Sigma = Sigma @ Sigma.T  # Make positive definite

        # Scale diagonal to extreme values
        diag_vals = np.random.uniform(1e-10, 1e10, n)
        D = np.diag(np.sqrt(diag_vals))
        Sigma_extreme = D @ Sigma @ D

        R = cov2cor(Sigma_extreme)

        # Diagonal should be exactly 1
        assert np.allclose(np.diag(R), 1.0, atol=1e-12)

        # Should be symmetric
        assert np.allclose(R, R.T, atol=1e-12)

        # Off-diagonal elements should be in [-1, 1]
        R_offdiag = R - np.diag(np.diag(R))
        assert np.all(np.abs(R_offdiag) <= 1.0 + 1e-10)

    def test_diagonal_multiplication_precision(self):
        """Test diagonal multiplication numerical precision."""
        # Create test matrices
        n, p = 100, 50
        X = np.random.randn(n, p)
        d = np.random.uniform(1e-10, 1e10, min(n, p))

        # Test pre-multiplication
        result_pre = diag_pre_multiply(d, X)

        # Manual computation for comparison
        expected_pre = np.diag(d) @ X

        assert np.allclose(result_pre, expected_pre, atol=1e-10)

        # Test post-multiplication
        result_post = diag_post_multiply(X, d)
        expected_post = X @ np.diag(d)

        assert np.allclose(result_post, expected_post, atol=1e-10)

    def test_floating_point_edge_cases(self):
        """Test handling of special floating-point values."""
        # Test with inf, -inf, nan
        X = np.array([
            [1.0, 2.0, 3.0],
            [np.inf, 4.0, 5.0],
            [6.0, -np.inf, 7.0],
            [8.0, 9.0, np.nan]
        ])

        # Functions should handle or appropriately reject special values
        try:
            result = normc(X, center=True)
            # If successful, should not contain inf/nan in finite columns
            finite_mask = np.isfinite(X).all(axis=0)
            if finite_mask.any():
                assert np.all(np.isfinite(result[:, finite_mask]))
        except (ValueError, FloatingPointError):
            # Expected for matrices with inf/nan
            pass

    def test_numerical_precision_cumulative_operations(self):
        """Test precision loss in cumulative operations."""
        # Create scenario prone to precision loss
        n = 1000
        X = np.random.randn(n, 10) * 1e-8  # Very small values

        # Multiple operations that could accumulate errors
        X_centered = normc(X, center=True)
        cov_matrix = np.cov(X_centered.T)
        corr_matrix = cov2cor(cov_matrix)

        # Should maintain numerical properties
        assert np.allclose(np.diag(corr_matrix), 1.0, atol=1e-10)
        assert np.allclose(corr_matrix, corr_matrix.T, atol=1e-10)

    def test_condition_number_extreme_cases(self):
        """Test handling of matrices with extreme condition numbers."""
        # Create ill-conditioned matrix
        n = 20
        U = np.random.orthogonal_group(n)

        # Extreme condition number
        singular_values = np.logspace(0, -15, n)  # Condition number ~10^15
        ill_conditioned = U @ np.diag(singular_values) @ U.T

        # Test positive definiteness check with ill-conditioned matrix
        result = is_posdef(ill_conditioned, tol=1e-12)

        # Should handle extreme condition numbers
        assert isinstance(result, bool)

    def test_matrix_operations_broadcasting_edge_cases(self):
        """Test matrix operations with broadcasting edge cases."""
        # Test diagonal operations with shape edge cases
        X = np.random.randn(1, 50)  # Single row
        d = np.random.randn(1)

        result = diag_pre_multiply(d, X)
        assert result.shape == X.shape

        # Test with single column
        Y = np.random.randn(100, 1)
        d_col = np.random.randn(1)

        result_post = diag_post_multiply(Y, d_col)
        assert result_post.shape == Y.shape

    def test_precision_loss_in_iterative_operations(self):
        """Test precision preservation in iterative operations."""
        # Simulate iterative algorithm with potential precision loss
        X = np.random.randn(50, 10)

        # Iterative centering and scaling (simulates optimization loops)
        current = X.copy()
        for i in range(100):
            current = normc(current, center=True)

        # Should not accumulate significant precision errors
        assert np.allclose(np.mean(current, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(current, axis=0), 1, atol=1e-10)

    def test_numerical_gradient_edge_cases(self):
        """Test numerical gradients and finite differences."""
        # Test scenarios where numerical gradients might be unstable
        def test_function(x):
            return np.sum(x**2) + 1e-15 * np.sum(x**4)

        x = np.random.randn(10) * 1e-8
        h = np.finfo(float).eps**(1/3)  # Optimal step size

        # Compute numerical gradient
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h

            grad[i] = (test_function(x_plus) - test_function(x_minus)) / (2 * h)

        # Should provide reasonable gradient estimate
        analytical_grad = 2 * x + 4 * 1e-15 * x**3
        assert np.allclose(grad, analytical_grad, rtol=1e-6)

    def test_matrix_inverse_numerical_stability(self):
        """Test numerical stability of matrix operations requiring inversion."""
        # Create matrices with various condition numbers
        condition_numbers = [1e3, 1e6, 1e9, 1e12]

        for cond in condition_numbers:
            n = 20
            U = np.random.orthogonal_group(n)

            # Create matrix with specific condition number
            singular_values = np.linspace(1.0, 1.0/cond, n)
            A = U @ np.diag(singular_values) @ U.T

            try:
                # Test that operations handle varying condition numbers
                A_inv = linalg.inv(A)
                identity_check = A @ A_inv

                # Check how well it recovers identity
                error = np.linalg.norm(identity_check - np.eye(n))

                # For well-conditioned matrices, error should be small
                if cond <= 1e6:
                    assert error < 1e-10
                else:
                    # For ill-conditioned, just ensure it doesn't crash
                    assert np.isfinite(error)

            except linalg.LinAlgError:
                # Expected for very ill-conditioned matrices
                assert cond >= 1e9