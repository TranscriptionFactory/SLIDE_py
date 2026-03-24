"""
Test coverage for private utility functions edge cases.
Focus: Boundary conditions and numerical edge cases in utility functions.
"""

import pytest
import numpy as np
import warnings
from scipy import linalg
from unittest.mock import patch

from loveslide.knockoff.utils import (
    diag_pre_multiply, diag_post_multiply, is_posdef,
    canonical_svd, normc, cov2cor
)


class TestDiagonalOperations:
    """Test diagonal matrix operations edge cases."""

    def test_diag_pre_multiply_empty_matrix(self):
        """Test diagonal pre-multiplication with empty arrays."""
        d = np.array([])
        X = np.empty((0, 5))
        result = diag_pre_multiply(d, X)
        assert result.shape == (0, 5)

    def test_diag_pre_multiply_broadcasting_edge(self):
        """Test broadcasting behavior with edge case dimensions."""
        d = np.array([1e-15, 1e15, 0.0])  # Extreme values
        X = np.ones((3, 1))
        result = diag_pre_multiply(d, X)
        expected = np.array([[1e-15], [1e15], [0.0]])
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_diag_post_multiply_inf_values(self):
        """Test post-multiplication with infinite values."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        d = np.array([np.inf, -np.inf])
        result = diag_post_multiply(X, d)
        assert np.isinf(result[0, 0])
        assert np.isneginf(result[0, 1])

    def test_diag_operations_memory_efficiency(self):
        """Test memory efficiency with large matrices."""
        # Large matrix to test memory handling
        n = 10000
        d = np.random.randn(n)
        X = np.random.randn(n, 100)

        # Should not raise memory errors
        result1 = diag_pre_multiply(d, X)
        result2 = diag_post_multiply(X.T, d)

        assert result1.shape == (n, 100)
        assert result2.shape == (100, n)


class TestPositiveDefiniteness:
    """Test positive-definiteness checking edge cases."""

    def test_is_posdef_singular_matrix(self):
        """Test with exactly singular matrix."""
        A = np.array([[1, 1], [1, 1]])  # rank 1, singular
        assert not is_posdef(A)

    def test_is_posdef_near_singular(self):
        """Test with nearly singular matrix."""
        A = np.array([[1, 1-1e-12], [1-1e-12, 1]])
        # Should handle numerical precision gracefully
        result = is_posdef(A, tol=1e-10)
        assert isinstance(result, bool)

    def test_is_posdef_tolerance_boundary(self):
        """Test tolerance boundary conditions."""
        # Matrix with smallest eigenvalue exactly at tolerance
        lambda_min = 1e-9
        A = np.diag([lambda_min, 1.0, 2.0])

        assert not is_posdef(A, tol=lambda_min)
        assert is_posdef(A, tol=lambda_min * 0.9)

    def test_is_posdef_large_matrix_fallback(self):
        """Test fallback to dense computation for large matrices."""
        # Mock scipy import failure to test fallback
        with patch('loveslide.knockoff.utils.eigsh', side_effect=ImportError):
            A = np.eye(600) + 0.1  # Large positive definite matrix
            result = is_posdef(A)
            assert result is True

    def test_is_posdef_numerical_precision_edge(self):
        """Test numerical precision at machine epsilon."""
        eps = np.finfo(float).eps
        A = np.diag([eps, 1.0, 1.0])  # Eigenvalue at machine precision

        result_strict = is_posdef(A, tol=eps * 10)
        result_loose = is_posdef(A, tol=eps * 0.1)

        assert not result_strict
        assert result_loose


class TestCanonicalSVD:
    """Test canonical SVD edge cases."""

    def test_canonical_svd_zero_matrix(self):
        """Test SVD of zero matrix."""
        X = np.zeros((5, 3))
        u, d, v = canonical_svd(X)

        assert u.shape == (5, 3)
        assert d.shape == (3,)
        assert v.shape == (3, 3)
        np.testing.assert_allclose(d, 0)

    def test_canonical_svd_rank_deficient(self):
        """Test SVD with rank-deficient matrix."""
        # Create rank-2 matrix
        X = np.outer([1, 2, 3], [1, 0]) + np.outer([0, 1, 1], [0, 1])
        u, d, v = canonical_svd(X)

        # Should handle rank deficiency gracefully
        assert len(d[d > 1e-12]) == 2  # Rank 2

    def test_canonical_svd_sign_consistency(self):
        """Test sign consistency across runs."""
        np.random.seed(42)
        X = np.random.randn(10, 5)

        u1, d1, v1 = canonical_svd(X)
        u2, d2, v2 = canonical_svd(X)

        # Should be deterministic
        np.testing.assert_allclose(u1, u2)
        np.testing.assert_allclose(v1, v2)

    def test_canonical_svd_extreme_aspect_ratios(self):
        """Test with extreme aspect ratios."""
        # Very wide matrix
        X_wide = np.random.randn(5, 100)
        u, d, v = canonical_svd(X_wide)
        assert u.shape == (5, 5)

        # Very tall matrix
        X_tall = np.random.randn(100, 5)
        u, d, v = canonical_svd(X_tall)
        assert u.shape == (100, 5)


class TestNormalizationFunctions:
    """Test matrix normalization edge cases."""

    def test_normc_constant_columns(self):
        """Test column normalization with constant columns."""
        X = np.array([[1, 5], [1, 5], [1, 5]])  # Constant columns

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = normc(X, center=True)

        # Should handle gracefully (may result in NaN or zeros)
        assert result.shape == X.shape

    def test_normc_single_observation(self):
        """Test with single observation."""
        X = np.array([[1.0, 2.0, 3.0]])  # Single row
        result = normc(X, center=True)

        # Should handle single observation case
        assert result.shape == (1, 3)

    def test_cov2cor_diagonal_covariance(self):
        """Test covariance to correlation with diagonal matrix."""
        Sigma = np.diag([1, 4, 9])  # Diagonal covariance
        result = cov2cor(Sigma)

        # Should be identity correlation matrix
        expected = np.eye(3)
        np.testing.assert_allclose(result, expected)

    def test_cov2cor_near_zero_variance(self):
        """Test with near-zero variance components."""
        Sigma = np.array([[1e-15, 0], [0, 1.0]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cov2cor(Sigma)

        # Should handle numerical issues gracefully
        assert result.shape == (2, 2)


class TestUtilityRobustness:
    """Test robustness of utility functions."""

    def test_utility_functions_with_complex_input(self):
        """Test utility functions reject complex input appropriately."""
        X_complex = np.array([[1+1j, 2], [3, 4+2j]])

        # Should handle complex numbers appropriately
        with pytest.raises((TypeError, ValueError)):
            canonical_svd(X_complex)

    def test_utility_functions_memory_stress(self):
        """Test behavior under memory stress conditions."""
        # Test with moderately large arrays
        n = 2000
        X = np.random.randn(n, n//2)

        # Should complete without memory errors
        result = normc(X)
        assert result.shape == X.shape

        # Test positive definiteness on large matrix
        A = X.T @ X + np.eye(n//2) * 0.1
        is_posdef_result = is_posdef(A)
        assert isinstance(is_posdef_result, bool)

    def test_numerical_stability_extreme_scales(self):
        """Test numerical stability with extreme scaling."""
        # Very large and very small numbers
        X = np.array([[1e-100, 1e100], [1e-50, 1e50]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u, d, v = canonical_svd(X)

        # Should not produce NaN or Inf
        assert np.all(np.isfinite(u))
        assert np.all(np.isfinite(v))
        # d might have extreme values but should be finite
        assert not np.any(np.isnan(d))


if __name__ == "__main__":
    pytest.main([__file__])