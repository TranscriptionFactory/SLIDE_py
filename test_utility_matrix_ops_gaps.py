"""
Test coverage for low-level utility function edge cases in SLIDE_py.

Critical gaps identified:
- Matrix utility functions boundary conditions
- Numerical precision in edge cases
- Memory efficiency in matrix operations
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from loveslide.knockoff.utils import (
    diag_pre_multiply, diag_post_multiply, is_posdef,
    canonical_svd, normc, cov2cor, rnorm_matrix
)


class TestDiagonalOperationsEdgeCases:
    """Test diagonal matrix operations with edge cases"""

    def test_diag_pre_multiply_zero_diagonal(self):
        """Test diag_pre_multiply with zero values in diagonal"""
        d = np.array([1.0, 0.0, 2.0])  # Zero in diagonal
        X = np.random.randn(3, 4)
        result = diag_pre_multiply(d, X)
        # Second row should be all zeros
        assert np.allclose(result[1, :], 0.0)

    def test_diag_pre_multiply_inf_values(self):
        """Test diagonal multiplication with infinite values"""
        d = np.array([1.0, np.inf, 2.0])
        X = np.ones((3, 4))

        with pytest.warns(RuntimeWarning):
            result = diag_pre_multiply(d, X)
            assert np.isinf(result[1, :]).all()

    def test_diag_post_multiply_dimension_mismatch(self):
        """Test dimension mismatch handling"""
        X = np.random.randn(3, 4)
        d = np.array([1.0, 2.0])  # Wrong dimension

        with pytest.raises(ValueError):
            diag_post_multiply(X, d)

    def test_diag_operations_memory_efficiency(self):
        """Test memory efficiency with large matrices"""
        n, p = 1000, 500
        X = np.random.randn(n, p)
        d = np.random.randn(n)

        # Should not create unnecessary copies
        import gc
        gc.collect()
        initial_objects = len(gc.get_objects())

        result = diag_pre_multiply(d, X)

        gc.collect()
        final_objects = len(gc.get_objects())

        # Should not create excessive intermediate objects
        assert final_objects - initial_objects < 10


class TestPositiveDefiniteChecksEdgeCases:
    """Test positive definiteness checking edge cases"""

    def test_is_posdef_near_singular(self):
        """Test with nearly singular matrices"""
        # Create nearly singular matrix
        A = np.array([[1e-12, 0], [0, 1.0]])

        # Default tolerance should catch this
        assert not is_posdef(A)

        # Very loose tolerance should pass
        assert is_posdef(A, tol=1e-15)

    def test_is_posdef_complex_eigenvalues(self):
        """Test with matrix having complex eigenvalues"""
        # Asymmetric matrix (should fail)
        A = np.array([[1.0, 2.0], [0.5, 1.0]])

        with pytest.warns(RuntimeWarning):
            result = is_posdef(A)
            assert not result

    def test_is_posdef_nan_values(self):
        """Test matrix with NaN values"""
        A = np.array([[1.0, np.nan], [np.nan, 1.0]])

        assert not is_posdef(A)

    def test_is_posdef_extreme_condition_numbers(self):
        """Test with extremely ill-conditioned matrices"""
        # Create ill-conditioned matrix
        U = np.random.randn(10, 10)
        s = np.logspace(-15, 0, 10)  # Very small to large eigenvalues
        A = U @ np.diag(s) @ U.T
        A = (A + A.T) / 2  # Ensure symmetry

        # Should handle extreme condition numbers gracefully
        result = is_posdef(A, tol=1e-12)
        assert isinstance(result, bool)


class TestSVDOperationsEdgeCases:
    """Test SVD operations with edge cases"""

    def test_canonical_svd_rank_deficient(self):
        """Test SVD with rank-deficient matrices"""
        # Create rank-deficient matrix
        X = np.array([[1, 2, 3], [2, 4, 6]])  # rank 1

        U, s, V = canonical_svd(X)

        # Should handle rank deficiency
        assert np.sum(s > 1e-10) == 1  # Only one significant singular value

    def test_canonical_svd_zero_matrix(self):
        """Test SVD with zero matrix"""
        X = np.zeros((5, 3))

        U, s, V = canonical_svd(X)

        assert np.allclose(s, 0.0)
        assert U.shape == (5, 3)
        assert V.shape == (3, 3)

    def test_canonical_svd_extreme_aspect_ratios(self):
        """Test SVD with extreme aspect ratios"""
        # Very tall matrix
        X = np.random.randn(1000, 2)
        U, s, V = canonical_svd(X)

        assert U.shape == (1000, 2)
        assert len(s) == 2
        assert V.shape == (2, 2)

        # Very wide matrix
        X = np.random.randn(2, 1000)
        U, s, V = canonical_svd(X)

        assert U.shape == (2, 2)
        assert len(s) == 2
        assert V.shape == (2, 1000)


class TestNormalizationEdgeCases:
    """Test normalization functions edge cases"""

    def test_normc_constant_columns(self):
        """Test normalization with constant columns"""
        X = np.array([[1, 5], [1, 5], [1, 5]])  # Constant columns

        result = normc(X, center=True)

        # Constant columns should become zero after centering
        assert np.allclose(result[:, 0], 0.0)
        assert np.allclose(result[:, 1], 0.0)

    def test_normc_single_observation(self):
        """Test normalization with single observation"""
        X = np.array([[1, 2, 3]])  # Single row

        result = normc(X, center=True)

        # Should handle single observation gracefully
        assert result.shape == (1, 3)
        assert np.allclose(result, 0.0)  # Centered single obs is zero

    def test_cov2cor_diagonal_covariance(self):
        """Test correlation from diagonal covariance matrix"""
        Sigma = np.diag([1, 4, 9])  # Diagonal covariance

        R = cov2cor(Sigma)

        # Should be identity matrix
        assert np.allclose(R, np.eye(3))

    def test_cov2cor_singular_covariance(self):
        """Test correlation with singular covariance matrix"""
        Sigma = np.array([[1, 1], [1, 1]])  # Singular

        with pytest.warns(RuntimeWarning):
            R = cov2cor(Sigma)
            # Should handle gracefully, possibly with NaN values
            assert R.shape == (2, 2)


class TestRandomGenerationEdgeCases:
    """Test random matrix generation edge cases"""

    def test_rnorm_matrix_extreme_parameters(self):
        """Test random matrix generation with extreme parameters"""
        # Very large mean and std
        X = rnorm_matrix(5, 3, mean=1e6, sd=1e3)

        assert X.shape == (5, 3)
        assert np.mean(X) > 1e5  # Should be around 1e6

    def test_rnorm_matrix_zero_std(self):
        """Test random matrix generation with zero standard deviation"""
        X = rnorm_matrix(3, 2, mean=5.0, sd=0.0)

        # Should be constant matrix
        assert np.allclose(X, 5.0)

    def test_rnorm_matrix_memory_large_matrices(self):
        """Test memory efficiency with large random matrices"""
        # Large matrix generation should not cause memory issues
        n, p = 1000, 500

        X = rnorm_matrix(n, p)

        assert X.shape == (n, p)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))


class TestNumericalPrecisionEdgeCases:
    """Test numerical precision in utility functions"""

    def test_operations_machine_epsilon(self):
        """Test operations near machine epsilon"""
        eps = np.finfo(float).eps

        # Matrix near machine precision
        X = np.array([[eps, 2*eps], [3*eps, 4*eps]])

        # Operations should handle gracefully
        result = normc(X)
        assert np.isfinite(result).all()

    def test_overflow_prevention(self):
        """Test overflow prevention in matrix operations"""
        # Create matrices that could overflow
        large_val = np.sqrt(np.finfo(float).max) / 2
        X = np.full((3, 3), large_val)
        d = np.full(3, 2.0)

        # Should not overflow
        result = diag_pre_multiply(d, X)
        assert np.isfinite(result).all()

    def test_underflow_handling(self):
        """Test underflow handling"""
        # Very small values
        tiny_val = np.finfo(float).tiny
        X = np.full((3, 3), tiny_val)

        # Operations should preserve or gracefully handle underflow
        result = normc(X)
        assert result.shape == (3, 3)


if __name__ == "__main__":
    pytest.main([__file__])