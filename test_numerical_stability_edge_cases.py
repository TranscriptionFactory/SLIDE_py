"""
Numerical stability and edge case testing for mathematical operations.

This module tests numerical edge cases that could lead to silent failures,
numerical instability, or incorrect mathematical computations.
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import patch

from src.loveslide.knockoff.utils import (
    diag_pre_multiply, diag_post_multiply, is_posdef,
    canonical_svd, normc, cov2cor, random_problem
)
from src.loveslide.knockoff.solve import (
    create_solve_equi, create_solve_sdp, _solve_sdp_cvxpy
)
from src.loveslide.love_python.love.cv import CV_delta, CalFittedSigma
from src.loveslide.love_python.love.est_pure_homo import (
    FindRowMax, FindPureNode, TestPure
)
from src.loveslide.love_python.love.utilities import threshA, offSum


class TestMatrixOperationEdgeCases:
    """Test edge cases in matrix operations."""

    def test_diag_pre_multiply_dimension_mismatch(self):
        """Test diag_pre_multiply with dimension mismatches."""
        d = np.array([1, 2, 3])
        X = np.random.randn(4, 5)  # Wrong number of rows

        with pytest.raises(ValueError):
            diag_pre_multiply(d, X)

    def test_diag_pre_multiply_zero_diagonal(self):
        """Test diag_pre_multiply with zero diagonal elements."""
        d = np.array([1, 0, 3])
        X = np.random.randn(3, 5)

        result = diag_pre_multiply(d, X)

        # Second row should be all zeros
        assert np.allclose(result[1, :], 0)
        assert not np.allclose(result[0, :], 0)
        assert not np.allclose(result[2, :], 0)

    def test_diag_pre_multiply_negative_diagonal(self):
        """Test diag_pre_multiply with negative diagonal elements."""
        d = np.array([-1, 2, -3])
        X = np.array([[1, 2], [3, 4], [5, 6]])

        result = diag_pre_multiply(d, X)
        expected = np.array([[-1, -2], [6, 8], [-15, -18]])

        assert np.allclose(result, expected)

    def test_diag_pre_multiply_very_large_values(self):
        """Test diag_pre_multiply with very large values."""
        d = np.array([1e15, 1e-15, 1.0])
        X = np.array([[1, 1], [1, 1], [1, 1]])

        result = diag_pre_multiply(d, X)

        assert result[0, 0] == 1e15
        assert result[1, 0] == 1e-15
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_diag_post_multiply_dimension_mismatch(self):
        """Test diag_post_multiply with dimension mismatches."""
        X = np.random.randn(3, 4)
        d = np.array([1, 2, 3])  # Wrong length

        with pytest.raises(ValueError):
            diag_post_multiply(X, d)

    def test_diag_post_multiply_inf_nan_handling(self):
        """Test diag_post_multiply with inf and nan values."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        d = np.array([np.inf, np.nan, 1.0])

        result = diag_post_multiply(X, d)

        assert np.isinf(result[:, 0]).all()
        assert np.isnan(result[:, 1]).all()
        assert np.array_equal(result[:, 2], X[:, 2])


class TestPositiveDefiniteChecking:
    """Test positive definite matrix checking edge cases."""

    def test_is_posdef_singular_matrix(self):
        """Test is_posdef with singular matrices."""
        # Rank deficient matrix
        A = np.array([[1, 1], [1, 1]])
        assert not is_posdef(A)

        # Zero matrix
        A_zero = np.zeros((3, 3))
        assert not is_posdef(A_zero)

    def test_is_posdef_nearly_singular(self):
        """Test is_posdef with nearly singular matrices."""
        # Matrix with very small eigenvalues
        A = np.array([[1, 0.999999], [0.999999, 1]])

        # Should be positive definite with default tolerance
        assert is_posdef(A)

        # Should not be positive definite with strict tolerance
        assert not is_posdef(A, tol=1e-3)

    def test_is_posdef_negative_definite(self):
        """Test is_posdef with negative definite matrices."""
        A = np.array([[-2, 0], [0, -3]])
        assert not is_posdef(A)

    def test_is_posdef_indefinite(self):
        """Test is_posdef with indefinite matrices."""
        A = np.array([[1, 0], [0, -1]])
        assert not is_posdef(A)

    def test_is_posdef_numerical_precision(self):
        """Test is_posdef with numerical precision issues."""
        # Create a matrix that should be PD but has numerical errors
        L = np.random.randn(5, 5)
        A = L @ L.T  # Should be PSD

        # Add tiny negative eigenvalue due to numerical error
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals[0] = -1e-14  # Tiny negative eigenvalue
        A_perturbed = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Should handle numerical precision appropriately
        assert not is_posdef(A_perturbed, tol=1e-12)
        assert is_posdef(A_perturbed, tol=1e-10)

    def test_is_posdef_non_symmetric(self):
        """Test is_posdef with non-symmetric matrices."""
        A = np.array([[2, 1], [0, 2]])  # Not symmetric

        # Should symmetrize or handle appropriately
        result = is_posdef(A)
        assert isinstance(result, bool)


class TestSVDStability:
    """Test SVD decomposition stability."""

    def test_canonical_svd_rank_deficient(self):
        """Test canonical_svd with rank deficient matrices."""
        # Create rank 2 matrix
        X = np.random.randn(100, 2) @ np.random.randn(2, 50)

        U, s, Vt = canonical_svd(X)

        # Check that small singular values are handled properly
        assert len(s) == min(X.shape)
        assert not np.any(np.isnan(s))
        assert not np.any(np.isinf(s))

    def test_canonical_svd_very_wide_matrix(self):
        """Test canonical_svd with very wide matrices."""
        X = np.random.randn(5, 1000)

        U, s, Vt = canonical_svd(X)

        assert U.shape[1] == 5  # Should be compact
        assert len(s) == 5

    def test_canonical_svd_very_tall_matrix(self):
        """Test canonical_svd with very tall matrices."""
        X = np.random.randn(1000, 5)

        U, s, Vt = canonical_svd(X)

        assert Vt.shape[0] == 5  # Should be compact
        assert len(s) == 5

    def test_canonical_svd_extreme_values(self):
        """Test canonical_svd with extreme values."""
        # Matrix with very large and very small values
        X = np.array([[1e10, 1e-10], [1e-10, 1e10]])

        U, s, Vt = canonical_svd(X)

        assert not np.any(np.isnan(U))
        assert not np.any(np.isnan(s))
        assert not np.any(np.isnan(Vt))
        assert not np.any(np.isinf(U))
        assert not np.any(np.isinf(s))
        assert not np.any(np.isinf(Vt))

    def test_canonical_svd_zero_matrix(self):
        """Test canonical_svd with zero matrix."""
        X = np.zeros((10, 5))

        U, s, Vt = canonical_svd(X)

        assert np.allclose(s, 0)
        assert not np.any(np.isnan(U))
        assert not np.any(np.isnan(Vt))


class TestNormalizationEdgeCases:
    """Test normalization function edge cases."""

    def test_normc_zero_variance_columns(self):
        """Test normc with zero variance columns."""
        X = np.array([[1, 1, 2], [1, 1, 4], [1, 1, 6]])  # First two columns constant

        result = normc(X)

        # Constant columns should remain unchanged or be handled gracefully
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_normc_single_value_columns(self):
        """Test normc with columns having single unique value."""
        X = np.array([[5], [5], [5], [5]])

        result = normc(X, center=True)

        # Should be all zeros after centering
        assert np.allclose(result, 0)

    def test_normc_extreme_values(self):
        """Test normc with extreme values."""
        X = np.array([[1e15], [-1e15], [0]])

        result = normc(X)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_normc_mixed_scales(self):
        """Test normc with mixed scale columns."""
        X = np.array([[1e-10, 1e10], [2e-10, 2e10], [3e-10, 3e10]])

        result = normc(X)

        # Both columns should be properly normalized
        assert np.allclose(np.std(result, axis=0), 1, atol=1e-10)


class TestCorrelationConversion:
    """Test correlation matrix conversion edge cases."""

    def test_cov2cor_singular_covariance(self):
        """Test cov2cor with singular covariance matrix."""
        # Create singular covariance matrix
        Sigma = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        # Should handle division by zero in diagonal
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cov2cor(Sigma)

        assert not np.any(np.isnan(result))

    def test_cov2cor_zero_diagonal(self):
        """Test cov2cor with zero diagonal elements."""
        Sigma = np.array([[0, 1], [1, 2]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cov2cor(Sigma)

        # Should handle zero diagonal appropriately
        assert result.shape == Sigma.shape

    def test_cov2cor_negative_diagonal(self):
        """Test cov2cor with negative diagonal elements."""
        Sigma = np.array([[-1, 0.5], [0.5, 2]])

        with pytest.raises((ValueError, RuntimeWarning)):
            cov2cor(Sigma)

    def test_cov2cor_numerical_precision(self):
        """Test cov2cor with numerical precision issues."""
        # Nearly perfect correlation
        eps = 1e-15
        Sigma = np.array([[1, 1-eps], [1-eps, 1]])

        result = cov2cor(Sigma)

        # Should maintain numerical stability
        assert not np.any(np.isnan(result))
        assert np.allclose(np.diag(result), 1)


class TestKnockoffSolverStability:
    """Test knockoff solver numerical stability."""

    def test_create_solve_equi_ill_conditioned(self):
        """Test equicorrelated knockoff solver with ill-conditioned matrices."""
        # Create ill-conditioned correlation matrix
        Sigma = np.eye(5)
        Sigma[0, 1:] = 0.99
        Sigma[1:, 0] = 0.99

        # Should handle ill-conditioning gracefully
        result = create_solve_equi(Sigma)

        assert result.shape == (5, 5)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_create_solve_equi_perfect_correlation(self):
        """Test equicorrelated solver with perfect correlation."""
        Sigma = np.ones((3, 3))  # Perfect correlation

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = create_solve_equi(Sigma)

        assert result.shape == (3, 3)

    def test_create_solve_sdp_degenerate_case(self):
        """Test SDP solver with degenerate cases."""
        # Create matrix that's barely positive definite
        Sigma = np.eye(4) * 1e-10

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = create_solve_sdp(Sigma)

        assert result.shape == (4, 4)


class TestLOVEAlgorithmStability:
    """Test LOVE algorithm numerical stability."""

    def test_cv_delta_extreme_values(self):
        """Test CV_delta with extreme parameter values."""
        X = np.random.randn(50, 10)

        # Very small delta values
        deltaGrids = np.array([1e-15, 1e-10, 1e-5])

        result = CV_delta(X, deltaGrids, diagonal=True, rep_cv=2)

        assert 'delta_opt' in result
        assert not np.isnan(result['delta_opt'])

    def test_cv_delta_singular_input(self):
        """Test CV_delta with singular input matrix."""
        # Create rank-deficient matrix
        X = np.ones((30, 5))  # All columns identical
        deltaGrids = np.array([0.1, 0.5, 0.9])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = CV_delta(X, deltaGrids, diagonal=True, rep_cv=2)

        assert 'delta_opt' in result

    def test_calfittedsigma_numerical_stability(self):
        """Test CalFittedSigma with numerical edge cases."""
        # Create problematic matrices
        Sigma = np.eye(5) + np.ones((5, 5)) * 0.99  # High correlation
        delta = 1e-10  # Very small delta
        Ms = np.random.randn(5, 3)
        Ms_sel = [True, False, True]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = CalFittedSigma(Sigma, delta, Ms, Ms_sel)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_findrowmax_degenerate_input(self):
        """Test FindRowMax with degenerate input."""
        # Matrix with identical rows
        Sigma = np.ones((4, 4))

        result = FindRowMax(Sigma)

        assert 'M' in result
        assert 'arg_M' in result
        assert not np.any(np.isnan(result['M']))

    def test_findpurenode_extreme_parameters(self):
        """Test FindPureNode with extreme parameters."""
        off_Sigma = np.random.randn(10, 10)
        off_Sigma = off_Sigma + off_Sigma.T  # Make symmetric

        Ms = np.random.randn(10, 5)
        Ms_sel = np.ones(5, dtype=bool)

        # Very small delta (extreme threshold)
        delta = 1e-15

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = FindPureNode(off_Sigma, delta, Ms, Ms_sel)

        assert isinstance(result, list)

    def test_testpure_boundary_conditions(self):
        """Test TestPure with boundary conditions."""
        Sigma_row = np.array([0.1, 0.9, 0.95, 0.99, 0.999])
        rowInd = 2
        Si = np.random.randn(3, 3)
        delta = 0.95  # Close to maximum correlation

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = TestPure(Sigma_row, rowInd, Si, delta)

        assert isinstance(result, bool)


class TestUtilityFunctionStability:
    """Test utility function numerical stability."""

    def test_thresha_extreme_thresholds(self):
        """Test threshA with extreme threshold values."""
        A = np.random.randn(10, 10)

        # Very large threshold (should zero out everything)
        result_large = threshA(A, mu=1e10)
        assert np.allclose(result_large, 0)

        # Very small threshold (should change very little)
        result_small = threshA(A, mu=1e-10)
        assert np.allclose(result_small, A, atol=1e-9)

    def test_thresha_zero_matrix(self):
        """Test threshA with zero matrix."""
        A = np.zeros((5, 5))
        mu = 0.5

        result = threshA(A, mu)
        assert np.allclose(result, 0)

    def test_offsum_weight_edge_cases(self):
        """Test offSum with edge case weights."""
        M = np.random.randn(8, 8)

        # Zero weights
        result_zero = offSum(M, weights=0.0)
        assert result_zero == 0.0

        # Very large weights
        result_large = offSum(M, weights=1e10)
        assert not np.isnan(result_large)
        assert not np.isinf(result_large)

        # Negative weights
        result_neg = offSum(M, weights=-1.0)
        assert isinstance(result_neg, float)

    def test_random_problem_extreme_parameters(self):
        """Test random_problem with extreme parameters."""
        # Very small dimensions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X, y, beta = random_problem(n=5, p=2, k=1)

        assert X.shape == (5, 2)
        assert len(y) == 5
        assert len(beta) == 2

        # Large dimensions (should handle efficiently)
        X_large, y_large, beta_large = random_problem(n=1000, p=500, k=10)
        assert X_large.shape == (1000, 500)
        assert len(y_large) == 1000
        assert np.sum(beta_large != 0) == 10


if __name__ == "__main__":
    pytest.main([__file__])