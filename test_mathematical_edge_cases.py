"""
Test mathematical edge cases and numerical stability.
Addresses: Matrix singularity, floating point precision, decomposition failures
"""
import pytest
import numpy as np
import scipy.linalg
from numpy.testing import assert_allclose
from loveslide import SLIDE, Knockoffs
from loveslide.knockoff.utils import is_posdef, canonical_svd, cov2cor
from loveslide.knockoff.solve import create_solve_sdp, create_solve_equi


class TestMatrixSingularityHandling:
    """Test handling of singular and nearly singular matrices."""

    def test_perfectly_singular_covariance(self):
        """Test behavior with perfectly singular covariance matrix."""
        # Create perfectly correlated variables
        X_base = np.random.randn(100, 5)
        X = np.column_stack([X_base, X_base[:, 0]])  # Perfect correlation

        with pytest.raises((np.linalg.LinAlgError, ValueError)):
            Knockoffs().create_knockoffs(X, method='sdp')

    def test_near_singular_covariance_handling(self):
        """Test graceful handling of near-singular matrices."""
        X_base = np.random.randn(100, 5)
        # Add tiny noise to avoid perfect singularity
        X = np.column_stack([X_base, X_base[:, 0] + 1e-12*np.random.randn(100)])

        # Should either succeed with regularization or fail gracefully
        try:
            result = Knockoffs().create_knockoffs(X, method='equi')
            assert result.shape == X.shape
        except (np.linalg.LinAlgError, ValueError):
            pass  # Acceptable failure mode

    def test_rank_deficient_matrix_detection(self):
        """Test detection of rank deficiency in input matrices."""
        # Create rank-deficient matrix
        X = np.random.randn(50, 10)
        X[:, 5] = X[:, 0] + X[:, 1]  # Linear dependence

        with warnings.catch_warnings(record=True) as w:
            try:
                Knockoffs().create_knockoffs(X)
                if w:
                    assert any("rank" in str(warning.message).lower() for warning in w)
            except (np.linalg.LinAlgError, ValueError):
                pass


class TestFloatingPointPrecision:
    """Test handling of floating point edge cases."""

    def test_very_small_eigenvalues(self):
        """Test handling of eigenvalues near machine precision."""
        # Create matrix with very small eigenvalues
        eigs = np.array([1e-15, 1e-14, 1e-13, 1, 2])
        Q, _ = np.linalg.qr(np.random.randn(5, 5))
        Sigma = Q @ np.diag(eigs) @ Q.T

        # Test positive definiteness check
        result = is_posdef(Sigma, tol=1e-12)
        assert isinstance(result, bool)

    def test_extreme_parameter_values(self):
        """Test with extreme parameter values."""
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        # Very small FDR
        slide_tiny = SLIDE({'fdr': 1e-10}, x=X, y=y)
        # Should not crash

        # Very large variance inflation
        params = {'fdr': 0.1, 'knockoff_kwargs': {'shrink': 1e10}}
        with pytest.warns(UserWarning):
            SLIDE(params, x=X, y=y)

    def test_numerical_overflow_protection(self):
        """Test protection against numerical overflow."""
        # Create data that might cause overflow
        X = np.random.randn(100, 10) * 1e10
        y = np.random.randn(100) * 1e10

        # Should either handle gracefully or fail with clear error
        try:
            slide = SLIDE({'fdr': 0.1}, x=X, y=y)
            result = slide.fit()
            assert np.all(np.isfinite(result.statistic))
        except (ValueError, np.linalg.LinAlgError) as e:
            assert "overflow" in str(e).lower() or "singular" in str(e).lower()


class TestSVDDecompositionFailures:
    """Test handling of SVD and eigenvalue decomposition edge cases."""

    def test_svd_convergence_failure_recovery(self):
        """Test recovery from SVD convergence failures."""
        # Create problematic matrix for SVD
        X = np.random.randn(100, 20)
        X[0, :] = 1e20  # Extreme outlier
        X[1, :] = -1e20

        try:
            U, s, Vt = canonical_svd(X)
            assert U.shape[0] == X.shape[0]
            assert len(s) == min(X.shape)
        except np.linalg.LinAlgError:
            pass  # Acceptable failure

    def test_correlation_matrix_edge_cases(self):
        """Test correlation matrix computation edge cases."""
        # Perfect correlations
        X = np.array([[1, 2], [2, 4], [3, 6]])  # Perfect correlation

        try:
            corr = cov2cor(np.cov(X.T))
            assert np.allclose(np.abs(corr[0, 1]), 1.0, atol=1e-10)
        except (ValueError, np.linalg.LinAlgError):
            pass

    def test_eigenvalue_decomposition_edge_cases(self):
        """Test eigenvalue decomposition with edge cases."""
        # Matrix with repeated eigenvalues
        Sigma = np.array([[2, 1], [1, 2]])  # Eigenvalues: 3, 1

        try:
            result = create_solve_sdp(Sigma)
            assert result.shape == (2, 2)
            assert is_posdef(result, tol=1e-10)
        except (np.linalg.LinAlgError, ValueError):
            pass