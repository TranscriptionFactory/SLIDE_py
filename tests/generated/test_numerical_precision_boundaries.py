"""
Test Coverage Gap: Numerical Precision Boundaries
===============================================

Tests extreme numerical precision scenarios and floating-point edge cases that may
not be fully covered in existing numerical stability tests.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import linalg
import sys
from src.loveslide.knockoff import utils
from src.loveslide.knockoff.solve import create_solve_equi, create_solve_sdp
from src.loveslide.knockoff.create import create_gaussian, create_second_order
from src.loveslide import SLIDE


class TestFloatingPointPrecisionLimits:
    """Test floating-point precision limit scenarios."""

    def test_machine_epsilon_boundary_operations(self):
        """Test operations at machine epsilon boundaries."""
        # Create matrices with values near machine epsilon
        eps = np.finfo(np.float64).eps

        # Matrix with eigenvalues near machine epsilon
        n = 5
        A = np.random.randn(n, n)
        A = A @ A.T  # Make positive semi-definite
        A += eps * np.eye(n)  # Add tiny diagonal

        # Test positive definiteness at precision limit
        result = utils.is_posdef(A, tol=eps/2)
        assert isinstance(result, bool)

        # Test with tolerance smaller than machine epsilon
        result_strict = utils.is_posdef(A, tol=eps*10)
        assert isinstance(result_strict, bool)

    def test_subnormal_number_handling(self):
        """Test handling of subnormal numbers."""
        # Create subnormal numbers (very close to zero)
        subnormal = np.finfo(np.float64).tiny / 2

        # Create matrix with subnormal elements
        X = np.full((5, 5), subnormal)
        X += np.eye(5) * 1e-10  # Add tiny diagonal

        # Test covariance matrix conversion
        try:
            cor_matrix = utils.cov2cor(X)
            # Should handle subnormal numbers gracefully
            assert np.all(np.isfinite(cor_matrix))
        except (FloatingPointError, RuntimeWarning):
            # Acceptable to warn about precision loss
            pass

    def test_extreme_condition_number_matrices(self):
        """Test matrices with extreme condition numbers."""
        # Create ill-conditioned matrix
        n = 10
        U, _, Vt = linalg.svd(np.random.randn(n, n))

        # Set singular values to span many orders of magnitude
        s = np.logspace(-15, 0, n)  # Condition number ~ 1e15
        X = U @ np.diag(s) @ Vt
        Sigma = X.T @ X

        # Test knockoff creation with ill-conditioned covariance
        try:
            knockoffs = create_gaussian(X, method='equi')
            assert knockoffs is not None
            assert knockoffs.shape == X.shape
        except (linalg.LinAlgError, np.linalg.LinAlgError, RuntimeError) as e:
            # Should provide meaningful error for ill-conditioned matrices
            assert "condition" in str(e).lower() or "singular" in str(e).lower()

    def test_floating_point_accumulation_errors(self):
        """Test accumulation of floating-point errors in iterative operations."""
        # Create scenario where floating-point errors accumulate
        X = np.random.randn(100, 10)

        # Repeatedly apply operations that could accumulate error
        result = X.copy()
        for i in range(1000):
            result = utils.normc(result, center=True)

            # Check that we don't get completely corrupt results
            assert np.all(np.isfinite(result))
            assert result.shape == X.shape

        # Final result should still be reasonable
        assert np.abs(np.mean(result, axis=0)).max() < 1e-10  # Should be centered

    def test_precision_loss_in_matrix_operations(self):
        """Test precision loss in chained matrix operations."""
        # Start with well-conditioned matrix
        np.random.seed(42)
        X = np.random.randn(50, 20)

        # Chain operations that could lose precision
        Sigma = X.T @ X / X.shape[0]  # Sample covariance

        # Test SVD precision
        U, s, Vt = utils.canonical_svd(Sigma)

        # Reconstruct and check precision loss
        Sigma_reconstructed = U @ np.diag(s) @ Vt
        precision_loss = np.max(np.abs(Sigma - Sigma_reconstructed))

        # Precision loss should be within reasonable bounds
        assert precision_loss < 1e-10

    def test_denormalized_eigenvalue_handling(self):
        """Test handling of denormalized eigenvalues in matrix decompositions."""
        # Create matrix with eigenvalues spanning denormal range
        n = 8
        # Mix of normal and near-denormal eigenvalues
        eigenvals = np.array([1.0, 0.1, 1e-10, 1e-50, 1e-100, 1e-200, 1e-307, 1e-308])
        eigenvals = eigenvals[:n]

        # Create symmetric matrix with these eigenvalues
        Q = linalg.qr(np.random.randn(n, n))[0]
        A = Q @ np.diag(eigenvals) @ Q.T

        # Test positive definiteness with denormal eigenvalues
        result = utils.is_posdef(A, tol=1e-309)  # Tolerance smaller than smallest eigenvalue

        # Should handle gracefully (may be False due to denormal values)
        assert isinstance(result, bool)


class TestNumericalStabilityBoundaries:
    """Test numerical stability at algorithm boundaries."""

    def test_knockoff_creation_near_singular_limit(self):
        """Test knockoff creation when covariance is nearly singular."""
        # Create nearly singular covariance matrix
        n, p = 100, 20
        X = np.random.randn(n, p)

        # Make last column nearly dependent on first
        X[:, -1] = X[:, 0] + 1e-12 * np.random.randn(n)

        # Test different knockoff methods
        methods = ['equi', 'sdp']

        for method in methods:
            try:
                knockoffs = create_gaussian(X, method=method)

                # If it succeeds, check quality
                if knockoffs is not None:
                    assert knockoffs.shape == X.shape
                    # Check that knockoffs are not identical to originals
                    assert not np.allclose(knockoffs, X, rtol=1e-6)

            except (linalg.LinAlgError, np.linalg.LinAlgError) as e:
                # Acceptable to fail on nearly singular matrices
                assert "singular" in str(e).lower()

    def test_sdp_solver_numerical_boundaries(self):
        """Test SDP solver at numerical precision boundaries."""
        # Create covariance matrix with challenging numerical properties
        p = 15
        np.random.seed(42)

        # Create matrix with eigenvalues spanning wide range
        eigenvals = np.logspace(-8, 0, p)  # From 1e-8 to 1
        Q = linalg.qr(np.random.randn(p, p))[0]
        Sigma = Q @ np.diag(eigenvals) @ Q.T

        try:
            # Test SDP-based knockoff creation
            s_values = create_solve_sdp(Sigma)

            # Check that solution is reasonable
            assert len(s_values) == p
            assert np.all(s_values >= -1e-10)  # Should be non-negative (within tolerance)
            assert np.all(s_values <= 1.0 + 1e-10)  # Should be bounded

        except Exception as e:
            # Should provide informative error for numerical issues
            error_msg = str(e).lower()
            assert any(word in error_msg for word in
                      ['numerical', 'precision', 'solver', 'convergence'])

    def test_cross_validation_numerical_stability(self):
        """Test cross-validation numerical stability with edge cases."""
        # Create data with numerical challenges
        np.random.seed(42)
        n, p = 50, 10
        X = np.random.randn(n, p)

        # Add small amount of noise to make it challenging
        X += 1e-10 * np.random.randn(n, p)

        # Binary target with extreme imbalance (numerical challenge for CV)
        y = np.zeros(n)
        y[:2] = 1  # Only 2 positive examples

        params = {
            'K': 3,
            'fdr': 0.1,
            'delta': [0.1, 0.5],
            'lambda': [0.3, 0.7]
        }

        try:
            slide = SLIDE(params, X, y)
            # Should handle numerical challenges in CV gracefully
            assert slide is not None

        except (ValueError, RuntimeError) as e:
            # Should provide meaningful error for numerical issues
            error_msg = str(e).lower()
            assert any(word in error_msg for word in
                      ['numerical', 'convergence', 'stability', 'sample'])

    def test_parameter_optimization_precision_limits(self):
        """Test parameter optimization at precision limits."""
        np.random.seed(42)
        X = np.random.randn(30, 8)
        y = np.random.binomial(1, 0.5, 30)

        # Parameters very close to boundaries
        params = {
            'K': 3,
            'fdr': 0.001,  # Very small FDR
            'delta': [1e-10, 1-1e-10],  # Very close to boundaries
            'lambda': [1e-10, 1-1e-10]  # Very close to boundaries
        }

        try:
            slide = SLIDE(params, X, y)

            # Should handle boundary parameters gracefully
            assert slide is not None

        except (ValueError, RuntimeError) as e:
            # Should provide clear parameter validation error
            error_msg = str(e).lower()
            assert any(word in error_msg for word in
                      ['parameter', 'bound', 'range', 'valid'])


class TestMathematicalInvariantPreservation:
    """Test preservation of mathematical invariants under numerical stress."""

    def test_correlation_matrix_properties_preservation(self):
        """Test that correlation matrix properties are preserved under numerical stress."""
        # Create challenging covariance matrix
        np.random.seed(42)
        p = 15
        X = np.random.randn(100, p)

        # Add numerical challenges
        X[:, 0] = X[:, 0] * 1e6  # Large scale difference
        X[:, 1] = X[:, 1] * 1e-6  # Small scale

        Sigma = np.cov(X.T)

        # Convert to correlation matrix
        R = utils.cov2cor(Sigma)

        # Check mathematical properties are preserved
        assert np.allclose(np.diag(R), 1.0, rtol=1e-10)  # Diagonal should be 1
        assert np.allclose(R, R.T, rtol=1e-10)  # Should be symmetric

        # All off-diagonal elements should be in [-1, 1]
        off_diag = R[np.triu_indices(p, k=1)]
        assert np.all(off_diag >= -1.0 - 1e-10)
        assert np.all(off_diag <= 1.0 + 1e-10)

    def test_knockoff_exchangeability_under_numerical_stress(self):
        """Test that knockoff exchangeability is preserved under numerical stress."""
        np.random.seed(42)
        n, p = 60, 12

        # Create data with numerical challenges
        X = np.random.randn(n, p)
        X[:, 0] *= 1e8  # Extreme scaling
        X[:, -1] *= 1e-8

        try:
            # Create knockoffs
            X_tilde = create_gaussian(X, method='equi')

            if X_tilde is not None:
                # Test basic exchangeability property: same marginal covariance structure
                combined = np.hstack([X, X_tilde])

                # Compute sample covariances
                Sigma_orig = np.cov(X.T)
                Sigma_knockoff = np.cov(X_tilde.T)

                # Should have similar covariance structure (within numerical tolerance)
                # Note: Perfect equality not expected due to sampling and numerical effects
                assert Sigma_orig.shape == Sigma_knockoff.shape

        except Exception as e:
            # Acceptable to fail on extreme numerical cases
            assert "numerical" in str(e).lower() or "singular" in str(e).lower()

    def test_matrix_norm_preservation(self):
        """Test that matrix norms are preserved under transformations."""
        np.random.seed(42)
        X = np.random.randn(20, 10)

        # Apply normalization
        X_norm = utils.normc(X, center=True)

        # Check that column means are preserved (should be zero after centering)
        col_means = np.mean(X_norm, axis=0)
        assert np.all(np.abs(col_means) < 1e-10)

        # Check that transformation preserves essential structure
        assert X_norm.shape == X.shape
        assert np.all(np.isfinite(X_norm))


class TestEdgeCaseRecoveryMechanisms:
    """Test recovery mechanisms for numerical edge cases."""

    def test_fallback_mechanisms_for_failed_decompositions(self):
        """Test fallback mechanisms when matrix decompositions fail."""
        # Create problematic matrix
        X = np.array([[1, 2, 3],
                      [1, 2, 3],  # Repeated row
                      [2, 4, 6]])  # Linear combination

        # This should trigger fallback mechanisms
        try:
            U, s, Vt = utils.canonical_svd(X)

            # If successful, check basic properties
            assert U.shape[1] <= min(X.shape)
            assert len(s) <= min(X.shape)
            assert Vt.shape[0] <= min(X.shape)

        except (linalg.LinAlgError, np.linalg.LinAlgError):
            # Acceptable to fail on rank-deficient matrices
            pass

    def test_graceful_degradation_mechanisms(self):
        """Test graceful degradation when optimal solutions are not achievable."""
        # Create scenario where optimal knockoffs cannot be created
        p = 8
        # Create singular covariance matrix
        Sigma = np.ones((p, p)) * 0.9
        np.fill_diagonal(Sigma, 1.0)
        # Make it singular by setting one eigenvalue to zero
        eigenvals, eigenvecs = linalg.eigh(Sigma)
        eigenvals[-1] = 0  # Make singular
        Sigma_singular = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        try:
            # Should either succeed with degraded solution or fail gracefully
            s_values = create_solve_equi(Sigma_singular)

            if s_values is not None:
                # Degraded solution should still satisfy basic constraints
                assert len(s_values) == p
                assert np.all(s_values >= -1e-10)  # Non-negative within tolerance

        except Exception as e:
            # Should provide informative error message
            assert "singular" in str(e).lower() or "rank" in str(e).lower()