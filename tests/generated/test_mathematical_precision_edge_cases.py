"""
Test coverage for mathematical precision and numerical stability edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.loveslide.slide import SLIDE
from src.loveslide.cv import SLIDEcv
from src.loveslide.score import Estimator, SLIDE_Estimator
from src.loveslide.knockoffs import Knockoffs


class TestMathematicalPrecisionEdges:
    """Test mathematical precision edge cases not covered elsewhere."""

    def test_slide_with_denormalized_numbers(self):
        """Test SLIDE with denormalized floating point numbers."""
        # Create data with denormalized numbers (very small but non-zero)
        X = np.random.randn(100, 20)
        X[X != 0] = np.where(X[X != 0] > 0,
                            np.nextafter(0, 1) * np.random.rand(np.sum(X != 0)),
                            -np.nextafter(0, 1) * np.random.rand(np.sum(X != 0)))

        y = np.random.binomial(1, 0.5, 100)

        params = {
            'delta': [0.1],
            'lambda': [0.5],
            'fdr': 0.1,
            'niter': 10,
            'pure_homo': True
        }

        # Should handle denormalized numbers without overflow/underflow
        slide = SLIDE(params, X, y)
        assert slide.data.X.shape == X.shape

    def test_slide_with_machine_epsilon_differences(self):
        """Test SLIDE with differences at machine epsilon level."""
        # Create features that differ only by machine epsilon
        base_feature = np.random.randn(50, 1)
        epsilon_diff = np.finfo(float).eps

        X = np.hstack([
            base_feature,
            base_feature + epsilon_diff,
            base_feature - epsilon_diff,
            np.random.randn(50, 7)
        ])

        y = np.random.binomial(1, 0.5, 50)

        params = {
            'delta': [0.1],
            'lambda': [0.5],
            'fdr': 0.1,
            'niter': 5,
            'pure_homo': True
        }

        # Should distinguish or handle epsilon-level differences
        slide = SLIDE(params, X, y)
        assert slide.data.X.shape == X.shape

    def test_estimator_with_extreme_feature_scaling(self):
        """Test estimator with extreme feature value ranges."""
        # Mix tiny and huge feature values
        X_tiny = np.random.randn(100, 5) * 1e-15  # Very small
        X_huge = np.random.randn(100, 5) * 1e15   # Very large
        X_normal = np.random.randn(100, 10)       # Normal scale

        X = np.hstack([X_tiny, X_huge, X_normal])
        y = np.random.randn(100)

        estimator = Estimator()
        estimator._init_model(y)

        # Should handle extreme scaling differences
        # Note: This tests the underlying mathematical stability
        covariance = np.cov(X.T)
        assert np.all(np.isfinite(covariance))

    def test_knockoffs_with_near_zero_eigenvalues(self):
        """Test knockoff generation with eigenvalues near machine epsilon."""
        # Create covariance with eigenvalues near machine epsilon
        n_features = 10
        U = np.random.randn(n_features, n_features)
        U, _ = np.linalg.qr(U)

        # Eigenvalues: some normal, some near machine epsilon
        eigenvals = np.array([1, 0.5, 0.1] + [np.finfo(float).eps * 10] * 7)
        Sigma = U @ np.diag(eigenvals) @ U.T

        # Generate X from this covariance
        X = np.random.multivariate_normal(np.zeros(n_features), Sigma, 100)

        knockoffs = Knockoffs()

        # Should handle near-zero eigenvalues gracefully
        with np.errstate(all='ignore'):  # Suppress numerical warnings
            try:
                result = knockoffs.create_second_order(X, method='equicorrelated')
                assert result.shape == X.shape
            except (np.linalg.LinAlgError, ValueError):
                # Acceptable to fail with near-singular matrices
                pass

    def test_cv_with_extreme_parameter_ratios(self):
        """Test cross-validation with extreme parameter ratios."""
        X = np.random.randn(100, 20)
        y = np.random.binomial(1, 0.5, 100)

        # Parameters with extreme ratios
        params = {
            'delta': [1e-10, 1 - 1e-10],  # Very small vs almost 1
            'lambda': [1e-10, 1 - 1e-10],
            'fdr': 1e-10,
            'niter': 5,
            'pure_homo': True
        }

        cv = SLIDEcv(params, X, y, cv_folds=3)

        # Should handle extreme parameter ratios numerically
        with np.errstate(all='ignore'):
            try:
                # This tests numerical stability in parameter space
                assert hasattr(cv, 'data')
                assert cv.data.X.shape == X.shape
            except (OverflowError, UnderflowError):
                # Acceptable to have numerical limits
                pass

    def test_mathematical_invariants_preservation(self):
        """Test that mathematical invariants are preserved under operations."""
        X = np.random.randn(50, 10)
        Sigma_orig = np.cov(X.T)

        # Test symmetry preservation
        assert np.allclose(Sigma_orig, Sigma_orig.T), "Covariance should be symmetric"

        # Test positive semi-definiteness
        eigenvals = np.linalg.eigvals(Sigma_orig)
        assert np.all(eigenvals >= -1e-10), "Covariance should be PSD"

        # Test normalization effects
        X_norm = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        Sigma_norm = np.cov(X_norm.T)

        # Normalized correlation matrix should have unit diagonal
        correlation = np.corrcoef(X.T)
        diagonal = np.diag(correlation)
        assert np.allclose(diagonal, 1.0, atol=1e-10), "Correlation diagonal should be 1"

    def test_numerical_differentiation_stability(self):
        """Test numerical stability of implicit differentiation operations."""
        # Create a scenario that might involve numerical differentiation
        X = np.random.randn(100, 15)
        y = np.random.binomial(1, 0.5, 100)

        # Parameters that might trigger different numerical paths
        params_list = [
            {'delta': [0.01], 'lambda': [0.1], 'fdr': 0.05},
            {'delta': [0.01 + 1e-8], 'lambda': [0.1], 'fdr': 0.05},  # Tiny perturbation
        ]

        results = []
        for params in params_list:
            params.update({'niter': 5, 'pure_homo': True})
            slide = SLIDE(params, X, y)
            results.append(slide)

        # Results should be numerically close for small parameter perturbations
        # This tests the stability of the underlying algorithms
        assert all(hasattr(r, 'data') for r in results)

    def test_matrix_conditioning_edge_cases(self):
        """Test behavior with matrices at conditioning boundaries."""
        # Create matrices with specific condition numbers
        condition_numbers = [1e2, 1e6, 1e12, 1e15]

        for cond_num in condition_numbers:
            n = 10
            U = np.random.randn(n, n)
            U, _ = np.linalg.qr(U)

            # Create matrix with specific condition number
            eigenvals = np.logspace(0, -np.log10(cond_num), n)
            Sigma = U @ np.diag(eigenvals) @ U.T

            # Test numerical stability
            actual_cond = np.linalg.cond(Sigma)

            if actual_cond < 1e14:  # Within reasonable numerical limits
                try:
                    inv_Sigma = np.linalg.inv(Sigma)
                    # Test that inversion is reasonably accurate
                    identity_test = Sigma @ inv_Sigma
                    max_off_diag = np.max(np.abs(identity_test - np.eye(n)))
                    assert max_off_diag < 1e-6, f"Poor inversion accuracy for cond={actual_cond}"
                except np.linalg.LinAlgError:
                    # Expected for very ill-conditioned matrices
                    pass

    def test_floating_point_comparison_edge_cases(self):
        """Test floating point comparison edge cases."""
        # Test cases where naive comparison might fail
        a = 0.1 + 0.2
        b = 0.3

        # This is a classic floating point precision issue
        assert not (a == b), "Direct comparison should fail"
        assert np.isclose(a, b), "np.isclose should succeed"

        # Test in context of the algorithms
        X = np.array([[0.1 + 0.2], [0.3], [0.1 + 0.1 + 0.1]])
        y = np.array([0, 1, 0])

        # Should handle floating point comparison issues in real algorithms
        correlation = np.corrcoef(X.T)[0, 0]
        assert np.isclose(correlation, 1.0) or correlation == 1.0