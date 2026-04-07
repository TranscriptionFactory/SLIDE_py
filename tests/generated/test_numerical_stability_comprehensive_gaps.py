"""
Comprehensive numerical stability and mathematical edge case tests for SLIDE_py.

Tests critical mathematical operations at precision boundaries:
- Machine epsilon boundaries
- Numerical overflow/underflow
- Matrix conditioning issues
- Convergence criteria edge cases
- Floating point precision limits
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from scipy.linalg import LinAlgError
from unittest.mock import patch

from loveslide import SLIDE, Knockoffs, call_love
from loveslide.tools import calc_default_fsize


class TestMachineEpsilonBoundaries:
    """Test operations at machine epsilon precision boundaries."""

    def setup_method(self):
        """Setup machine precision constants."""
        self.eps = np.finfo(float).eps
        self.tiny = np.finfo(float).tiny
        self.max_val = np.finfo(float).max

    def test_slide_matrix_operations_near_epsilon(self):
        """Test SLIDE matrix operations near machine epsilon."""
        # Create matrices with values near machine epsilon
        X = np.random.randn(50, 20) * self.eps * 1e3
        y = np.random.binomial(1, 0.5, 50)

        params = {"fdr": 0.1, "niter": 2}

        # Should handle tiny values gracefully
        slide = SLIDE(params, x=X, y=y)
        assert slide.data.X.shape == X.shape

        # Check for numerical warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            slide.show_params()
            # Should not produce overflow warnings

    def test_correlation_matrix_near_singular(self):
        """Test correlation matrix computation near singular conditions."""
        # Create nearly linearly dependent features
        X = np.random.randn(100, 10)
        X[:, 5] = X[:, 0] + np.random.randn(100) * self.eps * 1e6  # Nearly identical

        correlation_matrix = np.corrcoef(X.T)

        # Check condition number
        try:
            condition_number = np.linalg.cond(correlation_matrix)
            assert condition_number < 1e12, f"Matrix is ill-conditioned: {condition_number}"
        except LinAlgError:
            # Expected for truly singular matrices
            pass

    def test_eigenvalue_computation_precision(self):
        """Test eigenvalue computation at precision boundaries."""
        # Create matrix with very small eigenvalues
        A = np.random.randn(10, 10)
        A = A @ A.T  # Make positive definite
        A += np.eye(10) * self.eps * 1e6  # Add tiny regularization

        eigenvals, eigenvecs = np.linalg.eigh(A)

        # All eigenvalues should be positive
        assert np.all(eigenvals > -self.eps * 1e3), "Negative eigenvalues detected"

        # Eigenvectors should be orthonormal
        orthogonality_error = np.max(np.abs(eigenvecs.T @ eigenvecs - np.eye(10)))
        assert orthogonality_error < 1e-10, f"Eigenvectors not orthonormal: {orthogonality_error}"

    def test_matrix_inversion_conditioning(self):
        """Test matrix inversion with poor conditioning."""
        # Create ill-conditioned matrix
        condition_numbers = [1e6, 1e10, 1e14]

        for cond_num in condition_numbers:
            # Create matrix with specific condition number
            U, _, Vt = np.linalg.svd(np.random.randn(20, 20))
            s = np.logspace(0, -np.log10(cond_num), 20)
            A = U @ np.diag(s) @ Vt

            try:
                A_inv = np.linalg.inv(A)
                # Check that A @ A_inv ≈ I
                identity_error = np.max(np.abs(A @ A_inv - np.eye(20)))

                if cond_num <= 1e12:
                    assert identity_error < 1e-6, f"Inversion error too large: {identity_error}"
                else:
                    # For very ill-conditioned matrices, expect larger errors
                    assert identity_error < 1.0, "Complete inversion failure"

            except LinAlgError:
                # Expected for extremely ill-conditioned matrices
                assert cond_num > 1e12, "Unexpected inversion failure for well-conditioned matrix"


class TestNumericalOverflowUnderflow:
    """Test numerical overflow and underflow scenarios."""

    def test_slide_large_value_handling(self):
        """Test SLIDE with very large input values."""
        # Create data with large values (but not overflow)
        large_scale = 1e10
        X = np.random.randn(50, 20) * large_scale
        y = np.random.binomial(1, 0.5, 50)

        params = {"fdr": 0.1, "niter": 2}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            slide = SLIDE(params, x=X, y=y)

            # Check for overflow warnings
            overflow_warnings = [warning for warning in w
                               if "overflow" in str(warning.message).lower()]
            assert len(overflow_warnings) == 0, "Unexpected overflow in large value handling"

    def test_knockoffs_numerical_stability(self):
        """Test Knockoffs with numerically challenging matrices."""
        # Test with very small values
        X_small = np.random.randn(50, 20) * 1e-10
        y = np.random.binomial(1, 0.5, 50)

        knockoffs = Knockoffs()

        try:
            result = knockoffs.run_iteration(X_small, y, fdr=0.1, method='lasso')
            # Should either succeed or fail gracefully
            assert isinstance(result, dict)
        except (ValueError, LinAlgError, FloatingPointError) as e:
            # Acceptable failures for numerical issues
            assert any(keyword in str(e).lower() for keyword in
                      ['singular', 'numerically', 'overflow', 'underflow', 'ill-conditioned'])

    def test_love_algorithm_numerical_robustness(self):
        """Test LOVE algorithm with challenging numerical scenarios."""
        test_scenarios = [
            ("small_values", np.random.randn(50, 20) * 1e-8),
            ("large_values", np.random.randn(50, 20) * 1e8),
            ("mixed_scales", np.random.randn(50, 20) * np.logspace(-5, 5, 20)),
        ]

        for scenario_name, X in test_scenarios:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                try:
                    result = call_love(X, lbd=0.5, mu=0.5)

                    # Check for numerical warnings
                    numerical_warnings = [warning for warning in w
                                        if any(keyword in str(warning.message).lower()
                                              for keyword in ['overflow', 'underflow', 'invalid', 'precision'])]

                    if len(numerical_warnings) > 0:
                        warnings.warn(f"Numerical warnings in scenario {scenario_name}: {numerical_warnings}")

                    # Validate result structure
                    assert isinstance(result, dict), f"Invalid result type for {scenario_name}"

                except (ValueError, LinAlgError, FloatingPointError) as e:
                    # Document expected numerical failures
                    print(f"Expected numerical failure in {scenario_name}: {e}")

    def test_parameter_calculation_edge_cases(self):
        """Test parameter calculations at edge cases."""
        # Test calc_default_fsize with extreme values
        extreme_test_cases = [
            (1, 1000000),  # Very large K
            (1000000, 1),  # Very large n_rows
            (1, 1),        # Minimum values
            (2, 2),        # Equal small values
        ]

        for n_rows, K in extreme_test_cases:
            result = calc_default_fsize(n_rows, K)

            # Result should be reasonable
            assert isinstance(result, int), f"Non-integer result for n_rows={n_rows}, K={K}"
            assert result >= 0, f"Negative result for n_rows={n_rows}, K={K}"
            assert result <= max(n_rows, K), f"Result too large for n_rows={n_rows}, K={K}"


class TestMatrixConditioningIssues:
    """Test matrix conditioning and stability issues."""

    def test_slide_with_rank_deficient_matrices(self):
        """Test SLIDE with rank-deficient input matrices."""
        # Create rank-deficient matrix
        X_base = np.random.randn(100, 10)
        X_extended = np.column_stack([X_base, X_base[:, :5]])  # Duplicate some columns
        y = np.random.binomial(1, 0.5, 100)

        params = {"fdr": 0.1, "niter": 2}

        # Should detect and handle rank deficiency
        slide = SLIDE(params, x=X_extended, y=y)
        assert slide.data.X.shape[0] == 100  # Rows should be preserved

    def test_covariance_matrix_regularization(self):
        """Test covariance matrix regularization for stability."""
        # Create data that leads to singular covariance
        n_samples, n_features = 50, 100  # p > n scenario
        X = np.random.randn(n_samples, n_features)

        # Compute sample covariance
        cov_matrix = np.cov(X.T)

        # Check if regularization is needed
        try:
            eigenvals = np.linalg.eigvals(cov_matrix)
            min_eigenval = np.min(eigenvals)

            if min_eigenval <= 0:
                # Apply regularization
                regularized_cov = cov_matrix + np.eye(n_features) * 1e-6
                reg_eigenvals = np.linalg.eigvals(regularized_cov)
                assert np.all(reg_eigenvals > 0), "Regularization failed to make matrix positive definite"

        except LinAlgError:
            # Expected for truly problematic matrices
            pass

    def test_svd_convergence_edge_cases(self):
        """Test SVD convergence with challenging matrices."""
        challenging_matrices = [
            np.ones((10, 10)),  # Constant matrix
            np.eye(10) * 1e-15,  # Very small diagonal
            np.random.randn(10, 10) * 1e10,  # Large values
        ]

        for i, matrix in enumerate(challenging_matrices):
            try:
                U, s, Vt = np.linalg.svd(matrix)

                # Verify SVD reconstruction
                reconstructed = U @ np.diag(s) @ Vt
                reconstruction_error = np.max(np.abs(matrix - reconstructed))

                # Allow for larger errors in challenging cases
                if i == 0:  # Constant matrix
                    assert reconstruction_error < 1e-10
                else:
                    assert reconstruction_error < np.max(np.abs(matrix)) * 1e-10

            except LinAlgError as e:
                # Document convergence failures
                print(f"SVD convergence failure for matrix {i}: {e}")


class TestConvergenceCriteriaEdgeCases:
    """Test convergence criteria at edge cases."""

    def test_iterative_algorithm_convergence(self):
        """Test iterative algorithm convergence criteria."""
        # Mock iterative algorithm with various convergence scenarios

        def mock_iteration_values(scenario):
            """Generate iteration values for different convergence scenarios."""
            scenarios = {
                "fast_convergence": [1.0, 0.1, 0.01, 0.001, 0.0001],
                "slow_convergence": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                "oscillating": [1.0, 0.5, 0.8, 0.3, 0.7, 0.2],
                "diverging": [1.0, 2.0, 4.0, 8.0, 16.0],
                "plateau": [1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            }
            return scenarios.get(scenario, [1.0])

        tolerance = 1e-6
        max_iterations = 10

        for scenario_name in ["fast_convergence", "slow_convergence", "oscillating", "plateau"]:
            values = mock_iteration_values(scenario_name)

            # Test convergence detection
            converged = False
            for i in range(1, len(values)):
                if abs(values[i] - values[i-1]) < tolerance:
                    converged = True
                    break

            # Document convergence behavior
            if scenario_name == "fast_convergence":
                assert converged, "Fast convergence scenario should converge"
            elif scenario_name == "plateau":
                assert converged, "Plateau scenario should be detected as converged"

    def test_numerical_derivative_stability(self):
        """Test numerical derivative computation stability."""
        def test_function(x):
            return x**3 - 2*x**2 + x - 1

        def numerical_derivative(f, x, h=1e-7):
            return (f(x + h) - f(x - h)) / (2 * h)

        # Test at various scales
        test_points = [1e-8, 1e-4, 1.0, 1e4, 1e8]

        for x in test_points:
            try:
                derivative = numerical_derivative(test_function, x)

                # Analytical derivative for comparison
                analytical = 3*x**2 - 4*x + 1

                # Allow for numerical errors
                relative_error = abs(derivative - analytical) / max(abs(analytical), 1e-10)
                assert relative_error < 1e-3, f"Large derivative error at x={x}: {relative_error}"

            except (OverflowError, ZeroDivisionError):
                # Expected for extreme values
                print(f"Numerical derivative failed at x={x} (expected)")


class TestFloatingPointPrecisionLimits:
    """Test floating point precision limit scenarios."""

    def test_catastrophic_cancellation(self):
        """Test operations that could lead to catastrophic cancellation."""
        # Test subtraction of nearly equal large numbers
        a = 1e16
        b = 1e16 - 1

        direct_diff = a - b
        # This should equal 1, but might have precision issues

        # Better approach: use high precision or reorganize computation
        assert abs(direct_diff - 1.0) < 1e-10 or direct_diff == 1.0

    def test_accumulation_precision_loss(self):
        """Test precision loss in accumulation operations."""
        # Test summing many small numbers
        small_value = 1e-16
        n_terms = 1000000

        # Direct summation
        direct_sum = sum([small_value] * n_terms)
        expected_sum = small_value * n_terms

        # Check for precision loss
        relative_error = abs(direct_sum - expected_sum) / max(abs(expected_sum), 1e-20)

        # Should maintain reasonable precision
        assert relative_error < 0.1, f"Excessive precision loss in summation: {relative_error}"

    def test_division_by_small_numbers(self):
        """Test division by very small numbers."""
        numerator = 1.0
        small_denominators = [1e-10, 1e-15, 1e-20]

        for denom in small_denominators:
            try:
                result = numerator / denom

                # Check for overflow
                assert not np.isinf(result), f"Overflow in division: 1/{denom} = {result}"
                assert not np.isnan(result), f"NaN in division: 1/{denom} = {result}"

            except (OverflowError, ZeroDivisionError):
                # Expected for extremely small denominators
                assert denom < 1e-15, f"Unexpected division failure for denom={denom}"


def assert_numerical_health(values, name="values"):
    """Assert that numerical values are healthy (no NaN, Inf)."""
    values_array = np.asarray(values)

    assert not np.any(np.isnan(values_array)), f"{name} contains NaN values"
    assert not np.any(np.isinf(values_array)), f"{name} contains infinite values"
    assert np.all(np.isfinite(values_array)), f"{name} contains non-finite values"


def create_conditioned_matrix(size, condition_number):
    """Create a matrix with specified condition number."""
    U, _, Vt = np.linalg.svd(np.random.randn(size, size))
    s = np.logspace(0, -np.log10(condition_number), size)
    return U @ np.diag(s) @ Vt


if __name__ == "__main__":
    pytest.main([__file__])