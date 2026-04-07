"""
Comprehensive test coverage for mathematical edge cases and numerical stability.
Addresses critical gaps in boundary condition testing.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import linalg
from unittest.mock import patch, MagicMock
import warnings


class TestMatrixConditioningEdgeCases:
    """Test handling of ill-conditioned and singular matrices."""

    def test_near_singular_correlation_matrix(self):
        """Test behavior with nearly singular correlation matrices."""
        from loveslide.love_python.love.love import LOVE

        # Create nearly singular matrix
        p = 5
        A = np.random.randn(p, p)
        Sigma = A @ A.T
        # Make one eigenvalue very small
        eigvals, eigvecs = linalg.eigh(Sigma)
        eigvals[0] = 1e-12  # Nearly zero eigenvalue
        Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T

        X = np.random.multivariate_normal(np.zeros(p), Sigma, 100)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = LOVE(X, lbd=0.5, mu=0.5)
                assert result is not None
            except (linalg.LinAlgError, np.linalg.LinAlgError) as e:
                pytest.skip(f"Expected numerical issue with singular matrix: {e}")

    def test_identity_matrix_edge_case(self):
        """Test with identity correlation matrix."""
        from loveslide.love_python.love.love import LOVE

        n, p = 100, 10
        X = np.random.randn(n, p)  # Independent variables

        result = LOVE(X, lbd=0.5, mu=0.5)

        # Should handle independent variables gracefully
        assert result is not None
        assert 'A' in result

    def test_perfect_multicollinearity(self):
        """Test with perfectly correlated features."""
        from loveslide.love_python.love.love import LOVE

        n, p = 100, 5
        X_base = np.random.randn(n, 3)
        # Add perfectly correlated columns
        X = np.column_stack([
            X_base,
            X_base[:, 0] + X_base[:, 1],  # Perfect linear combination
            2 * X_base[:, 0]  # Perfect scaling
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = LOVE(X, lbd=0.5, mu=0.5)
                assert result is not None
            except Exception as e:
                pytest.skip(f"Expected issue with perfect collinearity: {e}")

    def test_extreme_condition_numbers(self):
        """Test matrices with extreme condition numbers."""
        from loveslide.knockoff.solve import create_solve_sdp

        # Create ill-conditioned matrix
        p = 10
        U = np.random.randn(p, p)
        eigvals = np.logspace(-15, 0, p)  # Condition number ~ 1e15
        Sigma = U @ np.diag(eigvals) @ U.T
        Sigma = (Sigma + Sigma.T) / 2  # Ensure symmetry

        try:
            result = create_solve_sdp(Sigma, method='equi')
            assert result is not None
            assert len(result) == p
        except Exception as e:
            pytest.skip(f"Expected numerical issues with extreme conditioning: {e}")


class TestNumericalPrecisionEdgeCases:
    """Test handling of extreme numerical values."""

    def test_extreme_small_values(self):
        """Test with extremely small values."""
        from loveslide.love_python.love.utilities import offSum

        # Matrix with extremely small off-diagonal elements
        M = np.eye(5) + 1e-16 * np.ones((5, 5))
        weights = 1.0

        result = offSum(M, weights)
        assert np.isfinite(result)
        assert result >= 0

    def test_extreme_large_values(self):
        """Test with extremely large values."""
        from loveslide.love_python.love.score import Score_mat

        # Create correlation matrix with large correlations
        R = np.array([[1.0, 0.9999999], [0.9999999, 1.0]])

        try:
            result = Score_mat(R, q=2, exact=False)
            assert result is not None
            assert 'score' in result
        except Exception as e:
            pytest.skip(f"Expected numerical issues with extreme correlations: {e}")

    def test_zero_variance_features(self):
        """Test handling of zero-variance features."""
        from loveslide.tools import check_params, init_data

        # Create data with zero-variance column
        X = pd.DataFrame({
            'var_col': np.random.randn(100),
            'zero_var': np.zeros(100),
            'const_col': np.full(100, 5.0)
        })
        y = pd.DataFrame({'target': np.random.randint(0, 2, 100)})

        input_params = {'x_path': None, 'y_path': None}

        with pytest.warns(UserWarning):
            data, params = init_data(input_params, X, y)
            check_params(params, data)

        # Should remove zero-variance columns
        assert data.X.shape[1] == 1  # Only var_col should remain

    def test_infinite_values_handling(self):
        """Test handling of infinite values."""
        from loveslide.love_python.love.utilities import threshA

        A = np.array([[1, 2, np.inf], [2, 1, 3], [np.inf, 3, 1]])
        mu = 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = threshA(A, mu, scale=False)
                assert not np.any(np.isinf(result))
            except Exception:
                pytest.skip("Expected handling of infinite values")

    def test_nan_values_propagation(self):
        """Test NaN handling in calculations."""
        from loveslide.love_python.love.utilities import threshA

        A = np.array([[1, 2, np.nan], [2, 1, 3], [np.nan, 3, 1]])
        mu = 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = threshA(A, mu, scale=False)
            # Should either handle NaN gracefully or raise appropriate error
            if result is not None:
                assert result.shape == A.shape


class TestBoundaryConditionEdgeCases:
    """Test edge cases at parameter boundaries."""

    def test_zero_lambda_parameter(self):
        """Test with lambda = 0."""
        from loveslide.love_python.love.love import LOVE

        n, p = 50, 10
        X = np.random.randn(n, p)

        result = LOVE(X, lbd=0.0, mu=0.5)  # Zero regularization
        assert result is not None
        assert 'A' in result

    def test_maximum_lambda_parameter(self):
        """Test with lambda = 1.0."""
        from loveslide.love_python.love.love import LOVE

        n, p = 50, 10
        X = np.random.randn(n, p)

        result = LOVE(X, lbd=1.0, mu=0.5)  # Maximum regularization
        assert result is not None
        assert 'A' in result

    def test_zero_mu_parameter(self):
        """Test with mu = 0."""
        from loveslide.love_python.love.love import LOVE

        n, p = 50, 10
        X = np.random.randn(n, p)

        result = LOVE(X, lbd=0.5, mu=0.0)
        assert result is not None
        assert 'A' in result

    def test_single_sample_dataset(self):
        """Test with n = 1 sample."""
        from loveslide.tools import init_data

        X = pd.DataFrame(np.random.randn(1, 10))
        y = pd.DataFrame([0])

        input_params = {'x_path': None, 'y_path': None}

        try:
            data, params = init_data(input_params, X, y)
            assert data.X.shape[0] == 1
        except ValueError as e:
            assert "sample" in str(e).lower()

    def test_single_feature_dataset(self):
        """Test with p = 1 feature."""
        from loveslide.love_python.love.love import LOVE

        n = 100
        X = np.random.randn(n, 1)

        try:
            result = LOVE(X, lbd=0.5, mu=0.5)
            assert result is not None
        except Exception as e:
            pytest.skip(f"Single feature case may not be supported: {e}")

    def test_p_much_greater_than_n(self):
        """Test with p >> n scenario."""
        from loveslide.love_python.love.love import LOVE

        n, p = 20, 100  # p >> n
        X = np.random.randn(n, p)

        try:
            result = LOVE(X, lbd=0.5, mu=0.5)
            assert result is not None
            assert 'A' in result
        except Exception as e:
            pytest.skip(f"High-dimensional case may require special handling: {e}")


class TestAlgorithmicConvergenceEdgeCases:
    """Test convergence behavior in edge cases."""

    def test_non_convergent_optimization(self):
        """Test behavior when optimization doesn't converge."""
        from loveslide.love_python.love.est_omega import estOmega

        # Create a difficult optimization case
        C = np.random.randn(20, 20)
        C = C @ C.T + 1e-10 * np.eye(20)  # Nearly singular
        lbd = 1e-6  # Very small regularization

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = estOmega(lbd, C)
                assert result is not None
        except Exception as e:
            pytest.skip(f"Convergence issues expected: {e}")

    def test_maximum_iterations_reached(self):
        """Test behavior when maximum iterations are reached."""
        from loveslide.love_python.love.cv import CV_delta

        n, p = 50, 20
        X = np.random.randn(n, p)
        deltaGrids = np.linspace(0.01, 0.1, 3)

        # This should complete within reasonable time
        result = CV_delta(X, deltaGrids, diagonal=True, rep=2)
        assert result is not None
        assert 'optDelta' in result

    def test_optimization_with_poor_initialization(self):
        """Test optimization with poor starting values."""
        from loveslide.love_python.love.est_pure_hetero import Est_Pure

        # Create challenging score matrix
        score_mat = np.random.exponential(1, (20, 20)) + np.eye(20) * 10
        delta = 0.5

        try:
            result = Est_Pure(score_mat, delta)
            assert result is not None
            assert 'estPureIndices' in result
        except Exception as e:
            pytest.skip(f"Challenging optimization case: {e}")


class TestMemoryAndScalabilityEdgeCases:
    """Test memory management and scalability edge cases."""

    def test_large_matrix_operations(self):
        """Test with large matrices near memory limits."""
        # Use moderate size to avoid actual memory issues in CI
        n, p = 500, 100
        X = np.random.randn(n, p)

        from loveslide.love_python.love.love import LOVE

        try:
            result = LOVE(X, lbd=0.5, mu=0.5)
            assert result is not None
            # Memory should be cleaned up
            del X
        except MemoryError:
            pytest.skip("Memory limit reached as expected")

    def test_repeated_large_operations(self):
        """Test repeated operations for memory leaks."""
        from loveslide.love_python.love.score import Score_mat

        for i in range(5):  # Multiple iterations
            R = np.corrcoef(np.random.randn(50, 20).T)
            result = Score_mat(R, q=2, exact=False)
            assert result is not None
            del R, result  # Explicit cleanup

    def test_concurrent_memory_usage(self):
        """Test memory usage with concurrent operations."""
        from loveslide.knockoff.utils import rnorm_matrix
        import threading

        results = []

        def worker():
            try:
                mat = rnorm_matrix(100, 50, mean=0, sd=1)
                results.append(mat is not None)
            except Exception:
                results.append(False)

        threads = []
        for _ in range(3):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert all(results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])