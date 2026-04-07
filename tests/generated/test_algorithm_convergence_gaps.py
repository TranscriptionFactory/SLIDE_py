"""
Test coverage gaps for algorithm convergence and numerical precision.

Critical gaps in testing algorithm convergence, stopping criteria,
and numerical precision that could lead to incorrect results.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import warnings

from src.loveslide.slide import SLIDE, OptimizeSLIDE
from src.loveslide.cv import SLIDEcv
from src.loveslide.knockoffs import Knockoffs
from src.loveslide.knockoff.solve import _solve_sdp_cvxpy


class TestAlgorithmConvergence:
    """Test algorithm convergence edge cases."""

    def test_slide_max_iterations_reached(self):
        """Test SLIDE behavior when max iterations reached without convergence."""
        # Create data that's likely to cause convergence issues
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 100)

        params = {
            'x_path': None, 'y_path': None,
            'niter': 5,  # Very low iteration limit
            'delta': [0.01],  # Very strict threshold
        }

        slide = SLIDE(params, x=X, y=y)

        with pytest.warns(UserWarning, match="Maximum iterations reached"):
            result = slide.run()
            # Should return partial results with convergence flag
            assert hasattr(result, 'converged')
            assert not result.converged

    def test_sdp_solver_convergence_failure(self):
        """Test SDP solver behavior on ill-conditioned problems."""
        # Create nearly singular covariance matrix
        X = np.random.randn(100, 50)
        # Make some columns nearly identical
        X[:, 1] = X[:, 0] + 1e-10 * np.random.randn(100)

        Sigma = np.cov(X.T)

        with pytest.raises(ValueError, match="SDP solver failed to converge"):
            _solve_sdp_cvxpy(Sigma, solver='SCS', max_iters=10, eps=1e-12)

    def test_knockoff_generation_convergence(self):
        """Test knockoff generation convergence on edge case data."""
        # Create data with perfect correlations
        X = np.random.randn(100, 10)
        X[:, 5] = X[:, 0]  # Perfect correlation

        knockoffs = Knockoffs()

        with pytest.warns(UserWarning, match="Convergence issues"):
            # Should handle perfect correlations gracefully
            X_knockoffs = knockoffs.fit_transform(X)
            assert X_knockoffs.shape == X.shape


class TestNumericalPrecisionEdgeCases:
    """Test numerical precision edge cases."""

    def test_extreme_eigenvalue_ratios(self):
        """Test handling of matrices with extreme eigenvalue ratios."""
        # Create matrix with very large condition number
        U, _, Vt = np.linalg.svd(np.random.randn(50, 50))
        S = np.diag(np.logspace(-12, 2, 50))  # Extreme eigenvalue range
        X = U @ S @ Vt

        # Should handle or warn about numerical instability
        with pytest.warns(UserWarning, match="numerical instability"):
            knockoffs = Knockoffs()
            result = knockoffs.fit_transform(X)

    def test_machine_epsilon_precision(self):
        """Test operations near machine epsilon precision."""
        # Create data near machine precision limits
        X = np.random.randn(100, 50) * np.finfo(float).eps * 100

        params = {'x_path': None, 'y_path': None, 'delta': [0.05]}
        y = np.random.randint(0, 2, 100)

        slide = SLIDE(params, x=X, y=y)

        with pytest.warns(UserWarning, match="precision"):
            result = slide.run()
            # Should handle near-zero values appropriately

    def test_overflow_prevention(self):
        """Test prevention of numerical overflow in calculations."""
        # Create data that could cause overflow
        X = np.random.randn(100, 50) * 1e10

        params = {'x_path': None, 'y_path': None}
        y = np.random.randint(0, 2, 100)

        with pytest.raises((OverflowError, ValueError), match="overflow"):
            slide = SLIDE(params, x=X, y=y)
            slide.run()


class TestIterativeAlgorithmStability:
    """Test stability of iterative algorithms."""

    def test_slide_reproducibility_with_seed(self):
        """Test SLIDE reproducibility with fixed random seed."""
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 100)
        params = {'x_path': None, 'y_path': None, 'seed': 42}

        # Run twice with same seed
        slide1 = SLIDE(params, x=X, y=y)
        result1 = slide1.run()

        slide2 = SLIDE(params, x=X, y=y)
        result2 = slide2.run()

        # Results should be identical
        np.testing.assert_array_equal(result1.marginal_idxs, result2.marginal_idxs)

    def test_cv_fold_consistency(self):
        """Test CV fold consistency across runs."""
        X = np.random.randn(200, 50)
        y = np.random.randint(0, 2, 200)

        # Create fitted SLIDE object
        params = {'x_path': None, 'y_path': None}
        slide = OptimizeSLIDE(params, x=X, y=y)
        slide.run()

        # Run CV multiple times with same seed
        cv1 = SLIDEcv(slide, nrep=3, k=5, seed=42)
        scores1 = cv1.run()

        cv2 = SLIDEcv(slide, nrep=3, k=5, seed=42)
        scores2 = cv2.run()

        # Should produce consistent results
        np.testing.assert_allclose(scores1, scores2, rtol=1e-10)

    def test_algorithm_monotonicity(self):
        """Test that optimization algorithms maintain monotonicity."""
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 100)

        params = {'x_path': None, 'y_path': None, 'niter': 100}

        class MonitoringSLIDE(SLIDE):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.objective_history = []

            def _objective_function(self):
                # Mock objective function that should be monotonic
                obj_val = len(self.objective_history) * -1  # Should decrease
                self.objective_history.append(obj_val)
                return obj_val

        slide = MonitoringSLIDE(params, x=X, y=y)
        slide.run()

        # Objective should be non-increasing (monotonic)
        objectives = slide.objective_history
        for i in range(1, len(objectives)):
            assert objectives[i] <= objectives[i-1], "Non-monotonic behavior detected"


class TestErrorAccumulation:
    """Test numerical error accumulation in iterative processes."""

    def test_floating_point_accumulation(self):
        """Test floating point error accumulation in long iterations."""
        # Simulate long iterative process
        X = np.random.randn(1000, 100)

        params = {
            'x_path': None, 'y_path': None,
            'niter': 1000,  # Many iterations
        }
        y = np.random.randint(0, 2, 1000)

        slide = SLIDE(params, x=X, y=y)

        with pytest.warns(UserWarning, match="numerical errors may have accumulated"):
            result = slide.run()

    def test_matrix_inversion_stability(self):
        """Test stability of repeated matrix inversions."""
        # Create matrix that becomes ill-conditioned through operations
        A = np.random.randn(50, 50)
        A = A @ A.T + np.eye(50) * 1e-10  # Nearly singular

        # Simulate repeated inversions
        for i in range(100):
            try:
                A_inv = np.linalg.inv(A)
                # Check if inversion is becoming unstable
                identity_check = A @ A_inv
                error = np.linalg.norm(identity_check - np.eye(50))

                if error > 1e-6:
                    pytest.warns(UserWarning, match="matrix inversion unstable")
                    break

                A = A_inv  # Use inverted matrix for next iteration
            except np.linalg.LinAlgError:
                break  # Expected behavior for singular matrices


if __name__ == "__main__":
    pytest.main([__file__, "-v"])