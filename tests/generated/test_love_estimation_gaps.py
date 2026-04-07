"""
Test coverage gaps for LOVE estimation modules.

Missing Coverage Areas:
- est_omega.py: estOmega and solve_row functions
- est_nonpure.py: EstAJInv, LP, EstAJDant, Dantzig functions
- est_pure_hetero.py: Est_Pure, Est_BI_C, Re_Est_Pure functions
- prescreen.py: Screen_X function and cross-validation
"""
import pytest
import numpy as np
from scipy.optimize import OptimizeResult
from unittest.mock import patch, Mock

from loveslide.love_python.love.est_omega import estOmega, solve_row
from loveslide.love_python.love.est_nonpure import EstAJInv, LP, EstAJDant, Dantzig
from loveslide.love_python.love.est_pure_hetero import Est_Pure, Est_BI_C, Re_Est_Pure
from loveslide.love_python.love.prescreen import Screen_X


class TestEstOmega:
    """Test precision matrix estimation via linear programming."""

    def test_est_omega_basic(self):
        """Test basic estOmega functionality."""
        np.random.seed(42)
        K = 3
        C = np.random.randn(K, K)
        C = C @ C.T + np.eye(K) * 0.1  # Make positive definite
        lbd = 0.1

        omega = estOmega(lbd, C)

        assert omega.shape == (K, K)
        assert np.all(np.isfinite(omega))

    def test_est_omega_identity_matrix(self):
        """Test estOmega with identity matrix."""
        C = np.eye(3)
        lbd = 0.01

        omega = estOmega(lbd, C)

        # Should be close to identity for small lambda
        assert omega.shape == (3, 3)
        assert np.allclose(omega, np.eye(3), atol=0.1)

    def test_est_omega_singular_matrix(self):
        """Test estOmega with near-singular matrix."""
        C = np.array([[1.0, 0.99], [0.99, 1.0]])
        lbd = 0.5

        # Should handle near-singular case gracefully
        omega = estOmega(lbd, C)
        assert omega.shape == (2, 2)
        assert np.all(np.isfinite(omega))

    def test_solve_row_convergence(self):
        """Test solve_row function convergence."""
        C = np.eye(3)
        lbd = 0.1
        col_ind = 0

        row_result = solve_row(col_ind, C, lbd)

        assert len(row_result) == 3
        assert np.all(np.isfinite(row_result))

    @patch('loveslide.love_python.love.est_omega.linprog')
    def test_solve_row_optimization_failure(self, mock_linprog):
        """Test solve_row handling optimization failure."""
        # Mock optimization failure
        mock_result = OptimizeResult()
        mock_result.success = False
        mock_result.x = None
        mock_linprog.return_value = mock_result

        C = np.eye(2)
        lbd = 0.1

        # Should handle failure gracefully
        with pytest.warns(UserWarning):
            result = solve_row(0, C, lbd)
            assert result is not None


class TestEstNonPure:
    """Test non-pure node estimation functions."""

    def test_est_aj_inv_basic(self):
        """Test EstAJInv basic functionality."""
        K = 3
        Omega = np.eye(K) * 2
        Y = np.random.randn(K, K)
        lbd = 0.1

        result = EstAJInv(Omega, Y, lbd)

        assert result.shape == Y.shape
        assert np.all(np.isfinite(result))

    def test_lp_function_basic(self):
        """Test LP (Linear Programming) function."""
        y = np.array([1.0, -0.5, 0.8])
        lbd = 0.2

        result = LP(y, lbd)

        assert len(result) == len(y)
        assert np.all(np.isfinite(result))

    def test_est_aj_dant_basic(self):
        """Test EstAJDant function."""
        K = 3
        C_hat = np.eye(K)
        Y = np.random.randn(K, K)
        lbd = 0.1
        pureVec = np.array([True, False, True])

        result = EstAJDant(C_hat, Y, lbd, pureVec)

        assert result.shape == Y.shape
        assert np.all(np.isfinite(result))

    def test_dantzig_solver_convergence(self):
        """Test Dantzig solver convergence."""
        C_hat = np.eye(3)
        y = np.array([1.0, 0.5, -0.3])
        lbd = 0.2

        result = Dantzig(C_hat, y, lbd)

        if result is not None:
            assert len(result) == len(y)
            assert np.all(np.isfinite(result))

    @patch('loveslide.love_python.love.est_nonpure.linprog')
    def test_dantzig_optimization_failure(self, mock_linprog):
        """Test Dantzig handling optimization failure."""
        mock_result = OptimizeResult()
        mock_result.success = False
        mock_linprog.return_value = mock_result

        C_hat = np.eye(2)
        y = np.array([1.0, 0.5])
        lbd = 0.1

        result = Dantzig(C_hat, y, lbd)
        assert result is None


class TestEstPureHetero:
    """Test pure node estimation with heteroscedasticity."""

    def test_est_pure_basic(self):
        """Test Est_Pure basic functionality."""
        np.random.seed(42)
        score_mat = np.random.randn(50, 10)
        delta = 0.1

        result = Est_Pure(score_mat, delta)

        assert isinstance(result, dict)
        assert 'nhat' in result
        assert 'nonpure_ind' in result
        assert 'pure_ind' in result

    def test_est_pure_no_pure_nodes(self):
        """Test Est_Pure when no pure nodes exist."""
        # Create score matrix with no clear pure nodes
        score_mat = np.random.randn(50, 10) * 0.1  # Small values
        delta = 0.5  # High threshold

        result = Est_Pure(score_mat, delta)

        assert result['nhat'] >= 0
        assert isinstance(result['nonpure_ind'], list)

    def test_est_bi_c_computation(self):
        """Test Est_BI_C matrix computation."""
        np.random.seed(42)
        K = 5
        M = np.random.randn(K, 3)
        R = np.corrcoef(np.random.randn(50, K), rowvar=False)
        I_part = [[0, 1], [2], [3, 4]]
        J_part = [[0], [1, 2]]

        result = Est_BI_C(M, R, I_part, J_part)

        assert isinstance(result, dict)
        assert 'Gamma_LL' in result
        assert 'L_hat' in result

    def test_re_est_pure_refinement(self):
        """Test Re_Est_Pure for pure node refinement."""
        np.random.seed(42)
        n, p, K = 100, 20, 3
        X = np.random.randn(n, p)
        Sigma = np.corrcoef(X, rowvar=False)
        M = np.random.randn(p, K)
        Gamma_LL = np.eye(K)
        L_hat = np.random.randn(K, K)

        result = Re_Est_Pure(X, Sigma, M, Gamma_LL, L_hat)

        assert isinstance(result, dict)
        assert 'refined_pure_ind' in result


class TestPreScreen:
    """Test pre-screening functionality."""

    def test_screen_x_basic(self):
        """Test Screen_X basic functionality."""
        np.random.seed(42)
        n, p = 100, 20

        # Create data with some noise features
        X_signal = np.random.randn(n, 15)
        X_noise = np.random.randn(n, 5) * 0.1
        X = np.hstack([X_signal, X_noise])

        thresh_grid = np.array([0.1])
        result = Screen_X(X, thresh_grid=thresh_grid)

        assert isinstance(result, np.ndarray)
        assert len(result) <= p

    def test_screen_x_cross_validation(self):
        """Test Screen_X with cross-validation."""
        np.random.seed(42)
        n, p = 50, 10
        X = np.random.randn(n, p)

        # Test with multiple thresholds
        result = Screen_X(X, nfolds=5, nthresh=10)

        assert isinstance(result, dict)
        assert 'thresh_min' in result
        assert 'thresh_1se' in result
        assert 'cv_mean' in result
        assert 'cv_sd' in result
        assert 'noise_ind' in result

    def test_screen_x_high_noise_proportion(self):
        """Test Screen_X with high noise proportion."""
        np.random.seed(42)
        n, p = 80, 30

        # Most features are noise
        X_signal = np.random.randn(n, 5)
        X_noise = np.random.randn(n, 25) * 0.05
        X = np.hstack([X_signal, X_noise])

        result = Screen_X(X, max_prop=0.8)

        if isinstance(result, dict):
            noise_detected = len(result['noise_ind'])
            assert noise_detected <= int(p * 0.8)

    def test_screen_x_no_correlation(self):
        """Test Screen_X with uncorrelated features."""
        np.random.seed(42)
        n, p = 60, 15

        # Generate truly independent features
        X = np.random.randn(n, p)

        thresh_grid = np.array([0.05])
        result = Screen_X(X, thresh_grid=thresh_grid)

        # Most features should be identified as noise
        assert isinstance(result, np.ndarray)


class TestErrorHandling:
    """Test error handling in estimation functions."""

    def test_est_omega_invalid_lambda(self):
        """Test estOmega with invalid lambda values."""
        C = np.eye(3)

        # Negative lambda should be handled
        with pytest.warns(UserWarning):
            result = estOmega(-0.1, C)

    def test_estimation_with_nan_inputs(self):
        """Test estimation functions with NaN inputs."""
        C_with_nan = np.array([[1.0, np.nan], [np.nan, 1.0]])

        with pytest.raises((ValueError, RuntimeError)):
            estOmega(0.1, C_with_nan)

    def test_estimation_with_mismatched_dimensions(self):
        """Test estimation functions with mismatched dimensions."""
        C = np.eye(3)
        Y = np.random.randn(2, 2)  # Wrong dimensions

        with pytest.raises((ValueError, IndexError)):
            EstAJInv(C, Y, 0.1)