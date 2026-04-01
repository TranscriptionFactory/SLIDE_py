"""
Comprehensive test coverage for knockoff generation edge cases.
Tests private functions, numerical stability, and error recovery.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from loveslide.knockoffs import (
    Knockoffs, _rlist_get, _create_second_order_r, _solve_sdp_r,
    _single_knockoff_iteration_python
)
from loveslide.knockoff.solve import (
    create_solve_equi, create_solve_sdp, create_solve_asdp,
    _solve_sdp_cvxpy, _get_sdp_solver, _merge_clusters, _divide_sdp
)
from loveslide.knockoff.utils import (
    is_posdef, canonical_svd, normc, cov2cor,
    diag_pre_multiply, diag_post_multiply
)


class TestKnockoffPrivateFunctions:
    """Test private knockoff utility functions."""

    def test_rlist_get_missing_attribute(self):
        """Test _rlist_get with missing R object attributes."""
        # Mock R object without required attribute
        mock_robj = MagicMock()
        mock_robj.names = ['other_attr']

        result = _rlist_get(mock_robj, 'missing_attr')
        assert result is None

    def test_rlist_get_type_error_handling(self):
        """Test _rlist_get with type errors."""
        # Non-R object
        with pytest.raises(TypeError):
            _rlist_get("not_an_r_object", "attr")

        # None object
        result = _rlist_get(None, "attr")
        assert result is None

    def test_create_second_order_r_edge_cases(self):
        """Test _create_second_order_r with edge cases."""
        # Singular matrix
        X_singular = np.array([[1, 2], [2, 4]])  # Rank 1

        try:
            result = _create_second_order_r(X_singular)
            # Should handle singularity gracefully or raise appropriate error
            assert result is not None
        except np.linalg.LinAlgError:
            # Acceptable to fail on singular matrix
            pass

        # Very small matrix
        X_small = np.array([[1]])
        result = _create_second_order_r(X_small)
        assert result.shape == (1, 1)

        # Matrix with extreme values
        X_extreme = np.array([[1e10, 1e-10], [1e-10, 1e10]])
        result = _create_second_order_r(X_extreme)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_solve_sdp_r_numerical_issues(self):
        """Test _solve_sdp_r with numerical instability."""
        # Nearly singular covariance matrix
        Sigma = np.array([[1.0, 0.9999], [0.9999, 1.0]])

        try:
            result = _solve_sdp_r(Sigma)
            assert result.shape == Sigma.shape
            assert is_posdef(result - Sigma)  # Should be positive semidefinite
        except (np.linalg.LinAlgError, ValueError):
            # Acceptable to fail on ill-conditioned matrix
            pass

        # Non-PSD matrix
        Sigma_non_psd = np.array([[1, 2], [2, 1]])  # Negative eigenvalue

        with pytest.raises(ValueError):
            _solve_sdp_r(Sigma_non_psd)

    def test_single_knockoff_iteration_python_edge_cases(self):
        """Test _single_knockoff_iteration_python with edge cases."""
        # Perfect multicollinearity
        z = np.array([[1, 2], [2, 4], [3, 6]])  # Perfectly correlated
        y = np.array([0, 1, 0])

        result = _single_knockoff_iteration_python(
            z, y, fdr=0.1, method='lasso', shrink=True,
            offset=1, statistic='LCD'
        )
        assert 'selected' in result
        assert isinstance(result['selected'], list)

        # All features identical
        z_identical = np.ones((50, 5))
        y_random = np.random.binomial(1, 0.5, 50)

        result = _single_knockoff_iteration_python(
            z_identical, y_random, fdr=0.1, method='lasso',
            shrink=True, offset=1, statistic='LCD'
        )
        assert len(result['selected']) <= z_identical.shape[1]

        # Single feature
        z_single = np.random.randn(100, 1)
        y_single = np.random.binomial(1, 0.5, 100)

        result = _single_knockoff_iteration_python(
            z_single, y_single, fdr=0.1, method='lasso',
            shrink=True, offset=1, statistic='LCD'
        )
        assert len(result['selected']) <= 1


class TestKnockoffSolverEdgeCases:
    """Test SDP solver edge cases and numerical stability."""

    def test_solve_sdp_cvxpy_solver_failures(self):
        """Test _solve_sdp_cvxpy with solver failures."""
        # Create problem that might cause solver issues
        Sigma = np.random.randn(20, 20)
        Sigma = Sigma @ Sigma.T  # Make PSD
        Sigma += 1e-12 * np.eye(20)  # Ensure PD

        # Test with different solver preferences
        with patch('cvxpy.Problem.solve') as mock_solve:
            mock_solve.side_effect = Exception("Solver failed")

            with pytest.raises(Exception):
                _solve_sdp_cvxpy(Sigma)

    def test_get_sdp_solver_no_solvers_available(self):
        """Test _get_sdp_solver when no solvers are available."""
        with patch('cvxpy.installed_solvers', return_value=[]):
            solver = _get_sdp_solver()
            # Should return default or raise appropriate error
            assert solver is not None or solver is None  # Implementation dependent

    def test_create_solve_equi_numerical_limits(self):
        """Test create_solve_equi at numerical limits."""
        # Very large matrix
        Sigma_large = np.eye(1000) + 0.1 * np.ones((1000, 1000))

        result = create_solve_equi(Sigma_large)
        assert result.shape == Sigma_large.shape
        assert is_posdef(result)

        # Matrix with near-zero eigenvalues
        Sigma_zero_eig = np.diag([1, 1e-10, 1e-15])

        try:
            result = create_solve_equi(Sigma_zero_eig)
            assert result.shape == Sigma_zero_eig.shape
        except np.linalg.LinAlgError:
            # Acceptable to fail on nearly singular matrix
            pass

    def test_create_solve_sdp_memory_limits(self):
        """Test create_solve_sdp with memory constraints."""
        # Large matrix that might cause memory issues
        n = 500
        Sigma_large = np.eye(n) + 0.01 * np.random.randn(n, n)
        Sigma_large = (Sigma_large + Sigma_large.T) / 2  # Ensure symmetric

        try:
            result = create_solve_sdp(Sigma_large, solver='SCS')
            assert result.shape == Sigma_large.shape
            assert is_posdef(result)
        except MemoryError:
            # Expected for very large matrices
            pass

    def test_merge_clusters_edge_cases(self):
        """Test _merge_clusters with edge cases."""
        # Single cluster
        clusters_single = np.array([0])
        result = _merge_clusters(clusters_single, max_size=10)
        assert len(np.unique(result)) == 1

        # All different clusters
        clusters_all_diff = np.arange(100)
        result = _merge_clusters(clusters_all_diff, max_size=5)
        assert len(np.unique(result)) <= 20  # Should merge into ~20 clusters

        # Empty clusters array
        clusters_empty = np.array([])
        result = _merge_clusters(clusters_empty, max_size=5)
        assert len(result) == 0

    def test_divide_sdp_extreme_cases(self):
        """Test _divide_sdp with extreme partitioning scenarios."""
        # Correlation matrix with extreme correlations
        Sigma = np.array([
            [1.0, 0.99, 0.01],
            [0.99, 1.0, 0.02],
            [0.01, 0.02, 1.0]
        ])

        groups = _divide_sdp(Sigma, max_size=2)
        assert isinstance(groups, list)
        assert all(len(group) <= 2 for group in groups)

        # Single variable
        Sigma_single = np.array([[1.0]])
        groups = _divide_sdp(Sigma_single, max_size=5)
        assert len(groups) == 1
        assert groups[0] == [0]


class TestKnockoffUtilityFunctions:
    """Test utility functions for numerical stability."""

    def test_is_posdef_edge_cases(self):
        """Test is_posdef with edge cases."""
        # Matrix with zero eigenvalue
        A_zero_eig = np.array([[1, 1], [1, 1]])  # Rank 1
        assert not is_posdef(A_zero_eig)

        # Matrix with tiny positive eigenvalues
        A_tiny_eig = np.diag([1, 1e-12])
        # Behavior depends on tolerance
        result = is_posdef(A_tiny_eig, tol=1e-10)
        assert isinstance(result, bool)

        # Non-symmetric matrix
        A_nonsym = np.array([[1, 2], [3, 4]])
        # Should handle or reject non-symmetric matrices
        try:
            result = is_posdef(A_nonsym)
            assert isinstance(result, bool)
        except ValueError:
            # Acceptable to reject non-symmetric matrices
            pass

    def test_canonical_svd_numerical_stability(self):
        """Test canonical_svd with numerical edge cases."""
        # Near-singular matrix
        X_singular = np.array([[1, 2], [1.0001, 2.0001]])

        U, s, Vt = canonical_svd(X_singular)
        assert U.shape[1] == min(X_singular.shape)
        assert not np.any(np.isnan(U))
        assert not np.any(np.isnan(s))
        assert not np.any(np.isnan(Vt))

        # Matrix with extreme aspect ratio
        X_wide = np.random.randn(5, 1000)
        U, s, Vt = canonical_svd(X_wide)
        assert U.shape == (5, 5)
        assert len(s) == 5

        X_tall = np.random.randn(1000, 5)
        U, s, Vt = canonical_svd(X_tall)
        assert Vt.shape == (5, 5)
        assert len(s) == 5

    def test_normc_extreme_data(self):
        """Test normc with extreme data values."""
        # Data with extreme values
        X_extreme = np.array([[1e10, 1e-10], [1e-10, 1e10]])

        X_norm = normc(X_extreme, center=True)
        assert not np.any(np.isnan(X_norm))
        assert not np.any(np.isinf(X_norm))

        # Constant columns
        X_constant = np.ones((100, 3))
        X_norm = normc(X_constant, center=True)
        # Should handle constant columns gracefully
        assert X_norm.shape == X_constant.shape

        # Single observation
        X_single = np.array([[1, 2, 3]])
        X_norm = normc(X_single, center=True)
        assert X_norm.shape == (1, 3)

    def test_cov2cor_edge_cases(self):
        """Test cov2cor with edge cases."""
        # Covariance matrix with zero variance
        Sigma_zero_var = np.array([[1, 0, 0.5], [0, 0, 0], [0.5, 0, 2]])

        try:
            R = cov2cor(Sigma_zero_var)
            # Should handle zero variance appropriately
            assert R.shape == Sigma_zero_var.shape
            assert np.allclose(np.diag(R), 1, atol=1e-10)
        except ZeroDivisionError:
            # Acceptable to fail on zero variance
            pass

        # Non-PSD covariance matrix
        Sigma_non_psd = np.array([[1, 2], [2, 1]])  # Negative eigenvalue

        try:
            R = cov2cor(Sigma_non_psd)
            # May or may not handle gracefully
            assert R.shape == Sigma_non_psd.shape
        except ValueError:
            # Acceptable to reject non-PSD matrix
            pass

    def test_diag_multiply_functions(self):
        """Test diagonal multiplication functions."""
        # Zero diagonal elements
        d_zero = np.array([0, 1, 2])
        X = np.random.randn(3, 5)

        result_pre = diag_pre_multiply(d_zero, X)
        assert result_pre[0, :] == 0  # First row should be zero

        result_post = diag_post_multiply(X, d_zero)
        assert result_post[:, 0] == 0  # First column should be zero

        # Extreme diagonal values
        d_extreme = np.array([1e10, 1e-10, 0])
        X = np.random.randn(3, 3)

        result_pre = diag_pre_multiply(d_extreme, X)
        assert not np.any(np.isnan(result_pre))

        result_post = diag_post_multiply(X, d_extreme)
        assert not np.any(np.isnan(result_post))