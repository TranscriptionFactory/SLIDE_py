"""
Comprehensive test coverage for knockoff statistics and utilities.

Missing Coverage Areas:
- knockoff/utils.py: diag_pre_multiply, diag_post_multiply, canonical_svd, normc, cov2cor
- knockoff/stats/*.py: Various statistic computation functions
- knockoff/_parallel.py: Parallel processing functions
- knockoff/create.py: Advanced knockoff creation methods
"""
import pytest
import numpy as np
from unittest.mock import patch, Mock
from concurrent.futures import Future
import multiprocessing

from loveslide.knockoff.utils import (
    diag_pre_multiply, diag_post_multiply, is_posdef, canonical_svd,
    normc, cov2cor, rnorm_matrix, random_problem, with_seed
)
from loveslide.knockoff.stats.base import (
    swap_columns, correct_for_swap, compute_difference_stat,
    compute_signed_max_stat, standardize
)
from loveslide.knockoff.stats.forward import stat_forward_selection
from loveslide.knockoff.stats.stability import stat_stability_selection
from loveslide.knockoff.stats.sqrt_lasso import stat_sqrt_lasso
from loveslide.knockoff._parallel import (
    _precompute_knockoff_params, _single_knockoff_iteration,
    knockoff_voting_parallel
)
from loveslide.knockoff.create import _decompose, _create_equicorrelated, _create_sdp


class TestUtilityFunctions:
    """Test utility functions comprehensively."""

    def test_diag_pre_multiply_basic(self):
        """Test diagonal pre-multiplication."""
        d = np.array([2, 3, 4])
        X = np.random.randn(3, 5)

        result = diag_pre_multiply(d, X)

        expected = np.diag(d) @ X
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_diag_pre_multiply_broadcasting(self):
        """Test diagonal pre-multiplication with broadcasting."""
        d = np.array([0.5, 2.0])
        X = np.array([[1, 2, 3], [4, 5, 6]])

        result = diag_pre_multiply(d, X)

        assert result.shape == X.shape
        np.testing.assert_allclose(result[0, :], X[0, :] * 0.5)
        np.testing.assert_allclose(result[1, :], X[1, :] * 2.0)

    def test_diag_post_multiply_basic(self):
        """Test diagonal post-multiplication."""
        X = np.random.randn(4, 3)
        d = np.array([2, 3, 4])

        result = diag_post_multiply(X, d)

        expected = X @ np.diag(d)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_diag_post_multiply_edge_cases(self):
        """Test diagonal post-multiplication edge cases."""
        # Zero diagonal element
        X = np.array([[1, 2], [3, 4]])
        d = np.array([0, 1])

        result = diag_post_multiply(X, d)

        assert result[0, 0] == 0
        assert result[1, 0] == 0
        assert result[0, 1] == 2
        assert result[1, 1] == 4

    def test_is_posdef_positive_definite(self):
        """Test is_posdef with positive definite matrix."""
        # Create known positive definite matrix
        A = np.array([[2, 1], [1, 2]])
        assert is_posdef(A) == True

    def test_is_posdef_negative_definite(self):
        """Test is_posdef with negative definite matrix."""
        A = np.array([[-2, -1], [-1, -2]])
        assert is_posdef(A) == False

    def test_is_posdef_singular(self):
        """Test is_posdef with singular matrix."""
        A = np.array([[1, 1], [1, 1]])  # Singular
        assert is_posdef(A) == False

    def test_is_posdef_tolerance(self):
        """Test is_posdef with different tolerance levels."""
        # Nearly singular matrix
        A = np.array([[1, 1-1e-8], [1-1e-8, 1]])

        assert is_posdef(A, tol=1e-10) == False
        assert is_posdef(A, tol=1e-6) == True

    def test_canonical_svd_basic(self):
        """Test canonical SVD decomposition."""
        np.random.seed(42)
        X = np.random.randn(10, 5)

        U, s, Vt = canonical_svd(X)

        # Verify SVD properties
        assert U.shape[1] == min(X.shape)
        assert len(s) == min(X.shape)
        assert Vt.shape[0] == min(X.shape)

        # Verify reconstruction
        X_reconstructed = U @ np.diag(s) @ Vt
        np.testing.assert_allclose(X, X_reconstructed, rtol=1e-10)

    def test_canonical_svd_tall_matrix(self):
        """Test canonical SVD with tall matrix."""
        X = np.random.randn(20, 3)
        U, s, Vt = canonical_svd(X)

        assert U.shape == (20, 3)
        assert len(s) == 3
        assert Vt.shape == (3, 3)

    def test_canonical_svd_wide_matrix(self):
        """Test canonical SVD with wide matrix."""
        X = np.random.randn(3, 20)
        U, s, Vt = canonical_svd(X)

        assert U.shape == (3, 3)
        assert len(s) == 3
        assert Vt.shape == (3, 20)

    def test_normc_centering_and_scaling(self):
        """Test normc function for normalization."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

        result = normc(X, center=True)

        # Check centering
        np.testing.assert_allclose(np.mean(result, axis=0), 0, atol=1e-10)

        # Check scaling
        np.testing.assert_allclose(np.std(result, axis=0), 1, rtol=1e-10)

    def test_normc_no_centering(self):
        """Test normc without centering."""
        X = np.array([[1, 2], [3, 4]], dtype=float)

        result = normc(X, center=False)

        # Should only scale, not center
        assert not np.allclose(np.mean(result, axis=0), 0)
        np.testing.assert_allclose(np.var(result, axis=0), 1, rtol=1e-10)

    def test_cov2cor_basic(self):
        """Test covariance to correlation conversion."""
        # Create covariance matrix
        Sigma = np.array([[4, 2], [2, 9]])

        R = cov2cor(Sigma)

        # Check diagonal is ones
        np.testing.assert_allclose(np.diag(R), 1, rtol=1e-10)

        # Check correlation calculation
        expected_cor = 2 / np.sqrt(4 * 9)
        assert abs(R[0, 1] - expected_cor) < 1e-10

    def test_cov2cor_identity_matrix(self):
        """Test cov2cor with identity matrix."""
        I = np.eye(3)
        R = cov2cor(I)

        np.testing.assert_allclose(R, I)

    def test_rnorm_matrix_basic(self):
        """Test random normal matrix generation."""
        np.random.seed(42)
        n, p = 10, 5
        mean, sd = 2.0, 0.5

        X = rnorm_matrix(n, p, mean=mean, sd=sd)

        assert X.shape == (n, p)
        assert abs(np.mean(X) - mean) < 0.2  # Allow some variance
        assert abs(np.std(X) - sd) < 0.2

    def test_random_problem_generation(self):
        """Test random problem generation for testing."""
        np.random.seed(42)
        n, p = 50, 20
        k = 5
        amplitude = 2.0

        X, y, beta, nonzero = random_problem(n, p, k, amplitude=amplitude)

        assert X.shape == (n, p)
        assert len(y) == n
        assert len(beta) == p
        assert len(nonzero) == k

        # Check that nonzero coefficients have correct amplitude
        assert np.all(np.abs(beta[nonzero]) >= amplitude)

    def test_with_seed_deterministic(self):
        """Test with_seed function for deterministic execution."""
        def random_func():
            return np.random.randn(5)

        # Same seed should give same result
        result1 = with_seed(42, random_func)
        result2 = with_seed(42, random_func)

        np.testing.assert_allclose(result1, result2)

    def test_with_seed_different_seeds(self):
        """Test with_seed with different seeds."""
        def random_func():
            return np.random.randn(5)

        result1 = with_seed(42, random_func)
        result2 = with_seed(43, random_func)

        assert not np.allclose(result1, result2)


class TestStatisticsBase:
    """Test base statistics functions."""

    def test_swap_columns_basic(self):
        """Test column swapping functionality."""
        X = np.array([[1, 2], [3, 4]])
        Xk = np.array([[5, 6], [7, 8]])

        X_swapped, Xk_swapped, swap = swap_columns(X, Xk, p=0.5)

        assert X_swapped.shape == X.shape
        assert Xk_swapped.shape == Xk.shape
        assert len(swap) == X.shape[1]

    def test_correct_for_swap_basic(self):
        """Test swap correction for statistics."""
        W = np.array([1.0, -2.0, 3.0])
        swap = np.array([False, True, False])

        W_corrected = correct_for_swap(W, swap)

        expected = np.array([1.0, 2.0, 3.0])  # Second element flipped
        np.testing.assert_array_equal(W_corrected, expected)

    def test_compute_difference_stat_basic(self):
        """Test difference statistic computation."""
        Z_orig = np.array([1.0, 2.0, -1.0])
        Z_ko = np.array([0.5, 1.5, -0.8])

        W = compute_difference_stat(Z_orig, Z_ko)

        expected = Z_orig - Z_ko
        np.testing.assert_allclose(W, expected)

    def test_compute_signed_max_stat_basic(self):
        """Test signed max statistic computation."""
        Z_orig = np.array([2.0, -1.0, 3.0])
        Z_ko = np.array([1.0, -2.0, 1.0])

        W = compute_signed_max_stat(Z_orig, Z_ko)

        # Should take max of absolute values with appropriate sign
        expected = np.array([2.0, 1.0, 3.0])  # Sign preserved from larger absolute value
        np.testing.assert_allclose(W, expected)

    def test_standardize_basic(self):
        """Test standardization function."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

        X_std = standardize(X)

        # Check mean is approximately zero
        np.testing.assert_allclose(np.mean(X_std, axis=0), 0, atol=1e-10)

        # Check std is approximately one
        np.testing.assert_allclose(np.std(X_std, axis=0, ddof=1), 1, rtol=1e-10)


class TestAdvancedStatistics:
    """Test advanced statistics methods."""

    def test_stat_forward_selection_basic(self):
        """Test forward selection statistic."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)
        Xk = np.random.randn(100, 20)

        W = stat_forward_selection(X, Xk, y)

        assert len(W) == 20
        assert np.all(np.isfinite(W))

    def test_stat_stability_selection_basic(self):
        """Test stability selection statistic."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)
        Xk = np.random.randn(50, 10)

        W = stat_stability_selection(X, Xk, y, n_bootstrap=10)

        assert len(W) == 10
        assert np.all(np.isfinite(W))

    def test_stat_sqrt_lasso_basic(self):
        """Test square-root lasso statistic."""
        np.random.seed(42)
        X = np.random.randn(60, 15)
        y = np.random.randn(60)
        Xk = np.random.randn(60, 15)

        W = stat_sqrt_lasso(X, Xk, y)

        assert len(W) == 15
        assert np.all(np.isfinite(W))


class TestParallelProcessing:
    """Test parallel processing functions."""

    def test_precompute_knockoff_params_basic(self):
        """Test knockoff parameter precomputation."""
        np.random.seed(42)
        z = np.random.randn(100, 20)

        params = _precompute_knockoff_params(z, method='equicorrelated')

        assert isinstance(params, dict)
        assert 'Sigma' in params

    def test_single_knockoff_iteration_basic(self):
        """Test single knockoff iteration."""
        np.random.seed(42)
        z = np.random.randn(50, 10)
        y = np.random.randn(50)

        params = {
            'method': 'equicorrelated',
            'fdr': 0.1,
            'offset': 1,
            'shrink': False
        }

        result = _single_knockoff_iteration(z, y, params, statistic='lasso_lambdadiff')

        assert 'W' in result
        assert 'selected' in result
        assert len(result['W']) == 10

    @patch('loveslide.knockoff._parallel.ProcessPoolExecutor')
    def test_knockoff_voting_parallel_mock(self, mock_executor):
        """Test parallel knockoff voting with mocked executor."""
        # Mock the executor and futures
        mock_future = Mock(spec=Future)
        mock_future.result.return_value = {
            'W': np.array([1.0, -0.5, 2.0]),
            'selected': np.array([0, 2]),
            'threshold': 1.0
        }

        mock_executor_instance = Mock()
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.submit.return_value = mock_future
        mock_executor.return_value = mock_executor_instance

        np.random.seed(42)
        z = np.random.randn(30, 3)
        y = np.random.randn(30)

        result = knockoff_voting_parallel(
            z, y,
            method='equicorrelated',
            fdr=0.1,
            n_knockoffs=2,
            statistic='lasso_lambdadiff',
            n_jobs=1
        )

        assert hasattr(result, 'selected')
        assert hasattr(result, 'W_votes')


class TestKnockoffCreation:
    """Test knockoff creation methods."""

    def test_decompose_basic(self):
        """Test matrix decomposition for knockoffs."""
        np.random.seed(42)
        X = np.random.randn(100, 10)

        decomp = _decompose(X)

        assert 'Sigma' in decomp
        assert 'Sigma_inv_one_half' in decomp
        assert decomp['Sigma'].shape == (10, 10)

    def test_create_equicorrelated_basic(self):
        """Test equicorrelated knockoff creation."""
        np.random.seed(42)
        X = np.random.randn(50, 8)

        Xk = _create_equicorrelated(X)

        assert Xk.shape == X.shape
        assert not np.allclose(X, Xk)  # Should be different

    def test_create_sdp_basic(self):
        """Test SDP knockoff creation."""
        np.random.seed(42)
        X = np.random.randn(40, 6)

        # Make correlation matrix well-conditioned
        X = X / np.std(X, axis=0)
        X = X - np.mean(X, axis=0)

        Xk = _create_sdp(X)

        assert Xk.shape == X.shape


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_utilities_with_invalid_inputs(self):
        """Test utility functions with invalid inputs."""
        # Test with mismatched dimensions
        with pytest.raises((ValueError, IndexError)):
            diag_pre_multiply(np.array([1, 2]), np.array([[1, 2, 3]]))

    def test_statistics_with_degenerate_cases(self):
        """Test statistics with degenerate input cases."""
        # Constant features
        X_constant = np.ones((50, 5))
        y = np.random.randn(50)
        Xk = np.random.randn(50, 5)

        # Should handle gracefully without crashing
        try:
            W = stat_forward_selection(X_constant, Xk, y)
            assert len(W) == 5
        except (np.linalg.LinAlgError, ValueError):
            # These exceptions are acceptable for degenerate cases
            pass

    def test_knockoff_creation_with_rank_deficient_matrix(self):
        """Test knockoff creation with rank-deficient input."""
        # Create rank-deficient matrix
        X = np.random.randn(30, 5)
        X[:, 4] = X[:, 0] + X[:, 1]  # Linear dependence

        # Should handle gracefully
        with pytest.warns(UserWarning):
            result = _decompose(X)
            assert 'Sigma' in result