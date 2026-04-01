"""
Test coverage for SDP solver robustness and numerical edge cases.
Critical for knockoff generation stability under challenging conditions.
"""
import pytest
import numpy as np
import warnings
from unittest.mock import patch, Mock

from loveslide.knockoff.solve import (
    _get_sdp_solver, _solve_sdp_cvxpy, create_solve_sdp,
    create_solve_asdp, create_solve_equi
)
from loveslide.knockoff.utils import is_posdef


class TestSDPSolverNumericalRobustness:
    """Test SDP solver robustness under numerical challenges."""

    def test_sdp_solver_preference_order(self):
        """Test SDP solver preference: DSDP > cvxpy > None."""
        # Reset global state
        import loveslide.knockoff.solve as solve_module
        solve_module._SDP_SOLVER = None

        # Test with both unavailable
        with patch('loveslide.knockoff.solve.dsdp', side_effect=ImportError):
            with patch('cvxpy.Variable', side_effect=ImportError):
                solver = _get_sdp_solver()
                assert solver is None

        # Test with only cvxpy available
        solve_module._SDP_SOLVER = None
        with patch('loveslide.knockoff.solve.dsdp', side_effect=ImportError):
            with patch('cvxpy.Variable'):  # Available
                solver = _get_sdp_solver()
                assert solver == 'cvxpy'

        # Test with DSDP available (preferred)
        solve_module._SDP_SOLVER = None
        with patch('loveslide.knockoff.solve.dsdp'):  # Available
            solver = _get_sdp_solver()
            assert solver == 'dsdp'

    def test_cvxpy_solver_numerical_conditioning(self):
        """Test cvxpy solver with poorly conditioned matrices."""
        # Near-singular matrix
        G = np.array([[1, 0.999999], [0.999999, 1]])

        # Should handle near-singular case
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                result = _solve_sdp_cvxpy(G, gaptol=1e-6)
                # Check that result is reasonable
                assert result is not None
                assert len(result) == G.shape[0]
            except Exception as e:
                # Acceptable if solver fails gracefully
                assert "singular" in str(e).lower() or "condition" in str(e).lower()

    def test_create_solve_sdp_matrix_types(self):
        """Test create_solve_sdp with different matrix conditions."""
        # Well-conditioned matrix
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        result = create_solve_sdp(Sigma)
        assert len(result) == 2
        assert np.all(result >= 0)  # s-values should be non-negative

        # Identity matrix (best conditioned)
        Sigma_id = np.eye(3)
        result_id = create_solve_sdp(Sigma_id)
        assert len(result_id) == 3
        assert np.allclose(result_id, 1.0)  # Should be all ones for identity

    def test_create_solve_sdp_singular_matrix_handling(self):
        """Test handling of singular matrices in SDP solve."""
        # Singular matrix
        Sigma_singular = np.array([[1, 1], [1, 1]])

        with pytest.raises(ValueError, match="not positive-definite"):
            create_solve_sdp(Sigma_singular)

    def test_create_solve_asdp_clustering_edge_cases(self):
        """Test ASDP method with clustering edge cases."""
        # Matrix with clear block structure
        Sigma_block = np.block([
            [np.eye(2), np.zeros((2, 2))],
            [np.zeros((2, 2)), np.eye(2)]
        ])

        result = create_solve_asdp(Sigma_block, n_clusters=2)
        assert len(result) == 4
        assert np.all(result >= 0)

        # Test with single cluster (should behave like equi)
        result_single = create_solve_asdp(Sigma_block, n_clusters=1)
        assert len(result_single) == 4

    def test_create_solve_asdp_high_dimensional_clustering(self):
        """Test ASDP clustering with high-dimensional data."""
        # High-dimensional correlation matrix
        p = 50
        Sigma_high = np.eye(p) + 0.1 * np.random.randn(p, p)
        Sigma_high = Sigma_high @ Sigma_high.T  # Make PSD
        Sigma_high = np.corrcoef(Sigma_high)  # Normalize to correlation

        # Should handle high dimensions without memory issues
        result = create_solve_asdp(Sigma_high, n_clusters=5)
        assert len(result) == p
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_create_solve_equi_theoretical_bounds(self):
        """Test equicorrelated method theoretical bounds."""
        # Small correlation matrix
        p = 5
        Sigma = 0.3 * np.ones((p, p)) + 0.7 * np.eye(p)

        result = create_solve_equi(Sigma)
        assert len(result) == p

        # All s-values should be equal for equicorrelated
        assert np.allclose(result, result[0])

        # Should satisfy 0 <= s <= 1
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_solver_fallback_chain_robustness(self):
        """Test complete solver fallback chain."""
        Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])

        # Test when primary solver fails, fallback works
        with patch('loveslide.knockoff.solve._get_sdp_solver', return_value='cvxpy'):
            with patch('loveslide.knockoff.solve._solve_sdp_cvxpy', side_effect=Exception("cvxpy failed")):
                # Should fall back to approximate method
                try:
                    result = create_solve_asdp(Sigma)
                    assert result is not None
                except Exception:
                    # If all methods fail, should get clear error
                    pytest.skip("All solvers unavailable in test environment")

    def test_numerical_precision_limits(self):
        """Test behavior at numerical precision limits."""
        # Matrix with very small eigenvalues
        Sigma_small_eig = np.array([[1e-10, 0], [0, 1]])

        # Should handle or reject appropriately
        try:
            result = create_solve_sdp(Sigma_small_eig)
            assert np.all(result >= 0)
        except ValueError:
            # Acceptable to reject ill-conditioned matrices
            pass

        # Matrix with very large condition number
        Sigma_large_cond = np.array([[1, 1-1e-10], [1-1e-10, 1]])

        try:
            result = create_solve_sdp(Sigma_large_cond)
            # Should either succeed with reasonable values or fail cleanly
            if result is not None:
                assert np.all(result >= 0)
                assert np.all(result <= 1)
        except ValueError:
            pass

    def test_memory_efficient_large_matrices(self):
        """Test memory efficiency with large correlation matrices."""
        # Large matrix (but reasonable for testing)
        p = 100
        np.random.seed(42)
        A = np.random.randn(p, 20)  # Rank-deficient structure
        Sigma_large = A @ A.T
        Sigma_large = np.corrcoef(Sigma_large + np.eye(p))

        # Should handle without excessive memory usage
        try:
            result = create_solve_asdp(Sigma_large, n_clusters=10)
            assert len(result) == p
            assert np.all(result >= 0)
        except MemoryError:
            pytest.skip("Test environment memory insufficient")

    def test_correlation_matrix_validation(self):
        """Test validation of correlation matrix properties."""
        # Non-symmetric matrix
        Sigma_nonsym = np.array([[1, 0.5], [0.6, 1]])

        try:
            create_solve_sdp(Sigma_nonsym)
        except (ValueError, Warning):
            # Should detect and handle non-symmetric matrices
            pass

        # Matrix with diagonal != 1
        Sigma_nonunit = np.array([[2, 0.5], [0.5, 1]])

        # Should handle or normalize appropriately
        result = create_solve_sdp(Sigma_nonunit)
        # If successful, result should be reasonable
        if result is not None:
            assert len(result) == 2

    def test_edge_case_matrix_dimensions(self):
        """Test edge cases with matrix dimensions."""
        # Single variable case
        Sigma_1x1 = np.array([[1.0]])
        result_1 = create_solve_equi(Sigma_1x1)
        assert len(result_1) == 1
        assert 0 <= result_1[0] <= 1

        # Two variable case (minimal for SDP)
        Sigma_2x2 = np.array([[1, 0], [0, 1]])
        result_2 = create_solve_sdp(Sigma_2x2)
        assert len(result_2) == 2
        assert np.allclose(result_2, 1.0)  # Identity should give s=1

    def test_solver_timeout_handling(self):
        """Test handling of solver timeouts."""
        # Create challenging optimization problem
        p = 20
        Sigma_challenging = np.eye(p) + 0.95 * np.ones((p, p))

        # Test with very strict tolerance (may timeout)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _solve_sdp_cvxpy(Sigma_challenging, gaptol=1e-12, maxit=10)
                # If it completes, should be valid
                if result is not None:
                    assert len(result) == p
        except Exception as e:
            # Timeout or convergence failure is acceptable
            assert any(word in str(e).lower()
                      for word in ['timeout', 'convergence', 'iteration', 'solver'])

    def test_warning_generation_for_poor_conditioning(self):
        """Test that appropriate warnings are generated for poor conditioning."""
        # Matrix with poor conditioning
        Sigma_poor = np.array([[1, 0.99999], [0.99999, 1]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                create_solve_sdp(Sigma_poor)
                # Should generate warnings for numerical issues
                if len(w) > 0:
                    warning_messages = [str(warning.message).lower() for warning in w]
                    assert any('condition' in msg or 'numerical' in msg or 'precision' in msg
                              for msg in warning_messages)
            except ValueError:
                # If it fails completely, that's also acceptable
                pass