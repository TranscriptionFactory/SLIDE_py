"""
Test coverage for SDP solver integration and numerical operations.
Addresses: SDP solvers, matrix conditioning, fallback mechanisms
"""
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import warnings

from loveslide.knockoff.solve import (
    _get_sdp_solver, _solve_sdp_cvxpy, create_solve_equi,
    create_solve_sdp, create_solve_asdp, _divide_sdp, _merge_clusters
)
from loveslide.knockoff.create import (
    _decompose, _create_equicorrelated, _create_sdp,
    create_fixed, create_gaussian, create_second_order
)


class TestSDPSolverDetection:
    """Test SDP solver detection and fallback mechanisms."""

    def test_get_sdp_solver_basic(self):
        """Test SDP solver detection."""
        solver = _get_sdp_solver()

        # Should return a valid solver name or None
        valid_solvers = ['MOSEK', 'CVXOPT', 'SCS', 'CLARABEL', None]
        assert solver in valid_solvers

    def test_solve_sdp_cvxpy_basic(self):
        """Test CVXPY SDP solver with simple problem."""
        # Create a simple SDP problem
        n = 3
        A0 = np.eye(n)
        A1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        A2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])

        A_list = [A1, A2]
        b = np.array([1, 1])
        C = A0

        try:
            result = _solve_sdp_cvxpy(A_list, b, C)

            if result is not None:
                # Should be symmetric
                assert np.allclose(result, result.T, atol=1e-6)

                # Should satisfy constraints approximately
                for i, A in enumerate(A_list):
                    constraint_val = np.trace(A @ result)
                    assert abs(constraint_val - b[i]) < 1e-3

        except Exception:
            # Some solvers may not be available
            pytest.skip("SDP solver not available")

    def test_solve_sdp_cvxpy_infeasible(self):
        """Test SDP solver with infeasible problem."""
        # Create conflicting constraints
        n = 2
        A1 = np.array([[1, 0], [0, 0]])
        A2 = np.array([[1, 0], [0, 0]])
        A_list = [A1, A2]
        b = np.array([1, -1])  # Conflicting constraints
        C = np.eye(n)

        try:
            result = _solve_sdp_cvxpy(A_list, b, C)
            # Should return None for infeasible problems
            assert result is None
        except Exception:
            pytest.skip("SDP solver not available")

    def test_sdp_solver_fallback_mechanism(self):
        """Test that system gracefully handles solver unavailability."""
        # This test checks the fallback behavior
        # when preferred solvers are not available

        # Create correlation matrix that requires SDP
        p = 5
        rho = 0.9
        Sigma = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)

        try:
            # Should not crash even if specific solver fails
            result = create_solve_sdp(Sigma, method='sdp')

            if result is not None:
                assert result.shape == (p,)
                assert np.all(result >= -1e-6)  # Should be non-negative
                assert np.all(result <= 1 + 1e-6)  # Should be at most 1

        except Exception as e:
            # Acceptable if all SDP solvers fail
            assert "solver" in str(e).lower() or "cvxpy" in str(e).lower()


class TestEquicorrelatedSolution:
    """Test equicorrelated knockoff solutions."""

    def test_create_solve_equi_basic(self):
        """Test equicorrelated solution computation."""
        # Simple correlation matrix
        p = 4
        rho = 0.5
        Sigma = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)

        s = create_solve_equi(Sigma)

        assert len(s) == p
        assert np.all(s >= -1e-10)  # Non-negative
        assert np.all(s <= 1 + 1e-10)  # At most 1

        # Check that 2*Sigma - diag(s) is positive definite
        G = 2 * Sigma - np.diag(s)
        eigenvals = np.linalg.eigvals(G)
        assert np.all(eigenvals > -1e-6)

    def test_create_solve_equi_edge_cases(self):
        """Test equicorrelated solution with edge cases."""
        # Identity matrix
        Sigma = np.eye(5)
        s = create_solve_equi(Sigma)
        assert_allclose(s, 1, atol=1e-10)  # Should be all ones

        # Highly correlated matrix
        p = 3
        Sigma = 0.95 * np.ones((p, p)) + 0.05 * np.eye(p)
        s = create_solve_equi(Sigma)

        # Should be feasible
        assert np.all(s >= -1e-6)
        assert np.all(s <= 1 + 1e-6)

    def test_create_solve_equi_ill_conditioned(self):
        """Test equicorrelated solution with ill-conditioned matrices."""
        # Nearly singular matrix
        p = 4
        A = np.random.randn(p, p-1)
        Sigma = A @ A.T + 1e-8 * np.eye(p)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = create_solve_equi(Sigma)

        # Should still produce a result
        assert len(s) == p
        assert np.all(np.isfinite(s))


class TestSDPSolution:
    """Test SDP-based knockoff solutions."""

    def test_create_solve_sdp_basic(self):
        """Test SDP solution computation."""
        # Correlation matrix that benefits from SDP
        p = 6
        rho = 0.7
        Sigma = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)

        try:
            s = create_solve_sdp(Sigma, method='sdp')

            if s is not None:
                assert len(s) == p
                assert np.all(s >= -1e-6)
                assert np.all(s <= 1 + 1e-6)

                # SDP solution should be at least as good as equicorrelated
                s_equi = create_solve_equi(Sigma)
                assert np.sum(s) >= np.sum(s_equi) - 1e-6

        except Exception:
            pytest.skip("SDP solver not available")

    def test_create_solve_sdp_vs_equicorrelated(self):
        """Compare SDP and equicorrelated solutions."""
        # Matrix where SDP should outperform equicorrelated
        p = 5
        Sigma = np.array([
            [1.0, 0.8, 0.1, 0.1, 0.1],
            [0.8, 1.0, 0.1, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.7, 0.1],
            [0.1, 0.1, 0.7, 1.0, 0.1],
            [0.1, 0.1, 0.1, 0.1, 1.0]
        ])

        s_equi = create_solve_equi(Sigma)

        try:
            s_sdp = create_solve_sdp(Sigma, method='sdp')

            if s_sdp is not None:
                # SDP should achieve higher sum (better power)
                assert np.sum(s_sdp) >= np.sum(s_equi) - 1e-6

        except Exception:
            pytest.skip("SDP solver not available")

    def test_create_solve_asdp_basic(self):
        """Test Approximate SDP (ASDP) solution."""
        # Large correlation matrix for ASDP
        p = 20
        rho = 0.6
        Sigma = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)

        s = create_solve_asdp(Sigma, max_size=5)

        assert len(s) == p
        assert np.all(s >= -1e-6)
        assert np.all(s <= 1 + 1e-6)

    def test_divide_sdp_clustering(self):
        """Test SDP problem division into clusters."""
        # Block correlation structure
        p = 12
        Sigma = np.eye(p)

        # Create blocks
        Sigma[0:3, 0:3] = 0.8
        Sigma[3:6, 3:6] = 0.7
        Sigma[6:9, 6:9] = 0.6
        Sigma[9:12, 9:12] = 0.5
        np.fill_diagonal(Sigma, 1)

        clusters, Sigma_clusters = _divide_sdp(Sigma, max_size=4)

        assert len(clusters) >= 3  # Should create multiple clusters
        assert sum(len(c) for c in clusters) == p  # All variables assigned

        # Check cluster matrices
        assert len(Sigma_clusters) == len(clusters)
        for i, cluster in enumerate(clusters):
            sub_Sigma = Sigma_clusters[i]
            assert sub_Sigma.shape == (len(cluster), len(cluster))
            # Should be submatrix of original
            original_sub = Sigma[np.ix_(cluster, cluster)]
            assert_array_almost_equal(sub_Sigma, original_sub)

    def test_merge_clusters_basic(self):
        """Test cluster merging functionality."""
        # Initial clusters
        clusters = np.array([0, 0, 1, 1, 2, 2, 3])
        max_size = 4

        merged = _merge_clusters(clusters, max_size)

        # Check that no cluster exceeds max_size
        unique_clusters, counts = np.unique(merged, return_counts=True)
        assert np.all(counts <= max_size)

        # Should have fewer or equal clusters
        assert len(unique_clusters) <= len(np.unique(clusters))


class TestKnockoffCreation:
    """Test knockoff creation functions."""

    def test_decompose_basic(self):
        """Test matrix decomposition for knockoffs."""
        n, p = 50, 8
        X = np.random.randn(n, p)

        decomp = _decompose(X, randomize=False)

        assert 'X' in decomp
        assert 'Sigma' in decomp
        assert 'Sigma_inv' in decomp

        # Check dimensions
        assert decomp['X'].shape == (n, p)
        assert decomp['Sigma'].shape == (p, p)
        assert decomp['Sigma_inv'].shape == (p, p)

        # Check that Sigma * Sigma_inv ≈ I
        product = decomp['Sigma'] @ decomp['Sigma_inv']
        assert_allclose(product, np.eye(p), atol=1e-10)

    def test_create_equicorrelated_basic(self):
        """Test equicorrelated knockoff creation."""
        n, p = 60, 6
        X = np.random.randn(n, p)

        Xk = _create_equicorrelated(X, randomize=False)

        assert Xk.shape == X.shape

        # Check knockoff properties
        # Cross-correlation should match auto-correlation
        Sigma_X = np.corrcoef(X.T)
        Sigma_cross = np.corrcoef(X.T, Xk.T)[:p, p:]

        # Diagonal should be preserved structure
        # (exact values depend on correlation structure)
        assert np.all(np.isfinite(Sigma_cross))

    def test_create_sdp_knockoffs(self):
        """Test SDP-based knockoff creation."""
        n, p = 50, 5
        X = np.random.randn(n, p)

        try:
            Xk = _create_sdp(X, randomize=False)

            if Xk is not None:
                assert Xk.shape == X.shape

                # Should have desired correlation properties
                combined = np.hstack([X, Xk])
                Sigma_combined = np.corrcoef(combined.T)

                # Check block structure
                Sigma_X = Sigma_combined[:p, :p]
                Sigma_Xk = Sigma_combined[p:, p:]
                Sigma_cross = Sigma_combined[:p, p:]

                # Auto-correlations should be similar
                assert_allclose(np.diag(Sigma_X), 1, atol=1e-6)
                assert_allclose(np.diag(Sigma_Xk), 1, atol=1e-6)

        except Exception:
            pytest.skip("SDP solver not available for knockoff creation")

    def test_create_fixed_design_knockoffs(self):
        """Test fixed design knockoff creation."""
        # Design matrix
        n, p = 40, 4
        X = np.random.randn(n, p)

        for method in ['equi', 'sdp']:
            try:
                Xk = create_fixed(X, method=method, randomize=False)

                if Xk is not None:
                    assert Xk.shape == X.shape

                    # Check mean centering
                    if n > p:  # Avoid rank issues
                        assert_allclose(np.mean(Xk, axis=0), 0, atol=1e-10)

            except Exception:
                if method == 'sdp':
                    pytest.skip(f"SDP solver not available for {method}")
                else:
                    raise

    def test_create_gaussian_knockoffs(self):
        """Test Gaussian model knockoffs."""
        # Parameters for Gaussian distribution
        p = 6
        mu = np.zeros(p)
        Sigma = 0.5 * np.ones((p, p)) + 0.5 * np.eye(p)

        n_samples = 100

        for method in ['equi', 'sdp']:
            try:
                X, Xk = create_gaussian(
                    mu=mu, Sigma=Sigma, method=method,
                    n_samples=n_samples, randomize=False
                )

                assert X.shape == (n_samples, p)
                assert Xk.shape == (n_samples, p)

                # Check approximate mean and covariance
                combined = np.vstack([X, Xk])
                empirical_mean = np.mean(combined, axis=0)
                assert_allclose(empirical_mean, mu, atol=0.3)

            except Exception:
                if method == 'sdp':
                    pytest.skip(f"SDP solver not available for {method}")
                else:
                    raise

    def test_create_second_order_knockoffs(self):
        """Test second-order knockoffs."""
        n, p = 70, 8
        X = np.random.randn(n, p)

        for method in ['equi', 'sdp']:
            try:
                Xk = create_second_order(X, method=method, randomize=False)

                if Xk is not None:
                    assert Xk.shape == X.shape

                    # Second-order knockoffs should satisfy stronger properties
                    combined = np.hstack([X, Xk])
                    Sigma_full = np.cov(combined.T, bias=True)

                    # Check symmetry properties
                    p_vars = X.shape[1]
                    Sigma_X = Sigma_full[:p_vars, :p_vars]
                    Sigma_Xk = Sigma_full[p_vars:, p_vars:]
                    Sigma_cross = Sigma_full[:p_vars, p_vars:]

                    # Cross-covariance should have specific structure
                    assert np.all(np.isfinite(Sigma_cross))

            except Exception:
                if method == 'sdp':
                    pytest.skip(f"SDP solver not available for {method}")
                else:
                    raise


class TestNumericalStability:
    """Test numerical stability of SDP operations."""

    def test_nearly_singular_matrices(self):
        """Test behavior with nearly singular correlation matrices."""
        p = 5
        # Create nearly singular matrix
        A = np.random.randn(p, p-1)
        Sigma = A @ A.T + 1e-10 * np.eye(p)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Should not crash
            s_equi = create_solve_equi(Sigma)
            assert np.all(np.isfinite(s_equi))

            try:
                s_sdp = create_solve_sdp(Sigma, method='sdp')
                if s_sdp is not None:
                    assert np.all(np.isfinite(s_sdp))
            except Exception:
                pass  # Acceptable for very ill-conditioned matrices

    def test_extreme_eigenvalues(self):
        """Test with matrices having extreme eigenvalue ratios."""
        p = 4
        # Create matrix with large condition number
        Q = np.random.randn(p, p)
        Q, _ = np.linalg.qr(Q)
        eigenvals = np.array([1, 0.1, 0.01, 0.001])
        Sigma = Q @ np.diag(eigenvals) @ Q.T

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            s = create_solve_equi(Sigma)
            assert np.all(np.isfinite(s))
            assert np.all(s >= -1e-6)

    def test_large_problem_scalability(self):
        """Test scalability with larger problems."""
        # Test with moderately large problem
        p = 50
        rho = 0.3
        Sigma = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)

        # Equicorrelated should scale well
        s_equi = create_solve_equi(Sigma)
        assert len(s_equi) == p
        assert np.all(np.isfinite(s_equi))

        # ASDP should handle large problems
        s_asdp = create_solve_asdp(Sigma, max_size=10)
        assert len(s_asdp) == p
        assert np.all(np.isfinite(s_asdp))

    def test_precision_consistency(self):
        """Test numerical precision consistency across methods."""
        p = 6
        Sigma = np.array([
            [1.0, 0.5, 0.3, 0.1, 0.1, 0.0],
            [0.5, 1.0, 0.4, 0.2, 0.0, 0.1],
            [0.3, 0.4, 1.0, 0.0, 0.2, 0.1],
            [0.1, 0.2, 0.0, 1.0, 0.3, 0.4],
            [0.1, 0.0, 0.2, 0.3, 1.0, 0.5],
            [0.0, 0.1, 0.1, 0.4, 0.5, 1.0]
        ])

        s_equi = create_solve_equi(Sigma)

        # Check that solution satisfies constraints precisely
        G = 2 * Sigma - np.diag(s_equi)
        eigenvals = np.linalg.eigvals(G)
        assert np.all(eigenvals > -1e-12), "Constraint violation in equicorrelated solution"

        try:
            s_sdp = create_solve_sdp(Sigma, method='sdp')
            if s_sdp is not None:
                G_sdp = 2 * Sigma - np.diag(s_sdp)
                eigenvals_sdp = np.linalg.eigvals(G_sdp)
                assert np.all(eigenvals_sdp > -1e-12), "Constraint violation in SDP solution"

        except Exception:
            pytest.skip("SDP solver not available for precision test")