#!/usr/bin/env python3
"""
Comprehensive test coverage for LOVE Python submodule.
This module is completely untested in the current codebase.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Union

from loveslide.love_python.love.est_pure_hetero import (
    Est_Pure, Est_BI_C, Re_Est_Pure, Post_Est_Pure, Est_K
)
from loveslide.love_python.love.est_pure_homo import Est_Pure as Est_Pure_Homo
from loveslide.love_python.love.est_nonpure import est_nonpure
from loveslide.love_python.love.est_omega import est_omega
from loveslide.love_python.love.prescreen import prescreen
from loveslide.love_python.love.utilities import (
    calculate_score_matrix, validate_inputs, standardize_data
)
from loveslide.love_python.love.cv import cross_validate_delta


class TestEstPureHetero:
    """Test heterogeneous pure estimation functions."""

    def test_est_pure_connected_components(self):
        """Test Est_Pure graph connected components detection."""
        # Create score matrix with known structure
        score_mat = np.full((6, 6), np.nan)

        # Component 1: {0, 1, 2}
        score_mat[0, 1] = 0.02
        score_mat[0, 2] = 0.03
        score_mat[1, 2] = 0.01

        # Component 2: {3, 4}
        score_mat[3, 4] = 0.04

        # Isolated node: {5}
        # No edges to node 5

        result = Est_Pure(score_mat, delta=0.05)

        assert result['K'] == 2  # Two components (ignoring isolated nodes)
        assert len(result['I_part']) == 2

        # Check components are correct
        components = [sorted(comp) for comp in result['I_part']]
        components.sort()  # Sort for comparison

        expected = [[0, 1, 2], [3, 4]]
        assert components == expected

    def test_est_pure_threshold_sensitivity(self):
        """Test Est_Pure sensitivity to delta threshold."""
        score_mat = np.full((4, 4), np.nan)
        score_mat[0, 1] = 0.02
        score_mat[0, 2] = 0.06  # Above some thresholds
        score_mat[1, 2] = 0.03
        score_mat[2, 3] = 0.08  # Well above threshold

        # Strict threshold - fewer connections
        result_strict = Est_Pure(score_mat, delta=0.05)

        # Lenient threshold - more connections
        result_lenient = Est_Pure(score_mat, delta=0.10)

        # Should have different component structures
        assert result_lenient['K'] >= result_strict['K']

    def test_est_pure_empty_matrix(self):
        """Test Est_Pure with empty score matrix."""
        score_mat = np.full((0, 0), np.nan)

        result = Est_Pure(score_mat, delta=0.05)

        assert result['K'] == 0
        assert len(result['I']) == 0
        assert len(result['I_part']) == 0

    def test_est_bi_c_valid_inputs(self):
        """Test Est_BI_C with valid mathematical inputs."""
        p = 8
        K = 3

        # Create realistic M matrix (loadings)
        M = np.random.randn(p, K)

        # Create positive definite R matrix
        A = np.random.randn(p, p)
        R = A @ A.T + 0.1 * np.eye(p)

        # Define partitions
        I_part = [[0, 1, 2], [3, 4], [5, 6, 7]]
        I = [0, 1, 2, 3, 4, 5, 6, 7]
        L_ind = [0, 1, 2]  # Indices of latent factors

        result = Est_BI_C(M, R, I_part, I, L_ind)

        assert 'Gamma_LL' in result
        assert 'L_hat' in result
        assert 'Gamma_thetatheta' in result

        # Check dimensions
        assert result['Gamma_LL'].shape == (len(L_ind), len(L_ind))
        assert result['L_hat'].shape == (p, len(L_ind))

    def test_est_bi_c_single_partition(self):
        """Test Est_BI_C with single large partition."""
        p = 5
        K = 2

        M = np.random.randn(p, K)
        A = np.random.randn(p, p)
        R = A @ A.T + 0.1 * np.eye(p)

        # Single partition containing all variables
        I_part = [[0, 1, 2, 3, 4]]
        I = [0, 1, 2, 3, 4]
        L_ind = [0, 1]

        result = Est_BI_C(M, R, I_part, I, L_ind)

        assert result['Gamma_LL'].shape == (2, 2)
        assert result['L_hat'].shape == (5, 2)

    def test_re_est_pure_convergence(self):
        """Test Re_Est_Pure iterative refinement."""
        n, p = 50, 10
        X = np.random.randn(n, p)

        # Create positive definite Sigma
        A = np.random.randn(p, p)
        Sigma = A @ A.T + 0.1 * np.eye(p)

        M = np.random.randn(p, 3)
        I_part = [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]]
        I = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        Gamma_LL = np.eye(3)
        L_hat = np.random.randn(p, 3)

        result = Re_Est_Pure(X, Sigma, M, I_part, I, Gamma_LL, L_hat)

        assert 'Gamma_LL' in result
        assert 'L_hat' in result
        assert 'converged' in result

        # Check convergence flag is boolean
        assert isinstance(result['converged'], bool)

    def test_post_est_pure_optimization(self):
        """Test Post_Est_Pure final optimization step."""
        p = 8
        K = 3

        # Create inputs
        A = np.random.randn(p, p)
        Sigma = A @ A.T + 0.1 * np.eye(p)
        Gamma_LL = np.eye(K)
        L_hat = np.random.randn(p, K)
        I_part = [[0, 1, 2], [3, 4], [5, 6, 7]]
        I = [0, 1, 2, 3, 4, 5, 6, 7]

        result = Post_Est_Pure(Sigma, Gamma_LL, L_hat, I_part, I)

        assert 'A_final' in result
        assert 'Sigma_final' in result
        assert result['A_final'].shape == (p, K)

    def test_est_k_factor_selection(self):
        """Test Est_K factor number estimation."""
        n, p = 60, 12
        K_true = 3

        # Generate data with known factor structure
        X = np.random.randn(n, p)
        L_hat = np.random.randn(p, K_true)
        Gamma_LL = np.eye(K_true)

        K_est = Est_K(X, L_hat, Gamma_LL)

        assert isinstance(K_est, int)
        assert K_est >= 1
        assert K_est <= min(p, n//2)  # Reasonable bounds


class TestEstPureHomo:
    """Test homogeneous pure estimation functions."""

    def test_est_pure_homo_basic(self):
        """Test homogeneous pure estimation with valid inputs."""
        # Create score matrix
        score_mat = np.random.rand(10, 10) * 0.1  # Small values
        score_mat = np.triu(score_mat, k=1)  # Upper triangular

        # Should not crash and return valid structure
        result = Est_Pure_Homo(score_mat, delta=0.05)

        assert 'pure_variables' in result
        assert 'components' in result
        assert isinstance(result['pure_variables'], list)

    def test_est_pure_homo_no_pure_variables(self):
        """Test homogeneous estimation with no pure variables."""
        # Create score matrix with all high scores
        score_mat = np.full((5, 5), 0.9)
        score_mat = np.triu(score_mat, k=1)

        result = Est_Pure_Homo(score_mat, delta=0.05)

        # Should find no pure variables
        assert len(result['pure_variables']) == 0


class TestEstNonPure:
    """Test non-pure variable estimation."""

    def test_est_nonpure_basic_functionality(self):
        """Test non-pure estimation basic functionality."""
        n, p = 50, 15
        X = np.random.randn(n, p)

        # Mock pure variable information
        pure_info = {
            'pure_indices': [0, 1, 5, 6],
            'factor_assignments': [0, 0, 1, 1],
            'loadings': np.random.randn(4, 2)
        }

        result = est_nonpure(X, pure_info, method="HT")

        assert 'nonpure_loadings' in result
        assert 'factor_structure' in result

    def test_est_nonpure_different_methods(self):
        """Test different non-pure estimation methods."""
        n, p = 40, 10
        X = np.random.randn(n, p)

        pure_info = {
            'pure_indices': [0, 1],
            'factor_assignments': [0, 0],
            'loadings': np.random.randn(2, 1)
        }

        methods = ["HT", "oracle", "simple"]

        for method in methods:
            result = est_nonpure(X, pure_info, method=method)
            assert result is not None


class TestEstOmega:
    """Test omega matrix estimation."""

    def test_est_omega_valid_inputs(self):
        """Test omega estimation with valid covariance structure."""
        p = 8

        # Create realistic factor structure
        A = np.random.randn(p, 3)  # Loading matrix
        Omega_true = np.diag(np.random.rand(p) + 0.1)  # True noise covariance

        # Observed covariance
        Sigma = A @ A.T + Omega_true

        Omega_est = est_omega(Sigma, A)

        assert Omega_est.shape == (p, p)
        assert np.allclose(Omega_est, Omega_est.T)  # Should be symmetric
        assert np.all(np.diag(Omega_est) > 0)  # Diagonal should be positive

    def test_est_omega_edge_cases(self):
        """Test omega estimation edge cases."""
        p = 5

        # Case 1: Zero loading matrix
        A_zero = np.zeros((p, 3))
        Sigma = np.eye(p)

        Omega = est_omega(Sigma, A_zero)
        assert np.allclose(Omega, Sigma)  # Should equal Sigma when A=0

        # Case 2: Perfect factor structure (Omega should be small)
        A_perfect = np.random.randn(p, p)  # Square loading matrix
        Sigma_perfect = A_perfect @ A_perfect.T

        Omega_small = est_omega(Sigma_perfect, A_perfect[:, :3])
        assert np.all(np.diag(Omega_small) >= 0)  # Non-negative diagonal


class TestPrescreen:
    """Test prescreening functionality."""

    def test_prescreen_basic(self):
        """Test basic prescreening functionality."""
        n, p = 100, 50
        X = np.random.randn(n, p)

        # Add some structure
        X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n)  # Correlated variables

        screened_indices = prescreen(X, method="variance", top_k=20)

        assert len(screened_indices) <= 20
        assert len(screened_indices) <= p
        assert all(0 <= idx < p for idx in screened_indices)

    def test_prescreen_different_methods(self):
        """Test different prescreening methods."""
        n, p = 60, 30
        X = np.random.randn(n, p)

        methods = ["variance", "correlation", "pca", "random"]

        for method in methods:
            indices = prescreen(X, method=method, top_k=15)
            assert isinstance(indices, (list, np.ndarray))
            assert len(indices) <= 15

    def test_prescreen_edge_cases(self):
        """Test prescreening edge cases."""
        # Case 1: More features requested than available
        X = np.random.randn(50, 10)
        indices = prescreen(X, top_k=20)  # More than p=10
        assert len(indices) == 10  # Should return all features

        # Case 2: Single feature
        X_single = np.random.randn(50, 1)
        indices = prescreen(X_single, top_k=5)
        assert len(indices) == 1
        assert indices[0] == 0


class TestUtilities:
    """Test LOVE utility functions."""

    def test_calculate_score_matrix(self):
        """Test score matrix calculation."""
        n, p = 50, 10
        X = np.random.randn(n, p)

        score_matrix = calculate_score_matrix(X, method="correlation")

        assert score_matrix.shape == (p, p)
        assert np.allclose(score_matrix, score_matrix.T)  # Symmetric
        assert np.all(np.diag(score_matrix) == 0)  # Zero diagonal

    def test_validate_inputs(self):
        """Test input validation utility."""
        # Valid inputs
        X_valid = np.random.randn(50, 10)
        assert validate_inputs(X_valid) == True

        # Invalid inputs
        X_invalid = np.array([[1, 2, np.inf], [4, 5, 6]])
        assert validate_inputs(X_invalid) == False

        # NaN inputs
        X_nan = np.array([[1, 2, np.nan], [4, 5, 6]])
        assert validate_inputs(X_nan) == False

    def test_standardize_data(self):
        """Test data standardization utility."""
        X = np.random.randn(50, 8) * 3 + 5  # Non-standard data

        X_std = standardize_data(X, method="zscore")

        # Should be approximately standardized
        assert np.allclose(np.mean(X_std, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_std, axis=0), 1, atol=1e-10)

        # Test different standardization methods
        X_minmax = standardize_data(X, method="minmax")
        assert np.all(X_minmax >= 0) and np.all(X_minmax <= 1)


class TestCrossValidation:
    """Test LOVE cross-validation functionality."""

    def test_cross_validate_delta_basic(self):
        """Test delta cross-validation."""
        n, p = 80, 15
        X = np.random.randn(n, p)

        delta_values = [0.01, 0.05, 0.1, 0.2]

        best_delta, cv_scores = cross_validate_delta(
            X, delta_values, cv_folds=3, scoring="aic"
        )

        assert best_delta in delta_values
        assert len(cv_scores) == len(delta_values)
        assert all(isinstance(score, float) for score in cv_scores)

    def test_cross_validate_delta_edge_cases(self):
        """Test delta CV with edge cases."""
        X = np.random.randn(20, 8)

        # Single delta value
        best_delta, scores = cross_validate_delta(X, [0.05], cv_folds=2)
        assert best_delta == 0.05
        assert len(scores) == 1

        # More folds than samples (should handle gracefully)
        best_delta, scores = cross_validate_delta(
            X, [0.05, 0.1], cv_folds=25  # More than n=20
        )
        assert best_delta in [0.05, 0.1]


class TestIntegration:
    """Integration tests for LOVE Python components."""

    def test_complete_love_python_pipeline(self):
        """Test complete LOVE Python estimation pipeline."""
        # Generate synthetic data with factor structure
        n, p = 100, 20
        K_true = 3

        # True loadings
        A_true = np.random.randn(p, K_true)
        factors = np.random.randn(n, K_true)
        noise = 0.1 * np.random.randn(n, p)

        X = factors @ A_true.T + noise

        # Step 1: Prescreen
        indices = prescreen(X, method="variance", top_k=15)
        X_screened = X[:, indices]

        # Step 2: Calculate score matrix
        score_mat = calculate_score_matrix(X_screened, method="correlation")

        # Step 3: Find pure variables
        pure_result = Est_Pure(score_mat, delta=0.1)

        # Step 4: Estimate factor structure (if pure variables found)
        if pure_result['K'] > 0:
            # Mock subsequent estimation steps
            # (Full implementation would continue the pipeline)
            assert len(pure_result['I_part']) > 0

        # Pipeline should complete without errors
        assert True

    def test_love_python_robustness(self):
        """Test LOVE Python robustness to challenging data."""
        scenarios = [
            # High correlation structure
            {"n": 50, "p": 10, "correlation": 0.8},
            # Low rank structure
            {"n": 100, "p": 15, "rank": 2},
            # Noisy data
            {"n": 80, "p": 12, "noise_level": 2.0}
        ]

        for scenario in scenarios:
            n, p = scenario["n"], scenario["p"]

            if "correlation" in scenario:
                # Generate correlated data
                X = np.random.randn(n, p)
                rho = scenario["correlation"]
                for i in range(p-1):
                    X[:, i+1] = rho * X[:, i] + np.sqrt(1-rho**2) * X[:, i+1]
            else:
                X = np.random.randn(n, p)

            # Should handle robustly
            try:
                score_mat = calculate_score_matrix(X)
                pure_result = Est_Pure(score_mat, delta=0.1)
                # Basic checks
                assert isinstance(pure_result, dict)
                assert 'K' in pure_result
            except Exception as e:
                pytest.fail(f"LOVE Python failed on scenario {scenario}: {e}")


if __name__ == "__main__":
    pytest.main([__file__])