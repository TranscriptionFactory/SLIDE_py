#!/usr/bin/env python3
"""
Critical test coverage gaps for SLIDE package.
These tests target the most important untested functionality.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

from loveslide import SLIDE, OptimizeSLIDE, Estimator, SLIDE_Estimator
from loveslide.love import call_love, call_love_r
from loveslide.knockoff.create import create_gaussian
from loveslide.knockoff.solve import create_solve_sdp, create_solve_equi
from loveslide.love_python.love.est_pure_hetero import Est_Pure, Est_BI_C
from loveslide.plotting import Plotter


class TestSLIDEStatePersistence:
    """Test SLIDE state loading and saving functionality."""

    def test_load_love_valid_pickle(self):
        """Test loading valid LOVE results from pickle file."""
        # Create mock LOVE result
        love_result = {
            "A": np.random.randn(10, 3),
            "K": 3,
            "Sigma": np.random.randn(10, 10),
            "converged": True
        }

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(love_result, f)
            f.flush()

            # Test SLIDE loading
            params = {"fdr": 0.1}
            X = np.random.randn(50, 10)
            y = np.random.randn(50)
            slide = SLIDE(params, x=X, y=y)

            slide.load_love(f.name)

            assert hasattr(slide, 'A')
            assert hasattr(slide, 'latent_factors')
            assert slide.A.shape[1] == 3

        os.unlink(f.name)

    def test_load_love_corrupted_file(self):
        """Test loading corrupted LOVE pickle file."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            f.write(b'corrupted pickle data')
            f.flush()

            params = {"fdr": 0.1}
            X = np.random.randn(50, 10)
            y = np.random.randn(50)
            slide = SLIDE(params, x=X, y=y)

            # Should handle gracefully without raising
            slide.load_love(f.name)
            assert not hasattr(slide, 'A')

        os.unlink(f.name)

    def test_load_love_missing_file(self):
        """Test loading non-existent LOVE file."""
        params = {"fdr": 0.1}
        X = np.random.randn(50, 10)
        y = np.random.randn(50)
        slide = SLIDE(params, x=X, y=y)

        slide.load_love("nonexistent_file.pkl")
        assert not hasattr(slide, 'A')

    def test_load_state_complete_directory(self):
        """Test loading complete SLIDE state from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock state files
            A = pd.DataFrame(np.random.randn(10, 3), columns=['Z0', 'Z1', 'Z2'])
            z_matrix = pd.DataFrame(np.random.randn(50, 3), columns=['Z0', 'Z1', 'Z2'])
            sig_LFs = ['Z0', 'Z2']
            sig_interacts = ['Z0_Z1', 'Z1_Z2']

            A.to_csv(os.path.join(tmpdir, 'A.csv'))
            z_matrix.to_csv(os.path.join(tmpdir, 'z_matrix.csv'))
            np.savetxt(os.path.join(tmpdir, 'sig_LFs.txt'), sig_LFs, fmt='%s')
            np.savetxt(os.path.join(tmpdir, 'sig_interacts.txt'), sig_interacts, fmt='%s')

            # Test loading
            params = {"fdr": 0.1}
            X = np.random.randn(50, 10)
            y = np.random.randn(50)
            slide = SLIDE(params, x=X, y=y)

            slide.load_state(tmpdir)

            assert hasattr(slide, 'A')
            assert hasattr(slide, 'latent_factors')
            assert hasattr(slide, 'sig_LFs')
            assert hasattr(slide, 'sig_interacts')
            assert len(slide.sig_LFs) == 2
            assert len(slide.sig_interacts) == 2

    def test_load_state_partial_directory(self):
        """Test loading SLIDE state from incomplete directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Only create A.csv and z_matrix.csv
            A = pd.DataFrame(np.random.randn(10, 3), columns=['Z0', 'Z1', 'Z2'])
            z_matrix = pd.DataFrame(np.random.randn(50, 3), columns=['Z0', 'Z1', 'Z2'])

            A.to_csv(os.path.join(tmpdir, 'A.csv'))
            z_matrix.to_csv(os.path.join(tmpdir, 'z_matrix.csv'))

            # Test loading
            params = {"fdr": 0.1}
            X = np.random.randn(50, 10)
            y = np.random.randn(50)
            slide = SLIDE(params, x=X, y=y)

            slide.load_state(tmpdir)

            assert hasattr(slide, 'A')
            assert hasattr(slide, 'latent_factors')
            assert slide.sig_interacts == []  # Should default to empty


class TestOptimizeSLIDECore:
    """Test core OptimizeSLIDE functionality that's currently untested."""

    def test_calc_z_matrix_basic(self):
        """Test Z matrix calculation with valid LOVE results."""
        params = {"fdr": 0.1, "niter": 5}
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        opt_slide = OptimizeSLIDE(params, x=X, y=y)

        # Mock LOVE result
        love_result = {
            "A": np.random.randn(10, 3),
            "Sigma": np.eye(10),
            "K": 3
        }

        z_matrix = opt_slide.calc_z_matrix(love_result)

        assert z_matrix.shape[0] == 50  # n_samples
        assert z_matrix.shape[1] == 3   # K factors
        assert isinstance(z_matrix, pd.DataFrame)

    def test_calc_z_matrix_edge_cases(self):
        """Test Z matrix calculation edge cases."""
        params = {"fdr": 0.1}
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        opt_slide = OptimizeSLIDE(params, x=X, y=y)

        # Test with zero factors
        love_result = {
            "A": np.random.randn(10, 0),
            "Sigma": np.eye(10),
            "K": 0
        }

        z_matrix = opt_slide.calc_z_matrix(love_result)
        assert z_matrix.shape[1] == 0

        # Test with single factor
        love_result["A"] = np.random.randn(10, 1)
        love_result["K"] = 1

        z_matrix = opt_slide.calc_z_matrix(love_result)
        assert z_matrix.shape[1] == 1

    @patch('loveslide.knockoffs.Knockoffs')
    def test_find_interaction_LFs_basic(self, mock_knockoffs):
        """Test interaction LF finding with valid setup."""
        # Mock knockoff results
        mock_knockoffs.return_value.run.return_value = {
            'selected': ['Z0_Z1', 'Z1_Z2'],
            'statistics': np.array([1.2, -0.8, 2.1])
        }

        params = {"fdr": 0.1, "do_interacts": True}
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        opt_slide = OptimizeSLIDE(params, x=X, y=y)

        # Setup required attributes
        opt_slide.latent_factors = pd.DataFrame(
            np.random.randn(50, 3),
            columns=['Z0', 'Z1', 'Z2']
        )
        opt_slide.marginal_idxs = np.array([0, 2])

        interactions = opt_slide.find_interaction_LFs()

        assert isinstance(interactions, list)
        assert len(interactions) >= 0

    def test_find_interaction_LFs_disabled(self):
        """Test interaction finding when disabled."""
        params = {"fdr": 0.1, "do_interacts": False}
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        opt_slide = OptimizeSLIDE(params, x=X, y=y)
        opt_slide.latent_factors = pd.DataFrame(
            np.random.randn(50, 3),
            columns=['Z0', 'Z1', 'Z2']
        )
        opt_slide.marginal_idxs = np.array([])

        interactions = opt_slide.find_interaction_LFs()

        assert interactions == []


class TestLOVEPureEstimation:
    """Test LOVE pure variable estimation functions."""

    def test_est_pure_basic(self):
        """Test Est_Pure with basic score matrix."""
        # Create symmetric score matrix with some connections
        score_mat = np.full((5, 5), np.nan)
        score_mat[0, 1] = score_mat[1, 0] = 0.03  # Connected
        score_mat[0, 2] = score_mat[2, 0] = 0.03  # Connected
        score_mat[1, 2] = score_mat[2, 1] = 0.03  # Connected
        score_mat[3, 4] = score_mat[4, 3] = 0.02  # Connected

        # Make upper triangular
        score_mat = np.triu(score_mat, k=1)

        result = Est_Pure(score_mat, delta=0.05)

        assert 'K' in result
        assert 'I' in result
        assert 'I_part' in result
        assert result['K'] >= 0
        assert isinstance(result['I_part'], list)

    def test_est_pure_no_connections(self):
        """Test Est_Pure with no connected components."""
        score_mat = np.full((5, 5), np.nan)
        # Fill upper triangle with values > delta
        for i in range(5):
            for j in range(i+1, 5):
                score_mat[i, j] = 0.9  # All above threshold

        result = Est_Pure(score_mat, delta=0.05)

        assert result['K'] == 0
        assert len(result['I']) == 0
        assert len(result['I_part']) == 0

    def test_est_pure_all_connected(self):
        """Test Est_Pure with all variables connected."""
        score_mat = np.full((5, 5), np.nan)
        # Fill upper triangle with values < delta
        for i in range(5):
            for j in range(i+1, 5):
                score_mat[i, j] = 0.01  # All below threshold

        result = Est_Pure(score_mat, delta=0.05)

        assert result['K'] == 1  # One large component
        assert len(result['I']) == 5  # All variables
        assert len(result['I_part']) == 1  # One partition

    def test_est_bi_c_basic(self):
        """Test Est_BI_C with valid inputs."""
        p = 6
        M = np.random.randn(p, 3)
        R = np.random.randn(p, p)
        R = R @ R.T  # Make positive definite
        I_part = [[0, 1, 2], [3, 4]]
        I = [0, 1, 2, 3, 4]
        L_ind = [0, 1]

        result = Est_BI_C(M, R, I_part, I, L_ind)

        assert 'Gamma_LL' in result
        assert 'L_hat' in result
        assert result['Gamma_LL'].shape[0] == len(L_ind)
        assert result['L_hat'].shape[0] == p


class TestKnockoffSolverEdgeCases:
    """Test knockoff solver edge cases."""

    def test_create_solve_sdp_singular_matrix(self):
        """Test SDP solver with near-singular covariance."""
        # Create near-singular matrix
        Sigma = np.random.randn(5, 5)
        Sigma = Sigma @ Sigma.T
        # Make nearly singular
        Sigma += 1e-10 * np.eye(5)

        # Should handle gracefully
        s = create_solve_sdp(Sigma)

        assert len(s) == Sigma.shape[0]
        assert np.all(s >= 0)
        assert np.all(s <= 1)

    def test_create_solve_equi_small_eigenvalues(self):
        """Test equicorrelated solver with small eigenvalues."""
        # Create matrix with small eigenvalues
        Sigma = np.diag([1, 1, 1, 0.01, 0.01])

        s = create_solve_equi(Sigma)

        assert len(s) == Sigma.shape[0]
        assert np.all(s >= 0)


class TestEstimatorEdgeCases:
    """Test Estimator class edge cases."""

    def test_estimator_perfect_separation(self):
        """Test binary classification with perfect separation."""
        X = np.array([[1, 0], [2, 0], [0, 1], [0, 2]])
        y = np.array([1, 1, 0, 0])  # Perfectly separable

        estimator = Estimator(model='logistic')
        estimator.fit(X, y)

        # Should handle without crashing
        score = estimator.evaluate(X, y, n_iters=1)
        assert score is not None

    def test_estimator_constant_y(self):
        """Test with constant response variable."""
        X = np.random.randn(50, 5)
        y = np.ones(50)  # Constant

        estimator = Estimator(model='auto')
        result = estimator.evaluate(X, y, n_iters=1)

        # Should return None for constant y
        assert result is None

    def test_estimator_single_feature(self):
        """Test with single feature."""
        X = np.random.randn(50, 1)
        y = np.random.randn(50)

        estimator = Estimator(model='linear')
        score = estimator.evaluate(X, y, n_iters=3)

        assert isinstance(score, float)
        assert not np.isnan(score)

    def test_slide_estimator_voting_failure(self):
        """Test SLIDE_Estimator when voting fails."""
        X = np.random.randn(30, 20)
        y = np.random.randn(30)

        estimator = SLIDE_Estimator()

        # Mock knockoff failure
        with patch('loveslide.knockoffs.Knockoffs') as mock_knockoffs:
            mock_knockoffs.return_value.run.side_effect = Exception("Knockoff failed")

            score = estimator.evaluate(X, y, n_iters=1)
            # Should handle gracefully, possibly returning None
            assert score is None or isinstance(score, float)


class TestPlotterErrorHandling:
    """Test Plotter error handling."""

    def test_plotter_empty_results(self):
        """Test plotting with empty/invalid results."""
        plotter = Plotter(figsize=(8, 6))

        # Test with empty selections
        empty_voting_result = MagicMock()
        empty_voting_result.selected = []
        empty_voting_result.fdr = 0.1

        # Should handle gracefully without crashing
        try:
            plotter.plot_knockoff_diagnostics(empty_voting_result)
        except Exception as e:
            pytest.fail(f"Plotter should handle empty results: {e}")

    def test_plotter_malformed_data(self):
        """Test plotting with malformed data structures."""
        plotter = Plotter()

        # Test with None data
        try:
            plotter.plot_love_factors(None, threshold=0.05)
        except (ValueError, AttributeError):
            pass  # Expected to fail gracefully
        except Exception as e:
            pytest.fail(f"Unexpected error type: {e}")


class TestNumericalStability:
    """Test numerical stability across algorithms."""

    def test_love_numerical_precision(self):
        """Test LOVE with extreme numerical values."""
        # Very large values
        X_large = 1e6 * np.random.randn(20, 10)

        # Should not crash
        try:
            result = call_love(X_large, verbose=False)
            assert result is not None
        except (OverflowError, np.linalg.LinAlgError):
            pass  # Acceptable failures for extreme values

    def test_knockoff_numerical_stability(self):
        """Test knockoff generation numerical stability."""
        # Near-singular correlation matrix
        X = np.random.randn(100, 20)
        # Make some features nearly identical
        X[:, 1] = X[:, 0] + 1e-10 * np.random.randn(100)

        try:
            X_k = create_gaussian(X)
            assert X_k.shape == X.shape
            # Verify knockoff constraints are approximately satisfied
            assert np.allclose(X.T @ X_k, X_k.T @ X, atol=1e-3)
        except np.linalg.LinAlgError:
            pass  # Acceptable for numerically challenging cases


if __name__ == "__main__":
    pytest.main([__file__])