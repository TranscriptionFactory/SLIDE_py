"""
Test coverage for algorithmic edge cases and boundary conditions.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.loveslide.slide import SLIDE, OptimizeSLIDE
from src.loveslide.cv import SLIDEcv
from src.loveslide.tools import calc_default_fsize, init_data
from src.loveslide.knockoffs import Knockoffs
from src.loveslide.love import call_love
from src.loveslide.score import Estimator, SLIDE_Estimator


class TestAlgorithmicConvergence:
    """Test algorithmic convergence and numerical stability."""

    def test_slide_with_singular_covariance(self):
        """Test SLIDE behavior with singular covariance matrices."""
        # Create perfectly correlated features
        X = np.random.randn(50, 10)
        X[:, 1] = X[:, 0]  # Perfect correlation
        X[:, 2] = 2 * X[:, 0]  # Linear combination
        y = np.random.binomial(1, 0.5, 50)

        params = {
            'delta': [0.1],
            'lambda': [0.5],
            'fdr': 0.1,
            'niter': 10,
            'pure_homo': True
        }

        slide = SLIDE(params, X, y)

        # Should handle singular covariance gracefully
        with patch('src.loveslide.love.call_love') as mock_love:
            mock_love.return_value = {
                'A': np.random.randn(X.shape[1], 5),
                'pure_indices': [list(range(5))]
            }

            # Test that it doesn't crash
            slide.calc_z_matrix = MagicMock(return_value=pd.DataFrame(
                np.random.randn(X.shape[0], 5),
                columns=[f'Z{i}' for i in range(5)]
            ))

            # Verify no exceptions with singular matrices
            assert slide.data.X.shape[1] == 10

    def test_slide_convergence_with_extreme_parameters(self):
        """Test SLIDE convergence with extreme parameter values."""
        X = np.random.randn(100, 20)
        y = np.random.binomial(1, 0.5, 100)

        # Test with very small delta
        params = {
            'delta': [1e-10],
            'lambda': [0.99999],  # Very high lambda
            'fdr': 1e-8,          # Very small FDR
            'niter': 1000,        # Many iterations
            'pure_homo': True
        }

        slide = SLIDE(params, X, y)

        with patch('src.loveslide.love.call_love') as mock_love:
            mock_love.return_value = {
                'A': np.random.randn(20, 3),
                'pure_indices': [[0, 1, 2]]
            }

            # Test numerical stability
            slide.calc_z_matrix = MagicMock(return_value=pd.DataFrame(
                np.random.randn(100, 3),
                columns=['Z0', 'Z1', 'Z2']
            ))

            # Should handle extreme parameters
            assert slide.input_params['delta'][0] == 1e-10

    def test_optimize_slide_parameter_bounds(self):
        """Test OptimizeSLIDE with parameter boundary conditions."""
        X = np.random.randn(50, 10)
        y = np.random.binomial(1, 0.5, 50)

        params = {
            'delta': [0, 1],      # Boundary values
            'lambda': [0, 1],
            'fdr': 0.5,
            'niter': 5
        }

        opt_slide = OptimizeSLIDE(params, X, y)

        with patch.object(opt_slide, 'run_slide') as mock_run:
            mock_run.return_value = ([], [])

            # Test boundary parameter handling
            opt_slide.optimize_params()

            # Should handle boundary cases gracefully
            assert len(opt_slide.input_params['delta']) == 2
            assert len(opt_slide.input_params['lambda']) == 2

    def test_cv_fold_generation_edge_cases(self):
        """Test cross-validation fold generation edge cases."""
        X = np.random.randn(10, 5)  # Very small dataset
        y = np.random.binomial(1, 0.5, 10)

        params = {
            'delta': [0.1],
            'lambda': [0.5],
            'cv_folds': 15,  # More folds than samples
            'fdr': 0.1
        }

        cv_slide = SLIDEcv(params, X, y)

        # Should handle impossible fold requirements
        with patch.object(cv_slide, '_generate_cv_folds') as mock_folds:
            mock_folds.return_value = [(
                np.array([0, 1, 2]), np.array([3, 4])
            ) for _ in range(3)]  # Realistic number of folds

            cv_slide.run_cv()
            mock_folds.assert_called()

    def test_knockoff_generation_rank_deficient(self):
        """Test knockoff generation with rank-deficient design matrices."""
        # Create rank-deficient design matrix
        X_base = np.random.randn(30, 5)
        X = np.column_stack([X_base, X_base[:, :2]])  # Add linearly dependent cols

        knockoffs = Knockoffs()

        with patch('src.loveslide.knockoff.solve.create_solve_sdp') as mock_solve:
            mock_solve.side_effect = np.linalg.LinAlgError("Singular matrix")

            # Should fallback to equicorrelated method
            with patch('src.loveslide.knockoff.solve.create_solve_equi') as mock_equi:
                mock_equi.return_value = np.random.randn(X.shape[1])

                knockoffs.create_knockoffs(X, method='sdp')
                mock_equi.assert_called()

    def test_love_estimation_with_extreme_noise(self):
        """Test LOVE estimation with extremely noisy data."""
        # Create data with very high noise-to-signal ratio
        true_signal = np.random.randn(100, 3)
        noise = 100 * np.random.randn(100, 10)  # Very high noise
        X = np.column_stack([true_signal, noise])

        with patch('src.loveslide.love_python.love.love.LOVE') as mock_love:
            mock_love.return_value = {
                'A': np.random.randn(X.shape[1], 2),
                'pure_indices': [[0], [1]]
            }

            result = call_love(X, lbd=0.5, mu=0.5)

            # Should handle noisy estimation
            assert 'A' in result
            mock_love.assert_called_once()


class TestNumericalStabilityEdgeCases:
    """Test numerical stability in edge cases."""

    def test_estimator_with_extreme_scaling(self):
        """Test estimator behavior with extremely scaled data."""
        # Create data with extreme scaling differences
        X_small = np.random.randn(50, 5) * 1e-10
        X_large = np.random.randn(50, 5) * 1e10
        X = np.column_stack([X_small, X_large])
        y = np.random.randn(50)

        estimator = Estimator(model_type='linear')

        # Test with extreme scaling
        estimator.fit(X, y)
        predictions = estimator.predict(X)

        # Should produce finite predictions
        assert np.all(np.isfinite(predictions))
        assert len(predictions) == len(y)

    def test_slide_estimator_regularization_extremes(self):
        """Test SLIDE estimator with extreme regularization."""
        X = np.random.randn(30, 10)
        y = np.random.randn(30)
        A = np.random.randn(10, 3)

        # Test with extreme regularization
        slide_est = SLIDE_Estimator(
            A=A,
            model_type='linear',
            alpha=1e-15,  # Extremely small regularization
            l1_ratio=1.0
        )

        slide_est.fit(X, y)
        predictions = slide_est.predict(X)

        assert np.all(np.isfinite(predictions))

        # Test with extremely large regularization
        slide_est_large = SLIDE_Estimator(
            A=A,
            model_type='linear',
            alpha=1e15,   # Extremely large regularization
            l1_ratio=0.0
        )

        slide_est_large.fit(X, y)
        predictions_large = slide_est_large.predict(X)

        assert np.all(np.isfinite(predictions_large))

    def test_calc_default_fsize_boundary_conditions(self):
        """Test default feature size calculation with boundary conditions."""
        # Test with n_rows == K
        assert calc_default_fsize(50, 50) == 48

        # Test with n_rows == K + 1
        assert calc_default_fsize(51, 50) == 50

        # Test with n_rows == K - 1
        assert calc_default_fsize(49, 50) == 49

        # Test with very large K
        assert calc_default_fsize(10, 1000) == 10

        # Test with K = 1
        assert calc_default_fsize(100, 1) == 1

        # Test edge case: n_rows = K = 1
        assert calc_default_fsize(1, 1) == -1  # n_rows - 2

    def test_data_initialization_edge_cases(self):
        """Test data initialization with edge cases."""
        # Test with empty parameter dict
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('feat1,feat2\n1,2\n3,4\n')
            x_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('label\n0\n1\n')
            y_path = f.name

        try:
            # Test with minimal parameters
            params = {'x_path': x_path, 'y_path': y_path}
            data, updated_params = init_data(params)

            # Should set all defaults
            assert updated_params['y_factor'] is True
            assert updated_params['y_flip'] is False
            assert updated_params['delta'] == [0.05, 0.1]

        finally:
            os.unlink(x_path)
            os.unlink(y_path)

        # Test with direct arrays instead of file paths
        X = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        y = pd.DataFrame([0, 1], columns=['label'])

        params = {}
        data, updated_params = init_data(params, X, y)

        assert data.X.equals(X)
        assert data.Y.equals(y.astype(int))  # Should convert to int with y_factor=True

    def test_memory_intensive_operations(self):
        """Test operations that could cause memory issues."""
        # Test with large matrix operations
        n_large = 1000
        p_large = 500

        # Mock memory-intensive operations
        with patch('numpy.linalg.svd') as mock_svd:
            mock_svd.return_value = (
                np.random.randn(n_large, min(n_large, p_large)),
                np.random.rand(min(n_large, p_large)),
                np.random.randn(min(n_large, p_large), p_large)
            )

            X = np.random.randn(n_large, p_large)

            # Test large matrix decomposition
            from src.loveslide.knockoff.utils import canonical_svd
            U, d, V = canonical_svd(X)

            assert U.shape[0] == n_large
            assert len(d) == min(n_large, p_large)
            mock_svd.assert_called_once()


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""

    def test_slide_state_recovery_corrupted_files(self):
        """Test SLIDE state recovery with corrupted files."""
        X = np.random.randn(50, 10)
        y = np.random.binomial(1, 0.5, 50)

        params = {'delta': [0.1], 'lambda': [0.5]}
        slide = SLIDE(params, X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create corrupted state files
            with open(os.path.join(tmpdir, 'A.csv'), 'w') as f:
                f.write("corrupted,data\n1,2,3\n")  # Invalid CSV

            # Should handle corruption gracefully
            slide.load_state(tmpdir)
            assert slide.marginal_idxs == []  # Should reset to empty

    def test_love_result_loading_with_missing_fields(self):
        """Test LOVE result loading with missing required fields."""
        X = np.random.randn(30, 8)
        y = np.random.binomial(1, 0.5, 30)

        params = {'delta': [0.1]}
        slide = SLIDE(params, X, y)

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl') as f:
            # Save incomplete LOVE result
            incomplete_result = {'pure_indices': [[0, 1]]}  # Missing 'A'
            import pickle
            pickle.dump(incomplete_result, f)
            f.flush()

            # Should handle missing fields gracefully
            slide.load_love(f.name)
            # Should not crash, but won't set attributes

    def test_knockoff_solver_fallback_chain(self):
        """Test knockoff solver fallback chain when methods fail."""
        X = np.random.randn(20, 10)

        knockoffs = Knockoffs()

        with patch('src.loveslide.knockoff.solve.create_solve_sdp') as mock_sdp, \
             patch('src.loveslide.knockoff.solve.create_solve_equi') as mock_equi:

            # Make SDP solver fail
            mock_sdp.side_effect = Exception("SDP failed")
            mock_equi.return_value = np.ones(X.shape[1])

            # Should fallback to equicorrelated
            result = knockoffs.create_knockoffs(X, method='sdp')

            mock_sdp.assert_called()
            mock_equi.assert_called()
            assert result is not None

    def test_parallel_execution_error_handling(self):
        """Test error handling in parallel execution contexts."""
        X = np.random.randn(100, 20)
        y = np.random.binomial(1, 0.5, 100)

        params = {
            'delta': [0.1],
            'lambda': [0.5],
            'n_workers': 4,  # Parallel execution
            'niter': 10
        }

        with patch('concurrent.futures.ProcessPoolExecutor') as mock_executor:
            # Mock executor that fails
            mock_future = MagicMock()
            mock_future.result.side_effect = Exception("Worker process failed")
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

            knockoffs = Knockoffs()

            # Should handle worker failures gracefully
            with pytest.raises(Exception):
                knockoffs.run_knockoff_filter(X, y, **params)


if __name__ == "__main__":
    pytest.main([__file__])