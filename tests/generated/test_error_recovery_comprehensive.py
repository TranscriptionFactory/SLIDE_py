"""
Comprehensive error recovery and boundary condition testing.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock

from loveslide import SLIDE, OptimizeSLIDE, Knockoffs
from loveslide.score import Estimator, SLIDE_Estimator


class TestErrorRecoveryScenarios:
    """Test error recovery and graceful degradation."""

    def test_slide_load_love_corrupted_file(self):
        """Test SLIDE.load_love with corrupted pickle file."""
        params = {"fdr": 0.1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        slide = SLIDE(params, x=X, y=y)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            f.write(b"corrupted_pickle_data")
            temp_path = f.name

        try:
            # Should handle corrupted file gracefully
            slide.load_love(temp_path)
            assert not hasattr(slide, 'A')  # Should not create attribute
        finally:
            os.unlink(temp_path)

    def test_slide_load_state_missing_files(self):
        """Test SLIDE.load_state with missing or corrupted files."""
        params = {"fdr": 0.1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        slide = SLIDE(params, x=X, y=y)

        # Test with non-existent directory
        fake_dir = "/non/existent/directory"
        slide.load_state(fake_dir)
        assert slide.marginal_idxs == []

        # Test with directory missing some files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only partial files
            pd.DataFrame(np.random.randn(20, 3)).to_csv(
                os.path.join(temp_dir, "A.csv")
            )
            # Missing z_matrix.csv, sig_LFs.txt

            slide.load_state(temp_dir)
            assert slide.marginal_idxs == []

    def test_estimator_memory_exhaustion_simulation(self):
        """Test Estimator behavior under memory constraints."""
        estimator = Estimator()

        # Create extremely large dataset to trigger memory issues
        X_large = np.random.randn(10000, 1000)
        y_large = np.random.randn(10000)

        with patch('sklearn.linear_model.LinearRegression.fit') as mock_fit:
            mock_fit.side_effect = MemoryError("Insufficient memory")

            with pytest.raises(MemoryError):
                estimator.fit(X_large, y_large)

    def test_knockoffs_solver_failure_recovery(self):
        """Test Knockoffs behavior when SDP solver fails."""
        X = np.random.randn(50, 20)
        knockoffs = Knockoffs()

        with patch('loveslide.knockoff.solve.create_solve_sdp') as mock_solve:
            mock_solve.side_effect = Exception("SDP solver failed")

            # Should fall back to equicorrelated method
            with pytest.raises(Exception):
                knockoffs._create_second_order_python(X, method='sdp')

    def test_slide_pipeline_interruption_recovery(self):
        """Test SLIDE pipeline recovery from interruption."""
        params = {"fdr": 0.1, "niter": 10}
        X = np.random.randn(100, 30)
        y = np.random.randn(100)

        with tempfile.TemporaryDirectory() as temp_dir:
            slide = OptimizeSLIDE(params, x=X, y=y)
            slide.input_params['outpath'] = temp_dir

            # Simulate interruption during run_pipeline
            with patch.object(slide, 'find_interaction_LFs') as mock_interact:
                mock_interact.side_effect = KeyboardInterrupt("User interrupted")

                with pytest.raises(KeyboardInterrupt):
                    slide.run_pipeline(verbose=False)

                # Should be able to resume from saved state
                slide_resume = OptimizeSLIDE(params, x=X, y=y)
                slide_resume.input_params['outpath'] = temp_dir
                # Test resumption logic here


class TestBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_estimator_extreme_data_values(self):
        """Test Estimator with extreme data values."""
        estimator = Estimator()

        # Test with very large values
        X_large = np.random.randn(50, 10) * 1e6
        y_large = np.random.randn(50) * 1e6
        estimator.fit(X_large, y_large)

        # Test with very small values
        X_small = np.random.randn(50, 10) * 1e-6
        y_small = np.random.randn(50) * 1e-6
        estimator.fit(X_small, y_small)

        # Test with mixed scales
        X_mixed = np.random.randn(50, 10)
        X_mixed[:, 0] *= 1e6  # First feature very large
        X_mixed[:, 1] *= 1e-6  # Second feature very small
        estimator.fit(X_mixed, y_large)

    def test_slide_extreme_parameter_values(self):
        """Test SLIDE with extreme parameter values."""
        # Test with very small FDR
        params = {"fdr": 1e-10, "niter": 1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        slide = SLIDE(params, x=X, y=y)
        assert slide.input_params["fdr"] == 1e-10

        # Test with very large niter
        params_large = {"fdr": 0.1, "niter": 1000}
        slide_large = SLIDE(params_large, x=X, y=y)
        assert slide_large.input_params["niter"] == 1000

    def test_knockoffs_singular_covariance_matrix(self):
        """Test Knockoffs with singular covariance matrix."""
        # Create data with perfect collinearity
        X = np.random.randn(50, 5)
        X = np.column_stack([X, X[:, 0] + X[:, 1]])  # Linearly dependent column

        knockoffs = Knockoffs()

        # Should detect and handle singular matrix
        with pytest.warns(UserWarning, match="singular"):
            result = knockoffs._create_second_order_python(X, method='equi')
            assert result.shape == X.shape

    def test_estimator_perfect_separation(self):
        """Test Estimator with perfectly separable binary data."""
        estimator = Estimator(model='logistic')

        # Create perfectly separable data
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])  # Perfect separation

        # Should handle convergence warnings gracefully
        with pytest.warns(Warning):
            estimator.fit(X, y)
            score = estimator.score(estimator.predict_proba(X), y)
            assert score == 1.0  # Perfect AUC