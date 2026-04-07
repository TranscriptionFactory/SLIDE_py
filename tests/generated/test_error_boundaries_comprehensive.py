"""
Comprehensive error boundary testing for SLIDE_py.

This module tests error handling in boundary conditions and exception scenarios
that may not be covered by existing tests.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.loveslide import SLIDE, OptimizeSLIDE, SLIDEcv, Knockoffs
from src.loveslide.tools import init_data, check_params, calc_default_fsize
from src.loveslide.love import call_love
from src.loveslide.score import Estimator, SLIDE_Estimator
from src.loveslide.plotting import Plotter


class TestDataLoadingErrorBoundaries:
    """Test error boundaries in data loading and initialization."""

    def test_init_data_missing_required_paths(self):
        """Test init_data with missing required paths."""
        input_params = {}

        with pytest.raises(ValueError, match="x_path is not provided"):
            init_data(input_params)

    def test_init_data_corrupted_file_reading(self):
        """Test init_data with corrupted file scenarios."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write corrupted CSV
            f.write("col1,col2\n1,2,3,4\ninvalid,data")
            corrupted_file = f.name

        input_params = {'x_path': corrupted_file}

        try:
            with pytest.raises((pd.errors.ParserError, ValueError)):
                init_data(input_params)
        finally:
            os.unlink(corrupted_file)

    def test_init_data_binary_file_as_csv(self):
        """Test init_data with binary file passed as CSV."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as f:
            # Write binary data
            f.write(b'\x00\x01\x02\x03\xff\xfe\xfd')
            binary_file = f.name

        input_params = {'x_path': binary_file}

        try:
            with pytest.raises((UnicodeDecodeError, pd.errors.ParserError)):
                init_data(input_params)
        finally:
            os.unlink(binary_file)

    def test_init_data_permission_denied(self):
        """Test init_data with permission denied scenario."""
        # Create a file then remove read permissions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n1,2\n3,4")
            restricted_file = f.name

        # Remove read permissions
        os.chmod(restricted_file, 0o000)

        input_params = {'x_path': restricted_file}

        try:
            with pytest.raises(PermissionError):
                init_data(input_params)
        finally:
            # Restore permissions and cleanup
            os.chmod(restricted_file, 0o644)
            os.unlink(restricted_file)

    def test_init_data_empty_file(self):
        """Test init_data with empty CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write only headers
            f.write("")
            empty_file = f.name

        input_params = {'x_path': empty_file}

        try:
            with pytest.raises((pd.errors.EmptyDataError, ValueError)):
                init_data(input_params)
        finally:
            os.unlink(empty_file)

    def test_init_data_mismatched_dimensions(self):
        """Test init_data with mismatched X and y dimensions."""
        # Create X file with 100 rows
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("x1,x2\n")
            for i in range(100):
                f.write(f"{i},{i+1}\n")
            x_file = f.name

        # Create y file with 50 rows
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("y\n")
            for i in range(50):
                f.write(f"{i}\n")
            y_file = f.name

        input_params = {'x_path': x_file, 'y_path': y_file}

        try:
            with pytest.raises(ValueError, match="dimension mismatch"):
                init_data(input_params)
        finally:
            os.unlink(x_file)
            os.unlink(y_file)


class TestSLIDEErrorBoundaries:
    """Test error boundaries in SLIDE algorithm execution."""

    def test_slide_initialization_with_invalid_data_types(self):
        """Test SLIDE initialization with invalid data types."""
        # Invalid input_params type
        with pytest.raises((TypeError, AttributeError)):
            SLIDE("invalid_string_param")

    def test_slide_calc_default_fsize_invalid_k(self):
        """Test calc_default_fsize with invalid K values."""
        with patch('src.loveslide.slide.init_data') as mock_init:
            mock_init.return_value = (
                {'X': np.random.randn(100, 50)}, {}
            )
            slide = SLIDE({})

            # Test with K <= 0
            with pytest.raises(ValueError):
                slide.calc_default_fsize(0)

            with pytest.raises(ValueError):
                slide.calc_default_fsize(-5)

            # Test with K greater than feature count
            result = slide.calc_default_fsize(100)
            assert result == 50  # Should be capped at number of features

    def test_slide_run_with_insufficient_memory(self):
        """Test SLIDE.run() under simulated memory constraints."""
        with patch('src.loveslide.slide.init_data') as mock_init:
            mock_init.return_value = (
                {'X': np.random.randn(1000, 500), 'y': np.random.randn(1000)},
                {'fsize': 100}
            )
            slide = SLIDE({})

            # Simulate memory error during processing
            with patch.object(slide, '_find_interaction_LFs_batch') as mock_batch:
                mock_batch.side_effect = MemoryError("Insufficient memory")

                with pytest.raises(MemoryError):
                    slide.run()

    def test_slide_run_with_singular_matrix(self):
        """Test SLIDE.run() with singular covariance matrix."""
        # Create perfectly correlated features
        X = np.ones((100, 10))  # All features identical
        y = np.random.randn(100)

        with patch('src.loveslide.slide.init_data') as mock_init:
            mock_init.return_value = ({'X': X, 'y': y}, {'fsize': 5})
            slide = SLIDE({})

            # Should handle singular matrix gracefully
            with pytest.warns(UserWarning):
                result = slide.run()

    def test_slide_invalid_latent_factor_access(self):
        """Test SLIDE with invalid latent factor access."""
        with patch('src.loveslide.slide.init_data') as mock_init:
            mock_init.return_value = (
                {'X': np.random.randn(50, 20), 'y': np.random.randn(50)}, {}
            )
            slide = SLIDE({})
            slide.A = np.random.randn(20, 5)  # 5 latent factors

            with pytest.raises(ValueError, match="Latent factor .* not found"):
                slide.get_feature_candidates(10)  # LF 10 doesn't exist


class TestKnockoffErrorBoundaries:
    """Test error boundaries in Knockoffs functionality."""

    def test_knockoffs_unsupported_model(self):
        """Test Knockoffs with unsupported model type."""
        X = np.random.randn(50, 10)
        knockoffs = Knockoffs(X)

        with pytest.raises(ValueError, match="Model not supported"):
            knockoffs.filter(
                y=np.random.randn(50),
                model='unsupported_model',
                statistic='lasso_lambdadiff'
            )

    def test_knockoffs_r_import_failure(self):
        """Test Knockoffs behavior when R import fails."""
        X = np.random.randn(30, 8)

        # Mock R import failure
        with patch('src.loveslide.knockoffs.importr') as mock_import:
            mock_import.side_effect = ImportError("R package not available")

            knockoffs = Knockoffs(X)

            # Should fallback to Python implementation
            with pytest.warns(UserWarning, match="falling back to Python"):
                result = knockoffs.filter(
                    y=np.random.randn(30),
                    model='second_order'
                )

    def test_knockoffs_sdp_solver_failure(self):
        """Test Knockoffs when SDP solver fails."""
        X = np.random.randn(20, 15)
        knockoffs = Knockoffs(X)

        # Mock SDP solver to always fail
        with patch('src.loveslide.knockoffs._solve_sdp_r') as mock_solve:
            mock_solve.side_effect = RuntimeError("SDP solver failed")

            # Should fallback to equicorrelated method
            with pytest.warns(UserWarning):
                result = knockoffs.filter(
                    y=np.random.randn(20),
                    model='second_order'
                )

    def test_knockoffs_extreme_correlation_structure(self):
        """Test Knockoffs with extreme correlation structures."""
        # Nearly perfect correlation
        X_base = np.random.randn(100, 1)
        noise = np.random.randn(100, 10) * 1e-10
        X = np.hstack([X_base] * 10) + noise

        knockoffs = Knockoffs(X)

        # Should handle near-singular correlation matrix
        with pytest.warns(UserWarning):
            result = knockoffs.filter(
                y=np.random.randn(100),
                model='equi'
            )


class TestCrossValidationErrorBoundaries:
    """Test error boundaries in cross-validation."""

    def test_slidecv_invalid_fold_specification(self):
        """Test SLIDEcv with invalid fold specifications."""
        input_params = {'n_folds': -1}

        with patch('src.loveslide.cv.init_data') as mock_init:
            mock_init.return_value = (
                {'X': np.random.randn(50, 10), 'y': np.random.randn(50)},
                input_params
            )

            with pytest.raises(ValueError):
                SLIDEcv(input_params)

    def test_slidecv_insufficient_samples_for_folds(self):
        """Test SLIDEcv when there are insufficient samples for the number of folds."""
        input_params = {'n_folds': 10}  # More folds than samples

        with patch('src.loveslide.cv.init_data') as mock_init:
            mock_init.return_value = (
                {'X': np.random.randn(5, 10), 'y': np.random.randn(5)},  # Only 5 samples
                input_params
            )

            cv = SLIDEcv(input_params)

            with pytest.raises(ValueError, match="insufficient samples"):
                cv.run()

    def test_slidecv_fold_validation_failure(self):
        """Test SLIDEcv when fold validation fails."""
        input_params = {'n_folds': 3}

        with patch('src.loveslide.cv.init_data') as mock_init:
            mock_init.return_value = (
                {'X': np.random.randn(50, 10), 'y': np.random.randn(50)},
                input_params
            )

            cv = SLIDEcv(input_params)

            # Mock invalid folds
            with patch.object(cv, '_folds_valid', return_value=False):
                with pytest.raises(ValueError, match="Invalid fold"):
                    cv.run()


class TestEstimatorErrorBoundaries:
    """Test error boundaries in Estimator classes."""

    def test_estimator_unsupported_model_type(self):
        """Test Estimator with unsupported model type."""
        with pytest.raises(ValueError, match="Invalid model"):
            estimator = Estimator(model_type='unsupported_model')
            estimator._init_model(np.random.randn(50))

    def test_estimator_predict_before_fit(self):
        """Test Estimator prediction before fitting."""
        estimator = Estimator(model_type='sklearn_linear')
        X = np.random.randn(30, 5)

        with pytest.raises(AttributeError):
            estimator.predict(X)

    def test_estimator_mismatched_dimensions_predict(self):
        """Test Estimator with mismatched dimensions in prediction."""
        estimator = Estimator(model_type='sklearn_linear')
        X_train = np.random.randn(50, 10)
        y_train = np.random.randn(50)
        X_test = np.random.randn(20, 5)  # Wrong number of features

        estimator.fit(X_train, y_train)

        with pytest.raises(ValueError):
            estimator.predict(X_test)

    def test_slide_estimator_initialization_errors(self):
        """Test SLIDE_Estimator initialization errors."""
        # Missing required parameters
        with pytest.raises((KeyError, TypeError)):
            SLIDE_Estimator()

        # Invalid love_mode parameter
        with pytest.raises(ValueError):
            SLIDE_Estimator(love_mode="invalid", fsize=10)


class TestLOVEErrorBoundaries:
    """Test error boundaries in LOVE algorithm."""

    def test_call_love_r_dependency_missing(self):
        """Test call_love when R dependencies are missing."""
        X = np.random.randn(50, 10)

        # Mock missing R package
        with patch('src.loveslide.love.importr') as mock_import:
            mock_import.side_effect = ImportError("R package not found")

            with pytest.raises(ImportError, match="R package not found"):
                call_love(X, implementation='R')

    def test_call_love_invalid_parameters(self):
        """Test call_love with invalid parameter combinations."""
        X = np.random.randn(30, 8)

        # Invalid FDR threshold
        with pytest.raises(ValueError):
            call_love(X, thresh_fdr=1.5)  # > 1.0

        with pytest.raises(ValueError):
            call_love(X, thresh_fdr=-0.1)  # < 0.0

        # Invalid lambda parameter
        with pytest.raises(ValueError):
            call_love(X, lbd=-0.5)  # < 0.0

    def test_call_love_singular_input_matrix(self):
        """Test call_love with singular input matrix."""
        # Create rank-deficient matrix
        X = np.ones((100, 10))  # All columns identical

        with pytest.warns(UserWarning):
            result = call_love(X, implementation='python')

    def test_call_love_extreme_dimensions(self):
        """Test call_love with extreme dimension ratios."""
        # More features than samples
        X = np.random.randn(10, 100)

        with pytest.warns(UserWarning):
            result = call_love(X, implementation='python')


class TestPlottingErrorBoundaries:
    """Test error boundaries in plotting functionality."""

    def test_plotter_missing_data(self):
        """Test Plotter with missing required data."""
        plotter = Plotter()

        # Try to plot without setting A matrix
        with pytest.raises(AttributeError):
            plotter.latent_factors()

    def test_plotter_invalid_plot_parameters(self):
        """Test Plotter with invalid plotting parameters."""
        plotter = Plotter()
        plotter.A = np.random.randn(20, 5)

        # Invalid figure size
        with pytest.raises((ValueError, TypeError)):
            plotter.latent_factors(figsize="invalid")

        # Invalid color scheme
        with pytest.raises((ValueError, KeyError)):
            plotter.latent_factors(cmap="nonexistent_colormap")

    def test_plotter_save_permission_error(self):
        """Test Plotter save with permission errors."""
        plotter = Plotter()
        plotter.A = np.random.randn(10, 3)

        # Try to save to restricted directory
        with pytest.raises((PermissionError, OSError)):
            plotter.latent_factors(save_path="/root/restricted_plot.png")


class TestParameterValidationErrorBoundaries:
    """Test parameter validation error boundaries."""

    def test_check_params_invalid_structure(self):
        """Test check_params with invalid parameter structures."""
        # None input
        with pytest.raises((TypeError, AttributeError)):
            check_params(None, {})

        # Non-dict input
        with pytest.raises((TypeError, AttributeError)):
            check_params("invalid", {})

    def test_calc_default_fsize_edge_cases(self):
        """Test calc_default_fsize with edge cases."""
        # Zero rows
        with pytest.raises(ValueError):
            calc_default_fsize(0, 10)

        # Zero K
        with pytest.raises(ValueError):
            calc_default_fsize(100, 0)

        # Negative values
        with pytest.raises(ValueError):
            calc_default_fsize(-10, 5)

        with pytest.raises(ValueError):
            calc_default_fsize(100, -5)


if __name__ == "__main__":
    pytest.main([__file__])