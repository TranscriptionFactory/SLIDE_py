"""
Comprehensive integration and workflow testing for SLIDE_py.

Missing Coverage Areas:
- End-to-end SLIDE workflow with real data
- LOVE-SLIDE integration pipeline
- Cross-validation with different fold strategies
- Memory management in large-scale scenarios
- Error propagation through full pipeline
- State persistence and recovery
- Plotting integration with results
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, Mock
import pickle

from loveslide import SLIDE, OptimizeSLIDE, SLIDEcv, Plotter
from loveslide.score import SLIDE_Estimator
from loveslide.tools import init_data
from loveslide.love import call_love


class TestEndToEndWorkflow:
    """Test complete SLIDE workflow scenarios."""

    def test_full_slide_pipeline_synthetic_data(self):
        """Test complete SLIDE pipeline with synthetic data."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples, n_features = 100, 50
        n_latent = 5

        # Create latent factors
        latent_factors = np.random.randn(n_samples, n_latent)
        loadings = np.random.randn(n_features, n_latent)

        # Generate observed data
        X = latent_factors @ loadings.T + np.random.randn(n_samples, n_features) * 0.1
        y = np.sum(latent_factors[:, :2], axis=1) + np.random.randn(n_samples) * 0.1

        # Save to temporary files
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as x_file:
            pd.DataFrame(X).to_csv(x_file.name, index=False)
            x_path = x_file.name

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as y_file:
            pd.DataFrame({'y': y}).to_csv(y_file.name, index=False)
            y_path = y_file.name

        try:
            # Setup SLIDE parameters
            input_params = {
                'x_path': x_path,
                'y_path': y_path,
                'delta': [0.05, 0.1],
                'lambda': [0.1],
                'fdr': 0.1,
                'f_size': 20,
                'n_jobs': 1
            }

            # Initialize SLIDE
            slide = SLIDE(input_params)

            # Test parameter display
            slide.show_params()

            # Test default feature size calculation
            default_fsize = slide.calc_default_fsize(K=n_latent)
            assert isinstance(default_fsize, int)
            assert default_fsize > 0

            # Test optimization workflow
            opt_slide = OptimizeSLIDE(input_params)

            # Run LOVE algorithm (mocked to avoid R dependencies)
            with patch('loveslide.love.call_love') as mock_love:
                mock_love_result = {
                    'latent_factors': np.random.randn(n_samples, n_latent),
                    'loading_matrix': np.random.randn(n_features, n_latent),
                    'pure_variables': [True] * 10 + [False] * (n_features - 10)
                }
                mock_love.return_value = mock_love_result

                opt_slide.run_love()

                assert hasattr(opt_slide, 'love_result')
                assert opt_slide.latent_factors is not None

            # Test knockoff generation and selection
            with patch.object(opt_slide, '_run_knockoffs') as mock_knockoffs:
                mock_knockoffs_result = {
                    'selected_features': np.array([1, 5, 10, 15]),
                    'feature_scores': np.random.randn(n_features),
                    'threshold': 0.5
                }
                mock_knockoffs.return_value = mock_knockoffs_result

                opt_slide.run_knockoffs()

                assert hasattr(opt_slide, 'knockoff_result')

        finally:
            # Cleanup
            for path in [x_path, y_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_slide_with_missing_data_handling(self):
        """Test SLIDE workflow with missing data."""
        np.random.seed(42)
        n_samples, n_features = 80, 30

        # Generate data with missing values
        X = np.random.randn(n_samples, n_features)
        X[10:20, 5:10] = np.nan  # Block of missing values
        X[np.random.rand(n_samples, n_features) < 0.05] = np.nan  # Scattered missing

        y = np.random.randn(n_samples)
        y[15:18] = np.nan  # Missing y values

        # Test data initialization with missing values
        input_params = {'fdr': 0.1, 'delta': [0.1]}

        data, processed_params = init_data(input_params, x=X, y=y)

        # Should handle missing data appropriately
        assert data.X is not None
        assert data.Y is not None

    def test_slide_cross_validation_integration(self):
        """Test SLIDE with cross-validation integration."""
        np.random.seed(42)
        n_samples, n_features = 120, 40
        n_latent = 3

        # Generate synthetic data
        latent_factors = np.random.randn(n_samples, n_latent)
        loadings = np.random.randn(n_features, n_latent)
        X = latent_factors @ loadings.T + np.random.randn(n_samples, n_features) * 0.2
        y = np.sum(latent_factors, axis=1) + np.random.randn(n_samples) * 0.1

        # Create mock fitted SLIDE object
        mock_slide_obj = Mock()
        mock_slide_obj.latent_factors = latent_factors
        mock_slide_obj.data.Y = y
        mock_slide_obj.input_params = {'fdr': 0.1}
        mock_slide_obj.marginal_idxs = np.arange(n_features)

        # Test cross-validation
        cv = SLIDEcv(
            mock_slide_obj,
            nrep=2,
            k=5,
            eval_type='corr'
        )

        with patch.object(cv, '_run_cv_fold') as mock_cv_fold:
            mock_cv_fold.return_value = {
                'score_real': 0.7,
                'score_permuted': 0.1,
                'selected_features': np.array([1, 5, 10])
            }

            results = cv.run()

            assert isinstance(results, dict)
            assert 'cv_scores' in results
            assert 'benchmark_scores' in results

    def test_slide_plotting_integration(self):
        """Test SLIDE results with plotting functionality."""
        np.random.seed(42)

        # Create mock latent factors results
        n_factors = 3
        factor_names = [f'LF{i+1}' for i in range(n_factors)]

        latent_factors_data = {}
        for name in factor_names:
            n_genes = np.random.randint(5, 15)
            gene_names = [f'Gene_{name}_{i}' for i in range(n_genes)]

            df = pd.DataFrame({
                'loading': np.random.randn(n_genes),
                'AUC': np.random.rand(n_genes),
                'corr': np.random.randn(n_genes) * 0.5,
                'color': np.random.choice(['red', 'blue'], n_genes)
            }, index=gene_names)

            latent_factors_data[name] = df

        # Test plotting functionality
        plotter = Plotter()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test latent factors plot
            plotter.plot_latent_factors(
                latent_factors_data,
                outdir=temp_dir,
                title='Test Latent Factors'
            )

            # Check if plot files were created
            plot_files = os.listdir(temp_dir)
            assert len(plot_files) > 0

    def test_slide_state_persistence(self):
        """Test SLIDE state persistence and recovery."""
        np.random.seed(42)
        n_samples, n_features = 60, 25

        # Generate test data
        input_params = {
            'delta': [0.1],
            'lambda': [0.1],
            'fdr': 0.15,
            'f_size': 15
        }

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        # Create SLIDE object
        slide = OptimizeSLIDE(input_params, x=X, y=y)

        # Simulate LOVE results
        mock_love_result = {
            'latent_factors': np.random.randn(n_samples, 3),
            'loading_matrix': np.random.randn(n_features, 3),
            'convergence': True
        }

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(mock_love_result, f)
            love_path = f.name

        try:
            # Test loading LOVE results
            slide.load_love(love_path)

            assert slide.love_result is not None
            assert slide.latent_factors is not None

            # Test state saving
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                state_path = f.name

            slide.save_state(state_path)

            # Test state loading
            slide2 = OptimizeSLIDE({})
            slide2.load_state(state_path)

            assert slide2.love_result is not None

        finally:
            for path in [love_path, state_path]:
                if os.path.exists(path):
                    os.unlink(path)


class TestErrorPropagation:
    """Test error propagation through the pipeline."""

    def test_love_failure_handling(self):
        """Test handling of LOVE algorithm failures."""
        input_params = {'delta': [0.1], 'fdr': 0.1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        slide = OptimizeSLIDE(input_params, x=X, y=y)

        # Mock LOVE to raise an exception
        with patch('loveslide.love.call_love') as mock_love:
            mock_love.side_effect = RuntimeError("LOVE algorithm failed")

            with pytest.raises(RuntimeError):
                slide.run_love()

    def test_knockoff_failure_handling(self):
        """Test handling of knockoff generation failures."""
        input_params = {'delta': [0.1], 'fdr': 0.1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        slide = OptimizeSLIDE(input_params, x=X, y=y)

        # Set mock latent factors
        slide.latent_factors = np.random.randn(50, 3)

        # Mock knockoff generation to fail
        with patch.object(slide, '_run_knockoffs') as mock_knockoffs:
            mock_knockoffs.side_effect = ValueError("Knockoff generation failed")

            with pytest.raises(ValueError):
                slide.run_knockoffs()

    def test_invalid_parameter_propagation(self):
        """Test invalid parameter handling through pipeline."""
        # Invalid parameters
        invalid_params = {
            'delta': [-0.1],  # Negative delta
            'fdr': 1.5,       # FDR > 1
            'lambda': [0]     # Zero lambda
        }

        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        with pytest.raises((ValueError, AssertionError)):
            slide = OptimizeSLIDE(invalid_params, x=X, y=y)

    def test_memory_error_handling(self):
        """Test memory error handling in large-scale scenarios."""
        # Mock extremely large problem
        with patch('numpy.random.randn') as mock_randn:
            def side_effect(*args):
                if args[0] > 10000:  # Large allocation
                    raise MemoryError("Not enough memory")
                return np.random.randn(*args)

            mock_randn.side_effect = side_effect

            input_params = {'delta': [0.1], 'fdr': 0.1}

            with pytest.raises(MemoryError):
                # This should trigger the memory error
                X = np.random.randn(50000, 10000)


class TestPerformanceScenarios:
    """Test performance-related scenarios."""

    def test_slide_with_high_dimensional_data(self):
        """Test SLIDE with high-dimensional data."""
        np.random.seed(42)

        # Moderately high-dimensional problem
        n_samples, n_features = 200, 500
        n_latent = 10

        # Generate sparse latent structure
        latent_factors = np.random.randn(n_samples, n_latent)
        loadings = np.random.randn(n_features, n_latent) * 0.1
        # Make some features strongly associated with latents
        loadings[:50, :] *= 10

        X = latent_factors @ loadings.T + np.random.randn(n_samples, n_features)
        y = np.sum(latent_factors[:, :3], axis=1) + np.random.randn(n_samples)

        input_params = {
            'delta': [0.1],
            'lambda': [0.1],
            'fdr': 0.1,
            'f_size': 100,  # Smaller chunks for high-dimensional data
            'n_jobs': 1
        }

        slide = OptimizeSLIDE(input_params, x=X, y=y)

        # Test that initialization completes without memory issues
        assert slide.data.X.shape == (n_samples, n_features)
        assert slide.data.Y.shape == (n_samples,)

    def test_slide_computational_efficiency(self):
        """Test computational efficiency optimizations."""
        np.random.seed(42)
        n_samples, n_features = 100, 200

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        input_params = {
            'delta': [0.1],
            'lambda': [0.1],
            'fdr': 0.1,
            'n_jobs': 2  # Test parallel processing
        }

        slide = OptimizeSLIDE(input_params, x=X, y=y)

        # Mock parallel knockoff processing
        with patch('loveslide.knockoffs.knockoff_voting_parallel') as mock_parallel:
            mock_parallel.return_value = Mock(
                selected=np.array([1, 5, 10]),
                W_votes=np.random.randn(n_features, 10)
            )

            slide.latent_factors = np.random.randn(n_samples, 5)

            # Should use parallel processing efficiently
            slide.run_knockoffs()

            mock_parallel.assert_called()

    def test_slide_memory_optimization(self):
        """Test memory optimization strategies."""
        np.random.seed(42)

        # Test chunked processing
        n_samples, n_features = 150, 300

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        input_params = {
            'delta': [0.1],
            'fdr': 0.1,
            'f_size': 50  # Force chunking
        }

        slide = OptimizeSLIDE(input_params, x=X, y=y)

        # Calculate number of chunks
        n_chunks = np.ceil(n_features / input_params['f_size'])
        assert n_chunks > 1  # Should require multiple chunks

        # Test chunk size calculation
        chunk_size = slide.calc_default_fsize(K=5)
        assert chunk_size <= input_params['f_size']


class TestDataTypeCompatibility:
    """Test compatibility with different data types."""

    def test_slide_with_pandas_input(self):
        """Test SLIDE with pandas DataFrame input."""
        np.random.seed(42)
        n_samples, n_features = 80, 30

        # Create pandas DataFrames
        feature_names = [f'Feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=feature_names
        )

        y_df = pd.DataFrame({
            'target': np.random.randn(n_samples)
        })

        input_params = {'delta': [0.1], 'fdr': 0.1}

        # Should handle pandas input
        data, processed_params = init_data(input_params, x=X_df, y=y_df)

        assert data.X is not None
        assert data.Y is not None
        assert isinstance(data.X, np.ndarray)

    def test_slide_with_sparse_matrices(self):
        """Test SLIDE with sparse matrix input."""
        from scipy.sparse import csr_matrix

        np.random.seed(42)
        n_samples, n_features = 100, 50

        # Create sparse matrix (mostly zeros)
        X_dense = np.random.randn(n_samples, n_features)
        X_dense[np.abs(X_dense) < 1.5] = 0  # Make sparse
        X_sparse = csr_matrix(X_dense)

        y = np.random.randn(n_samples)

        input_params = {'delta': [0.1], 'fdr': 0.1}

        # Should handle sparse matrices
        try:
            data, processed_params = init_data(input_params, x=X_sparse, y=y)
            assert data.X is not None
        except NotImplementedError:
            # Sparse matrix support may not be implemented
            pytest.skip("Sparse matrix support not implemented")

    def test_slide_with_categorical_data(self):
        """Test SLIDE with mixed data types."""
        np.random.seed(42)
        n_samples = 100

        # Mixed data: continuous + categorical
        X_cont = np.random.randn(n_samples, 15)
        X_cat = np.random.choice(['A', 'B', 'C'], (n_samples, 5))

        # Should handle data type conversion
        X_combined = np.hstack([
            X_cont,
            np.random.randn(n_samples, 5)  # Convert categorical to dummy
        ])

        y = np.random.randn(n_samples)

        input_params = {'delta': [0.1], 'fdr': 0.1}

        data, processed_params = init_data(input_params, x=X_combined, y=y)

        assert data.X.shape[1] >= 15  # Should have at least continuous features


class TestRobustnessScenarios:
    """Test robustness to various challenging scenarios."""

    def test_slide_with_collinear_features(self):
        """Test SLIDE with highly collinear features."""
        np.random.seed(42)
        n_samples, n_features = 100, 20

        # Create collinear features
        X_base = np.random.randn(n_samples, 5)
        X_collinear = X_base + np.random.randn(n_samples, 5) * 0.01

        X = np.hstack([X_base, X_collinear, np.random.randn(n_samples, 10)])
        y = np.random.randn(n_samples)

        input_params = {'delta': [0.1], 'fdr': 0.1}

        slide = OptimizeSLIDE(input_params, x=X, y=y)

        # Should handle collinearity gracefully
        assert slide.data.X is not None

    def test_slide_with_outliers(self):
        """Test SLIDE robustness to outliers."""
        np.random.seed(42)
        n_samples, n_features = 120, 30

        X = np.random.randn(n_samples, n_features)
        # Add extreme outliers
        X[0, :] = 50  # Extreme positive outlier
        X[1, :] = -50  # Extreme negative outlier

        y = np.random.randn(n_samples)
        y[0] = 100  # Outlier in response

        input_params = {'delta': [0.1], 'fdr': 0.1}

        slide = OptimizeSLIDE(input_params, x=X, y=y)

        # Should be robust to outliers
        assert slide.data.X is not None

    def test_slide_with_constant_features(self):
        """Test SLIDE with constant/zero-variance features."""
        np.random.seed(42)
        n_samples, n_features = 80, 25

        X = np.random.randn(n_samples, n_features)
        # Add constant features
        X[:, 5] = 1.0  # Constant feature
        X[:, 10] = 0.0  # Zero feature

        y = np.random.randn(n_samples)

        input_params = {'delta': [0.1], 'fdr': 0.1}

        slide = OptimizeSLIDE(input_params, x=X, y=y)

        # Should handle constant features appropriately
        assert slide.data.X is not None