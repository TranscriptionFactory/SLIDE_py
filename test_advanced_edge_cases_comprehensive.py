"""
Advanced edge cases and boundary condition testing for SLIDE_py.

Missing Coverage Areas:
- Numerical precision edge cases
- Boundary value testing for parameters
- Algorithmic convergence edge cases
- Platform-specific behavior
- Extreme data characteristics
- Resource exhaustion scenarios
"""
import pytest
import numpy as np
import pandas as pd
import sys
import gc
from unittest.mock import patch, Mock
import warnings

from loveslide import SLIDE, OptimizeSLIDE, SLIDEcv
from loveslide.score import Estimator, SLIDE_Estimator
from loveslide.cv import SLIDEcv
from loveslide.tools import init_data, calc_default_fsize
from loveslide.knockoffs import Knockoffs
from loveslide.love_python.love.love import LOVE


class TestNumericalPrecision:
    """Test numerical precision and floating-point edge cases."""

    def test_slide_with_very_small_values(self):
        """Test SLIDE with very small numerical values."""
        np.random.seed(42)
        n_samples, n_features = 100, 20

        # Very small values near machine epsilon
        X = np.random.randn(n_samples, n_features) * 1e-15
        y = np.random.randn(n_samples) * 1e-15

        input_params = {'delta': [1e-16], 'fdr': 0.1}

        # Should handle very small values without numerical issues
        data, processed_params = init_data(input_params, x=X, y=y)

        assert np.all(np.isfinite(data.X))
        assert np.all(np.isfinite(data.Y))

    def test_slide_with_very_large_values(self):
        """Test SLIDE with very large numerical values."""
        np.random.seed(42)
        n_samples, n_features = 80, 15

        # Very large values
        scale = 1e10
        X = np.random.randn(n_samples, n_features) * scale
        y = np.random.randn(n_samples) * scale

        input_params = {'delta': [0.1], 'fdr': 0.1}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May generate numerical warnings

            data, processed_params = init_data(input_params, x=X, y=y)

            assert data.X is not None
            assert data.Y is not None

    def test_slide_with_extreme_correlations(self):
        """Test SLIDE with extreme correlation structures."""
        np.random.seed(42)
        n_samples, n_features = 100, 10

        # Create perfectly correlated features
        base_feature = np.random.randn(n_samples)
        X = np.column_stack([
            base_feature,
            base_feature + np.random.randn(n_samples) * 1e-12,  # Almost identical
            base_feature * -1,  # Perfect negative correlation
            np.random.randn(n_samples, n_features - 3)
        ])

        y = np.random.randn(n_samples)

        input_params = {'delta': [0.1], 'fdr': 0.1}

        slide = OptimizeSLIDE(input_params, x=X, y=y)

        # Should handle extreme correlations
        assert slide.data.X is not None

    def test_estimator_with_near_singular_matrices(self):
        """Test estimator with near-singular covariance matrices."""
        np.random.seed(42)
        n_samples, n_features = 50, 40

        # Create near-singular design matrix
        X_base = np.random.randn(n_samples, 20)
        X_dependent = X_base[:, :20] + np.random.randn(n_samples, 20) * 1e-10
        X = np.hstack([X_base, X_dependent])

        y = np.random.randn(n_samples)

        estimator = Estimator()

        # Should handle near-singular matrices gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            estimator.fit(X, y)

            # Should complete without crashing
            assert estimator.model is not None

    def test_love_algorithm_convergence_edge_cases(self):
        """Test LOVE algorithm with convergence edge cases."""
        np.random.seed(42)

        # Data that's difficult to converge
        n_samples, n_features = 100, 30
        X = np.random.randn(n_samples, n_features) * 0.01  # Very small signal

        # Mock LOVE algorithm with convergence issues
        love = LOVE()

        with patch.object(love, '_check_convergence') as mock_conv:
            mock_conv.return_value = False  # Never converges

            # Should handle non-convergence gracefully
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                try:
                    result = love.fit(X, max_iter=5)  # Limited iterations
                    # Should return some result even without convergence
                    assert result is not None
                except RuntimeError:
                    # Acceptable to raise error for non-convergence
                    pass

    def test_knockoffs_with_degenerate_covariance(self):
        """Test knockoffs with degenerate covariance structures."""
        np.random.seed(42)
        n_samples, n_features = 60, 20

        # Create rank-deficient data
        X_rank_def = np.random.randn(n_samples, 10)
        X = np.hstack([X_rank_def, X_rank_def])  # Duplicate columns

        y = np.random.randn(n_samples)

        knockoffs = Knockoffs()

        # Should handle rank deficiency
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                result = knockoffs.fit(X, y)
                assert result is not None
            except (np.linalg.LinAlgError, ValueError):
                # Acceptable to fail with degenerate covariance
                pass


class TestBoundaryValues:
    """Test boundary values for parameters and inputs."""

    def test_slide_with_zero_delta(self):
        """Test SLIDE with delta=0."""
        np.random.seed(42)
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        input_params = {'delta': [0.0], 'fdr': 0.1}

        # Should handle delta=0 boundary case
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            slide = OptimizeSLIDE(input_params, x=X, y=y)
            assert slide.input_params['delta'] == [0.0]

    def test_slide_with_maximum_delta(self):
        """Test SLIDE with delta=1."""
        np.random.seed(42)
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        input_params = {'delta': [1.0], 'fdr': 0.1}

        slide = OptimizeSLIDE(input_params, x=X, y=y)
        assert slide.input_params['delta'] == [1.0]

    def test_slide_with_extreme_fdr(self):
        """Test SLIDE with extreme FDR values."""
        np.random.seed(42)
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        # Test minimum FDR
        input_params = {'delta': [0.1], 'fdr': 1e-10}

        slide = OptimizeSLIDE(input_params, x=X, y=y)
        assert slide.input_params['fdr'] == 1e-10

        # Test maximum FDR (close to 1)
        input_params = {'delta': [0.1], 'fdr': 0.999}

        slide = OptimizeSLIDE(input_params, x=X, y=y)
        assert slide.input_params['fdr'] == 0.999

    def test_slide_with_minimum_feature_size(self):
        """Test SLIDE with f_size=1."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        input_params = {'delta': [0.1], 'fdr': 0.1, 'f_size': 1}

        slide = OptimizeSLIDE(input_params, x=X, y=y)

        # Should handle minimum chunk size
        assert slide.input_params['f_size'] == 1

    def test_slide_with_maximum_feature_size(self):
        """Test SLIDE with f_size larger than number of features."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        input_params = {'delta': [0.1], 'fdr': 0.1, 'f_size': 100}

        slide = OptimizeSLIDE(input_params, x=X, y=y)

        # Should handle oversized chunk
        effective_fsize = slide.calc_default_fsize(K=3)
        assert effective_fsize <= X.shape[1]

    def test_estimator_with_extreme_learning_rates(self):
        """Test estimator with extreme learning rates/regularization."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        # Test with very high regularization
        estimator = SLIDE_Estimator(regularization=1e10)

        estimator.fit(X, y)

        # Should handle extreme regularization
        predictions = estimator.predict(X)
        assert np.all(np.isfinite(predictions))

    def test_cv_with_extreme_fold_numbers(self):
        """Test cross-validation with extreme fold numbers."""
        np.random.seed(42)
        n_samples = 100

        # Create mock slide object
        mock_slide_obj = Mock()
        mock_slide_obj.latent_factors = np.random.randn(n_samples, 5)
        mock_slide_obj.data.Y = np.random.randn(n_samples)
        mock_slide_obj.input_params = {'fdr': 0.1}
        mock_slide_obj.marginal_idxs = np.arange(20)

        # Test with k=2 (minimum meaningful CV)
        cv = SLIDEcv(mock_slide_obj, nrep=1, k=2)

        with patch.object(cv, '_run_cv_fold') as mock_cv_fold:
            mock_cv_fold.return_value = {
                'score_real': 0.5,
                'score_permuted': 0.1,
                'selected_features': np.array([1, 5])
            }

            results = cv.run()
            assert isinstance(results, dict)

        # Test with k=n_samples (leave-one-out)
        cv = SLIDEcv(mock_slide_obj, nrep=1, k=n_samples)

        with patch.object(cv, '_run_cv_fold') as mock_cv_fold:
            mock_cv_fold.return_value = {
                'score_real': 0.5,
                'score_permuted': 0.1,
                'selected_features': np.array([1])
            }

            # Should handle leave-one-out CV
            results = cv.run()
            assert isinstance(results, dict)


class TestExtremeDataCharacteristics:
    """Test with extreme data characteristics."""

    def test_slide_with_single_sample(self):
        """Test SLIDE with n=1 sample."""
        X = np.random.randn(1, 20)
        y = np.random.randn(1)

        input_params = {'delta': [0.1], 'fdr': 0.1}

        # Should handle single sample gracefully (or raise appropriate error)
        with pytest.raises((ValueError, RuntimeError)):
            slide = OptimizeSLIDE(input_params, x=X, y=y)

    def test_slide_with_single_feature(self):
        """Test SLIDE with p=1 feature."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = np.random.randn(100)

        input_params = {'delta': [0.1], 'fdr': 0.1}

        slide = OptimizeSLIDE(input_params, x=X, y=y)

        # Should handle single feature
        assert slide.data.X.shape[1] == 1

    def test_slide_with_more_features_than_samples(self):
        """Test SLIDE with p >> n scenario."""
        np.random.seed(42)
        n_samples, n_features = 50, 200

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        input_params = {'delta': [0.1], 'fdr': 0.1, 'f_size': 30}

        slide = OptimizeSLIDE(input_params, x=X, y=y)

        # Should handle high-dimensional case
        assert slide.data.X.shape == (n_samples, n_features)

    def test_slide_with_all_zero_features(self):
        """Test SLIDE with all-zero feature matrix."""
        n_samples, n_features = 100, 20

        X = np.zeros((n_samples, n_features))
        y = np.random.randn(n_samples)

        input_params = {'delta': [0.1], 'fdr': 0.1}

        # Should handle all-zero features appropriately
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            slide = OptimizeSLIDE(input_params, x=X, y=y)
            assert slide.data.X is not None

    def test_slide_with_all_zero_response(self):
        """Test SLIDE with all-zero response."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.zeros(100)

        input_params = {'delta': [0.1], 'fdr': 0.1}

        slide = OptimizeSLIDE(input_params, x=X, y=y)

        # Should handle zero response
        assert np.all(slide.data.Y == 0)

    def test_slide_with_identical_samples(self):
        """Test SLIDE with all identical samples."""
        n_samples, n_features = 100, 20

        # All rows are identical
        sample = np.random.randn(n_features)
        X = np.tile(sample, (n_samples, 1))
        y = np.ones(n_samples)  # Constant response

        input_params = {'delta': [0.1], 'fdr': 0.1}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            slide = OptimizeSLIDE(input_params, x=X, y=y)
            assert slide.data.X is not None

    def test_slide_with_infinite_values(self):
        """Test SLIDE with infinite values in data."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        # Introduce infinite values
        X[10, 5] = np.inf
        X[20, 10] = -np.inf
        y[15] = np.inf

        input_params = {'delta': [0.1], 'fdr': 0.1}

        # Should handle infinite values (clean or raise error)
        with pytest.raises((ValueError, RuntimeError)):
            slide = OptimizeSLIDE(input_params, x=X, y=y)

    def test_slide_with_nan_values(self):
        """Test SLIDE with NaN values in data."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        # Introduce NaN values
        X[5:10, 2:5] = np.nan
        y[25:30] = np.nan

        input_params = {'delta': [0.1], 'fdr': 0.1}

        # Should handle NaN values appropriately
        data, processed_params = init_data(input_params, x=X, y=y)

        # Either clean the data or raise appropriate error
        assert data.X is not None
        assert data.Y is not None


class TestResourceExhaustionScenarios:
    """Test resource exhaustion and system limits."""

    def test_slide_memory_pressure_simulation(self):
        """Test SLIDE under simulated memory pressure."""
        np.random.seed(42)

        # Moderately sized problem that might stress memory
        n_samples, n_features = 500, 100

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        input_params = {'delta': [0.1], 'fdr': 0.1, 'f_size': 20}

        # Mock memory allocation to raise MemoryError
        with patch('numpy.zeros') as mock_zeros:
            def memory_pressure_side_effect(*args, **kwargs):
                # Allow small allocations, fail on large ones
                total_size = np.prod(args)
                if total_size > 10000:
                    raise MemoryError("Simulated memory pressure")
                return np.zeros(*args, **kwargs)

            mock_zeros.side_effect = memory_pressure_side_effect

            # Should handle memory pressure gracefully
            with pytest.raises(MemoryError):
                slide = OptimizeSLIDE(input_params, x=X, y=y)

    def test_slide_with_maximum_iterations(self):
        """Test SLIDE with iteration limits."""
        np.random.seed(42)
        X = np.random.randn(100, 30)
        y = np.random.randn(100)

        # Mock LOVE to never converge
        with patch('loveslide.love_python.love.love.LOVE') as mock_love_class:
            mock_love = Mock()
            mock_love.fit.return_value = {
                'latent_factors': np.random.randn(100, 5),
                'convergence': False,  # Never converges
                'iterations': 1000     # Maximum iterations
            }
            mock_love_class.return_value = mock_love

            slide = OptimizeSLIDE({'delta': [0.1], 'fdr': 0.1}, x=X, y=y)

            # Should handle max iterations gracefully
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                slide.run_love()

                # Should complete despite non-convergence
                assert slide.love_result is not None

    def test_slide_thread_safety(self):
        """Test SLIDE thread safety with concurrent access."""
        import threading
        import time

        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        input_params = {'delta': [0.1], 'fdr': 0.1}

        results = []
        errors = []

        def worker():
            try:
                slide = OptimizeSLIDE(input_params.copy(), x=X.copy(), y=y.copy())
                results.append(slide)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should handle concurrent access
        assert len(errors) == 0 or all(isinstance(e, (ValueError, RuntimeError)) for e in errors)
        assert len(results) <= 3

    def test_slide_garbage_collection_stress(self):
        """Test SLIDE with aggressive garbage collection."""
        np.random.seed(42)

        # Force garbage collection frequently
        original_threshold = gc.get_threshold()

        try:
            gc.set_threshold(1, 1, 1)  # Very aggressive GC

            for i in range(5):
                X = np.random.randn(50, 15)
                y = np.random.randn(50)

                slide = OptimizeSLIDE({'delta': [0.1], 'fdr': 0.1}, x=X, y=y)

                # Force GC
                gc.collect()

                # Should handle frequent GC
                assert slide.data.X is not None

                # Explicitly delete to test cleanup
                del slide, X, y

        finally:
            gc.set_threshold(*original_threshold)


class TestPlatformSpecificBehavior:
    """Test platform-specific behaviors."""

    def test_slide_with_different_numpy_dtypes(self):
        """Test SLIDE with different numpy data types."""
        np.random.seed(42)

        dtypes = [np.float32, np.float64, np.int32, np.int64]

        for dtype in dtypes:
            if np.issubdtype(dtype, np.integer):
                # Integer data
                X = np.random.randint(-10, 10, (50, 20), dtype=dtype)
                y = np.random.randint(-5, 5, 50, dtype=dtype)
            else:
                # Float data
                X = np.random.randn(50, 20).astype(dtype)
                y = np.random.randn(50).astype(dtype)

            input_params = {'delta': [0.1], 'fdr': 0.1}

            # Should handle different dtypes
            data, processed_params = init_data(input_params, x=X, y=y)

            assert data.X is not None
            assert data.Y is not None

    def test_slide_with_different_array_orders(self):
        """Test SLIDE with different array memory layouts."""
        np.random.seed(42)
        n_samples, n_features = 100, 20

        # C-order (row-major)
        X_c = np.random.randn(n_samples, n_features)
        assert X_c.flags.c_contiguous

        # Fortran-order (column-major)
        X_f = np.asfortranarray(X_c)
        assert X_f.flags.f_contiguous

        y = np.random.randn(n_samples)

        input_params = {'delta': [0.1], 'fdr': 0.1}

        # Should handle both memory layouts
        for X in [X_c, X_f]:
            data, processed_params = init_data(input_params, x=X, y=y)
            assert data.X is not None

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_slide_with_unix_specific_features(self):
        """Test SLIDE with Unix-specific features."""
        # This could test features like fork(), memory mapping, etc.
        # For now, just ensure basic functionality on Unix
        np.random.seed(42)
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        input_params = {'delta': [0.1], 'fdr': 0.1}

        slide = OptimizeSLIDE(input_params, x=X, y=y)
        assert slide.data.X is not None

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_slide_with_windows_specific_features(self):
        """Test SLIDE with Windows-specific features."""
        # Test Windows-specific behaviors
        np.random.seed(42)
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        input_params = {'delta': [0.1], 'fdr': 0.1}

        slide = OptimizeSLIDE(input_params, x=X, y=y)
        assert slide.data.X is not None


class TestAlgorithmicEdgeCases:
    """Test algorithmic edge cases and convergence scenarios."""

    def test_slide_with_pathological_data_distributions(self):
        """Test SLIDE with pathological data distributions."""
        np.random.seed(42)

        # Heavy-tailed distribution
        X_heavy = np.random.standard_t(df=2, size=(100, 20))
        y_heavy = np.random.standard_t(df=2, size=100)

        # Skewed distribution
        X_skew = np.random.exponential(scale=2, size=(100, 20))
        y_skew = np.random.exponential(scale=1, size=100)

        # Multimodal distribution
        X_multi = np.concatenate([
            np.random.randn(50, 20) - 3,
            np.random.randn(50, 20) + 3
        ])
        y_multi = np.concatenate([
            np.random.randn(50) - 2,
            np.random.randn(50) + 2
        ])

        input_params = {'delta': [0.1], 'fdr': 0.1}

        for X, y in [(X_heavy, y_heavy), (X_skew, y_skew), (X_multi, y_multi)]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                slide = OptimizeSLIDE(input_params, x=X, y=y)
                assert slide.data.X is not None

    def test_slide_convergence_with_poor_initialization(self):
        """Test SLIDE convergence with poor starting conditions."""
        np.random.seed(42)
        X = np.random.randn(100, 30)
        y = np.random.randn(100)

        # Mock LOVE with poor initialization
        with patch('loveslide.love_python.love.love.LOVE') as mock_love_class:
            mock_love = Mock()

            # Poor initial conditions
            mock_love.fit.return_value = {
                'latent_factors': np.ones((100, 5)) * 1e-10,  # Nearly zero
                'loading_matrix': np.ones((30, 5)) * 1e-10,
                'convergence': True
            }
            mock_love_class.return_value = mock_love

            slide = OptimizeSLIDE({'delta': [0.1], 'fdr': 0.1}, x=X, y=y)

            slide.run_love()

            # Should handle poor initialization
            assert slide.latent_factors is not None

    def test_slide_with_numerical_instabilities(self):
        """Test SLIDE robustness to numerical instabilities."""
        np.random.seed(42)

        # Create ill-conditioned correlation matrix
        n_features = 20
        X = np.random.randn(100, n_features)

        # Make features nearly linearly dependent
        for i in range(1, n_features):
            X[:, i] = X[:, 0] + np.random.randn(100) * 1e-8

        y = np.random.randn(100)

        input_params = {'delta': [0.1], 'fdr': 0.1}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            slide = OptimizeSLIDE(input_params, x=X, y=y)

            # Should handle ill-conditioning
            assert slide.data.X is not None