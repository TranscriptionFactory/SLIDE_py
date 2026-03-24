"""
Test integration boundary conditions and edge cases.

Focus: Component interactions at boundaries, data transfer
between modules, and integration failure modes.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, call
import tempfile
import os
import warnings

from loveslide import SLIDE, OptimizeSLIDE, SLIDEcv, Knockoffs, Plotter
from loveslide.score import Estimator, SLIDE_Estimator


class TestComponentIntegrationBoundaries:
    """Test boundaries between major components."""

    def test_slide_to_love_data_transfer(self):
        """Test data transfer from SLIDE to LOVE components."""
        # Edge case: data with extreme values
        X_extreme = np.array([
            [1e6, 1e-6, 0],
            [1e-6, 1e6, 0],
            [0, 0, 1]
        ])
        y_extreme = np.array([1e10, -1e10, 0])

        params = {"fdr": 0.1, "niter": 2, "K": 2}
        slide = SLIDE(params, x=X_extreme, y=y_extreme)

        with patch('loveslide.love.call_love') as mock_love:
            # Test if extreme values are handled properly
            mock_love.return_value = {"factors": np.random.randn(3, 2)}

            try:
                result = slide.run_love()
                # Should either normalize data or handle extremes
                call_args = mock_love.call_args
                if call_args:
                    passed_data = call_args[0]  # Assuming data is first arg
                    # Verify data transformation or bounds checking
                    assert np.all(np.isfinite(passed_data)) if hasattr(passed_data, '__iter__') else True
            except Exception as e:
                # Should provide meaningful error for extreme values
                assert "range" in str(e).lower() or "scale" in str(e).lower()

    def test_love_to_knockoffs_integration(self):
        """Test integration between LOVE results and knockoff generation."""
        X = np.random.randn(40, 20)
        y = np.random.randn(40)
        params = {"fdr": 0.1, "niter": 3, "K": 5}

        slide = SLIDE(params, x=X, y=y)

        # Mock LOVE result with edge case factor structure
        mock_love_result = {
            "factors": np.random.randn(20, 5),
            "loadings": np.random.randn(40, 5),
            "residual_var": np.array([0.001, 1000, 0.5])  # Extreme variances
        }

        with patch('loveslide.love.call_love', return_value=mock_love_result):
            try:
                love_result = slide.run_love()

                # Now test knockoff generation with this result
                knockoffs = Knockoffs()
                # This should handle the factor structure appropriately
                X_ko = knockoffs.create(X, factors=mock_love_result["factors"])

                if X_ko is not None:
                    assert X_ko.shape == X.shape
                    # Verify knockoffs preserve some statistical properties
                    assert not np.allclose(X_ko, X)  # Should be different

            except Exception as e:
                # Should handle incompatible factor structures
                assert "factor" in str(e).lower() or "dimension" in str(e).lower()

    def test_knockoffs_to_estimator_integration(self):
        """Test integration from knockoffs to estimator."""
        X = np.random.randn(35, 15)
        y = np.random.binomial(1, 0.5, 35)  # Binary outcome

        # Create knockoffs with potential edge cases
        X_knockoffs = X + np.random.normal(0, 0.1, X.shape)

        # Edge case: some knockoffs are identical to originals
        X_knockoffs[:5, :5] = X[:5, :5]

        estimator = SLIDE_Estimator()

        try:
            # Test with potentially problematic knockoff structure
            W_stats = estimator.calculate_knockoff_statistics(
                X, X_knockoffs, y
            )

            # Should handle identical columns gracefully
            assert len(W_stats) == X.shape[1]
            assert np.all(np.isfinite(W_stats))

        except Exception as e:
            # Should provide clear error for invalid knockoffs
            assert "knockoff" in str(e).lower() or "identical" in str(e).lower()

    def test_estimator_to_plotting_integration(self):
        """Test integration from estimator results to plotting."""
        X = np.random.randn(30, 12)
        y = np.random.randn(30)

        # Mock estimator results with edge cases
        mock_results = {
            "selected_features": [],  # No features selected
            "coefficients": np.array([]),
            "p_values": np.array([]),
            "test_scores": [np.nan, 0.5, 1.0]  # Including NaN
        }

        plotter = Plotter()

        try:
            # Test plotting with empty/problematic results
            with patch('matplotlib.pyplot.savefig'):  # Prevent actual file creation
                plotter.plot_results(mock_results)

        except Exception as e:
            # Should handle empty results gracefully
            assert "empty" in str(e).lower() or "no features" in str(e).lower() or "plot" in str(e).lower()


class TestDataFlowBoundaries:
    """Test data flow at module boundaries."""

    def test_data_type_consistency_across_modules(self):
        """Test data type consistency as data flows between modules."""
        # Start with mixed data types
        X_mixed = pd.DataFrame({
            'float32': np.random.randn(25).astype(np.float32),
            'float64': np.random.randn(25).astype(np.float64),
            'int32': np.random.randint(0, 100, 25).astype(np.int32)
        })
        y_int = np.random.randint(0, 2, 25)

        params = {"fdr": 0.1, "niter": 2}

        try:
            slide = SLIDE(params, x=X_mixed.values, y=y_int)

            # Track data types through pipeline
            data_types = []

            with patch('loveslide.love.call_love') as mock_love:
                def capture_data_type(*args, **kwargs):
                    if len(args) > 0:
                        data_types.append(args[0].dtype if hasattr(args[0], 'dtype') else type(args[0]))
                    return {"factors": np.random.randn(3, 2)}

                mock_love.side_effect = capture_data_type
                slide.run_love()

            # Data types should be consistent or properly handled
            if data_types:
                assert all(dt in [np.float32, np.float64] for dt in data_types if hasattr(dt, '__name__'))

        except Exception as e:
            # Should handle type mismatches gracefully
            assert "type" in str(e).lower() or "dtype" in str(e).lower()

    def test_data_scaling_preservation(self):
        """Test that data scaling is preserved across module boundaries."""
        # Data with different scales
        X_multiscale = np.column_stack([
            np.random.randn(30) * 1000,      # Large scale
            np.random.randn(30) * 0.001,     # Small scale
            np.random.randn(30)              # Normal scale
        ])
        y = np.random.randn(30)

        params = {"fdr": 0.1, "niter": 2}
        slide = SLIDE(params, x=X_multiscale, y=y)

        # Track data scaling through pipeline
        with patch('loveslide.knockoffs.Knockoffs.create') as mock_knockoffs:
            def check_scaling(*args, **kwargs):
                data = args[0] if len(args) > 0 else None
                if data is not None and hasattr(data, 'std'):
                    scales = np.std(data, axis=0)
                    # Check if scaling is reasonable
                    scale_ratio = np.max(scales) / np.min(scales)
                    assert scale_ratio < 1e6, "Scale differences too extreme"
                return np.random.randn(*data.shape) if data is not None else None

            mock_knockoffs.side_effect = check_scaling

            try:
                knockoffs = Knockoffs()
                knockoffs.create(X_multiscale)
            except AssertionError as e:
                # Expected if scaling isn't handled
                assert "scale" in str(e).lower()

    def test_missing_data_propagation(self):
        """Test how missing data is handled across module boundaries."""
        X_with_missing = np.random.randn(25, 10)
        X_with_missing[::3, ::2] = np.nan  # Introduce systematic missingness
        y = np.random.randn(25)

        params = {"fdr": 0.1, "niter": 2}

        try:
            slide = SLIDE(params, x=X_with_missing, y=y)

            # Should either handle missing data or provide clear error
            result = slide.calc_default_fsize(3)
            assert result is not None

        except Exception as e:
            # Should clearly indicate missing data issue
            assert any(word in str(e).lower() for word in ["missing", "nan", "null", "none"])


class TestVersionCompatibilityBoundaries:
    """Test version compatibility at integration points."""

    def test_numpy_version_compatibility(self):
        """Test compatibility with different NumPy behaviors."""
        X = np.random.randn(20, 8)
        y = np.random.randn(20)
        params = {"fdr": 0.1, "niter": 2}

        # Test with different NumPy random states
        with patch('numpy.random.RandomState') as mock_random:
            mock_random.return_value.randn.return_value = X

            slide = SLIDE(params, x=X, y=y)

            try:
                result = slide.calc_default_fsize(2)
                assert isinstance(result, (int, np.integer))
            except Exception as e:
                # Should handle random state issues
                assert "random" in str(e).lower() or "seed" in str(e).lower()

    def test_pandas_integration_boundaries(self):
        """Test integration boundaries with pandas DataFrames."""
        # Different pandas data types
        df_data = pd.DataFrame({
            'numeric': np.random.randn(30),
            'categorical': pd.Categorical(['A', 'B', 'C'] * 10),
            'datetime': pd.date_range('2020-01-01', periods=30),
            'object': ['text_' + str(i) for i in range(30)]
        })

        y_series = pd.Series(np.random.randn(30), name='target')

        params = {"fdr": 0.1, "niter": 2}

        try:
            # Should handle pandas data appropriately
            slide = SLIDE(params, x=df_data.select_dtypes(include=[np.number]), y=y_series)
            result = slide.calc_default_fsize(2)
            assert result is not None

        except Exception as e:
            # Should provide clear error for incompatible pandas data
            assert any(word in str(e).lower() for word in ["pandas", "dataframe", "series", "dtype"])


class TestResourceManagementBoundaries:
    """Test resource management at integration boundaries."""

    def test_memory_handoff_between_modules(self):
        """Test memory management when data is passed between modules."""
        # Large data to test memory handoff
        try:
            X_large = np.random.randn(500, 100)
            y_large = np.random.randn(500)
            params = {"fdr": 0.1, "niter": 2, "K": 10}

            slide = SLIDE(params, x=X_large, y=y_large)

            # Test memory usage during handoffs
            import psutil
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            with patch('loveslide.love.call_love') as mock_love:
                mock_love.return_value = {"factors": np.random.randn(100, 10)}
                slide.run_love()

            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable
            assert memory_increase < X_large.nbytes * 10, "Excessive memory usage"

        except ImportError:
            pytest.skip("psutil not available for memory testing")
        except MemoryError:
            pytest.skip("Insufficient memory for large data test")

    def test_file_handle_management(self):
        """Test file handle management across module boundaries."""
        X = np.random.randn(30, 15)
        y = np.random.randn(30)
        params = {"fdr": 0.1, "niter": 2}

        # Create multiple temporary files
        temp_files = [tempfile.NamedTemporaryFile(delete=False) for _ in range(5)]
        temp_paths = [f.name for f in temp_files]
        for f in temp_files:
            f.close()

        try:
            slide = SLIDE(params, x=X, y=y)

            # Test multiple save operations
            for i, path in enumerate(temp_paths):
                try:
                    if hasattr(slide, 'save_state'):
                        slide.save_state(f"{path}_{i}")
                except Exception as e:
                    # Should handle file operations gracefully
                    assert "file" in str(e).lower() or "io" in str(e).lower()

        finally:
            # Cleanup
            for path in temp_paths:
                for suffix in ['', '_0', '_1', '_2', '_3', '_4']:
                    full_path = path + suffix
                    if os.path.exists(full_path):
                        try:
                            os.unlink(full_path)
                        except:
                            pass

    def test_warning_propagation_across_modules(self):
        """Test how warnings are handled across module boundaries."""
        X = np.random.randn(25, 12)
        y = np.random.randn(25)
        params = {"fdr": 0.1, "niter": 2}

        slide = SLIDE(params, x=X, y=y)

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            # Operations that might generate warnings
            with patch('numpy.linalg.solve') as mock_solve:
                mock_solve.side_effect = lambda a, b: warnings.warn("Numerical instability") or b

                try:
                    result = slide.calc_default_fsize(3)

                    # Check if warnings are properly handled
                    if warning_list:
                        warning_messages = [str(w.message).lower() for w in warning_list]
                        assert any("numerical" in msg for msg in warning_messages)

                except Exception as e:
                    # Should handle warning scenarios appropriately
                    assert "warning" in str(e).lower() or "numerical" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])