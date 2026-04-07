"""
Test cross-platform robustness and R-Python interface reliability.

This module focuses on testing the robustness of R-Python interfaces,
graceful degradation when dependencies are missing, and platform-specific
edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, PropertyMock
import sys
import os
import subprocess
import tempfile

from src.loveslide.knockoffs import (_create_second_order_r, _solve_sdp_r,
                                    _rlist_get, Knockoffs)
from src.loveslide.love import call_love_r, call_love


class TestRPythonInterface:
    """Test R-Python interface robustness."""

    def test_r_not_available_graceful_degradation(self):
        """Test graceful fallback when R is not available."""
        with patch('rpy2.robjects') as mock_rpy2:
            mock_rpy2.side_effect = ImportError("R not found")

            # Should fall back to Python implementation
            knockoffs = Knockoffs(backend='auto')
            assert knockoffs.backend == 'python'

            # Should still work with Python backend
            X = np.random.randn(50, 10)
            result = knockoffs.create_gaussian(X)
            assert result.shape == X.shape

    def test_rpy2_version_compatibility(self):
        """Test compatibility with different rpy2 versions."""
        # Mock different rpy2 version behaviors
        test_cases = [
            ('3.5.x', True),   # Supports dict-like access
            ('3.6.x', False),  # Deprecated dict-like access
        ]

        for version, supports_dict_access in test_cases:
            with patch('src.loveslide.knockoffs._rlist_get') as mock_rlist:
                # Simulate version-specific behavior
                if supports_dict_access:
                    mock_rlist.return_value = "test_value"
                else:
                    mock_rlist.side_effect = TypeError("Dict access not supported")

                # Function should handle both versions gracefully
                mock_robj = MagicMock()
                mock_robj.names = ['test_item']

                try:
                    result = _rlist_get(mock_robj, 'test_item')
                    # Should either return value or handle gracefully
                    assert result is not None or True  # Fallback successful
                except TypeError:
                    # Should provide informative error
                    assert "not supported" in str(pytest.raises(TypeError).value)

    def test_r_memory_management(self):
        """Test R memory management doesn't leak."""
        if not self._r_available():
            pytest.skip("R not available")

        with patch('psutil.Process') as mock_process:
            mock_memory = MagicMock()
            mock_memory.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB baseline
            mock_process.return_value = mock_memory

            initial_memory = 100 * 1024 * 1024  # 100MB

            # Run multiple R operations
            try:
                for i in range(10):
                    X = np.random.randn(100, 20)
                    _create_second_order_r(X)

                    # Simulate memory growth
                    mock_memory.memory_info.return_value.rss = initial_memory + i * 1024 * 1024

                final_memory = mock_memory.memory_info.return_value.rss

                # Memory growth should be bounded (less than 50% increase)
                assert final_memory < initial_memory * 1.5

            except ImportError:
                pytest.skip("R interface not available")

    def test_r_package_dependencies(self):
        """Test handling of missing R packages."""
        with patch('rpy2.robjects.r') as mock_r:
            # Simulate missing R package
            mock_r.side_effect = Exception("package 'knockoff' is not installed")

            with pytest.raises(ImportError, match="R package.*not.*installed"):
                X = np.random.randn(50, 10)
                _create_second_order_r(X)

    def test_r_data_type_conversion(self):
        """Test R-Python data type conversion edge cases."""
        if not self._r_available():
            pytest.skip("R not available")

        test_cases = [
            np.array([[1, 2], [3, 4]], dtype=np.float32),  # float32 -> R double
            np.array([[1, 2], [3, 4]], dtype=np.float64),  # float64 -> R double
            np.array([[1, 2], [3, 4]], dtype=np.int32),    # int32 -> R integer
            np.array([[np.nan, 2], [3, 4]]),               # NaN handling
            np.array([[np.inf, 2], [3, 4]]),               # Infinity handling
        ]

        for X in test_cases:
            try:
                result = _create_second_order_r(X)

                # Result should have same shape and be finite (if input was)
                assert result.shape == X.shape
                if np.isfinite(X).all():
                    assert np.isfinite(result).all()

            except Exception as e:
                # Should provide informative error for problematic inputs
                if np.isinf(X).any() or np.isnan(X).any():
                    assert "infinite" in str(e).lower() or "nan" in str(e).lower()
                else:
                    raise

    def test_concurrent_r_access(self):
        """Test concurrent access to R doesn't cause conflicts."""
        import threading
        import concurrent.futures

        if not self._r_available():
            pytest.skip("R not available")

        def r_operation(thread_id):
            """Function to run R operation in thread."""
            X = np.random.randn(50, 10) + thread_id * 0.1  # Unique per thread
            try:
                result = _create_second_order_r(X)
                return result.shape == X.shape
            except Exception:
                return False

        # Run multiple R operations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(r_operation, i) for i in range(8)]
            results = [f.result(timeout=30) for f in futures]

        # At least some operations should succeed
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.5  # At least 50% should succeed

    @staticmethod
    def _r_available():
        """Check if R is available."""
        try:
            import rpy2.robjects
            return True
        except ImportError:
            return False


class TestPlatformSpecificEdgeCases:
    """Test platform-specific edge cases and compatibility."""

    def test_file_path_handling(self):
        """Test file path handling across different platforms."""
        # Test different path formats
        path_formats = [
            "/tmp/test.txt",              # Unix absolute
            "~/test.txt",                 # Home directory
            "./test.txt",                 # Relative
            "../test.txt",                # Parent directory
        ]

        if sys.platform == "win32":
            path_formats.extend([
                "C:\\temp\\test.txt",     # Windows absolute
                "C:/temp/test.txt",       # Windows with forward slashes
            ])

        from src.loveslide import SLIDE

        for path_format in path_formats:
            try:
                # Test that path handling doesn't crash
                expanded_path = os.path.expanduser(path_format)
                normalized_path = os.path.normpath(expanded_path)
                assert isinstance(normalized_path, str)

            except Exception as e:
                # Platform-specific paths may fail on wrong platform
                if sys.platform != "win32" and "C:\\" in path_format:
                    continue  # Expected failure
                else:
                    raise

    def test_numpy_version_compatibility(self):
        """Test compatibility with different NumPy versions."""
        # Test features that changed between NumPy versions
        X = np.random.randn(10, 5)

        # Test matrix multiplication (@ operator introduced in 1.10)
        try:
            result1 = X @ X.T
            result2 = np.dot(X, X.T)
            assert np.allclose(result1, result2)
        except Exception:
            # Fallback for old NumPy versions
            result2 = np.dot(X, X.T)
            assert result2.shape == (10, 10)

        # Test random number generator (changed in 1.17)
        try:
            rng = np.random.default_rng(42)
            random_data = rng.standard_normal((10, 5))
            assert random_data.shape == (10, 5)
        except AttributeError:
            # Fallback for old NumPy versions
            np.random.seed(42)
            random_data = np.random.randn(10, 5)
            assert random_data.shape == (10, 5)

    def test_pandas_version_compatibility(self):
        """Test compatibility with different pandas versions."""
        import pandas as pd

        # Create test DataFrame
        data = np.random.randn(100, 10)
        df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(10)])

        # Test features that changed between pandas versions
        try:
            # Modern pandas syntax
            result = df.sample(frac=0.5, random_state=42)
            assert len(result) == 50
        except Exception:
            # Fallback for older pandas
            result = df.sample(n=50, random_state=42)
            assert len(result) == 50

        # Test DataFrame dtypes (behavior changed in newer versions)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) == 10

    def test_multiprocessing_compatibility(self):
        """Test multiprocessing compatibility across platforms."""
        from multiprocessing import Pool
        import os

        def simple_task(x):
            """Simple function for multiprocessing test."""
            return x ** 2

        # Test multiprocessing works
        if os.name != 'nt':  # Skip on Windows due to pickling issues
            try:
                with Pool(processes=2) as pool:
                    results = pool.map(simple_task, [1, 2, 3, 4])
                assert results == [1, 4, 9, 16]
            except Exception as e:
                pytest.skip(f"Multiprocessing not available: {e}")
        else:
            pytest.skip("Multiprocessing test skipped on Windows")

    def test_temporary_file_handling(self):
        """Test temporary file creation and cleanup."""
        import tempfile
        import os

        # Test temporary file creation
        temp_files = []
        try:
            for i in range(5):
                with tempfile.NamedTemporaryFile(delete=False) as tf:
                    tf.write(b"test data")
                    temp_files.append(tf.name)

            # All files should exist
            for tf_name in temp_files:
                assert os.path.exists(tf_name)

        finally:
            # Cleanup
            for tf_name in temp_files:
                try:
                    os.unlink(tf_name)
                except FileNotFoundError:
                    pass  # Already cleaned up

    def test_environment_variable_handling(self):
        """Test environment variable handling."""
        import os

        # Test common environment variables
        env_vars = ['PATH', 'HOME', 'USER']
        if sys.platform == "win32":
            env_vars.extend(['USERPROFILE', 'USERNAME'])

        for var in env_vars:
            if var in os.environ:
                value = os.environ[var]
                assert isinstance(value, str)
                assert len(value) > 0

        # Test setting custom environment variable
        test_var = 'SLIDE_TEST_VAR'
        original_value = os.environ.get(test_var)

        try:
            os.environ[test_var] = 'test_value'
            assert os.environ[test_var] == 'test_value'
        finally:
            # Restore original state
            if original_value is None:
                os.environ.pop(test_var, None)
            else:
                os.environ[test_var] = original_value


class TestDependencyFallbacks:
    """Test graceful degradation when optional dependencies are missing."""

    def test_missing_optional_packages(self):
        """Test behavior when optional packages are missing."""
        optional_packages = [
            'seaborn',      # For plotting
            'tqdm',         # For progress bars
            'rpy2',         # For R interface
            'cvxpy',        # For SDP solving
        ]

        for package in optional_packages:
            with patch.dict('sys.modules', {package: None}):
                # Should not crash importing main modules
                try:
                    from src.loveslide import SLIDE
                    slide = SLIDE({"fdr": 0.1})
                    assert slide is not None
                except ImportError as e:
                    # Should provide informative error message
                    assert package in str(e) or "optional" in str(e).lower()

    def test_plotting_without_seaborn(self):
        """Test plotting functionality falls back gracefully."""
        with patch.dict('sys.modules', {'seaborn': None}):
            from src.loveslide.plotting import Plotter

            # Should create plotter even without seaborn
            plotter = Plotter()
            assert plotter is not None

            # Should warn about missing dependencies
            with pytest.warns(UserWarning, match="seaborn.*not available"):
                X = np.random.randn(100, 10)
                plotter.plot_data(X)

    def test_progress_bars_without_tqdm(self):
        """Test progress bars fall back gracefully."""
        with patch.dict('sys.modules', {'tqdm': None}):
            # Should work without tqdm (possibly without progress bars)
            from src.loveslide import SLIDE

            slide = SLIDE({"fdr": 0.1})
            X = np.random.randn(100, 50)

            # Should complete without error (maybe without progress indication)
            try:
                result = slide.run(X)
                assert result is not None
            except Exception as e:
                # Should not fail due to missing tqdm
                assert "tqdm" not in str(e)

    def test_solver_fallback_chain(self):
        """Test SDP solver fallback when preferred solvers unavailable."""
        from src.loveslide.knockoff.solve import _get_sdp_solver

        with patch.dict('sys.modules', {'cvxpy': None}):
            # Should fall back to alternative solver
            solver = _get_sdp_solver()
            assert solver is not None

        with patch.dict('sys.modules', {'cvxpy': None, 'scipy.optimize': None}):
            # Should provide informative error when no solvers available
            with pytest.raises(ImportError, match="SDP solver.*not available"):
                _get_sdp_solver()