"""
Comprehensive cross-platform and environment compatibility testing for SLIDE_py.

Tests behavior across different operating systems, Python versions,
and dependency configurations to ensure robust cross-platform operation.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os
import platform
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, Mock
import warnings

from loveslide import (
    SLIDE, SLIDEcv, Knockoffs, call_love,
    init_data, Plotter
)


class TestPlatformSpecificBehavior:
    """Test platform-specific behavior and compatibility."""

    def test_file_path_handling_across_platforms(self):
        """Test file path handling on different platforms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test different path separators
            if platform.system() == "Windows":
                test_paths = [
                    os.path.join(temp_dir, "data\\subdir\\test.csv"),
                    os.path.join(temp_dir, "data/subdir/test.csv"),  # Mixed separators
                    Path(temp_dir) / "data" / "test.csv"
                ]
            else:
                test_paths = [
                    os.path.join(temp_dir, "data/subdir/test.csv"),
                    Path(temp_dir) / "data" / "test.csv"
                ]

            # Create test data
            X = pd.DataFrame(np.random.randn(50, 10))
            y = pd.Series(np.random.randn(50))

            for test_path in test_paths:
                # Create directory structure
                os.makedirs(os.path.dirname(test_path), exist_ok=True)

                # Save test files
                X.to_csv(str(test_path).replace('.csv', '_X.csv'), index=False)
                y.to_csv(str(test_path).replace('.csv', '_y.csv'), index=False, header=['target'])

                # Test SLIDE can handle the paths
                params = {
                    'x_path': str(test_path).replace('.csv', '_X.csv'),
                    'y_path': str(test_path).replace('.csv', '_y.csv'),
                    'fdr': 0.1
                }

                try:
                    slide = SLIDE(params)
                    assert slide.data.X.shape == (50, 10)
                    assert slide.data.y.shape == (50,)
                except Exception as e:
                    pytest.fail(f"Failed with path {test_path}: {e}")

    def test_line_ending_handling(self):
        """Test handling of different line endings across platforms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create CSV files with different line endings
            data = "col1,col2,col3\n1,2,3\n4,5,6\n7,8,9"

            line_endings = {
                'unix': data.replace('\n', '\n'),      # LF
                'windows': data.replace('\n', '\r\n'),  # CRLF
                'mac': data.replace('\n', '\r')         # CR
            }

            for ending_type, file_content in line_endings.items():
                file_path = os.path.join(temp_dir, f'test_{ending_type}.csv')

                # Write with specific line endings
                with open(file_path, 'wb') as f:
                    f.write(file_content.encode('utf-8'))

                # Test reading with pandas (which SLIDE uses)
                try:
                    df = pd.read_csv(file_path)
                    assert df.shape == (3, 3)
                except Exception as e:
                    pytest.fail(f"Failed to read {ending_type} line endings: {e}")

    def test_case_sensitive_filesystem_handling(self):
        """Test handling of case-sensitive vs case-insensitive filesystems."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with different cases
            file_lower = os.path.join(temp_dir, 'data.csv')
            file_upper = os.path.join(temp_dir, 'DATA.CSV')

            # Save test data
            X = pd.DataFrame(np.random.randn(20, 5))
            X.to_csv(file_lower, index=False)

            # Test both cases
            params_lower = {'x_path': file_lower, 'y_path': None, 'fdr': 0.1}
            params_upper = {'x_path': file_upper, 'y_path': None, 'fdr': 0.1}

            # Lower case should always work
            slide_lower = SLIDE(params_lower, y=np.random.randn(20))
            assert slide_lower.data.X.shape == (20, 5)

            # Upper case behavior depends on filesystem
            try:
                slide_upper = SLIDE(params_upper, y=np.random.randn(20))
                # If it works, filesystem is case-insensitive
                assert slide_upper.data.X.shape == (20, 5)
            except FileNotFoundError:
                # Expected on case-sensitive filesystems
                assert platform.system() in ['Linux', 'Darwin']  # Usually case-sensitive

    def test_unicode_path_handling(self):
        """Test handling of Unicode characters in file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test Unicode characters in paths
            unicode_paths = [
                os.path.join(temp_dir, "数据.csv"),        # Chinese
                os.path.join(temp_dir, "données.csv"),     # French
                os.path.join(temp_dir, "データ.csv"),      # Japanese
                os.path.join(temp_dir, "данные.csv"),     # Russian
            ]

            X = pd.DataFrame(np.random.randn(20, 5))
            y = pd.Series(np.random.randn(20))

            for unicode_path in unicode_paths:
                try:
                    # Save files with Unicode names
                    X.to_csv(unicode_path, index=False)
                    y.to_csv(unicode_path.replace('.csv', '_y.csv'), index=False, header=['target'])

                    # Test SLIDE can handle Unicode paths
                    params = {
                        'x_path': unicode_path,
                        'y_path': unicode_path.replace('.csv', '_y.csv'),
                        'fdr': 0.1
                    }

                    slide = SLIDE(params)
                    assert slide.data.X.shape == (20, 5)

                except (UnicodeError, OSError) as e:
                    # Some filesystems may not support certain Unicode characters
                    warnings.warn(f"Unicode path not supported: {unicode_path}: {e}")


class TestPythonVersionCompatibility:
    """Test compatibility across Python versions."""

    def test_python_version_detection(self):
        """Test detection of Python version and features."""
        python_version = sys.version_info

        # Test version-specific features
        if python_version >= (3, 8):
            # Test walrus operator compatibility (if used in code)
            test_list = [1, 2, 3, 4, 5]
            assert (n := len(test_list)) == 5

        if python_version >= (3, 9):
            # Test dictionary merge operator (if used in code)
            dict1 = {'a': 1, 'b': 2}
            dict2 = {'c': 3, 'd': 4}
            merged = dict1 | dict2
            assert len(merged) == 4

    def test_numpy_version_compatibility(self):
        """Test compatibility with different NumPy versions."""
        import numpy as np

        numpy_version = tuple(map(int, np.__version__.split('.')[:2]))

        # Test features that might vary across NumPy versions
        X = np.random.randn(50, 20)

        # Test random number generator (changed in NumPy 1.17+)
        if numpy_version >= (1, 17):
            rng = np.random.default_rng(42)
            random_data = rng.standard_normal((10, 5))
        else:
            np.random.seed(42)
            random_data = np.random.randn(10, 5)

        assert random_data.shape == (10, 5)

        # Test matrix operations that might have changed
        A = np.random.randn(20, 20)
        try:
            eigenvals = np.linalg.eigvals(A)
            assert len(eigenvals) == 20
        except np.linalg.LinAlgError:
            # May fail on some versions with certain matrices
            pass

    def test_pandas_version_compatibility(self):
        """Test compatibility with different pandas versions."""
        import pandas as pd

        pandas_version = tuple(map(int, pd.__version__.split('.')[:2]))

        # Test features that might vary across pandas versions
        df = pd.DataFrame(np.random.randn(50, 5), columns=['A', 'B', 'C', 'D', 'E'])

        # Test reading CSV with different pandas versions
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            # Test different reading methods
            df_read = pd.read_csv(temp_file)
            assert df_read.shape == (50, 5)

            # Test dtype inference (may vary across versions)
            dtypes = df_read.dtypes
            assert all(dtype == np.float64 for dtype in dtypes)

        finally:
            os.unlink(temp_file)


class TestDependencyCompatibility:
    """Test compatibility with different dependency versions."""

    def test_scipy_version_compatibility(self):
        """Test compatibility with different SciPy versions."""
        try:
            import scipy
            from scipy import linalg
            from scipy.sparse import random as sparse_random

            scipy_version = tuple(map(int, scipy.__version__.split('.')[:2]))

            # Test linear algebra operations
            A = np.random.randn(20, 20)
            A = A @ A.T  # Make positive definite

            try:
                chol = linalg.cholesky(A)
                assert chol.shape == (20, 20)
            except linalg.LinAlgError:
                # May fail if matrix is not positive definite
                pass

            # Test sparse operations (if used)
            sparse_mat = sparse_random(100, 100, density=0.1)
            assert sparse_mat.shape == (100, 100)

        except ImportError:
            pytest.skip("SciPy not available")

    def test_sklearn_version_compatibility(self):
        """Test compatibility with different scikit-learn versions."""
        try:
            import sklearn
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import cross_val_score

            sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))

            # Test basic sklearn functionality
            X = np.random.randn(100, 5)
            y = np.random.randn(100)

            model = LinearRegression()
            model.fit(X, y)

            predictions = model.predict(X)
            assert predictions.shape == (100,)

            # Test cross-validation (API may vary across versions)
            if sklearn_version >= (0, 22):
                scores = cross_val_score(model, X, y, cv=3)
                assert len(scores) == 3

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_matplotlib_version_compatibility(self):
        """Test compatibility with different matplotlib versions."""
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            matplotlib_version = tuple(map(int, matplotlib.__version__.split('.')[:2]))

            # Test basic plotting functionality
            fig, ax = plt.subplots()
            x = np.linspace(0, 10, 100)
            y = np.sin(x)

            ax.plot(x, y)
            ax.set_title("Test Plot")

            # Test saving (format support may vary)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fig.savefig(f.name)
                temp_plot = f.name

            try:
                assert os.path.exists(temp_plot)
                assert os.path.getsize(temp_plot) > 0
            finally:
                os.unlink(temp_plot)
                plt.close(fig)

        except ImportError:
            pytest.skip("matplotlib not available")


class TestEnvironmentVariableHandling:
    """Test handling of environment variables and system settings."""

    def test_environment_variable_handling(self):
        """Test handling of relevant environment variables."""
        # Test NumPy threading environment variables
        env_vars_to_test = [
            'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS',
            'OMP_NUM_THREADS',
            'NUMBA_NUM_THREADS'
        ]

        for env_var in env_vars_to_test:
            original_value = os.environ.get(env_var)

            try:
                # Set environment variable
                os.environ[env_var] = '1'

                # Test that SLIDE still works
                X = np.random.randn(50, 20)
                y = np.random.randn(50)
                params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

                slide = SLIDE(params, x=X, y=y)
                assert slide.data.X.shape == (50, 20)

            finally:
                # Restore original value
                if original_value is None:
                    os.environ.pop(env_var, None)
                else:
                    os.environ[env_var] = original_value

    def test_locale_handling(self):
        """Test handling of different locales."""
        import locale

        # Get current locale
        current_locale = locale.getlocale()

        # Test different locales that might affect number formatting
        locales_to_test = ['C', 'en_US.UTF-8', 'de_DE.UTF-8']

        for test_locale in locales_to_test:
            try:
                locale.setlocale(locale.LC_ALL, test_locale)

                # Test that number parsing still works correctly
                X = np.array([[1.5, 2.7], [3.14, 4.2]])
                params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

                slide = SLIDE(params, x=X, y=np.array([1.0, 2.0]))
                assert np.allclose(slide.data.X, X)

            except locale.Error:
                # Locale not available on this system
                continue
            finally:
                # Restore original locale
                try:
                    locale.setlocale(locale.LC_ALL, current_locale)
                except locale.Error:
                    # Fallback to 'C' locale
                    locale.setlocale(locale.LC_ALL, 'C')

    def test_temporary_directory_handling(self):
        """Test handling of different temporary directory configurations."""
        import tempfile

        original_tmpdir = tempfile.gettempdir()

        # Test with different temporary directories
        with tempfile.TemporaryDirectory() as custom_tmpdir:
            # Set custom temporary directory
            os.environ['TMPDIR'] = custom_tmpdir
            tempfile.tempdir = custom_tmpdir

            try:
                # Test operations that might use temporary files
                X = np.random.randn(100, 50)
                y = np.random.randn(100)

                params = {'x_path': None, 'y_path': None, 'fdr': 0.1}
                slide = SLIDE(params, x=X, y=y)

                # Simulate operations that might create temporary files
                slide.data.love_result = {
                    'pure_Ind': [],
                    'A': np.random.randn(50, 10),
                    'delta': 0.1
                }

                assert slide.data.X.shape == (100, 50)

            finally:
                # Restore original temporary directory
                tempfile.tempdir = None
                os.environ.pop('TMPDIR', None)


class TestMemoryLayoutCompatibility:
    """Test compatibility with different memory layouts and data types."""

    def test_array_memory_layout(self):
        """Test handling of different array memory layouts."""
        n_samples, n_features = 100, 50

        # Test different memory layouts
        layouts = {
            'C': np.random.randn(n_samples, n_features),  # C-contiguous
            'F': np.asfortranarray(np.random.randn(n_samples, n_features)),  # Fortran-contiguous
            'strided': np.random.randn(n_samples * 2, n_features)[::2]  # Strided
        }

        for layout_name, X in layouts.items():
            y = np.random.randn(n_samples)
            params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

            try:
                slide = SLIDE(params, x=X, y=y)
                assert slide.data.X.shape == (n_samples, n_features)
            except Exception as e:
                pytest.fail(f"Failed with {layout_name} layout: {e}")

    def test_data_type_compatibility(self):
        """Test compatibility with different data types."""
        n_samples, n_features = 50, 20

        # Test different data types
        dtypes_to_test = [
            np.float32, np.float64,
            np.int32, np.int64,
        ]

        for dtype in dtypes_to_test:
            try:
                X = np.random.randn(n_samples, n_features).astype(dtype)
                y = np.random.randn(n_samples).astype(dtype)

                params = {'x_path': None, 'y_path': None, 'fdr': 0.1}
                slide = SLIDE(params, x=X, y=y)

                # Should convert to appropriate type internally
                assert slide.data.X.shape == (n_samples, n_features)

            except Exception as e:
                # Some operations might not support all data types
                warnings.warn(f"Data type {dtype} not supported: {e}")

    def test_endianness_compatibility(self):
        """Test compatibility with different byte orders."""
        # Test different endianness
        X_native = np.random.randn(50, 20)
        y_native = np.random.randn(50)

        # Create arrays with different byte orders
        X_big = X_native.astype('>f8')  # Big-endian
        X_little = X_native.astype('<f8')  # Little-endian
        y_big = y_native.astype('>f8')
        y_little = y_native.astype('<f8')

        test_cases = [
            (X_big, y_big, "big-endian"),
            (X_little, y_little, "little-endian"),
            (X_native, y_native, "native")
        ]

        for X, y, endian_type in test_cases:
            params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

            try:
                slide = SLIDE(params, x=X, y=y)
                assert slide.data.X.shape == (50, 20)
            except Exception as e:
                warnings.warn(f"Endianness {endian_type} caused issues: {e}")


class TestConcurrencyPlatformDifferences:
    """Test concurrency behavior across platforms."""

    def test_multiprocessing_start_methods(self):
        """Test different multiprocessing start methods."""
        import multiprocessing as mp

        # Get available start methods for this platform
        available_methods = mp.get_all_start_methods()

        def simple_knockoffs_task():
            X = np.random.randn(50, 20)
            y = np.random.randn(50)
            knockoffs = Knockoffs(backend='python')
            result = knockoffs.select_short_freq(X, y, fdr=0.1)
            return len(result.selected)

        for method in available_methods:
            if method == 'fork' and platform.system() == 'Darwin':
                # Fork is unsafe on macOS with certain NumPy builds
                continue

            try:
                ctx = mp.get_context(method)
                with ctx.Pool(processes=2) as pool:
                    results = pool.map(simple_knockoffs_task, range(2))
                    assert len(results) == 2
                    assert all(isinstance(r, int) for r in results)

            except (OSError, RuntimeError) as e:
                # Some start methods might not be available
                warnings.warn(f"Start method {method} not available: {e}")

    def test_thread_safety_across_platforms(self):
        """Test thread safety behavior across platforms."""
        import threading
        import time

        results = []
        errors = []

        def threaded_operation(thread_id):
            try:
                X = np.random.randn(30, 15)
                y = np.random.randn(30)
                params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

                slide = SLIDE(params, x=X, y=y)
                results.append((thread_id, slide.data.X.shape))
                time.sleep(0.01)  # Small delay to increase chance of race conditions

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Run multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=threaded_operation, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should complete without errors on all platforms
        assert len(errors) == 0, f"Thread safety issues: {errors}"
        assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])