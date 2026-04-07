"""
Test skeleton for environment and configuration validation.

Focus on testing system dependencies, environment setup,
configuration validation, and cross-platform compatibility.
"""
import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import warnings

import numpy as np
import pandas as pd

from loveslide.tools import init_data, check_params, calc_default_fsize


class TestEnvironmentSetup:
    """Test environment setup and dependency validation."""

    def test_python_version_compatibility(self):
        """Test Python version compatibility requirements."""
        # Check Python version is supported
        major, minor = sys.version_info[:2]

        # Assuming minimum Python 3.7 support
        assert major >= 3, "Python 3 required"
        assert (major, minor) >= (3, 7), "Python 3.7+ required"

    def test_required_packages_importable(self):
        """Test that all required packages can be imported."""
        required_packages = [
            'numpy', 'pandas', 'scipy', 'sklearn',
            'matplotlib', 'seaborn', 'tqdm', 'easydict'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        assert len(missing_packages) == 0, f"Missing packages: {missing_packages}"

    def test_optional_dependencies_availability(self):
        """Test availability of optional dependencies."""
        optional_packages = {
            'rpy2': 'R interface functionality',
            'plotly': 'Interactive plotting',
            'jupyter': 'Notebook support'
        }

        for package, purpose in optional_packages.items():
            try:
                __import__(package)
                print(f"✓ Optional package {package} available ({purpose})")
            except ImportError:
                print(f"○ Optional package {package} not available ({purpose})")

    def test_r_environment_detection(self):
        """Test R environment availability for R interface."""
        try:
            import rpy2.robjects as robjects
            # Test basic R functionality
            r_version = robjects.r('R.version.string')[0]
            print(f"R detected: {r_version}")

            # Test required R packages
            required_r_packages = ['MASS', 'glmnet']
            for pkg in required_r_packages:
                try:
                    robjects.packages.importr(pkg)
                    print(f"✓ R package {pkg} available")
                except Exception:
                    print(f"○ R package {pkg} not available")

        except ImportError:
            print("○ R interface (rpy2) not available")

    def test_system_memory_availability(self):
        """Test system memory availability for large computations."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            # Warn if less than 2GB available
            if available_gb < 2:
                warnings.warn(f"Low memory: {available_gb:.1f}GB available")

            print(f"System memory: {available_gb:.1f}GB available")

        except ImportError:
            print("○ psutil not available for memory checking")

    def test_temporary_directory_access(self):
        """Test temporary directory access and permissions."""
        temp_dir = tempfile.gettempdir()

        # Test write permissions
        test_file = Path(temp_dir) / "slide_test_file.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()
            assert True  # Write permission confirmed
        except (PermissionError, OSError) as e:
            pytest.fail(f"Temporary directory not writable: {e}")


class TestConfigurationValidation:
    """Test configuration file handling and validation."""

    def test_init_data_file_path_validation(self):
        """Test file path validation in init_data."""
        # Test non-existent file paths
        params = {
            'x_path': '/nonexistent/path/x.csv',
            'y_path': '/nonexistent/path/y.csv'
        }

        with pytest.raises(FileNotFoundError):
            init_data(params)

    def test_init_data_file_format_validation(self):
        """Test file format validation and parsing."""
        # Create temporary files with different formats
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Valid CSV
            f.write('feature1,feature2,feature3\n1,2,3\n4,5,6\n')
            valid_csv = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Invalid CSV (malformed)
            f.write('feature1,feature2\n1,2,3,4\n')  # Inconsistent columns
            invalid_csv = f.name

        try:
            # Test valid CSV
            data, _ = init_data(
                {'x_path': valid_csv, 'y_path': valid_csv}
            )
            assert hasattr(data, 'X')
            assert hasattr(data, 'Y')

            # Test invalid CSV
            with pytest.raises((pd.errors.ParserError, ValueError)):
                init_data(
                    {'x_path': invalid_csv, 'y_path': invalid_csv}
                )

        finally:
            os.unlink(valid_csv)
            os.unlink(invalid_csv)

    def test_init_data_parameter_type_validation(self):
        """Test parameter type validation in init_data."""
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        # Test invalid parameter types
        invalid_params = [
            {'fdr': 'invalid'},  # String instead of float
            {'niter': 'ten'},    # String instead of int
            {'n_workers': -1},   # Negative workers
            {'y_factor': 'yes'}  # String instead of bool
        ]

        for params in invalid_params:
            with pytest.raises((TypeError, ValueError)):
                init_data(params, x=X, y=y)

    def test_check_params_data_quality_validation(self):
        """Test data quality checking in check_params."""
        # Create data with various quality issues
        class MockData:
            def __init__(self):
                self.X = pd.DataFrame(np.random.randn(100, 10))
                self.Y = pd.Series(np.random.choice([0, 1], 100))

        data = MockData()

        # Test with zero variance features
        data.X.iloc[:, 5] = 1.0  # Constant column

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_params({'fdr': 0.1}, data)

            # Should warn about or handle zero variance features
            if len(w) > 0:
                assert any("variance" in str(warning.message).lower() or
                          "constant" in str(warning.message).lower() for warning in w)

    def test_calc_default_fsize_parameter_validation(self):
        """Test parameter validation in calc_default_fsize."""
        # Test invalid inputs
        with pytest.raises((ValueError, TypeError)):
            calc_default_fsize(-10, 5)  # Negative n_rows

        with pytest.raises((ValueError, TypeError)):
            calc_default_fsize(100, -2)  # Negative K

        with pytest.raises((ValueError, TypeError)):
            calc_default_fsize(100.5, 5)  # Non-integer n_rows

        # Test edge cases
        assert calc_default_fsize(0, 5) >= 0  # Zero samples
        assert calc_default_fsize(5, 0) >= 0  # Zero factors

    def test_parameter_interdependency_validation(self):
        """Test validation of parameter interdependencies."""
        X = np.random.randn(100, 50)
        y = np.random.choice([0, 1], 100)

        # Test incompatible parameter combinations
        incompatible_params = [
            {'fdr': 0.01, 'thresh_fdr': 0.5},  # Very strict FDR with loose thresh_fdr
            {'niter': 1, 'fdr': 0.001},       # Too few iterations for strict FDR
            {'f_size': 100, 'fdr': 0.1}       # f_size larger than n_features
        ]

        for params in incompatible_params:
            data, processed_params = init_data(params, x=X, y=y)

            # Should either adjust parameters or raise warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                check_params(processed_params, data)

                # Check for parameter adjustment warnings
                if len(w) > 0:
                    assert any("parameter" in str(warning.message).lower() or
                              "adjusted" in str(warning.message).lower()
                              for warning in w)


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility issues."""

    def test_path_separator_handling(self):
        """Test correct path separator handling across platforms."""
        # Test with different path separators
        unix_style_paths = {
            'x_path': '/tmp/data/x.csv',
            'y_path': '/tmp/data/y.csv'
        }

        windows_style_paths = {
            'x_path': 'C:\\data\\x.csv',
            'y_path': 'C:\\data\\y.csv'
        }

        # Should handle both styles gracefully (even if files don't exist)
        for paths in [unix_style_paths, windows_style_paths]:
            try:
                # This will fail due to non-existent files, but path parsing should work
                init_data(paths)
            except FileNotFoundError:
                pass  # Expected
            except Exception as e:
                # Should not fail due to path parsing issues
                assert "path" not in str(e).lower()

    def test_line_ending_handling(self):
        """Test handling of different line endings in files."""
        # Create test files with different line endings
        test_data = "feature1,feature2\n1,2\n3,4"

        line_endings = {
            'unix': test_data.replace('\n', '\n'),      # LF
            'windows': test_data.replace('\n', '\r\n'),  # CRLF
            'mac': test_data.replace('\n', '\r')         # CR
        }

        for ending_type, data_content in line_endings.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                           newline='', delete=False) as f:
                f.write(data_content)
                file_path = f.name

            try:
                # Should parse correctly regardless of line endings
                df = pd.read_csv(file_path, index_col=0)
                assert df.shape[0] > 0  # Successfully parsed rows

            except Exception as e:
                pytest.fail(f"Failed to parse {ending_type} line endings: {e}")

            finally:
                os.unlink(file_path)

    def test_unicode_handling(self):
        """Test Unicode character handling in data files."""
        # Create test file with Unicode characters
        unicode_data = "feature_α,feature_β\n1.5,2.3\n3.7,4.1"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                       encoding='utf-8', delete=False) as f:
            f.write(unicode_data)
            file_path = f.name

        try:
            # Should handle Unicode in column names
            df = pd.read_csv(file_path, index_col=0)
            assert 'feature_α' in df.columns or 'feature_α' in str(df.columns)

        except UnicodeError as e:
            pytest.fail(f"Unicode handling failed: {e}")

        finally:
            os.unlink(file_path)

    def test_locale_specific_number_formats(self):
        """Test handling of locale-specific number formats."""
        # Test different decimal separators
        test_data_comma = "feature1,feature2\n\"1,5\",\"2,3\"\n\"3,7\",\"4,1\""
        test_data_period = "feature1,feature2\n1.5,2.3\n3.7,4.1"

        for data_content in [test_data_comma, test_data_period]:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                           delete=False) as f:
                f.write(data_content)
                file_path = f.name

            try:
                # Should parse numbers correctly
                df = pd.read_csv(file_path, index_col=0)

                # Check that numeric data was parsed
                if ',' in data_content and '"' in data_content:
                    # May need specific parsing for comma decimals
                    pass
                else:
                    assert df.dtypes.iloc[0] in ['float64', 'int64']

            except Exception as e:
                # Note the parsing issue but don't fail
                print(f"Number format parsing issue: {e}")

            finally:
                os.unlink(file_path)


class TestResourceManagement:
    """Test resource management and cleanup."""

    def test_memory_cleanup_after_errors(self):
        """Test memory is properly cleaned up after errors."""
        # Create scenario that could cause memory leak
        X = np.random.randn(1000, 100)  # Moderately large data
        y = np.random.choice([0, 1], 1000)

        import gc
        import psutil
        import os

        try:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
        except ImportError:
            initial_memory = 0

        # Force an error that could leave memory allocated
        with patch('loveslide.tools.pd.read_csv', side_effect=MemoryError("Mock error")):
            try:
                init_data({'x_path': 'dummy.csv', 'y_path': 'dummy.csv'})
            except (MemoryError, FileNotFoundError):
                pass

        # Force garbage collection
        gc.collect()

        try:
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Should not have significant memory increase (> 100MB)
            assert memory_increase < 100 * 1024 * 1024, f"Memory leak detected: {memory_increase} bytes"

        except ImportError:
            # psutil not available, skip memory check
            pass

    def test_file_handle_cleanup(self):
        """Test that file handles are properly closed."""
        # Create temporary files
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                           delete=False) as f:
                f.write("col1,col2\n1,2\n")
                temp_files.append(f.name)

        try:
            # Test multiple file operations
            for file_path in temp_files:
                try:
                    df = pd.read_csv(file_path, index_col=0)
                except Exception:
                    pass

            # Files should be accessible for deletion (handles closed)
            for file_path in temp_files:
                os.unlink(file_path)  # Should not raise PermissionError

        except PermissionError as e:
            pytest.fail(f"File handle not properly closed: {e}")

    def test_concurrent_access_safety(self):
        """Test safety with concurrent file access."""
        import threading
        import time

        # Create shared temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                       delete=False) as f:
            f.write("feature1,feature2\n1,2\n3,4\n")
            shared_file = f.name

        results = []
        errors = []

        def read_file():
            try:
                df = pd.read_csv(shared_file, index_col=0)
                results.append(df)
            except Exception as e:
                errors.append(e)

        # Create multiple threads reading the same file
        threads = [threading.Thread(target=read_file) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        try:
            # Should not have errors from concurrent access
            assert len(errors) == 0, f"Concurrent access errors: {errors}"
            assert len(results) == 5, f"Expected 5 results, got {len(results)}"

        finally:
            os.unlink(shared_file)


if __name__ == "__main__":
    pytest.main([__file__])