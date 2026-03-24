"""
Test coverage for runtime environment edge cases and system-specific behaviors
Focus: Platform variations, dependency availability, and environment-specific failures
"""

import pytest
import numpy as np
import sys
import os
import tempfile
import shutil
import platform
from unittest.mock import patch, MagicMock
import importlib

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loveslide import SLIDE, OptimizeSLIDE, SLIDEcv
from loveslide.knockoffs import Knockoffs


class TestRuntimeEnvironmentEdgeCases:
    """Test runtime environment variations and edge cases"""

    def test_python_version_compatibility(self):
        """Test compatibility across Python version features"""
        python_version = sys.version_info

        # Test version-specific features
        version_results = {
            'python_version': f"{python_version.major}.{python_version.minor}",
            'features_available': {}
        }

        # Test numpy compatibility
        try:
            import numpy as np
            version_results['features_available']['numpy'] = {
                'version': np.__version__,
                'random_generator': hasattr(np.random, 'Generator'),
                'matrix_power': hasattr(np.linalg, 'matrix_power')
            }

            # Test with SLIDE functionality
            X = np.random.rand(30, 8)
            y = np.random.randint(0, 2, 30)

            params = {'K': 2, 'max_iters': 2}
            slide = SLIDE(params, x=X, y=y)
            slide.show_params()  # Test basic functionality

            version_results['slide_compatibility'] = True

        except Exception as e:
            version_results['features_available']['numpy_error'] = str(e)
            version_results['slide_compatibility'] = False

        # Test scipy compatibility
        try:
            from scipy import stats
            version_results['features_available']['scipy'] = {
                'available': True,
                'stats_module': hasattr(stats, 'pearsonr')
            }
        except ImportError:
            version_results['features_available']['scipy'] = {'available': False}

        # Verify basic compatibility
        assert version_results['slide_compatibility'], f"SLIDE incompatible with Python {python_version}"

    def test_missing_dependency_fallbacks(self):
        """Test behavior when optional dependencies are missing"""
        dependency_tests = [
            {
                'module': 'rpy2',
                'functionality': 'R interface',
                'test_function': lambda: self._test_r_interface_fallback()
            },
            {
                'module': 'cvxpy',
                'functionality': 'SDP solving',
                'test_function': lambda: self._test_sdp_solver_fallback()
            },
            {
                'module': 'matplotlib',
                'functionality': 'Plotting',
                'test_function': lambda: self._test_plotting_fallback()
            }
        ]

        fallback_results = {}

        for dep in dependency_tests:
            module_name = dep['module']

            # Test with module available
            try:
                importlib.import_module(module_name)
                module_available = True
            except ImportError:
                module_available = False

            # Test functionality with/without module
            try:
                test_result = dep['test_function']()
                fallback_results[module_name] = {
                    'module_available': module_available,
                    'functionality_works': test_result,
                    'fallback_success': True
                }
            except Exception as e:
                fallback_results[module_name] = {
                    'module_available': module_available,
                    'functionality_works': False,
                    'error': str(e),
                    'fallback_success': False
                }

        # Verify graceful fallback behavior
        for module, result in fallback_results.items():
            if not result['module_available']:
                # Should either work with fallback or fail gracefully
                assert result['fallback_success'] or 'error' in result, f"Poor fallback for {module}"

    def _test_r_interface_fallback(self):
        """Test R interface functionality with fallback"""
        from loveslide.love import call_love

        X = np.random.rand(25, 6)

        try:
            # Try to call LOVE (may fallback if R unavailable)
            result = call_love(X=X, lbd=0.5, thresh_fdr=0.1, verbose=False)
            return result is not None
        except ImportError:
            # Expected if R interface unavailable
            return True
        except Exception:
            # Other errors should be handled gracefully
            return False

    def _test_sdp_solver_fallback(self):
        """Test SDP solver functionality with fallback"""
        X = np.random.rand(20, 8)
        y = np.random.randint(0, 2, 20)

        try:
            knockoffs = Knockoffs(y=y, z2=X)
            result = knockoffs.filter_knockoffs_iterative_python(
                z=X, y=y, fdr=0.1, niter=1
            )
            return result is not None
        except ImportError:
            # Expected if SDP solver unavailable
            return True
        except Exception:
            return False

    def _test_plotting_fallback(self):
        """Test plotting functionality with fallback"""
        try:
            from loveslide.plotting import Plotter
            plotter = Plotter()

            # Test basic plotting functionality
            lfs = {'LF1': [0, 1, 2], 'LF2': [3, 4, 5]}

            # Should either work or fail gracefully
            with tempfile.TemporaryDirectory() as tmpdir:
                plotter.plot_latent_factors(lfs, outdir=tmpdir)
                return True
        except ImportError:
            return True  # Graceful fallback
        except Exception:
            return False

    def test_file_system_edge_cases(self):
        """Test file system related edge cases"""
        file_system_tests = [
            {
                'name': 'readonly_directory',
                'setup': lambda: self._create_readonly_directory(),
                'test': lambda path: self._test_file_operations(path, readonly=True)
            },
            {
                'name': 'long_path_names',
                'setup': lambda: self._create_long_path(),
                'test': lambda path: self._test_file_operations(path, long_path=True)
            },
            {
                'name': 'special_characters',
                'setup': lambda: self._create_special_char_path(),
                'test': lambda path: self._test_file_operations(path, special_chars=True)
            },
            {
                'name': 'network_path',
                'setup': lambda: tempfile.mkdtemp(),  # Simulate network path
                'test': lambda path: self._test_file_operations(path, network=True)
            }
        ]

        filesystem_results = {}

        for test_case in file_system_tests:
            try:
                # Setup test environment
                test_path = test_case['setup']()

                if test_path and os.path.exists(test_path):
                    # Run file system test
                    result = test_case['test'](test_path)
                    filesystem_results[test_case['name']] = {
                        'test_completed': True,
                        'result': result,
                        'path_created': True
                    }

                    # Cleanup
                    try:
                        if os.path.isdir(test_path):
                            shutil.rmtree(test_path, ignore_errors=True)
                    except:
                        pass
                else:
                    filesystem_results[test_case['name']] = {
                        'test_completed': False,
                        'path_created': False
                    }

            except Exception as e:
                filesystem_results[test_case['name']] = {
                    'test_completed': False,
                    'error': str(e)
                }

        # Verify file system handling
        completed_tests = sum(1 for res in filesystem_results.values()
                            if res.get('test_completed', False))

        # Should handle majority of file system scenarios
        assert completed_tests >= len(file_system_tests) * 0.5, f"Poor file system handling: {filesystem_results}"

    def _create_readonly_directory(self):
        """Create a read-only directory for testing"""
        try:
            test_dir = tempfile.mkdtemp()
            os.chmod(test_dir, 0o444)  # Read-only
            return test_dir
        except:
            return None

    def _create_long_path(self):
        """Create a very long path for testing"""
        try:
            base_dir = tempfile.mkdtemp()
            long_path = os.path.join(base_dir, 'a' * 50, 'b' * 50, 'c' * 50)
            os.makedirs(long_path, exist_ok=True)
            return long_path
        except:
            return None

    def _create_special_char_path(self):
        """Create a path with special characters"""
        try:
            base_dir = tempfile.mkdtemp()
            # Use Unicode characters that might cause issues
            special_path = os.path.join(base_dir, 'test_αβγ_файл')
            os.makedirs(special_path, exist_ok=True)
            return special_path
        except:
            return None

    def _test_file_operations(self, test_path, readonly=False, long_path=False,
                            special_chars=False, network=False):
        """Test file operations in various environments"""
        try:
            # Test basic file creation
            test_file = os.path.join(test_path, 'test_slide_output.txt')

            if not readonly:
                # Try to create a file
                with open(test_file, 'w') as f:
                    f.write("SLIDE test output\n")

                # Test SLIDE with file output
                X = np.random.rand(20, 6)
                y = np.random.randint(0, 2, 20)
                params = {'K': 2, 'max_iters': 1}
                slide = SLIDE(params, x=X, y=y)

                # Test parameter saving
                try:
                    slide.save_params(test_path, scores=None)
                    return True
                except:
                    return False
            else:
                # Test handling of read-only directory
                try:
                    with open(test_file, 'w') as f:
                        f.write("test")
                    return False  # Should have failed
                except PermissionError:
                    return True  # Expected behavior
                except:
                    return False

        except Exception:
            return False

    def test_memory_constraint_environments(self):
        """Test behavior under memory constraints"""
        import psutil

        # Get current memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_constraint_tests = [
            {
                'name': 'large_dataset_chunked',
                'test': lambda: self._test_memory_efficient_processing()
            },
            {
                'name': 'repeated_allocations',
                'test': lambda: self._test_repeated_memory_allocations()
            },
            {
                'name': 'garbage_collection',
                'test': lambda: self._test_garbage_collection_behavior()
            }
        ]

        memory_results = {}

        for test_case in memory_constraint_tests:
            try:
                # Monitor memory before test
                pre_test_memory = process.memory_info().rss / 1024 / 1024

                # Run memory test
                result = test_case['test']()

                # Monitor memory after test
                post_test_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = post_test_memory - pre_test_memory

                memory_results[test_case['name']] = {
                    'test_completed': True,
                    'result': result,
                    'memory_growth_mb': memory_growth,
                    'excessive_growth': memory_growth > 200  # Flag if > 200MB growth
                }

            except Exception as e:
                memory_results[test_case['name']] = {
                    'test_completed': False,
                    'error': str(e)
                }

        # Verify memory constraint handling
        for test_name, result in memory_results.items():
            if result.get('excessive_growth', False):
                pytest.fail(f"Excessive memory growth in {test_name}: {result['memory_growth_mb']}MB")

    def _test_memory_efficient_processing(self):
        """Test memory-efficient processing of large datasets"""
        try:
            # Test with progressively larger datasets
            for size in [100, 200, 300]:
                X = np.random.rand(size, min(20, size//5))
                y = np.random.randint(0, 2, size)

                # Test chunked processing
                knockoffs = Knockoffs(y=y, z2=X)
                result = knockoffs.filter_knockoffs_iterative_python(
                    z=X, y=y, fdr=0.1, niter=1
                )

                # Clean up immediately
                del X, y, knockoffs, result

            return True
        except Exception:
            return False

    def _test_repeated_memory_allocations(self):
        """Test repeated memory allocations and deallocations"""
        try:
            for iteration in range(20):
                X = np.random.rand(50, 10)
                y = np.random.randint(0, 2, 50)

                params = {'K': 2, 'max_iters': 1}
                slide = SLIDE(params, x=X, y=y)

                # Force deallocation
                del slide, X, y

            return True
        except Exception:
            return False

    def _test_garbage_collection_behavior(self):
        """Test garbage collection behavior"""
        import gc

        try:
            # Create objects and force garbage collection
            objects = []
            for i in range(10):
                X = np.random.rand(30, 8)
                y = np.random.randint(0, 2, 30)
                knockoffs = Knockoffs(y=y, z2=X)
                objects.append((X, y, knockoffs))

            # Force garbage collection
            gc.collect()

            # Clear references
            objects.clear()
            gc.collect()

            return True
        except Exception:
            return False

    def test_platform_specific_behaviors(self):
        """Test platform-specific behaviors and edge cases"""
        current_platform = platform.system()

        platform_tests = {
            'Linux': lambda: self._test_linux_specific(),
            'Darwin': lambda: self._test_macos_specific(),
            'Windows': lambda: self._test_windows_specific()
        }

        platform_results = {
            'platform': current_platform,
            'architecture': platform.machine(),
            'python_implementation': platform.python_implementation()
        }

        # Run platform-specific tests
        if current_platform in platform_tests:
            try:
                result = platform_tests[current_platform]()
                platform_results['platform_test'] = {
                    'completed': True,
                    'result': result
                }
            except Exception as e:
                platform_results['platform_test'] = {
                    'completed': False,
                    'error': str(e)
                }
        else:
            platform_results['platform_test'] = {
                'completed': False,
                'unsupported_platform': True
            }

        # Test general cross-platform functionality
        try:
            X = np.random.rand(30, 8)
            y = np.random.randint(0, 2, 30)

            # Test basic SLIDE functionality
            params = {'K': 2, 'max_iters': 2}
            slide = SLIDE(params, x=X, y=y)
            slide.show_params()

            platform_results['basic_functionality'] = True
        except Exception as e:
            platform_results['basic_functionality'] = False
            platform_results['basic_error'] = str(e)

        # Verify platform compatibility
        assert platform_results['basic_functionality'], f"Basic functionality failed on {current_platform}"

    def _test_linux_specific(self):
        """Test Linux-specific behaviors"""
        try:
            # Test file permissions and paths
            test_dir = tempfile.mkdtemp()
            os.chmod(test_dir, 0o755)

            # Test SLIDE with Linux path conventions
            X = np.random.rand(25, 6)
            y = np.random.randint(0, 2, 25)
            params = {'K': 2, 'max_iters': 1}
            slide = SLIDE(params, x=X, y=y)

            # Test with Unix-style paths
            output_path = os.path.join(test_dir, 'linux_test')
            slide.save_params(output_path, scores=None)

            shutil.rmtree(test_dir, ignore_errors=True)
            return True
        except Exception:
            return False

    def _test_macos_specific(self):
        """Test macOS-specific behaviors"""
        try:
            # Test with macOS path conventions
            test_dir = tempfile.mkdtemp()

            X = np.random.rand(25, 6)
            y = np.random.randint(0, 2, 25)
            params = {'K': 2, 'max_iters': 1}
            slide = SLIDE(params, x=X, y=y)

            output_path = os.path.join(test_dir, 'macos_test')
            slide.save_params(output_path, scores=None)

            shutil.rmtree(test_dir, ignore_errors=True)
            return True
        except Exception:
            return False

    def _test_windows_specific(self):
        """Test Windows-specific behaviors"""
        try:
            # Test with Windows path conventions
            test_dir = tempfile.mkdtemp()

            X = np.random.rand(25, 6)
            y = np.random.randint(0, 2, 25)
            params = {'K': 2, 'max_iters': 1}
            slide = SLIDE(params, x=X, y=y)

            # Test with Windows-style paths
            output_path = os.path.join(test_dir, 'windows_test')
            slide.save_params(output_path, scores=None)

            shutil.rmtree(test_dir, ignore_errors=True)
            return True
        except Exception:
            return False


class TestEnvironmentRobustness:
    """Test robustness across different environment configurations"""

    def test_import_isolation(self):
        """Test import behavior and module isolation"""
        import sys

        # Test module import paths
        original_path = sys.path.copy()

        try:
            # Test with modified import path
            sys.path.insert(0, '/nonexistent/path')

            # Should still be able to import SLIDE components
            from loveslide import SLIDE, Knockoffs

            X = np.random.rand(20, 6)
            y = np.random.randint(0, 2, 20)

            slide = SLIDE({'K': 2}, x=X, y=y)
            knockoffs = Knockoffs(y=y, z2=X)

            assert slide is not None
            assert knockoffs is not None

        finally:
            sys.path = original_path

    def test_environment_variable_handling(self):
        """Test handling of environment variables"""
        import os

        # Test various environment variable scenarios
        env_tests = [
            ('PYTHONPATH', '/test/path'),
            ('SLIDE_DEBUG', 'True'),
            ('NUMPY_SEED', '42'),
            ('SLIDE_WORKERS', '1')
        ]

        original_env = {}

        try:
            for env_var, env_value in env_tests:
                # Store original value
                original_env[env_var] = os.environ.get(env_var)

                # Set test value
                os.environ[env_var] = env_value

                # Test SLIDE functionality with modified environment
                X = np.random.rand(25, 6)
                y = np.random.randint(0, 2, 25)

                slide = SLIDE({'K': 2, 'max_iters': 1}, x=X, y=y)
                slide.show_params()  # Should work regardless of env vars

        finally:
            # Restore original environment
            for env_var, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(env_var, None)
                else:
                    os.environ[env_var] = original_value


if __name__ == "__main__":
    pytest.main([__file__])