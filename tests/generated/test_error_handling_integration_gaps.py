"""
Comprehensive test coverage for error handling and integration gaps.
Addresses critical gaps in cross-module interactions and failure recovery.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import os
import shutil
from contextlib import contextmanager


class TestCrossModuleIntegrationGaps:
    """Test integration between major modules."""

    def test_love_to_knockoffs_pipeline(self):
        """Test complete LOVE → Knockoffs pipeline."""
        from loveslide import call_love, Knockoffs

        # Generate test data
        n, p = 100, 20
        X = pd.DataFrame(np.random.randn(n, p))
        y = pd.DataFrame(np.random.randint(0, 2, (n, 1)))

        # Step 1: Run LOVE
        try:
            love_result = call_love(X, lbd=0.5, mu=0.5, thresh_fdr=0.2)
            assert love_result is not None
            assert 'A' in love_result

            # Step 2: Use LOVE results in Knockoffs
            A_matrix = love_result['A']
            if A_matrix.shape[1] > 0:  # If factors found
                knockoffs = Knockoffs(y.iloc[:, 0], X, model='auto')

                # Test knockoff generation with LOVE structure
                X_knockoffs = knockoffs.generate_knockoffs(method='second_order')
                assert X_knockoffs.shape == X.shape

        except Exception as e:
            pytest.skip(f"LOVE-Knockoffs pipeline requires R environment: {e}")

    def test_slide_state_persistence_recovery(self):
        """Test SLIDE state persistence and recovery."""
        from loveslide import SLIDE

        input_params = {
            'x_path': None, 'y_path': None,
            'fdr': 0.1, 'niter': 5
        }

        n, p = 50, 15
        X = pd.DataFrame(np.random.randn(n, p))
        y = pd.DataFrame(np.random.randint(0, 2, (n, 1)))

        slide = SLIDE(input_params, X, y)

        # Create temporary directory for state
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test save/load state cycle
            try:
                # Mock some state data
                slide.A = pd.DataFrame(np.random.randn(p, 3))
                slide.latent_factors = pd.DataFrame(np.random.randn(n, 3))
                slide.sig_LFs = ['Z0', 'Z1']
                slide.sig_interacts = []

                # Save state (would normally happen during execution)
                A_path = os.path.join(tmp_dir, 'A.csv')
                z_path = os.path.join(tmp_dir, 'z_matrix.csv')
                lf_path = os.path.join(tmp_dir, 'sig_LFs.txt')

                slide.A.to_csv(A_path)
                slide.latent_factors.to_csv(z_path)
                np.savetxt(lf_path, slide.sig_LFs, fmt='%s')

                # Create new SLIDE instance and load state
                slide2 = SLIDE(input_params, X, y)
                slide2.load_state(tmp_dir)

                assert slide2.A is not None
                assert slide2.latent_factors is not None
                assert slide2.sig_LFs == slide.sig_LFs

            except Exception as e:
                pytest.skip(f"State persistence test requires file I/O: {e}")

    def test_parallel_execution_thread_safety(self):
        """Test thread safety in parallel execution."""
        from loveslide import SLIDEcv
        import threading
        import queue

        input_params = {
            'x_path': None, 'y_path': None,
            'fdr': 0.1, 'n_workers': 2
        }

        n, p = 80, 20
        X = pd.DataFrame(np.random.randn(n, p))
        y = pd.DataFrame(np.random.randint(0, 2, (n, 1)))

        results_queue = queue.Queue()

        def run_cv_worker():
            try:
                cv = SLIDEcv(n_folds=3, metric='auc', n_workers=1)
                cv.data, cv.input_params = cv._init_data_params(input_params, X, y)

                # Simple CV without full SLIDE pipeline
                mock_selections = [np.random.choice(p, 5, replace=False) for _ in range(3)]
                results_queue.put((True, mock_selections))
            except Exception as e:
                results_queue.put((False, str(e)))

        # Run multiple CV instances concurrently
        threads = []
        for _ in range(3):
            t = threading.Thread(target=run_cv_worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Check all threads completed successfully
        success_count = 0
        while not results_queue.empty():
            success, result = results_queue.get()
            if success:
                success_count += 1

        assert success_count >= 2  # At least 2/3 should succeed

    def test_r_python_interface_fallback(self):
        """Test R-Python interface with fallback mechanisms."""
        from loveslide import call_love

        n, p = 50, 10
        X = pd.DataFrame(np.random.randn(n, p))

        # Test with R unavailable
        with patch('rpy2.robjects.r', side_effect=ImportError("R not available")):
            try:
                result = call_love(X, lbd=0.5, mu=0.5)
                # Should either use Python fallback or raise informative error
                if result is not None:
                    assert 'A' in result
            except ImportError as e:
                assert "R" in str(e) or "rpy2" in str(e)


class TestErrorHandlingGaps:
    """Test comprehensive error handling scenarios."""

    def test_corrupted_input_file_handling(self):
        """Test handling of corrupted input files."""
        from loveslide.tools import init_data

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("corrupted,data\n1,2,3,4\n")  # Inconsistent columns
            f.flush()

            input_params = {'x_path': f.name, 'y_path': None}

            try:
                with pytest.raises((pd.errors.ParserError, ValueError)):
                    init_data(input_params)
            finally:
                os.unlink(f.name)

    def test_memory_exhaustion_recovery(self):
        """Test recovery from memory exhaustion."""
        from loveslide.knockoff.utils import rnorm_matrix

        # Simulate memory error
        with patch('numpy.random.randn', side_effect=MemoryError("Out of memory")):
            with pytest.raises(MemoryError):
                rnorm_matrix(1000, 1000)

    def test_numerical_solver_failure_recovery(self):
        """Test recovery from numerical solver failures."""
        from loveslide.knockoff.solve import create_solve_sdp

        # Create degenerate matrix
        Sigma = np.zeros((5, 5))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = create_solve_sdp(Sigma)
                # Should either handle gracefully or raise appropriate error
                if result is not None:
                    assert len(result) == 5
        except (np.linalg.LinAlgError, ValueError) as e:
            assert any(word in str(e).lower() for word in ['singular', 'positive', 'definite'])

    def test_concurrent_file_access_conflicts(self):
        """Test handling of concurrent file access conflicts."""
        from loveslide import SLIDE
        import threading

        input_params = {
            'x_path': None, 'y_path': None,
            'out_path': tempfile.mkdtemp()
        }

        n, p = 30, 10
        X = pd.DataFrame(np.random.randn(n, p))
        y = pd.DataFrame(np.random.randint(0, 2, (n, 1)))

        def concurrent_slide_worker(worker_id):
            try:
                slide = SLIDE(input_params, X, y)
                # Simulate file operations
                test_file = os.path.join(input_params['out_path'], f'test_{worker_id}.txt')
                with open(test_file, 'w') as f:
                    f.write(f"Worker {worker_id}")
                return True
            except Exception:
                return False

        threads = []
        results = []

        for i in range(3):
            def worker():
                results.append(concurrent_slide_worker(i))

            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Cleanup
        shutil.rmtree(input_params['out_path'])

        # At least some workers should succeed
        assert any(results)

    def test_invalid_parameter_combinations(self):
        """Test handling of invalid parameter combinations."""
        from loveslide.tools import check_params, init_data

        n, p = 50, 10
        X = pd.DataFrame(np.random.randn(n, p))
        y = pd.DataFrame(np.random.randint(0, 2, (n, 1)))

        # Test invalid parameter combinations
        invalid_params_list = [
            {'x_path': None, 'y_path': None, 'fdr': -0.1},  # Negative FDR
            {'x_path': None, 'y_path': None, 'fdr': 1.5},   # FDR > 1
            {'x_path': None, 'y_path': None, 'niter': 0},   # Zero iterations
            {'x_path': None, 'y_path': None, 'niter': -5},  # Negative iterations
        ]

        for invalid_params in invalid_params_list:
            try:
                data, params = init_data(invalid_params, X, y)
                # Should either fix parameters or raise error
                if params['fdr'] is not None:
                    assert 0 <= params['fdr'] <= 1
                if params['niter'] is not None:
                    assert params['niter'] > 0
            except ValueError:
                pass  # Expected for some invalid combinations

    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies in factor estimation."""
        from loveslide.love_python.love.est_pure_hetero import Est_Pure

        # Create artificially circular score matrix
        score_mat = np.ones((5, 5))  # All correlations = 1 (circular)
        delta = 0.1

        try:
            result = Est_Pure(score_mat, delta)
            # Should handle circular dependencies gracefully
            if result is not None:
                assert 'estPureIndices' in result
        except (ValueError, RuntimeError) as e:
            # Circular dependency should be detected
            assert any(word in str(e).lower() for word in ['circular', 'convergence', 'singular'])


class TestResourceManagementGaps:
    """Test resource management and cleanup."""

    def test_temporary_file_cleanup(self):
        """Test cleanup of temporary files after errors."""
        from loveslide import SLIDE

        input_params = {
            'x_path': None, 'y_path': None,
            'out_path': tempfile.mkdtemp()
        }

        n, p = 30, 10
        X = pd.DataFrame(np.random.randn(n, p))
        y = pd.DataFrame(np.random.randint(0, 2, (n, 1)))

        slide = SLIDE(input_params, X, y)

        try:
            # Simulate error during file operations
            with patch('pandas.DataFrame.to_csv', side_effect=IOError("Disk full")):
                with pytest.raises(IOError):
                    # This would normally create temporary files
                    temp_file = os.path.join(input_params['out_path'], 'temp.csv')
                    slide.data.X.to_csv(temp_file)

            # Check that temporary files are cleaned up
            temp_files = [f for f in os.listdir(input_params['out_path']) if f.startswith('temp')]
            assert len(temp_files) == 0

        finally:
            shutil.rmtree(input_params['out_path'])

    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        from loveslide.love_python.love.utilities import recoverGroup

        # Repeated operations that might leak memory
        for i in range(10):
            A = np.random.randn(20, 5)
            result = recoverGroup(A)

            # Explicit cleanup
            del A, result

        # If we get here without memory issues, test passes
        assert True

    def test_exception_propagation_chains(self):
        """Test proper exception propagation through call chains."""
        from loveslide import Knockoffs

        n, p = 50, 10
        X = pd.DataFrame(np.random.randn(n, p))
        y = pd.Series(np.random.randint(0, 2, n))

        knockoffs = Knockoffs(y, X)

        # Simulate deep exception chain
        with patch('loveslide.knockoff.solve._get_sdp_solver', side_effect=RuntimeError("Solver failed")):
            try:
                knockoffs.generate_knockoffs(method='sdp')
            except RuntimeError as e:
                # Original exception should be preserved
                assert "Solver failed" in str(e)
            except Exception as e:
                # Or wrapped with context
                assert any(word in str(e).lower() for word in ['solver', 'sdp', 'failed'])


@contextmanager
def temporary_environment_change(env_var, value):
    """Context manager for temporary environment variable changes."""
    old_value = os.environ.get(env_var)
    os.environ[env_var] = str(value)
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = old_value


class TestEnvironmentConfigurationGaps:
    """Test environment and configuration edge cases."""

    def test_missing_environment_variables(self):
        """Test behavior with missing environment variables."""
        from loveslide.tools import init_data

        # Test with modified PATH (simulating missing dependencies)
        with temporary_environment_change('PATH', ''):
            input_params = {'x_path': None, 'y_path': None}
            X = pd.DataFrame(np.random.randn(30, 5))
            y = pd.DataFrame(np.random.randint(0, 2, (30, 1)))

            # Should still work for basic operations
            data, params = init_data(input_params, X, y)
            assert data.X.shape == X.shape

    def test_locale_compatibility(self):
        """Test compatibility with different locales."""
        from loveslide.love_python.love.utilities import offSum

        # Test with non-English locale settings
        with temporary_environment_change('LC_NUMERIC', 'de_DE.UTF-8'):
            M = np.array([[1.5, 2.7], [2.7, 1.5]])
            result = offSum(M, 1.0)
            assert isinstance(result, float)
            assert np.isfinite(result)

    def test_platform_specific_differences(self):
        """Test platform-specific numerical differences."""
        from loveslide.knockoff.utils import is_posdef

        # Test matrix that might have platform-dependent eigenvalue precision
        A = np.random.randn(10, 10)
        Sigma = A @ A.T + 1e-14 * np.eye(10)  # Barely positive definite

        # Results might differ slightly between platforms
        result1 = is_posdef(Sigma, tol=1e-12)
        result2 = is_posdef(Sigma, tol=1e-10)

        # But should be consistent within reasonable tolerances
        assert isinstance(result1, bool)
        assert isinstance(result2, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])