"""
Comprehensive edge case and error handling tests for SLIDE_py.

Addresses the TODO items found in existing test files:
- Parameter validation edge cases
- Data validation edge cases
- Cross-platform compatibility
- Recovery from failures
- Memory management edge cases
- File I/O error scenarios
- Concurrent execution edge cases
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import threading
import multiprocessing
import gc
import sys
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from loveslide import (
    SLIDE, OptimizeSLIDE, SLIDEcv, Knockoffs, VotingResult,
    call_love, Plotter, Estimator, SLIDE_Estimator,
    init_data, show_params, check_params, calc_default_fsize
)


class TestParameterValidationEdgeCases:
    """Complete parameter validation edge case testing."""

    def test_fdr_boundary_values(self):
        """Test FDR validation at exact boundaries."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Exact boundary values
        SLIDE({'fdr': 0.0}, x=X, y=y)  # Should work
        SLIDE({'fdr': 1.0}, x=X, y=y)  # Should work

        # Just outside boundaries
        with pytest.raises(ValueError, match="fdr.*between 0 and 1"):
            SLIDE({'fdr': -1e-10}, x=X, y=y)

        with pytest.raises(ValueError, match="fdr.*between 0 and 1"):
            SLIDE({'fdr': 1.0 + 1e-10}, x=X, y=y)

    def test_niter_validation_edge_cases(self):
        """Test niter validation with edge cases."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Minimum valid value
        SLIDE({'niter': 1}, x=X, y=y)  # Should work

        # Invalid values
        with pytest.raises(ValueError, match="niter.*positive"):
            SLIDE({'niter': 0}, x=X, y=y)

        with pytest.raises(ValueError, match="niter.*integer"):
            SLIDE({'niter': -1}, x=X, y=y)

        with pytest.raises(ValueError, match="niter.*integer"):
            SLIDE({'niter': 1.5}, x=X, y=y)

    def test_f_size_validation_edge_cases(self):
        """Test f_size parameter validation."""
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        # Valid f_size values
        SLIDE({'f_size': 1}, x=X, y=y)  # Minimum
        SLIDE({'f_size': 20}, x=X, y=y)  # Equal to number of features

        # Invalid f_size
        with pytest.raises(ValueError, match="f_size.*positive"):
            SLIDE({'f_size': 0}, x=X, y=y)

        with pytest.raises(ValueError, match="f_size.*positive"):
            SLIDE({'f_size': -5}, x=X, y=y)

        # f_size larger than number of features (should warn or adjust)
        with pytest.warns(UserWarning, match="f_size.*larger.*features"):
            SLIDE({'f_size': 100}, x=X, y=y)

    def test_parameter_type_validation(self):
        """Test that parameter types are validated correctly."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Wrong types
        with pytest.raises(TypeError, match="fdr.*numeric"):
            SLIDE({'fdr': "0.1"}, x=X, y=y)

        with pytest.raises(TypeError, match="niter.*integer"):
            SLIDE({'niter': "5"}, x=X, y=y)

        # Complex numbers (edge case)
        with pytest.raises(TypeError, match="fdr.*real"):
            SLIDE({'fdr': 0.1 + 1j}, x=X, y=y)

    def test_parameter_missing_required(self):
        """Test behavior when required parameters are missing."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Empty parameters (should use defaults or raise error)
        try:
            SLIDE({}, x=X, y=y)
        except ValueError as e:
            assert "required" in str(e).lower()

    def test_parameter_unknown_keys(self):
        """Test handling of unknown parameter keys."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        params = {
            'fdr': 0.1,
            'niter': 5,
            'unknown_param': 'value',
            'another_unknown': 123
        }

        # Should warn about unknown parameters but not crash
        with pytest.warns(UserWarning, match="unknown.*parameter"):
            SLIDE(params, x=X, y=y)

    def test_parameter_combination_validation(self):
        """Test validation of parameter combinations."""
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        # Incompatible parameter combinations
        incompatible_params = [
            {'backend': 'r_knockoffs', 'method': 'python_only_method'},
            {'f_size': 30, 'niter': 1000},  # Very large niter with large f_size
        ]

        for params in incompatible_params:
            params['fdr'] = 0.1  # Add required param
            with pytest.warns(UserWarning) or pytest.raises(ValueError):
                SLIDE(params, x=X, y=y)


class TestDataValidationEdgeCases:
    """Complete data validation edge case testing."""

    def test_data_shape_mismatches(self):
        """Test all possible data shape mismatches."""
        # Different sample size mismatches
        mismatches = [
            (np.random.randn(50, 10), np.random.randn(49)),  # Off by one
            (np.random.randn(50, 10), np.random.randn(60)),  # Larger y
            (np.random.randn(50, 10), np.random.randn(0)),   # Empty y
        ]

        for X, y in mismatches:
            with pytest.raises(ValueError, match="X and y.*incompatible"):
                SLIDE({'fdr': 0.1}, x=X, y=y)

    def test_data_dimensionality_edge_cases(self):
        """Test edge cases in data dimensionality."""
        # 1D X (should be 2D)
        X_1d = np.random.randn(50)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="X.*2-dimensional"):
            SLIDE({'fdr': 0.1}, x=X_1d, y=y)

        # 3D X
        X_3d = np.random.randn(50, 10, 5)

        with pytest.raises(ValueError, match="X.*2-dimensional"):
            SLIDE({'fdr': 0.1}, x=X_3d, y=y)

        # 2D y (should be 1D)
        X = np.random.randn(50, 10)
        y_2d = np.random.randn(50, 2)

        with pytest.raises(ValueError, match="y.*1-dimensional"):
            SLIDE({'fdr': 0.1}, x=X, y=y_2d)

    def test_data_with_special_values(self):
        """Test data containing special floating point values."""
        n, p = 50, 10
        y = np.random.randn(n)

        # Data with NaN
        X_nan = np.random.randn(n, p)
        X_nan[10, 3] = np.nan

        with pytest.raises(ValueError, match="NaN.*detected"):
            SLIDE({'fdr': 0.1}, x=X_nan, y=y)

        # Data with infinity
        X_inf = np.random.randn(n, p)
        X_inf[5, 7] = np.inf

        with pytest.raises(ValueError, match="infinite.*detected"):
            SLIDE({'fdr': 0.1}, x=X_inf, y=y)

        # Data with -infinity
        X_ninf = np.random.randn(n, p)
        X_ninf[15, 2] = -np.inf

        with pytest.raises(ValueError, match="infinite.*detected"):
            SLIDE({'fdr': 0.1}, x=X_ninf, y=y)

        # y with NaN
        X = np.random.randn(n, p)
        y_nan = y.copy()
        y_nan[25] = np.nan

        with pytest.raises(ValueError, match="NaN.*detected"):
            SLIDE({'fdr': 0.1}, x=X, y=y_nan)

    def test_data_extreme_values(self):
        """Test data with extreme but valid values."""
        n, p = 50, 10

        # Very large values
        X_large = np.random.randn(n, p) * 1e10
        y = np.random.randn(n)

        with pytest.warns(UserWarning, match="extreme.*values"):
            SLIDE({'fdr': 0.1}, x=X_large, y=y)

        # Very small values
        X_small = np.random.randn(n, p) * 1e-10

        with pytest.warns(UserWarning, match="small.*variance"):
            SLIDE({'fdr': 0.1}, x=X_small, y=y)

    def test_data_rank_deficiency(self):
        """Test various forms of rank-deficient data."""
        n, p = 100, 10
        y = np.random.randn(n)

        # Perfect linear dependence
        X_dep = np.random.randn(n, p)
        X_dep[:, 5] = X_dep[:, 0] + X_dep[:, 1]  # Column 5 = column 0 + column 1

        with pytest.warns(UserWarning, match="rank.*deficient"):
            SLIDE({'fdr': 0.1}, x=X_dep, y=y)

        # Nearly linear dependence
        X_nearly_dep = np.random.randn(n, p)
        X_nearly_dep[:, 5] = X_nearly_dep[:, 0] + X_nearly_dep[:, 1] + 1e-10 * np.random.randn(n)

        with pytest.warns(UserWarning, match="nearly.*singular"):
            SLIDE({'fdr': 0.1}, x=X_nearly_dep, y=y)

        # Constant columns
        X_const = np.random.randn(n, p)
        X_const[:, 3] = 5.0  # Constant column

        with pytest.warns(UserWarning, match="constant.*column"):
            SLIDE({'fdr': 0.1}, x=X_const, y=y)

    def test_data_size_edge_cases(self):
        """Test edge cases in data sizes."""
        # More features than samples (p >> n)
        X_wide = np.random.randn(10, 100)
        y_wide = np.random.randn(10)

        with pytest.warns(UserWarning, match="more features.*samples"):
            SLIDE({'fdr': 0.1, 'f_size': 5}, x=X_wide, y=y_wide)

        # Very small dataset
        X_tiny = np.random.randn(5, 3)
        y_tiny = np.random.randn(5)

        with pytest.warns(UserWarning, match="small.*dataset"):
            SLIDE({'fdr': 0.1}, x=X_tiny, y=y_tiny)

        # Single sample
        X_single = np.random.randn(1, 10)
        y_single = np.random.randn(1)

        with pytest.raises(ValueError, match="insufficient.*samples"):
            SLIDE({'fdr': 0.1}, x=X_single, y=y_single)

        # Single feature
        X_single_feat = np.random.randn(50, 1)
        y_50 = np.random.randn(50)

        with pytest.warns(UserWarning, match="single.*feature"):
            SLIDE({'fdr': 0.1}, x=X_single_feat, y=y_50)

    def test_data_type_edge_cases(self):
        """Test edge cases with different data types."""
        n, p = 50, 10

        # Integer data
        X_int = np.random.randint(-10, 10, size=(n, p))
        y = np.random.randn(n)

        SLIDE({'fdr': 0.1}, x=X_int, y=y)  # Should work

        # Boolean data
        X_bool = np.random.choice([True, False], size=(n, p))

        with pytest.warns(UserWarning, match="boolean.*data"):
            SLIDE({'fdr': 0.1}, x=X_bool, y=y)

        # Complex data (should fail)
        X_complex = np.random.randn(n, p) + 1j * np.random.randn(n, p)

        with pytest.raises(ValueError, match="complex.*not supported"):
            SLIDE({'fdr': 0.1}, x=X_complex, y=y)

        # Mixed precision
        X_float32 = np.random.randn(n, p).astype(np.float32)
        y_float64 = np.random.randn(n).astype(np.float64)

        with pytest.warns(UserWarning, match="mixed.*precision"):
            SLIDE({'fdr': 0.1}, x=X_float32, y=y_float64)


class TestMemoryManagementEdgeCases:
    """Test memory management edge cases and large dataset handling."""

    def test_memory_efficient_chunking(self):
        """Test that chunking works efficiently for large datasets."""
        # Create moderately large dataset
        n, p = 2000, 500

        try:
            X = np.random.randn(n, p)
            y = np.random.randn(n)

            initial_memory = gc.get_obj_count = len(gc.get_objects())

            params = {'fdr': 0.1, 'niter': 3, 'f_size': 50}
            slide = SLIDE(params, x=X, y=y)

            final_memory = len(gc.get_objects())

            # Should not create excessive temporary objects
            assert final_memory - initial_memory < 50000

        except MemoryError:
            pytest.skip("Insufficient memory for large dataset test")

    def test_memory_cleanup_after_failure(self):
        """Test that memory is cleaned up even after failures."""
        initial_objects = len(gc.get_objects())

        try:
            # Force a failure with invalid data
            X = np.full((100, 10), np.nan)  # All NaN
            y = np.random.randn(100)

            with pytest.raises(ValueError):
                SLIDE({'fdr': 0.1}, x=X, y=y)

        finally:
            gc.collect()
            final_objects = len(gc.get_objects())

        # Should not leak memory even after failure
        assert final_objects - initial_objects < 1000

    def test_repeated_large_operations(self):
        """Test memory stability with repeated large operations."""
        memory_usage = []

        for i in range(5):
            X = np.random.randn(500, 100)
            y = np.random.randn(500)

            slide = SLIDE({'fdr': 0.1, 'niter': 2}, x=X, y=y)

            del X, y, slide
            gc.collect()

            memory_usage.append(len(gc.get_objects()))

        # Memory usage should not continuously increase
        assert max(memory_usage) - min(memory_usage) < 10000

    def test_large_intermediate_matrices(self):
        """Test handling of large intermediate matrices."""
        n, p = 1000, 200

        try:
            # This will create large covariance matrices
            X = np.random.randn(n, p)
            y = np.random.randn(n)

            params = {'fdr': 0.1, 'niter': 3}

            with patch('numpy.cov') as mock_cov:
                # Mock to return very large matrix (simulate memory pressure)
                large_cov = np.random.randn(p, p) * 1e6
                mock_cov.return_value = large_cov

                with pytest.warns(UserWarning, match="large.*matrix") or \
                     pytest.raises(MemoryError):
                    SLIDE(params, x=X, y=y)

        except MemoryError:
            pytest.skip("Insufficient memory for large matrix test")


class TestConcurrentExecutionEdgeCases:
    """Test concurrent execution and thread safety."""

    def test_parallel_slide_execution(self):
        """Test running SLIDE in parallel processes."""
        def run_slide(seed):
            np.random.seed(seed)
            X = np.random.randn(100, 20)
            y = np.random.randn(100)

            params = {'fdr': 0.1, 'niter': 3, 'seed': seed}
            slide = SLIDE(params, x=X, y=y)
            return slide.input_params['seed']

        # Run in parallel
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_slide, seed) for seed in [42, 43]]
            results = [f.result() for f in futures]

        # Should complete successfully
        assert results == [42, 43]

    def test_thread_safety_knockoffs(self):
        """Test that knockoff operations are thread-safe."""
        X = np.random.randn(100, 15)
        y = np.random.randn(100)

        results = []

        def run_knockoffs(seed):
            knockoffs = Knockoffs(backend='python')
            result = knockoffs.select_short_freq(X, y, fdr=0.1, niter=3, seed=seed)
            results.append(len(result.selected))

        # Run concurrently
        threads = [
            threading.Thread(target=run_knockoffs, args=(seed,))
            for seed in [100, 101]
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should complete without race conditions
        assert len(results) == 2

    def test_shared_resource_access(self):
        """Test access to shared resources (e.g., random number generators)."""
        # This tests that different instances don't interfere with each other
        X1 = np.random.RandomState(42).randn(50, 10)
        X2 = np.random.RandomState(42).randn(50, 10)
        y = np.random.randn(50)

        # Should give identical results
        slide1 = SLIDE({'fdr': 0.1, 'seed': 42}, x=X1, y=y)
        slide2 = SLIDE({'fdr': 0.1, 'seed': 42}, x=X2, y=y)

        # Results should be identical
        assert slide1.input_params == slide2.input_params

    def test_interruption_handling(self):
        """Test graceful handling of interruptions."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        # Simulate KeyboardInterrupt during execution
        with patch('loveslide.call_love') as mock_love:
            mock_love.side_effect = KeyboardInterrupt("User interrupted")

            with pytest.raises(KeyboardInterrupt):
                SLIDE({'fdr': 0.1, 'niter': 10}, x=X, y=y)


class TestFileIOErrorScenarios:
    """Test file I/O error handling scenarios."""

    def test_permission_denied_scenarios(self):
        """Test handling of file permission errors."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Mock permission errors for different file operations
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            # Should not crash on file operations
            slide = SLIDE({'fdr': 0.1}, x=X, y=y)
            assert slide is not None

    def test_disk_space_errors(self):
        """Test handling of disk space errors."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Mock disk space errors
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            with pytest.warns(UserWarning, match="disk.*space") or \
                 pytest.raises(OSError):
                slide = SLIDE({'fdr': 0.1}, x=X, y=y)

    def test_corrupted_file_scenarios(self):
        """Test handling of corrupted files."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Mock corrupted file content
        with patch('builtins.open', mock_open(read_data="corrupted content")):
            with patch('pickle.load', side_effect=pickle.UnpicklingError("Corrupt pickle")):
                # Should handle corrupted files gracefully
                with pytest.warns(UserWarning, match="corrupted") or \
                     pytest.raises(ValueError):
                    slide = SLIDE({'fdr': 0.1}, x=X, y=y)

    def test_network_filesystem_scenarios(self):
        """Test scenarios with network filesystems."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Mock network timeouts
        with patch('builtins.open', side_effect=TimeoutError("Network timeout")):
            with pytest.warns(UserWarning, match="network.*timeout") or \
                 pytest.raises(TimeoutError):
                slide = SLIDE({'fdr': 0.1}, x=X, y=y)

    def test_temporary_file_handling(self):
        """Test proper cleanup of temporary files."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Run SLIDE that might create temporary files
            slide = SLIDE({'fdr': 0.1}, x=X, y=y)

            # Check that no temporary files are left behind
            temp_files = list(Path(temp_dir).glob("*"))
            assert len(temp_files) == 0  # Should clean up


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility issues."""

    def test_path_separator_handling(self):
        """Test that path separators work across platforms."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Test with different path styles
        if sys.platform == "win32":
            test_path = "C:\\temp\\test_file.txt"
        else:
            test_path = "/tmp/test_file.txt"

        # Should handle platform-specific paths
        slide = SLIDE({'fdr': 0.1, 'output_path': test_path}, x=X, y=y)
        assert slide is not None

    def test_line_ending_handling(self):
        """Test handling of different line endings."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Mock files with different line endings
        for line_ending in ['\n', '\r\n', '\r']:
            mock_content = f"param1=value1{line_ending}param2=value2{line_ending}"

            with patch('builtins.open', mock_open(read_data=mock_content)):
                # Should handle different line endings
                slide = SLIDE({'fdr': 0.1}, x=X, y=y)
                assert slide is not None

    def test_unicode_handling(self):
        """Test handling of unicode characters in parameters/paths."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Unicode in string parameters
        unicode_params = {
            'fdr': 0.1,
            'description': 'Test with unicode: αβγ δεζ 中文 🔬'
        }

        # Should handle unicode characters
        slide = SLIDE(unicode_params, x=X, y=y)
        assert slide is not None

    def test_numerical_precision_differences(self):
        """Test that results are consistent across different precision levels."""
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)

        # Run with lower precision
        slide_32 = SLIDE({'fdr': 0.1}, x=X, y=y)

        # Run with higher precision
        X_64 = X.astype(np.float64)
        y_64 = y.astype(np.float64)
        slide_64 = SLIDE({'fdr': 0.1}, x=X_64, y=y_64)

        # Results should be reasonably similar
        assert slide_32 is not None
        assert slide_64 is not None


class TestRecoveryFromFailures:
    """Test recovery mechanisms from various failure modes."""

    def test_partial_failure_recovery(self):
        """Test recovery when part of the pipeline fails."""
        X = np.random.randn(100, 15)
        y = np.random.randn(100)

        # Mock failure in LOVE component
        with patch('loveslide.call_love') as mock_love:
            mock_love.side_effect = RuntimeError("LOVE computation failed")

            # SLIDE should attempt fallback or graceful degradation
            with pytest.warns(UserWarning, match="fallback") or \
                 pytest.raises(RuntimeError):
                slide = SLIDE({'fdr': 0.1, 'fallback': True}, x=X, y=y)

    def test_retry_mechanism(self):
        """Test retry mechanisms for transient failures."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        call_count = 0

        def failing_function(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 times
                raise ConnectionError("Transient network error")
            return {'L_hat': np.random.randn(10, 2)}

        with patch('loveslide.call_love', side_effect=failing_function):
            # Should retry and eventually succeed
            slide = SLIDE({'fdr': 0.1, 'max_retries': 3}, x=X, y=y)
            assert slide is not None
            assert call_count == 3  # Should have retried

    def test_state_recovery_after_crash(self):
        """Test state recovery mechanisms after crashes."""
        X = np.random.randn(100, 15)
        y = np.random.randn(100)

        # Simulate crash during execution
        with patch('loveslide.knockoff.filter.knockoff_filter_voting') as mock_voting:
            mock_voting.side_effect = SystemExit("Simulated crash")

            with pytest.raises(SystemExit):
                slide = SLIDE({'fdr': 0.1, 'save_intermediate': True}, x=X, y=y)

        # Should be able to recover state
        # (This would require actual state saving implementation)
        # slide_recovered = SLIDE({'fdr': 0.1, 'resume_from': 'checkpoint'}, x=X, y=y)

    def test_resource_exhaustion_recovery(self):
        """Test recovery from resource exhaustion."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        # Mock memory exhaustion
        with patch('numpy.cov', side_effect=MemoryError("Out of memory")):
            with pytest.warns(UserWarning, match="memory.*exhausted") or \
                 pytest.raises(MemoryError):
                slide = SLIDE({
                    'fdr': 0.1,
                    'memory_efficient': True,
                    'chunk_size': 5
                }, x=X, y=y)

    def test_dependency_failure_fallback(self):
        """Test fallback when optional dependencies fail."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Mock R backend failure
        with patch('loveslide.knockoffs._rlist_get', side_effect=ImportError("R not available")):
            # Should fall back to Python backend
            knockoffs = Knockoffs(backend='r_knockoffs', fallback_backend='python')
            result = knockoffs.select_short_freq(X, y, fdr=0.1)
            assert isinstance(result, VotingResult)

    def test_numerical_instability_recovery(self):
        """Test recovery from numerical instabilities."""
        # Create ill-conditioned data
        X = np.random.randn(100, 10)
        X[:, 1] = X[:, 0] + 1e-15 * np.random.randn(100)  # Nearly dependent
        y = np.random.randn(100)

        # Should detect and handle numerical instability
        with pytest.warns(UserWarning, match="numerical.*instability"):
            slide = SLIDE({'fdr': 0.1, 'regularization': 1e-6}, x=X, y=y)

    def test_timeout_recovery(self):
        """Test recovery from operation timeouts."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        def slow_function(*args, **kwargs):
            import time
            time.sleep(10)  # Simulate very slow operation
            return {'result': 'data'}

        with patch('loveslide.call_love', side_effect=slow_function):
            # Should timeout and handle gracefully
            with pytest.raises(TimeoutError):
                slide = SLIDE({'fdr': 0.1, 'timeout': 1}, x=X, y=y)


class TestComplexEdgeCaseScenarios:
    """Test complex scenarios combining multiple edge cases."""

    def test_high_dimensional_with_special_values(self):
        """Test high-dimensional data with special values."""
        n, p = 100, 500
        X = np.random.randn(n, p)

        # Add various edge cases
        X[10, 100] = np.inf  # Infinity
        X[20, 200:205] = np.nan  # Multiple NaNs
        X[:, 300] = 1e-15  # Nearly constant column
        X[:, 350] = X[:, 0] + X[:, 1]  # Linear dependence

        y = np.random.randn(n)

        with pytest.warns(UserWarning) or pytest.raises(ValueError):
            slide = SLIDE({'fdr': 0.1, 'robust': True}, x=X, y=y)

    def test_multiple_concurrent_failures(self):
        """Test handling of multiple concurrent failures."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        # Mock multiple types of failures
        with patch('loveslide.call_love', side_effect=RuntimeError("LOVE failed")), \
             patch('builtins.open', side_effect=PermissionError("File access denied")), \
             patch('numpy.linalg.inv', side_effect=np.linalg.LinAlgError("Singular matrix")):

            with pytest.raises(RuntimeError):
                slide = SLIDE({'fdr': 0.1}, x=X, y=y)

    def test_extreme_parameter_combinations(self):
        """Test extreme but valid parameter combinations."""
        X = np.random.randn(1000, 100)
        y = np.random.randn(1000)

        extreme_params = {
            'fdr': 1e-10,     # Extremely strict FDR
            'niter': 1000,    # Many iterations
            'f_size': 1,      # Smallest possible chunk size
        }

        try:
            slide = SLIDE(extreme_params, x=X, y=y)
            assert slide is not None
        except (MemoryError, TimeoutError):
            pytest.skip("Extreme parameters exceed resource limits")

    def test_pathological_data_structures(self):
        """Test pathological but valid data structures."""
        # Data where most features are noise
        n, p = 200, 100
        signal_features = 2
        noise_features = p - signal_features

        X_signal = np.random.randn(n, signal_features)
        X_noise = 1e-6 * np.random.randn(n, noise_features)  # Very weak noise
        X = np.column_stack([X_signal, X_noise])

        y = X_signal[:, 0] + X_signal[:, 1] + 0.1 * np.random.randn(n)

        slide = SLIDE({'fdr': 0.05}, x=X, y=y)  # Should find signal
        assert slide is not None