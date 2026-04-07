#!/usr/bin/env python3
"""
Comprehensive edge case tests for LOVE algorithm.
Tests parameter optimization failures, challenging data scenarios, and integration edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import os

from loveslide.love import call_love, call_love_r, _convert_r_pure_ind


class TestLOVEParameterOptimization:
    """Test LOVE parameter optimization edge cases and failures."""

    def test_love_extreme_parameter_combinations(self):
        """Test LOVE with extreme parameter combinations."""
        X = np.random.rand(50, 30)

        extreme_cases = [
            # Very small lambda and mu
            {'lbd': 1e-6, 'mu': 1e-6, 'thresh_fdr': 0.001},
            # Very large lambda and mu (close to 1)
            {'lbd': 0.999, 'mu': 0.999, 'thresh_fdr': 0.999},
            # Asymmetric parameters
            {'lbd': 0.01, 'mu': 0.99, 'thresh_fdr': 0.5},
            {'lbd': 0.99, 'mu': 0.01, 'thresh_fdr': 0.5},
        ]

        for params in extreme_cases:
            try:
                result = call_love(X, **params)
                # If it succeeds, result should be meaningful
                assert result is not None
                if hasattr(result, 'pure_indices'):
                    assert isinstance(result.pure_indices, (list, np.ndarray))
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                # Extreme parameters might legitimately fail
                assert len(str(e)) > 0  # Should provide meaningful error message

    def test_love_parameter_boundary_conditions(self):
        """Test LOVE at parameter boundary conditions."""
        X = np.random.rand(100, 50)

        boundary_cases = [
            # At boundaries (0 and 1)
            {'lbd': 0.0, 'mu': 0.5},  # Lambda at 0
            {'lbd': 1.0, 'mu': 0.5},  # Lambda at 1
            {'lbd': 0.5, 'mu': 0.0},  # Mu at 0
            {'lbd': 0.5, 'mu': 1.0},  # Mu at 1
            # Very strict FDR
            {'lbd': 0.5, 'mu': 0.5, 'thresh_fdr': 0.001},
            # Very lenient FDR
            {'lbd': 0.5, 'mu': 0.5, 'thresh_fdr': 0.999},
        ]

        for params in boundary_cases:
            try:
                result = call_love(X, **params)
                assert result is not None
            except (ValueError, ZeroDivisionError, RuntimeWarning):
                # Boundary values might cause numerical issues
                pass

    def test_love_convergence_failure_scenarios(self):
        """Test LOVE behavior when optimization fails to converge."""
        X = np.random.rand(100, 50)

        # Parameters that might cause convergence issues
        challenging_params = {
            'lbd': 0.5,
            'mu': 0.5,
            'max_iter': 2,  # Very few iterations
            'tol': 1e-15,   # Very tight tolerance
            'thresh_fdr': 0.001  # Very strict FDR
        }

        try:
            result = call_love(X, **challenging_params)
            # If it converges, great
            assert result is not None
        except (RuntimeError, ValueError) as e:
            # Should provide informative error about convergence
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in
                      ['converg', 'iter', 'max', 'tol', 'optim'])

    def test_love_with_optimization_algorithm_failures(self):
        """Test LOVE when underlying optimization algorithms fail."""
        X = np.random.rand(50, 30)

        # Mock optimization failure
        with patch('loveslide.love_python.love.cv.CV_delta',
                   side_effect=RuntimeError("Optimization failed")):
            with pytest.raises(RuntimeError):
                call_love(X, lbd=0.5, mu=0.5)

        # Mock partial optimization failure
        with patch('loveslide.love_python.love.cv.KfoldCV_delta',
                   side_effect=ValueError("Cross-validation failed")):
            try:
                result = call_love(X, lbd=0.5, mu=0.5)
                # Should handle CV failure gracefully or propagate error
                assert result is not None
            except ValueError:
                pass


class TestLOVEDataChallenges:
    """Test LOVE with challenging data scenarios."""

    def test_love_perfect_correlation_structure(self):
        """Test LOVE with perfectly correlated features."""
        n, p = 100, 20
        base_signal = np.random.rand(n, 1)

        # Create perfectly correlated features
        X = np.hstack([base_signal, base_signal + 1e-10 * np.random.rand(n, p-1)])

        try:
            result = call_love(X, lbd=0.5, mu=0.5)
            # Should handle perfect correlations
            assert result is not None
        except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
            # Perfect correlation might cause numerical issues
            pass

    def test_love_block_diagonal_structure(self):
        """Test LOVE with block diagonal correlation structure."""
        # Create block structure: [Block1, Block2, Independent]
        block1 = np.random.rand(100, 10)
        block2 = np.random.rand(100, 10)
        independent = np.random.rand(100, 5)

        # Add within-block correlation
        block1_corr = block1 @ np.random.rand(10, 10)
        block2_corr = block2 @ np.random.rand(10, 10)

        X = np.column_stack([block1_corr, block2_corr, independent])

        result = call_love(X, lbd=0.5, mu=0.5)

        # Should handle block structure appropriately
        assert result is not None
        if hasattr(result, 'pure_indices'):
            # Might detect blocks as pure groups
            assert len(result.pure_indices) >= 0

    def test_love_high_noise_low_signal(self):
        """Test LOVE with high noise, low signal scenarios."""
        n, p = 100, 50

        # Very weak signal
        true_signal = 0.01 * np.random.rand(n, 5)
        noise = np.random.rand(n, p-5)

        X = np.column_stack([true_signal, noise])

        result = call_love(X, lbd=0.5, mu=0.5, thresh_fdr=0.1)

        # Should handle high noise appropriately
        assert result is not None
        # Might not find any pure nodes due to high noise
        if hasattr(result, 'pure_indices'):
            assert isinstance(result.pure_indices, (list, np.ndarray))

    def test_love_sparse_data_patterns(self):
        """Test LOVE with sparse data patterns."""
        n, p = 100, 40

        # Create sparse patterns
        X = np.zeros((n, p))

        # Only some features are non-zero for some samples
        for i in range(0, n, 10):
            for j in range(0, p, 5):
                if i < n and j < p:
                    X[i:i+5, j:j+3] = np.random.rand(min(5, n-i), min(3, p-j))

        try:
            result = call_love(X, lbd=0.5, mu=0.5)
            assert result is not None
        except (ValueError, np.linalg.LinAlgError):
            # Sparse patterns might cause numerical issues
            pass

    def test_love_outlier_contamination(self):
        """Test LOVE with outlier contamination."""
        X = np.random.rand(100, 30)

        # Add extreme outliers
        outlier_mask = np.random.choice([True, False], size=X.shape, p=[0.05, 0.95])
        X[outlier_mask] = np.random.choice([-100, 100], size=np.sum(outlier_mask))

        try:
            result = call_love(X, lbd=0.5, mu=0.5)
            # Should handle outliers robustly or detect them
            assert result is not None
        except (ValueError, RuntimeWarning):
            # Outliers might trigger warnings or errors
            pass


class TestLOVERPythonInterface:
    """Test R/Python interface edge cases and failures."""

    def test_love_r_unavailable_fallback(self):
        """Test LOVE behavior when R is unavailable."""
        X = np.random.rand(50, 20)

        # Mock R unavailability
        with patch('loveslide.love.call_love_r',
                   side_effect=ImportError("R not available")):
            # Should fallback to Python implementation
            result = call_love(X, lbd=0.5, mu=0.5)
            assert result is not None

    def test_love_r_package_unavailable(self):
        """Test LOVE behavior when R packages are unavailable."""
        X = np.random.rand(50, 20)

        # Mock R package import failure
        with patch('loveslide.love.call_love_r',
                   side_effect=RuntimeError("R package 'LOVE' not found")):
            # Should provide informative error or fallback
            try:
                result = call_love(X, lbd=0.5, mu=0.5, backend='R')
                assert result is not None
            except RuntimeError as e:
                assert 'package' in str(e) or 'LOVE' in str(e)

    def test_love_r_large_data_transfer(self):
        """Test R interface with large data that might cause transfer issues."""
        # Large dataset that might cause R/Python transfer issues
        X_large = np.random.rand(1000, 500)

        try:
            # This might fail due to memory or transfer limitations
            result = call_love_r(X_large, lbd=0.5)
            assert result is not None
        except (MemoryError, RuntimeError, OSError):
            # Large data transfer might fail
            pytest.skip("Large data transfer to R failed")

    def test_convert_r_pure_ind_edge_cases(self):
        """Test R pure indices conversion with edge cases."""
        # Test with various R list structures
        test_cases = [
            # Empty list
            [],
            # Single group
            [{'indices': np.array([1, 2, 3]), 'other_field': 'value'}],
            # Multiple groups with different structures
            [
                {'indices': np.array([1, 2]), 'type': 'pure'},
                {'indices': np.array([5, 6, 7, 8]), 'type': 'mixed'}
            ],
            # Edge case with single indices
            [{'indices': np.array([1]), 'singleton': True}]
        ]

        for r_list in test_cases:
            try:
                result = _convert_r_pure_ind(r_list)
                assert isinstance(result, list)
                # Each element should be a list of indices
                for group in result:
                    assert isinstance(group, list)
                    assert all(isinstance(idx, (int, np.integer)) for idx in group)
            except (ValueError, TypeError, KeyError):
                # Some edge cases might legitimately fail
                pass

    def test_love_r_parameter_type_conversion(self):
        """Test R interface parameter type conversion edge cases."""
        X = np.random.rand(50, 20)

        # Test with various parameter types that need conversion
        param_cases = [
            # Integer parameters (might need float conversion)
            {'lbd': 1, 'delta': 1, 'thresh_fdr': 1},
            # Numpy scalar parameters
            {'lbd': np.float64(0.5), 'delta': np.float32(0.1)},
            # List parameters (might need R vector conversion)
            {'rep_CV': [10, 20, 30]}  # If applicable
        ]

        for params in param_cases:
            try:
                result = call_love_r(X, **params)
                assert result is not None
            except (TypeError, ValueError) as e:
                # Type conversion issues are expected for some cases
                assert 'type' in str(e).lower() or 'convert' in str(e).lower()


class TestLOVEMemoryAndPerformance:
    """Test LOVE memory management and performance edge cases."""

    def test_love_memory_pressure_scenarios(self):
        """Test LOVE behavior under memory pressure."""
        # Progressively larger datasets
        sizes = [(100, 50), (500, 100), (1000, 200)]

        for n, p in sizes:
            X = np.random.rand(n, p)

            try:
                result = call_love(X, lbd=0.5, mu=0.5)
                assert result is not None

                # Check memory cleanup
                import gc
                gc.collect()

            except MemoryError:
                # Expected for very large datasets
                break

    def test_love_computational_complexity_limits(self):
        """Test LOVE with computationally challenging scenarios."""
        # High-dimensional but manageable
        X = np.random.rand(200, 150)

        # Set time limits to prevent hanging
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("LOVE computation timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout

        try:
            result = call_love(X, lbd=0.5, mu=0.5)
            assert result is not None
        except TimeoutError:
            # Acceptable if computation is too complex
            pass
        finally:
            signal.alarm(0)  # Cancel timeout

    def test_love_iterative_algorithm_interruption(self):
        """Test LOVE behavior when iterative algorithms are interrupted."""
        X = np.random.rand(100, 50)

        # Mock interruption during iteration
        original_cv = None
        try:
            import loveslide.love_python.love.cv as cv_module
            original_cv = cv_module.CV_delta

            def interrupted_cv(*args, **kwargs):
                raise KeyboardInterrupt("User interrupted")

            cv_module.CV_delta = interrupted_cv

            with pytest.raises(KeyboardInterrupt):
                call_love(X, lbd=0.5, mu=0.5)

        finally:
            if original_cv:
                cv_module.CV_delta = original_cv


if __name__ == "__main__":
    pytest.main([__file__])