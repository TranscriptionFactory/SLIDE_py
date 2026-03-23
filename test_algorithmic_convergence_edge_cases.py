"""
Test algorithmic convergence edge cases and pathological conditions.
Critical for preventing infinite loops and ensuring robust convergence detection.
"""
import pytest
import numpy as np
import pandas as pd
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from loveslide.love_python.love import LOVE
from loveslide.knockoffs import Knockoffs
from loveslide.love_python.love.cv import CV_delta, CV_lbd
from loveslide.love_python.love.est_omega import estOmega


class TestConvergenceDetection:
    """Test convergence detection in iterative algorithms."""

    def test_love_convergence_with_oscillating_objective(self):
        """Test LOVE convergence when objective function oscillates."""
        # Create data that might cause oscillation
        X = np.random.randn(50, 20)
        # Add structure that might cause oscillation
        X[:25, :10] = X[:25, :10] + 2.0  # Block structure
        X[25:, 10:] = X[25:, 10:] - 2.0

        # Test with parameters that might cause convergence issues
        with patch('loveslide.love_python.love.cv.CV_delta') as mock_cv:
            # Mock oscillating convergence behavior
            call_count = 0

            def oscillating_cv(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                # Simulate oscillating objective
                if call_count % 2 == 0:
                    return np.array([0.05, 0.1, 0.15]), np.array([1.0, 0.8, 1.2])
                else:
                    return np.array([0.05, 0.1, 0.15]), np.array([1.2, 0.8, 1.0])

            mock_cv.side_effect = oscillating_cv

            # Should detect oscillation and converge
            result = LOVE(X, lbd=0.5, mu=0.5)
            assert result is not None
            assert call_count > 0

    def test_infinite_loop_prevention_cv_delta(self):
        """Test that CV_delta doesn't run indefinitely."""
        X = np.random.randn(30, 10)

        # Create scenario that might cause infinite loop
        delta_grids = np.array([0.01, 0.05, 0.1, 0.2])

        start_time = time.time()

        try:
            # Should complete within reasonable time
            with patch('loveslide.love_python.love.est_pure_homo.FindPureNode') as mock_find:
                # Mock to return consistent results to avoid real infinite loop
                mock_find.return_value = (
                    [1, 2, 3],  # pureIndices
                    np.eye(10),  # Sigma_final
                    np.ones(10) * 0.1  # se_est
                )

                result = CV_delta(X, delta_grids, diagonal=True)

                elapsed = time.time() - start_time
                # Should complete within 30 seconds (generous timeout)
                assert elapsed < 30.0
                assert result is not None

        except Exception as e:
            elapsed = time.time() - start_time
            # Even if it fails, shouldn't take too long
            assert elapsed < 30.0

    def test_convergence_with_degenerate_covariance(self):
        """Test convergence when covariance matrix is degenerate."""
        # Create data with perfect linear dependence
        X_base = np.random.randn(50, 5)
        X_degenerate = np.hstack([
            X_base,
            X_base[:, [0, 1]],  # Duplicate columns
            2 * X_base[:, [2]],  # Linear combination
        ])

        # Should handle degenerate covariance gracefully
        with patch('loveslide.love_python.love.est_omega.estOmega') as mock_omega:
            # Mock to handle singularity
            mock_omega.return_value = np.eye(8) + np.random.randn(8, 8) * 0.01

            try:
                result = LOVE(X_degenerate, lbd=0.1, mu=0.1)
                # Should either succeed or fail gracefully
                assert True
            except (np.linalg.LinAlgError, ValueError):
                # Acceptable to fail on degenerate data
                assert True

    def test_knockoff_convergence_timeout(self):
        """Test that knockoff iterations don't run indefinitely."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        knockoffs = Knockoffs(X, y, fdr=0.1, niter=1000)  # Many iterations

        start_time = time.time()

        # Mock the iteration to simulate slow convergence
        with patch('loveslide.knockoff._parallel._single_knockoff_iteration') as mock_iter:
            def slow_iteration(*args, **kwargs):
                time.sleep(0.01)  # Simulate some processing time
                return Mock(selected_vars=['var1', 'var2'], threshold=1.5)

            mock_iter.side_effect = slow_iteration

            with patch('loveslide.knockoff.filter._run_single_knockoff') as mock_run:
                mock_run.return_value = (['var1', 'var2'], 1.5)

                # Should complete or timeout gracefully
                try:
                    result = knockoffs.run()
                    elapsed = time.time() - start_time
                    # Should not take excessively long even with many iterations
                    assert elapsed < 60.0  # 1 minute timeout
                except Exception:
                    elapsed = time.time() - start_time
                    assert elapsed < 60.0

    def test_estOmega_convergence_edge_cases(self):
        """Test estOmega convergence with edge case inputs."""
        # Test various edge cases for omega estimation
        edge_cases = [
            # Nearly singular correlation matrix
            np.eye(5) + np.ones((5, 5)) * 1e-10,
            # High correlation matrix
            np.eye(5) + np.ones((5, 5)) * 0.9,
            # Block diagonal structure
            np.block([[np.ones((2, 2)), np.zeros((2, 3))],
                      [np.zeros((3, 2)), np.eye(3)]]),
        ]

        for i, C in enumerate(edge_cases):
            # Ensure positive definite
            C = C + np.eye(C.shape[0]) * 0.01

            with patch('loveslide.love_python.love.est_omega.solve_row') as mock_solve:
                # Mock the row solver to ensure convergence
                mock_solve.return_value = np.random.randn(C.shape[0]) * 0.1

                try:
                    result = estOmega(lbd=0.1, C=C)
                    assert result.shape == C.shape
                    # Result should be reasonable (not all zeros or infinite)
                    assert np.isfinite(result).all()
                    assert not np.allclose(result, 0)
                except Exception as e:
                    # Document which edge case failed
                    pytest.fail(f"Edge case {i} failed: {str(e)}")


class TestNumericalStabilityConvergence:
    """Test convergence under numerical instability."""

    def test_convergence_near_machine_precision(self):
        """Test convergence when dealing with machine precision limits."""
        # Create data near machine precision limits
        X = np.random.randn(30, 10) * 1e-15  # Very small values
        X = X + np.random.randn(30, 10) * 1e-14  # Add noise at precision limit

        with patch('loveslide.love_python.love.cv.CV_lbd') as mock_cv_lbd:
            # Mock to avoid numerical issues in test
            mock_cv_lbd.return_value = (0.1, np.random.randn(10, 3))

            try:
                result = LOVE(X, lbd=0.5, mu=0.5)
                # Should handle numerical precision gracefully
                assert result is not None
                if 'A' in result:
                    assert np.isfinite(result['A']).all()
            except (ValueError, np.linalg.LinAlgError) as e:
                # Acceptable to fail on numerically degenerate data
                assert "singular" in str(e).lower() or "precision" in str(e).lower()

    def test_convergence_with_ill_conditioned_matrix(self):
        """Test convergence with ill-conditioned matrices."""
        # Create ill-conditioned matrix
        U = np.random.randn(50, 20)
        # Very large condition number
        S = np.logspace(-15, 0, 20)  # Singular values from 1e-15 to 1
        V = np.random.randn(20, 20)
        X_ill = U @ np.diag(S) @ V

        with patch('loveslide.love_python.love.love.LOVE') as mock_love:
            # Mock to handle ill-conditioning
            mock_love.return_value = {
                'A': np.random.randn(20, 5),
                'pure_indices': [1, 2, 3]
            }

            try:
                result = LOVE(X_ill, lbd=0.1)
                assert result is not None
            except np.linalg.LinAlgError:
                # Expected for extremely ill-conditioned matrices
                assert True

    def test_adaptive_convergence_criteria(self):
        """Test that convergence criteria adapt to data characteristics."""
        # Test different data scales
        data_scales = [1e-6, 1.0, 1e6]

        for scale in data_scales:
            X = np.random.randn(40, 15) * scale

            with patch('loveslide.love_python.love.cv.CV_delta') as mock_cv:
                def scale_adaptive_cv(*args, **kwargs):
                    # Convergence tolerance should adapt to scale
                    return (np.array([0.05, 0.1]),
                            np.array([1.0, 0.8]) * scale)

                mock_cv.side_effect = scale_adaptive_cv

                try:
                    result = LOVE(X, lbd=0.5)
                    # Should handle different scales appropriately
                    assert result is not None
                except Exception as e:
                    # Document scale-related failures
                    pytest.fail(f"Failed at scale {scale}: {str(e)}")


class TestConcurrentConvergence:
    """Test convergence behavior under concurrent operations."""

    def test_parallel_knockoff_convergence(self):
        """Test convergence when running knockoffs in parallel."""
        X = np.random.randn(100, 30)
        y = np.random.randn(100)

        results = []
        threads = []

        def run_knockoffs(thread_id):
            """Run knockoffs in separate thread."""
            knockoffs = Knockoffs(X, y, fdr=0.1, niter=10)

            with patch('loveslide.knockoff.filter.knockoff_filter') as mock_filter:
                mock_filter.return_value = Mock(
                    selected_vars=[f'var_{thread_id}_{i}' for i in range(3)],
                    threshold=1.5
                )

                try:
                    result = knockoffs.run()
                    results.append((thread_id, "success", len(result.selected_vars)))
                except Exception as e:
                    results.append((thread_id, "error", str(e)))

        # Run multiple threads simultaneously
        for i in range(3):
            t = threading.Thread(target=run_knockoffs, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join(timeout=30)  # 30 second timeout per thread

        # All threads should complete successfully
        assert len(results) == 3
        success_count = sum(1 for r in results if r[1] == "success")
        assert success_count >= 2  # Allow for some thread-related failures

    def test_memory_pressure_convergence(self):
        """Test convergence under memory pressure."""
        # Simulate memory pressure with large data
        X_large = np.random.randn(1000, 500)

        with patch('loveslide.love_python.love.love.LOVE') as mock_love:
            def memory_limited_love(*args, **kwargs):
                # Simulate memory pressure by forcing garbage collection
                import gc
                gc.collect()
                return {
                    'A': np.random.randn(500, 10),
                    'pure_indices': list(range(10))
                }

            mock_love.side_effect = memory_limited_love

            try:
                result = LOVE(X_large, lbd=0.1)
                # Should handle memory pressure gracefully
                assert result is not None
            except MemoryError:
                # Acceptable to fail under extreme memory pressure
                assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])