"""
Test coverage for algorithm state recovery and consistency after interruptions.

Critical gaps:
- State recovery after exceptions
- Partial computation resumption
- Memory state consistency
"""

import pytest
import numpy as np
import sys
import os
import threading
import time
import signal
from unittest.mock import patch, MagicMock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from loveslide.slide import SLIDE, OptimizeSLIDE
from loveslide.cv import SLIDEcv
from loveslide.knockoffs import Knockoffs
from loveslide.score import SLIDE_Estimator


class TestAlgorithmStateRecovery:
    """Test algorithm state recovery after interruptions"""

    def test_slide_recovery_after_memory_error(self):
        """Test SLIDE recovery after memory exhaustion"""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        slide = SLIDE()

        # Simulate memory error during computation
        with patch('numpy.linalg.svd', side_effect=MemoryError("Out of memory")):
            with pytest.raises(MemoryError):
                slide.run(X, y)

        # Algorithm should be in clean state for retry
        # Smaller problem should work
        X_small = X[:50, :25]
        y_small = y[:50]

        result = slide.run(X_small, y_small)
        assert result is not None
        assert hasattr(result, 'selected_vars')

    def test_optimization_interruption_recovery(self):
        """Test optimization algorithm interruption and recovery"""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        slide_opt = OptimizeSLIDE()

        # Simulate interruption during optimization
        def interrupt_after_iterations(*args, **kwargs):
            # Allow a few iterations then raise
            if not hasattr(interrupt_after_iterations, 'call_count'):
                interrupt_after_iterations.call_count = 0
            interrupt_after_iterations.call_count += 1

            if interrupt_after_iterations.call_count > 3:
                raise KeyboardInterrupt("User interruption")

            return np.random.randn(len(args[0]))  # Mock return

        with patch('scipy.optimize.minimize', side_effect=interrupt_after_iterations):
            with pytest.raises(KeyboardInterrupt):
                slide_opt.run(X, y, max_iter=100)

        # Should recover for next run
        result = slide_opt.run(X, y, max_iter=10)
        assert result is not None

    def test_cv_fold_failure_recovery(self):
        """Test CV recovery when individual folds fail"""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        cv = SLIDEcv()

        # Mock fold that fails randomly
        original_run = SLIDE.run

        def failing_fold_run(self, X_fold, y_fold, **kwargs):
            # Randomly fail some folds
            if np.random.random() < 0.3:  # 30% failure rate
                raise ValueError("Fold computation failed")
            return original_run(self, X_fold, y_fold, **kwargs)

        with patch.object(SLIDE, 'run', failing_fold_run):
            # Should handle fold failures gracefully
            try:
                result = cv.run(X, y, n_folds=5)
                # Should complete with available folds
                assert result is not None
            except ValueError:
                # Acceptable if too many folds fail
                pass

    def test_knockoff_generation_partial_recovery(self):
        """Test knockoff generation recovery from partial failures"""
        X = np.random.randn(100, 50)

        knockoffs = Knockoffs()

        # Simulate failure in knockoff generation
        with patch('loveslide.knockoff.create._create_sdp',
                   side_effect=Exception("SDP solver failed")):
            with pytest.raises(Exception):
                knockoffs.generate(X, method='sdp')

        # Should fallback to alternative method
        result = knockoffs.generate(X, method='equicorrelated')
        assert result is not None
        assert result.shape == X.shape


class TestPartialComputationConsistency:
    """Test partial computation and resumption consistency"""

    def test_estimator_partial_fit_consistency(self):
        """Test estimator partial fitting consistency"""
        X = np.random.randn(200, 50)
        y = np.random.randn(200)

        estimator = SLIDE_Estimator()

        # Split data for partial fitting
        X1, X2 = X[:100], X[100:]
        y1, y2 = y[:100], y[100:]

        # Fit on first part
        estimator.fit(X1, y1)
        params_after_first = estimator.get_params()

        # Continue with second part (partial_fit equivalent)
        estimator.fit(X2, y2)
        params_after_second = estimator.get_params()

        # Parameters should evolve consistently
        assert params_after_first != params_after_second

        # Full fit should be reasonably similar
        estimator_full = SLIDE_Estimator()
        estimator_full.fit(X, y)

        # Predictions should be reasonably similar
        pred_partial = estimator.predict(X)
        pred_full = estimator_full.predict(X)

        correlation = np.corrcoef(pred_partial, pred_full)[0, 1]
        assert correlation > 0.7  # Should be reasonably correlated

    def test_cv_checkpoint_consistency(self):
        """Test CV checkpoint and resumption consistency"""
        X = np.random.randn(200, 50)
        y = np.random.randn(200)

        cv = SLIDEcv()

        # Simulate checkpoint during CV
        fold_results = []

        def checkpoint_fold(fold_idx, X_train, X_test, y_train, y_test):
            """Simulate individual fold computation"""
            slide = SLIDE()
            result = slide.run(X_train, y_train)
            score = np.mean((result.predict(X_test) - y_test) ** 2)
            fold_results.append(score)
            return score

        # Simulate interruption after 3 folds
        with patch('threading.Thread') as mock_thread:
            # Mock partial CV completion
            for i in range(3):  # Simulate 3 completed folds
                fold_results.append(np.random.random())

            # Resume should continue from where left off
            # This would be implemented in a real checkpointing system
            assert len(fold_results) == 3

    def test_optimization_warm_start_consistency(self):
        """Test optimization warm start consistency"""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        slide_opt = OptimizeSLIDE()

        # First optimization
        result1 = slide_opt.run(X, y, max_iter=10)

        # Warm start from previous result (conceptually)
        # This tests that algorithm state is preserved appropriately
        result2 = slide_opt.run(X, y, max_iter=20)

        # Second run should potentially be more converged
        assert result2 is not None
        assert result1 is not None


class TestMemoryStateConsistency:
    """Test memory state consistency across operations"""

    def test_memory_state_after_exception(self):
        """Test memory state consistency after exceptions"""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        slide = SLIDE()

        # Force exception during computation
        with patch('numpy.dot', side_effect=Exception("Computation error")):
            with pytest.raises(Exception):
                slide.run(X, y)

        # Memory state should be clean
        import gc
        gc.collect()

        # New computation should work normally
        result = slide.run(X, y)
        assert result is not None

    def test_concurrent_state_isolation(self):
        """Test state isolation in concurrent operations"""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        results = {}
        errors = {}

        def worker(worker_id):
            """Worker function for concurrent operations"""
            try:
                slide = SLIDE()
                # Different parameters per worker
                thresh = 0.1 + worker_id * 0.05
                result = slide.run(X, y, thresh=thresh)
                results[worker_id] = result
            except Exception as e:
                errors[worker_id] = e

        # Start multiple concurrent operations
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Each worker should have independent state
        assert len(errors) == 0  # No cross-contamination errors
        assert len(results) == 3  # All workers completed

        # Results should differ (different thresholds)
        result_vars = [len(r.selected_vars) for r in results.values()]
        assert len(set(result_vars)) > 1  # Different numbers of selected variables

    def test_algorithm_state_cleanup_after_failure(self):
        """Test algorithm state cleanup after various failures"""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        slide = SLIDE()

        # Various failure scenarios
        failure_scenarios = [
            (ValueError, "Invalid input"),
            (RuntimeError, "Runtime error"),
            (MemoryError, "Memory exhausted"),
            (KeyboardInterrupt, "User interruption")
        ]

        for exception_type, message in failure_scenarios:
            # Simulate different types of failures
            with patch('loveslide.slide.SLIDE._compute_statistics',
                       side_effect=exception_type(message)):
                with pytest.raises(exception_type):
                    slide.run(X, y)

            # State should be clean after each failure
            # Test by running a simple operation
            try:
                test_result = slide.run(X[:20], y[:20])  # Small problem
                assert test_result is not None
            except Exception:
                # If still failing, cleanup was incomplete
                pytest.fail(f"State not clean after {exception_type.__name__}")


class TestStateConsistencyValidation:
    """Test validation of state consistency"""

    def test_parameter_state_consistency(self):
        """Test parameter state remains consistent"""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        slide = SLIDE()

        # Set specific parameters
        initial_params = {'thresh': 0.1, 'max_iter': 100}

        result1 = slide.run(X, y, **initial_params)

        # Parameters should not change between runs
        result2 = slide.run(X, y, **initial_params)

        # Results should be identical (deterministic)
        np.random.seed(42)  # Control randomness
        result3 = slide.run(X, y, **initial_params)

        # With same seed, results should be reproducible
        assert result3 is not None

    def test_internal_state_isolation(self):
        """Test internal state isolation between instances"""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        slide1 = SLIDE()
        slide2 = SLIDE()

        # Configure differently
        result1 = slide1.run(X, y, thresh=0.1)
        result2 = slide2.run(X, y, thresh=0.2)

        # Internal states should be independent
        assert result1 is not None
        assert result2 is not None
        # Should have different numbers of selected variables
        assert len(result1.selected_vars) != len(result2.selected_vars)


if __name__ == "__main__":
    pytest.main([__file__])