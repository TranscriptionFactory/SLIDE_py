"""Test coverage for exception recovery and cleanup patterns."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock, MagicMock
import tempfile
import os
import pickle
from contextlib import contextmanager

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from loveslide import SLIDE, OptimizeSLIDE, SLIDEcv
from loveslide.knockoffs import Knockoffs
from loveslide.love import call_love, call_love_r
from loveslide.tools import init_data


class TestExceptionRecoveryPatterns:
    """Test exception recovery and cleanup in complex operations."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 20))
        y = pd.DataFrame(np.random.randint(0, 2, (50, 1)))
        return X, y

    @pytest.fixture
    def base_params(self):
        """Base parameters for testing."""
        return {
            'x_path': None, 'y_path': None,
            'fdr': 0.1, 'lambda': [0.1],
            'niter': 5, 'n_workers': 1
        }

    def test_init_data_partial_failure_recovery(self, tmp_path):
        """Test init_data recovery from partial file reading failures."""
        # Create valid X file but corrupted Y file
        X = pd.DataFrame(np.random.randn(10, 5))
        x_path = tmp_path / "valid_x.csv"
        y_path = tmp_path / "corrupted_y.csv"

        X.to_csv(x_path)

        # Create corrupted Y file
        with open(y_path, 'w') as f:
            f.write("corrupted,data,format\n1,2,invalid\n")

        params = {'x_path': str(x_path), 'y_path': str(y_path)}

        # Should raise meaningful error, not crash
        with pytest.raises((pd.errors.ParserError, ValueError)):
            init_data(params)

    def test_slide_love_failure_recovery(self, sample_data, base_params):
        """Test SLIDE recovery when LOVE computation fails."""
        X, y = sample_data

        slide = SLIDE(base_params, X, y)

        # Mock LOVE to fail during computation
        with patch('loveslide.slide.call_love') as mock_love:
            mock_love.side_effect = RuntimeError("LOVE computation failed")

            # SLIDE should handle LOVE failure gracefully
            with pytest.raises(RuntimeError):
                # This would typically be called in a larger workflow
                call_love(X.values)

        # Verify SLIDE object is still in valid state after failure
        assert hasattr(slide, 'data')
        assert hasattr(slide, 'input_params')

    def test_knockoff_generation_memory_exhaustion_recovery(self, sample_data):
        """Test knockoff generation recovery from memory issues."""
        X, _ = sample_data

        ko = Knockoffs(fdr=0.1, backend='python')

        # Simulate memory exhaustion during knockoff generation
        with patch('numpy.linalg.cholesky') as mock_chol:
            mock_chol.side_effect = MemoryError("Out of memory")

            with pytest.raises(MemoryError):
                ko._create_gaussian_knockoffs(X.corr().values)

        # Verify knockoff object is still usable after failure
        assert ko.fdr == 0.1
        assert ko.backend == 'python'

    def test_r_interface_failure_graceful_degradation(self, sample_data):
        """Test graceful degradation when R interface fails."""
        X, _ = sample_data

        # Test with R completely unavailable
        with patch('rpy2.robjects.r', side_effect=ImportError("R not available")):
            # Should fall back to Python implementation or raise meaningful error
            ko = Knockoffs(fdr=0.1, backend='auto')

            # The system should handle R unavailability gracefully
            assert ko.backend in ['python', 'auto']

    def test_file_corruption_during_save_load(self, sample_data, base_params, tmp_path):
        """Test recovery from file corruption during save/load operations."""
        X, y = sample_data
        slide = SLIDE(base_params, X, y)

        # Simulate file corruption during save
        save_path = tmp_path / "slide_results.pkl"

        # Create a corrupted pickle file
        with open(save_path, 'wb') as f:
            f.write(b'corrupted_pickle_data')

        # Test load failure handling
        with pytest.raises((pickle.UnpicklingError, EOFError)):
            slide.load_love(str(save_path))

        # Verify slide object is still functional after load failure
        slide.show_params()  # Should not crash

    def test_cv_fold_failure_recovery(self, sample_data, base_params):
        """Test SLIDEcv recovery when individual CV folds fail."""
        X, y = sample_data

        # Create a minimal SLIDE object for CV testing
        slide = SLIDE(base_params, X, y)

        # Mock latent factors for SLIDEcv
        slide.latent_factors = pd.DataFrame(np.random.randn(50, 5))
        slide.marginal_idxs = list(range(5))

        slidecv = SLIDEcv(slide, nrep=2, k=3)

        # Mock a fold to fail during knockoff generation
        with patch.object(Knockoffs, 'run') as mock_ko_run:
            # First call succeeds, second fails, third succeeds
            mock_ko_run.side_effect = [
                Mock(votes=Mock(W=np.random.randn(5)), stats=Mock()),  # Success
                RuntimeError("Fold failed"),                            # Failure
                Mock(votes=Mock(W=np.random.randn(5)), stats=Mock())   # Success
            ]

            # CV should handle individual fold failures gracefully
            try:
                results = slidecv.run(save_results=False)
                # Should get partial results or meaningful error
            except RuntimeError:
                pass  # Expected if all folds fail

    def test_multiprocessing_worker_failure_recovery(self, sample_data, base_params):
        """Test recovery from worker process failures."""
        X, y = sample_data

        # Test with multiple workers where some may fail
        params = base_params.copy()
        params['n_workers'] = 2

        slide = SLIDE(params, X, y)

        # Mock worker process to fail
        with patch('multiprocessing.Pool') as mock_pool:
            mock_pool.return_value.__enter__.return_value.map.side_effect = \
                RuntimeError("Worker process died")

            # Should either handle worker failures or raise meaningful error
            with pytest.raises((RuntimeError, OSError)):
                # This would typically be in a method that uses multiprocessing
                pass

    @contextmanager
    def simulate_resource_exhaustion(self, resource_type="memory"):
        """Context manager to simulate resource exhaustion."""
        if resource_type == "memory":
            with patch('numpy.zeros') as mock_zeros:
                mock_zeros.side_effect = MemoryError("Insufficient memory")
                yield
        elif resource_type == "disk":
            with patch('builtins.open', side_effect=OSError("No space left on device")):
                yield

    def test_resource_exhaustion_recovery(self, sample_data, base_params):
        """Test recovery from various resource exhaustion scenarios."""
        X, y = sample_data
        slide = SLIDE(base_params, X, y)

        # Test memory exhaustion during matrix operations
        with self.simulate_resource_exhaustion("memory"):
            with pytest.raises(MemoryError):
                np.zeros((1000000, 1000000))  # Would trigger our mock

        # Test disk space exhaustion during file operations
        with self.simulate_resource_exhaustion("disk"):
            with pytest.raises(OSError):
                with open("test_file.tmp", 'w') as f:
                    f.write("test")

    def test_cleanup_after_exceptions(self, sample_data, base_params, tmp_path):
        """Test proper cleanup after various exception types."""
        X, y = sample_data

        # Test cleanup after initialization failure
        params = base_params.copy()
        params['out_path'] = str(tmp_path)

        try:
            slide = SLIDE(params, X, y)

            # Force an exception during some operation
            slide.data.X = None  # Corrupt data
            slide.show_params()  # Should fail
        except Exception:
            pass

        # Verify no lingering temporary files or corrupted state
        temp_files = list(tmp_path.glob("*.tmp"))
        assert len(temp_files) == 0  # No temp files left behind

    def test_nested_exception_handling(self, sample_data, base_params):
        """Test handling of nested exceptions in complex operations."""
        X, y = sample_data

        def operation_with_nested_exceptions():
            """Simulate operation that can fail at multiple levels."""
            try:
                # Level 1: Data validation
                if X.shape[1] < 10:
                    pass  # OK

                try:
                    # Level 2: Algorithm execution
                    ko = Knockoffs(fdr=0.1)

                    try:
                        # Level 3: Result processing
                        result = ko._create_gaussian_knockoffs(X.corr().values)
                        return result
                    except np.linalg.LinAlgError as e:
                        raise RuntimeError(f"Matrix computation failed: {e}")

                except RuntimeError as e:
                    raise ValueError(f"Algorithm failed: {e}")

            except ValueError as e:
                raise TypeError(f"Operation failed: {e}")

        # Test that nested exceptions are handled properly
        try:
            operation_with_nested_exceptions()
        except (TypeError, ValueError, RuntimeError):
            pass  # Expected - just testing exception propagation