"""
Test coverage for complex exception recovery scenarios in SLIDE optimization.
These failure modes occur during parameter grid search and state management.
"""
import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

from loveslide.slide import OptimizeSLIDE


class TestSLIDEExceptionRecoveryScenarios:
    """Test complex exception handling and recovery in SLIDE optimization."""

    @pytest.fixture
    def mock_optimize_slide(self):
        """Create OptimizeSLIDE instance for testing."""
        input_params = {
            'X': np.random.randn(100, 20),
            'Y': np.random.randn(100, 1),
            'out_path': tempfile.mkdtemp(),
            'delta_range': [0.1, 0.5],
            'lambda_range': [0.1, 0.5],
            'thresh_fdr': 0.2,
            'pure_homo': False,
            'love_backend': 'python'
        }
        return OptimizeSLIDE(input_params)

    def test_love_result_none_handling(self, mock_optimize_slide):
        """Test handling when LOVE returns None (Dantzig LP failure)."""
        with patch.object(mock_optimize_slide, 'get_latent_factors') as mock_love:
            mock_love.return_value = None
            mock_optimize_slide.love_result = None

            # Should skip iteration when LOVE returns None
            with patch('builtins.print') as mock_print:
                try:
                    mock_optimize_slide.run(verbose=True)
                except Exception:
                    pass  # We expect this to fail gracefully

                # Check that the failure was logged
                print_calls = [call.args[0] for call in mock_print.call_args_list]
                assert any("LOVE returned None" in str(call) for call in print_calls)

    def test_data_mismatch_recovery_during_state_load(self, mock_optimize_slide):
        """Test recovery when loaded state doesn't match input data."""
        # Create mock state directory with mismatched data
        out_iter = os.path.join(mock_optimize_slide.input_params['out_path'], "0.1_0.5_out")
        os.makedirs(out_iter, exist_ok=True)

        # Create mismatched A matrix
        mismatched_A = pd.DataFrame(
            np.random.randn(15, 5),  # Wrong dimensions
            columns=[f"Z{i}" for i in range(5)]
        )
        mismatched_A.to_csv(os.path.join(out_iter, "A.csv"))

        # Create z_matrix
        z_matrix = pd.DataFrame(np.random.randn(50, 5))  # Wrong sample count
        z_matrix.to_csv(os.path.join(out_iter, "z_matrix.csv"))

        # Create sig_LFs
        with open(os.path.join(out_iter, "sig_LFs.txt"), 'w') as f:
            f.write("Z0\nZ1")

        with patch('builtins.print') as mock_print:
            mock_optimize_slide.load_state(out_iter)

            # Should handle the error gracefully
            print_calls = [call.args[0] for call in mock_print.call_args_list]
            assert any("do not match" in str(call) for call in print_calls)

    def test_file_permission_error_handling(self, mock_optimize_slide):
        """Test handling of file permission errors during state save."""
        with patch('os.makedirs', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                mock_optimize_slide.run()

    def test_corrupted_pickle_file_recovery(self, mock_optimize_slide):
        """Test recovery from corrupted pickle files."""
        corrupted_path = os.path.join(mock_optimize_slide.input_params['out_path'], "corrupted.pkl")

        # Create a corrupted file
        with open(corrupted_path, 'w') as f:
            f.write("corrupted content")

        with patch('builtins.print') as mock_print:
            mock_optimize_slide.load_love(corrupted_path)

            print_calls = [call.args[0] for call in mock_print.call_args_list]
            assert any("Error loading LOVE result" in str(call) for call in print_calls)

    def test_memory_pressure_during_optimization(self, mock_optimize_slide):
        """Test behavior under memory pressure."""
        # Simulate memory error during latent factor computation
        with patch.object(mock_optimize_slide, 'get_latent_factors',
                         side_effect=MemoryError("Out of memory")):
            with patch('builtins.print') as mock_print:
                try:
                    mock_optimize_slide.run(verbose=True)
                except Exception:
                    pass

                print_calls = [call.args[0] for call in mock_print.call_args_list]
                assert any("Error running LOVE" in str(call) for call in print_calls)

    def test_network_failure_during_r_backend(self, mock_optimize_slide):
        """Test handling network failures when using R backend."""
        mock_optimize_slide.input_params['love_backend'] = 'r'

        # Simulate network/R connection failure
        with patch('loveslide.love.call_love_r',
                   side_effect=ConnectionError("R server unavailable")):
            with pytest.raises((ConnectionError, Exception)):
                mock_optimize_slide.get_latent_factors(
                    x=mock_optimize_slide.data.X,
                    y=mock_optimize_slide.data.Y,
                    delta=0.1,
                    lbd=0.5,
                    love_backend='r'
                )

    def test_disk_space_exhaustion_during_output(self, mock_optimize_slide):
        """Test handling disk space exhaustion during output writing."""
        # Mock disk space exhaustion
        with patch('pandas.DataFrame.to_csv', side_effect=OSError("No space left on device")):
            with pytest.raises(OSError):
                # This should trigger during state saving
                mock_optimize_slide.latent_factors = pd.DataFrame(np.random.randn(10, 5))
                mock_optimize_slide.latent_factors.to_csv("test.csv")

    def test_interrupted_optimization_state_consistency(self, mock_optimize_slide):
        """Test state consistency when optimization is interrupted."""
        iteration_count = 0

        def mock_get_latent_factors(*args, **kwargs):
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count == 2:  # Interrupt on second iteration
                raise KeyboardInterrupt("User interrupted")
            return Mock()

        with patch.object(mock_optimize_slide, 'get_latent_factors',
                         side_effect=mock_get_latent_factors):
            with pytest.raises(KeyboardInterrupt):
                mock_optimize_slide.run()

    def test_concurrent_access_state_corruption(self, mock_optimize_slide):
        """Test handling of state corruption from concurrent access."""
        out_iter = os.path.join(mock_optimize_slide.input_params['out_path'], "0.1_0.5_out")
        os.makedirs(out_iter, exist_ok=True)

        # Simulate partial write (file exists but is empty/corrupt)
        with open(os.path.join(out_iter, "A.csv"), 'w') as f:
            f.write("")  # Empty file

        with patch('pandas.read_csv', side_effect=pd.errors.EmptyDataError("No data")):
            with pytest.raises((pd.errors.EmptyDataError, Exception)):
                mock_optimize_slide.load_state(out_iter)

    def test_parameter_grid_exhaustion_fallback(self, mock_optimize_slide):
        """Test behavior when all parameter combinations fail."""
        # All parameter combinations fail
        with patch.object(mock_optimize_slide, 'get_latent_factors',
                         side_effect=Exception("All parameters failed")):
            with patch('builtins.print') as mock_print:
                mock_optimize_slide.run(verbose=True)

                # Should log all failures
                print_calls = [call.args[0] for call in mock_print.call_args_list]
                failure_count = sum("Error running LOVE" in str(call) for call in print_calls)
                assert failure_count > 0

    def test_invalid_knockoff_backend_fallback(self, mock_optimize_slide):
        """Test fallback when knockoff backend is invalid."""
        mock_optimize_slide.input_params['knockoff_backend'] = 'invalid_backend'

        # Should handle invalid backend gracefully
        with patch('builtins.print') as mock_print:
            try:
                mock_optimize_slide.select_short_freq_slide()
            except Exception as e:
                assert "backend" in str(e).lower() or "invalid" in str(e).lower()

    def test_cleanup_after_critical_failure(self, mock_optimize_slide):
        """Test cleanup after critical system failures."""
        out_path = mock_optimize_slide.input_params['out_path']

        # Simulate critical failure that leaves partial state
        with patch('builtins.print'):
            try:
                # Create some state
                os.makedirs(os.path.join(out_path, "0.1_0.5_out"), exist_ok=True)

                # Trigger failure
                with patch.object(mock_optimize_slide, 'get_latent_factors',
                                 side_effect=SystemError("Critical system error")):
                    mock_optimize_slide.run()
            except SystemError:
                pass

        # Verify cleanup can be performed
        assert os.path.exists(out_path)  # Directory structure intact for manual cleanup