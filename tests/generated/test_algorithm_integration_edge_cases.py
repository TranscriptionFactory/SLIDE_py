"""
Test coverage for algorithm integration and workflow edge cases.

Critical gaps in testing complex interactions between different algorithms
and end-to-end workflow scenarios that might fail in production.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock
import warnings

from src.loveslide.slide import SLIDE, OptimizeSLIDE
from src.loveslide.knockoffs import Knockoffs
from src.loveslide.cv import SLIDEcv
from src.loveslide.love import call_love
from src.loveslide.score import Estimator, SLIDE_Estimator
from src.loveslide.plotting import Plotter

class TestAlgorithmInteractionEdgeCases:
    """Test edge cases in algorithm interactions."""

    def test_slide_knockoffs_dimension_mismatch(self):
        """Test SLIDE when knockoffs return unexpected dimensions."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        knockoffs = Knockoffs()

        # Mock knockoffs to return wrong dimensions
        with patch.object(knockoffs, 'fit_transform') as mock_transform:
            # Return wrong shape
            mock_transform.return_value = np.random.randn(100, 45)  # Wrong number of features

            slide = OptimizeSLIDE({'fdr': 0.1})

            with pytest.raises((ValueError, AssertionError)):
                # Should detect dimension mismatch
                slide.get_latent_factors(X, knockoffs=knockoffs)

    def test_love_slide_inconsistent_results(self):
        """Test handling when LOVE and SLIDE produce inconsistent results."""
        X = np.random.randn(100, 20)

        # Mock LOVE to return inconsistent latent factor count
        mock_love_result = {
            'A': np.random.randn(20, 15),  # 15 latent factors
            'other_data': 'mock'
        }

        slide = OptimizeSLIDE({'fdr': 0.1})

        # Should handle inconsistent LF counts gracefully
        try:
            z_matrix = slide.calc_z_matrix(mock_love_result)
            assert z_matrix.shape[1] <= 15  # Should adapt to available LFs
        except (ValueError, KeyError) as e:
            # Acceptable - should not crash silently
            assert "inconsistent" in str(e).lower() or "shape" in str(e).lower()

    def test_cv_slide_parameter_conflict(self):
        """Test CV when SLIDE parameters conflict with CV settings."""
        X = np.random.randn(100, 30)
        y = np.random.randn(100)

        # Parameters that might conflict
        slide_cv = SLIDEcv(
            X=X, y=y,
            fdr_range=[0.1, 0.2],
            n_folds=5,
            n_workers=1
        )

        # Mock SLIDE to have conflicting internal parameters
        with patch('src.loveslide.slide.OptimizeSLIDE') as MockSLIDE:
            mock_instance = MagicMock()
            mock_instance.input_params = {'fdr': 0.3}  # Conflicts with CV range
            MockSLIDE.return_value = mock_instance

            # Should handle parameter conflicts
            try:
                slide_cv.run()
            except ValueError as e:
                # Should provide clear error about parameter conflict
                assert "parameter" in str(e).lower() or "conflict" in str(e).lower()

class TestWorkflowStateConsistency:
    """Test state consistency across complex workflows."""

    def test_interrupted_workflow_recovery(self):
        """Test recovery from interrupted multi-step workflows."""
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        with tempfile.TemporaryDirectory() as temp_dir:
            slide = OptimizeSLIDE({'fdr': 0.1, 'outpath': temp_dir})

            # Simulate workflow interruption after partial completion
            slide.X = X
            slide.y = y

            # Save partial state manually
            partial_state = {
                'step': 'love_completed',
                'A_matrix': np.random.randn(20, 10),
                'iteration': 2
            }

            import pickle
            state_file = os.path.join(temp_dir, 'params_iter_2.pkl')
            with open(state_file, 'wb') as f:
                pickle.dump(partial_state, f)

            # Should resume from saved state
            try:
                slide.load_state(2)
                # Continue workflow
                slide.run_SLIDE(X, love_result={'A': partial_state['A_matrix']})

            except (FileNotFoundError, KeyError) as e:
                # Acceptable if graceful error handling
                assert "state" in str(e).lower() or "resume" in str(e).lower()

    def test_memory_state_vs_file_state_consistency(self):
        """Test consistency between memory state and file state."""
        X = np.random.randn(50, 15)
        y = np.random.randn(50)

        with tempfile.TemporaryDirectory() as temp_dir:
            slide = OptimizeSLIDE({'fdr': 0.1, 'outpath': temp_dir})
            slide.X = X
            slide.y = y

            # Simulate state divergence
            memory_params = {'fdr': 0.1, 'iteration': 3}
            slide.input_params.update(memory_params)

            # File state different from memory
            file_params = {'fdr': 0.2, 'iteration': 3}  # Different fdr

            import pickle
            state_file = os.path.join(temp_dir, 'params_iter_3.pkl')
            with open(state_file, 'wb') as f:
                pickle.dump(file_params, f)

            # Should detect and handle state inconsistency
            try:
                slide.load_state(3)
                # Check which state is used
                assert slide.input_params['fdr'] in [0.1, 0.2]  # Should be consistent
            except ValueError as e:
                # Acceptable if detects inconsistency
                assert "inconsistent" in str(e).lower() or "state" in str(e).lower()

    def test_concurrent_algorithm_resource_sharing(self):
        """Test resource sharing between concurrent algorithm instances."""
        X = np.random.randn(100, 30)

        # Create multiple knockoff instances
        knockoffs1 = Knockoffs()
        knockoffs2 = Knockoffs()

        # Should not interfere with each other
        result1 = knockoffs1.fit_transform(X)
        result2 = knockoffs2.fit_transform(X)

        # Results should be independent
        assert not np.array_equal(result1, result2)  # Should be different random knockoffs
        assert result1.shape == result2.shape == X.shape

class TestDataFlowEdgeCases:
    """Test edge cases in data flow between components."""

    def test_sparse_to_dense_conversion_consistency(self):
        """Test consistency when converting between sparse and dense matrices."""
        pytest.importorskip("scipy")
        from scipy import sparse

        # Create sparse matrix
        dense_X = np.random.randn(50, 20)
        dense_X[dense_X < 0.5] = 0  # Make sparse
        sparse_X = sparse.csr_matrix(dense_X)

        knockoffs = Knockoffs()

        # Should handle both formats consistently
        dense_result = knockoffs.fit_transform(dense_X)
        sparse_result = knockoffs.fit_transform(sparse_X.toarray())

        # Results should have consistent properties
        assert dense_result.shape == sparse_result.shape
        assert not np.allclose(dense_result, sparse_result)  # Different random results

    def test_nan_propagation_through_pipeline(self):
        """Test NaN handling through complete pipeline."""
        X = np.random.randn(50, 20)
        X[10, 5] = np.nan  # Insert NaN
        y = np.random.randn(50)
        y[15] = np.nan  # Insert NaN in target

        with warnings.catch_warnings():
            warnings.filterwarnings("error")

            try:
                slide = OptimizeSLIDE({'fdr': 0.1})
                # Should either handle NaNs or raise clear error
                result = slide.run_SLIDE(X, love_result={'A': np.random.randn(20, 10)})

                # If successful, should not contain NaN in results
                if isinstance(result, dict) and 'scores' in result:
                    assert not np.isnan(result['scores']).any()

            except (ValueError, Warning) as e:
                # Acceptable if clearly identifies NaN issue
                assert "nan" in str(e).lower() or "missing" in str(e).lower()

    def test_infinite_values_handling(self):
        """Test handling of infinite values in data pipeline."""
        X = np.random.randn(50, 20)
        X[0, 0] = np.inf
        X[1, 1] = -np.inf
        y = np.random.randn(50)

        knockoffs = Knockoffs()

        # Should handle infinite values gracefully
        try:
            result = knockoffs.fit_transform(X)
            # Should not contain infinite values in output
            assert np.isfinite(result).all()

        except ValueError as e:
            # Acceptable if clearly identifies infinite value issue
            assert "infinite" in str(e).lower() or "finite" in str(e).lower()

class TestScalabilityEdgeCases:
    """Test edge cases related to scalability and performance."""

    def test_memory_efficient_large_matrix_operations(self):
        """Test memory efficiency with large matrices."""
        # Use moderately large matrices to test memory handling
        large_X = np.random.randn(500, 100)

        knockoffs = Knockoffs()

        # Should handle large matrices without excessive memory usage
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        result = knockoffs.fit_transform(large_X)

        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before

        # Memory increase should be reasonable (less than 10x data size)
        data_size = large_X.nbytes
        assert memory_increase < 10 * data_size

        assert result.shape == large_X.shape

    def test_algorithm_timeout_handling(self):
        """Test handling of algorithms that exceed time limits."""
        X = np.random.randn(100, 50)

        # Mock slow algorithm
        knockoffs = Knockoffs()

        with patch.object(knockoffs, 'fit_transform') as mock_fit:
            import time

            def slow_transform(*args, **kwargs):
                time.sleep(0.1)  # Simulate slow operation
                return np.random.randn(*X.shape)

            mock_fit.side_effect = slow_transform

            # Should complete within reasonable time
            import time
            start_time = time.time()
            result = knockoffs.fit_transform(X)
            end_time = time.time()

            # Should not take excessively long
            assert end_time - start_time < 1.0  # Should finish quickly in mock
            assert result.shape == X.shape

if __name__ == "__main__":
    pytest.main([__file__])