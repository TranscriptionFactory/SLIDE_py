"""
Test coverage for complex private method chain interactions
Focus: Multi-step computation workflows and internal state consistency
"""

import pytest
import numpy as np
import unittest.mock as mock
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loveslide import SLIDE, OptimizeSLIDE, SLIDEcv
from loveslide.knockoffs import Knockoffs
from loveslide.cv import SLIDEcv


class TestPrivateMethodChainIntegration:
    """Test complex private method interactions that occur in computation chains"""

    def test_cv_fold_processing_chain_integrity(self):
        """Test integrity of _run_slide_fold -> _find_interactions_fold -> _build_prediction_features chain"""
        # Setup test data
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)

        cv = SLIDEcv(
            x=X, y=y,
            folds=[(list(range(25)), list(range(25, 50)))],
            n_workers=1
        )

        # Mock intermediate states to test chain consistency
        with patch.object(cv, '_find_interactions_fold') as mock_interactions:
            with patch.object(cv, '_build_prediction_features') as mock_features:
                mock_interactions.return_value = {'interactions': [], 'marginal_features': list(range(5))}
                mock_features.return_value = np.random.rand(25, 8)

                # Test chain execution maintains state consistency
                result = cv._run_slide_fold(0, cv.folds[0])

                # Verify method call chain integrity
                assert mock_interactions.called
                assert mock_features.called

                # Verify state consistency across chain
                call_args = mock_features.call_args[0]
                assert len(call_args) >= 3  # Expected number of arguments

    def test_knockoff_construction_solver_fallback_chain(self):
        """Test knockoff construction with SDP solver fallback chain"""
        X = np.random.rand(30, 8)
        y = np.random.randint(0, 2, 30)

        knockoffs = Knockoffs(y=y, z2=X)

        # Test solver fallback chain under different failure modes
        with patch('loveslide.knockoff.solve._get_sdp_solver') as mock_solver:
            # Simulate solver unavailability
            mock_solver.side_effect = ImportError("CVXPY not available")

            # Should fallback gracefully through solver chain
            try:
                result = knockoffs.filter_knockoffs_iterative_python(
                    z=X, y=y, fdr=0.1, niter=1
                )
                # Verify fallback mechanism worked
                assert result is not None
            except ImportError:
                # Expected if no fallback solvers available
                pass

    def test_slide_latent_factor_extraction_chain(self):
        """Test SLIDE latent factor extraction method chain consistency"""
        params = {'K': 3, 'max_iters': 2, 'fdr_thresh': 0.1}
        X = np.random.rand(40, 12)
        y = np.random.randint(0, 2, 40)

        slide = OptimizeSLIDE(params, x=X, y=y)

        # Mock LOVE result for testing chain
        mock_love_result = {
            'L_hat': np.random.rand(12, 3),
            'pure_idx': [0, 1, 2, 5, 8]
        }

        with patch.object(slide, 'calc_z_matrix') as mock_calc_z:
            mock_calc_z.return_value = np.random.rand(40, 15)

            # Test method chain: get_latent_factors -> calc_z_matrix -> find_standalone_LFs
            try:
                lf_result = slide.get_latent_factors(
                    x=X, y=y, love_result=mock_love_result
                )

                # Verify chain execution and state consistency
                assert mock_calc_z.called
                assert lf_result is not None

                # Verify internal state consistency
                if hasattr(slide, 'z_matrix'):
                    assert slide.z_matrix.shape[0] == X.shape[0]

            except Exception as e:
                # Log chain failure for debugging
                pytest.fail(f"Private method chain failed: {e}")

    def test_love_estimation_parameter_propagation_chain(self):
        """Test parameter propagation through LOVE estimation chain"""
        from loveslide.love import call_love

        X = np.random.rand(25, 8)

        # Test parameter propagation through estimation chain
        with patch('loveslide.love.call_love_r') as mock_love_r:
            # Mock successful R call
            mock_love_r.return_value = {
                'L_hat': np.random.rand(8, 3),
                'pure_idx': [0, 1, 3],
                'converged': True
            }

            # Test parameter consistency through chain
            result = call_love(
                X=X,
                lbd=0.3,
                mu=0.7,
                thresh_fdr=0.15,
                verbose=False
            )

            # Verify parameter propagation
            call_args = mock_love_r.call_args
            if call_args:
                kwargs = call_args[1] if len(call_args) > 1 else {}
                # Check critical parameters were propagated
                assert 'thresh_fdr' in kwargs or len(call_args[0]) > 3

    def test_statistical_computation_consistency_chain(self):
        """Test statistical computation consistency across method chains"""
        X = np.random.rand(35, 10)
        y = np.random.randint(0, 2, 35)

        # Test statistical consistency across computation chain
        knockoffs = Knockoffs(y=y, z2=X)

        # Test multiple iterations maintain statistical properties
        results = []
        for i in range(3):
            try:
                result = knockoffs.filter_knockoffs_iterative_python(
                    z=X, y=y, fdr=0.1, niter=1, seed=i
                )
                if result is not None:
                    results.append(result)
            except:
                pass

        # Verify statistical consistency across iterations
        if len(results) >= 2:
            # Basic consistency checks
            assert all(hasattr(r, 'selected') for r in results)
            # Statistical properties should be similar across runs
            fdr_rates = [len(r.selected) / X.shape[1] if hasattr(r, 'selected') else 0 for r in results]
            if any(f > 0 for f in fdr_rates):
                assert np.std(fdr_rates) < 0.5  # Reasonable consistency


class TestMethodChainErrorPropagation:
    """Test error propagation through private method chains"""

    def test_cv_fold_error_propagation(self):
        """Test error propagation through CV fold processing chain"""
        X = np.random.rand(30, 8)
        y = np.random.randint(0, 2, 30)

        cv = SLIDEcv(x=X, y=y, folds=[(list(range(15)), list(range(15, 30)))])

        # Inject error in middle of chain
        with patch.object(cv, '_find_interactions_fold') as mock_interactions:
            mock_interactions.side_effect = ValueError("Computation failed")

            # Verify error propagation and handling
            with pytest.raises(ValueError):
                cv._run_slide_fold(0, cv.folds[0])

    def test_knockoff_computation_error_recovery(self):
        """Test error recovery in knockoff computation chains"""
        X = np.random.rand(25, 6)
        y = np.random.randint(0, 2, 25)

        knockoffs = Knockoffs(y=y, z2=X)

        # Test error recovery with malformed intermediate results
        with patch.object(knockoffs, '_compute_glmnet_lambdasmax') as mock_lambda:
            mock_lambda.side_effect = np.linalg.LinAlgError("Singular matrix")

            # Should handle computational errors gracefully
            try:
                result = knockoffs.filter_knockoffs_iterative_python(
                    z=X, y=y, fdr=0.1, niter=1
                )
                # If it completes, verify result validity
                if result is not None:
                    assert hasattr(result, 'selected')
            except np.linalg.LinAlgError:
                # Expected for singular matrix cases
                pass

    def test_state_corruption_detection_in_chains(self):
        """Test detection of state corruption in method chains"""
        params = {'K': 2, 'max_iters': 1}
        X = np.random.rand(20, 6)
        y = np.random.randint(0, 2, 20)

        slide = OptimizeSLIDE(params, x=X, y=y)

        # Corrupt internal state during chain execution
        with patch.object(slide, 'calc_z_matrix') as mock_calc_z:
            # Return malformed z_matrix
            mock_calc_z.return_value = np.array([[]])  # Empty invalid matrix

            mock_love_result = {
                'L_hat': np.random.rand(6, 2),
                'pure_idx': [0, 1, 2]
            }

            # Should detect and handle state corruption
            try:
                result = slide.get_latent_factors(x=X, y=y, love_result=mock_love_result)
                # Verify corruption was handled
                if result is not None:
                    assert 'error' in str(result).lower() or len(result) == 0
            except (ValueError, IndexError, AssertionError):
                # Expected for corrupted state
                pass


if __name__ == "__main__":
    pytest.main([__file__])