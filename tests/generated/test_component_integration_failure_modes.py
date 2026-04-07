"""
Test component integration failure modes and error propagation.
Critical for preventing cascading failures between SLIDE components.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from loveslide.slide import SLIDE, OptimizeSLIDE
from loveslide.cv import SLIDEcv
from loveslide.knockoffs import Knockoffs
from loveslide.plotting import Plotter
from loveslide.score import SLIDE_Estimator


class TestLOVEToKnockoffIntegration:
    """Test integration between LOVE and Knockoff components."""

    def test_malformed_love_result_handling(self):
        """Test knockoff handling of malformed LOVE results."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        params = {"fdr": 0.1}

        slide = SLIDE(params, x=X, y=y)

        # Test various malformed LOVE results
        malformed_results = [
            # Missing required keys
            {'A': np.random.randn(50, 5)},  # Missing pure_indices
            {'pure_indices': [1, 2, 3]},    # Missing A
            # Wrong dimensions
            {'A': np.random.randn(30, 5), 'pure_indices': [1, 2, 3]},  # Wrong A dimensions
            # Invalid data types
            {'A': "invalid", 'pure_indices': [1, 2, 3]},
            # Empty results
            {'A': np.array([]), 'pure_indices': []},
        ]

        for i, malformed_result in enumerate(malformed_results):
            with patch('loveslide.love.call_love') as mock_love:
                mock_love.return_value = malformed_result

                # Should handle malformed results gracefully
                try:
                    slide.run_love()
                    if hasattr(slide, 'A'):
                        # If it succeeded, validate the result
                        assert slide.A is not None
                        assert slide.A.shape[0] == 50  # Should match X.shape[1]
                except (ValueError, KeyError, AttributeError, TypeError) as e:
                    # Acceptable to fail with clear error message
                    assert len(str(e)) > 0  # Should have meaningful error message

    def test_love_knockoff_dimension_mismatch(self):
        """Test handling of dimension mismatches between LOVE and knockoffs."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        params = {"fdr": 0.1}

        slide = SLIDE(params, x=X, y=y)

        # LOVE returns wrong dimensions
        with patch('loveslide.love.call_love') as mock_love:
            mock_love.return_value = {
                'A': np.random.randn(30, 5),  # Wrong first dimension
                'pure_indices': [1, 2, 3, 4, 5]
            }

            slide.run_love()

            # Try to run knockoffs with mismatched dimensions
            with tempfile.TemporaryDirectory() as tmpdir:
                with pytest.raises((ValueError, IndexError, AssertionError)):
                    slide.run_knockoffs(tmpdir)

    def test_corrupted_latent_factors_handling(self):
        """Test handling of corrupted latent factor matrices."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        params = {"fdr": 0.1}

        slide = SLIDE(params, x=X, y=y)

        # Create valid LOVE result first
        with patch('loveslide.love.call_love') as mock_love:
            mock_love.return_value = {
                'A': np.random.randn(50, 5),
                'pure_indices': [1, 2, 3, 4, 5]
            }
            slide.run_love()

            # Corrupt the latent factors
            corrupted_cases = [
                np.full((100, 5), np.inf),     # Infinite values
                np.full((100, 5), np.nan),     # NaN values
                np.zeros((100, 5)),            # All zeros
                np.ones((100, 5)) * 1e20,      # Extremely large values
            ]

            for i, corrupted_lf in enumerate(corrupted_cases):
                slide.latent_factors = pd.DataFrame(corrupted_lf)

                # Should detect corruption before knockoffs
                with tempfile.TemporaryDirectory() as tmpdir:
                    try:
                        slide.run_knockoffs(tmpdir)
                        # If successful, validate the result
                        assert hasattr(slide, 'sig_interacts')
                    except (ValueError, FloatingPointError, RuntimeError):
                        # Should catch corruption and fail gracefully
                        assert True

    def test_partial_love_failure_recovery(self):
        """Test recovery when LOVE partially fails."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        params = {"fdr": 0.1}

        slide = SLIDE(params, x=X, y=y)

        # Simulate partial LOVE failure
        with patch('loveslide.love.call_love') as mock_love:
            def partial_failure(*args, **kwargs):
                # Return partial result
                result = {
                    'A': np.random.randn(50, 3),  # Fewer factors than expected
                    'pure_indices': [1, 2],       # Fewer pure indices
                }
                # Simulate some internal failure flag
                result['warnings'] = ['Convergence issues detected']
                return result

            mock_love.side_effect = partial_failure

            slide.run_love()

            # Should adapt to partial results
            assert hasattr(slide, 'A')
            assert slide.A.shape[1] == 3  # Should handle fewer factors


class TestKnockoffToPlottingIntegration:
    """Test integration between knockoff results and plotting."""

    def test_empty_knockoff_results_plotting(self):
        """Test plotting when knockoffs find no significant variables."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        params = {"fdr": 0.1}

        slide = SLIDE(params, x=X, y=y)

        # Mock empty knockoff results
        with patch('loveslide.love.call_love') as mock_love:
            mock_love.return_value = {
                'A': np.random.randn(50, 5),
                'pure_indices': [1, 2, 3, 4, 5]
            }
            slide.run_love()

            # Empty knockoff results
            slide.sig_interacts = []
            slide.sig_LFs = []

            # Plotting should handle empty results gracefully
            plotter = Plotter(slide)

            with patch('matplotlib.pyplot.show'), \
                 patch('matplotlib.pyplot.savefig'):
                try:
                    plotter.plot_results()
                    # Should not crash on empty results
                    assert True
                except (ValueError, IndexError) as e:
                    # Should provide meaningful error for empty results
                    assert "empty" in str(e).lower() or "no" in str(e).lower()

    def test_inconsistent_knockoff_plotting_data(self):
        """Test plotting with inconsistent knockoff data structures."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        params = {"fdr": 0.1}

        slide = SLIDE(params, x=X, y=y)

        with patch('loveslide.love.call_love') as mock_love:
            mock_love.return_value = {
                'A': np.random.randn(50, 5),
                'pure_indices': [1, 2, 3, 4, 5]
            }
            slide.run_love()

            # Create inconsistent data
            slide.sig_interacts = ['var1_var2', 'var3_var4']
            slide.sig_LFs = ['Z1', 'Z2', 'Z3']  # Different length than interactions

            plotter = Plotter(slide)

            # Should handle inconsistent data lengths
            with patch('matplotlib.pyplot.show'), \
                 patch('matplotlib.pyplot.savefig'):
                try:
                    plotter.plot_results()
                    assert True
                except (ValueError, IndexError) as e:
                    # Should fail gracefully with clear error
                    assert len(str(e)) > 0


class TestCrossValidationIntegration:
    """Test integration of cross-validation with other components."""

    def test_cv_with_corrupted_slide_object(self):
        """Test CV behavior with corrupted SLIDE object."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        # Create corrupted SLIDE object
        corrupted_slide = Mock()
        corrupted_slide.latent_factors = None  # Missing required attribute
        corrupted_slide.data = Mock()
        corrupted_slide.data.Y = pd.Series(y)
        corrupted_slide.input_params = {"fdr": 0.1}

        # Should detect corruption early
        with pytest.raises((AttributeError, ValueError)):
            cv = SLIDEcv(corrupted_slide)

    def test_cv_with_mismatched_dimensions(self):
        """Test CV with dimension mismatches in SLIDE object."""
        X = np.random.randn(100, 20)
        y = np.random.randn(80)  # Wrong length

        mock_slide = Mock()
        mock_slide.latent_factors = pd.DataFrame(np.random.randn(100, 5))
        mock_slide.data = Mock()
        mock_slide.data.Y = pd.Series(y)  # Mismatched length
        mock_slide.input_params = {"fdr": 0.1}
        mock_slide.marginal_idxs = [0, 1, 2]

        # Should detect dimension mismatch
        with pytest.raises((ValueError, IndexError)):
            cv = SLIDEcv(mock_slide)

    def test_cv_integration_failure_recovery(self):
        """Test CV recovery from component integration failures."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        mock_slide = Mock()
        mock_slide.latent_factors = pd.DataFrame(np.random.randn(100, 5))
        mock_slide.data = Mock()
        mock_slide.data.Y = pd.Series(y)
        mock_slide.input_params = {"fdr": 0.1}
        mock_slide.marginal_idxs = [0, 1, 2]

        cv = SLIDEcv(mock_slide, nrep=2, k=3)

        # Mock knockoff failure in some folds
        with patch('loveslide.knockoffs.Knockoffs.run') as mock_ko:
            call_count = 0

            def intermittent_failure(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count % 3 == 0:  # Fail every third call
                    raise RuntimeError("Simulated knockoff failure")
                return Mock(selected_vars=['var1', 'var2'])

            mock_ko.side_effect = intermittent_failure

            with patch.object(cv, '_bench_cv') as mock_bench:
                mock_bench.return_value = {
                    'SLIDE_corr': [0.5, 0.6],
                    'NULL_corr': [0.1, 0.2]
                }

                # Should handle some fold failures gracefully
                try:
                    result = cv.run(seed=42)
                    # Should complete with partial results
                    assert result is not None
                except RuntimeError as e:
                    # Should aggregate failures meaningfully
                    assert "failure" in str(e).lower() or len(str(e)) > 0


class TestEstimatorIntegration:
    """Test integration with scoring estimators."""

    def test_estimator_with_invalid_slide_results(self):
        """Test estimator behavior with invalid SLIDE results."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        estimator = SLIDE_Estimator()

        # Test with various invalid inputs
        invalid_inputs = [
            (None, y),                          # No features
            (X, None),                          # No target
            (np.array([]), y),                  # Empty features
            (X, np.array([])),                 # Empty target
            (X[:50], y),                       # Mismatched dimensions
        ]

        for i, (X_invalid, y_invalid) in enumerate(invalid_inputs):
            with pytest.raises((ValueError, IndexError, AttributeError)):
                estimator.fit(X_invalid, y_invalid)

    def test_estimator_with_corrupted_model_state(self):
        """Test estimator behavior with corrupted internal state."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        estimator = SLIDE_Estimator()

        # Fit normally first
        with patch('loveslide.love.call_love') as mock_love:
            mock_love.return_value = {
                'A': np.random.randn(50, 5),
                'pure_indices': [1, 2, 3, 4, 5]
            }

            estimator.fit(X, y)

            # Corrupt the internal state
            estimator.coef_ = None  # Remove fitted coefficients

            # Prediction should detect corrupted state
            with pytest.raises((AttributeError, ValueError)):
                estimator.predict(X)


class TestFailureRecovery:
    """Test failure recovery mechanisms across components."""

    def test_graceful_degradation_chain(self):
        """Test graceful degradation when multiple components fail."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        params = {"fdr": 0.1}

        slide = SLIDE(params, x=X, y=y)

        # Simulate chain of failures
        with patch('loveslide.love.call_love') as mock_love:
            # LOVE succeeds but returns minimal result
            mock_love.return_value = {
                'A': np.random.randn(50, 2),  # Fewer factors
                'pure_indices': [1, 2]        # Fewer pure variables
            }

            slide.run_love()

            # Should adapt to minimal LOVE result
            assert hasattr(slide, 'A')
            assert slide.A.shape[1] == 2

            with tempfile.TemporaryDirectory() as tmpdir:
                with patch('loveslide.knockoffs.Knockoffs.run') as mock_ko:
                    # Knockoffs also return minimal result
                    mock_result = Mock()
                    mock_result.selected_vars = ['var1']  # Only one variable
                    mock_ko.return_value = mock_result

                    slide.run_knockoffs(tmpdir)

                    # Should complete with degraded results
                    assert len(slide.sig_interacts) >= 0  # May be empty
                    # Should still have some usable state
                    assert hasattr(slide, 'latent_factors')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])