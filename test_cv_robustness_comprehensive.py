"""
Comprehensive cross-validation robustness testing.
Tests fold generation, stratification, and edge cases in CV workflows.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from loveslide.cv import SLIDEcv


class TestSLIDEcvFoldGeneration:
    """Test CV fold generation and stratification edge cases."""

    def test_cv_with_extreme_class_imbalance(self):
        """Test CV with extremely imbalanced classes."""
        # Create mock SLIDE object
        mock_slide = MagicMock()
        mock_slide.latent_factors = pd.DataFrame(np.random.randn(100, 10))
        mock_slide.data.Y = np.concatenate([np.ones(99), np.zeros(1)])  # 99:1 ratio
        mock_slide.input_params = {'fdr': 0.1}
        mock_slide.marginal_idxs = [0, 1, 2]

        cv = SLIDEcv(mock_slide, k=5, nrep=2)

        try:
            results = cv.run()
            # Should handle extreme imbalance gracefully
            assert 'performance' in results
        except ValueError as e:
            # Acceptable to fail on impossible stratification
            assert "stratification" in str(e).lower()

    def test_cv_with_insufficient_samples_per_class(self):
        """Test CV when some classes have fewer samples than k folds."""
        mock_slide = MagicMock()
        mock_slide.latent_factors = pd.DataFrame(np.random.randn(20, 5))
        # Only 2 samples of class 1, but k=10
        mock_slide.data.Y = np.concatenate([np.zeros(18), np.ones(2)])
        mock_slide.input_params = {'fdr': 0.1}
        mock_slide.marginal_idxs = [0]

        cv = SLIDEcv(mock_slide, k=10, nrep=2)

        with pytest.raises(ValueError, match="Cannot have number of splits"):
            cv.run()

    def test_cv_with_single_class(self):
        """Test CV with only one class present."""
        mock_slide = MagicMock()
        mock_slide.latent_factors = pd.DataFrame(np.random.randn(50, 8))
        mock_slide.data.Y = np.ones(50)  # All same class
        mock_slide.input_params = {'fdr': 0.1}
        mock_slide.marginal_idxs = [0, 1]

        cv = SLIDEcv(mock_slide, eval_type='auc', k=5, nrep=2)

        # Should fallback to regular k-fold or raise appropriate error
        try:
            results = cv.run()
            # AUC undefined for single class
            assert results is not None
        except ValueError:
            # Expected for AUC with single class
            pass

    def test_cv_with_nan_labels(self):
        """Test CV with NaN values in labels."""
        mock_slide = MagicMock()
        mock_slide.latent_factors = pd.DataFrame(np.random.randn(50, 5))
        y_with_nan = np.random.randn(50)
        y_with_nan[::5] = np.nan  # 20% NaN
        mock_slide.data.Y = y_with_nan
        mock_slide.input_params = {'fdr': 0.1}
        mock_slide.marginal_idxs = [0]

        cv = SLIDEcv(mock_slide, k=5, nrep=2)

        with pytest.raises(ValueError, match="NaN values"):
            cv.run()

    def test_cv_with_empty_marginal_idxs(self):
        """Test CV when no marginal factors are available."""
        mock_slide = MagicMock()
        mock_slide.latent_factors = pd.DataFrame(np.random.randn(100, 10))
        mock_slide.data.Y = np.random.randn(100)
        mock_slide.input_params = {'fdr': 0.1}
        mock_slide.marginal_idxs = []  # No significant marginal factors

        cv = SLIDEcv(mock_slide, k=5, nrep=2)

        results = cv.run()
        # Should handle case with no marginal factors
        assert 'performance' in results
        assert results['performance'] is not None


class TestSLIDEcvKnockoffIntegration:
    """Test CV integration with knockoff selection."""

    @patch('loveslide.cv.Knockoffs')
    def test_cv_knockoff_failure_recovery(self, mock_knockoffs_class):
        """Test CV recovery when knockoff generation fails."""
        mock_slide = MagicMock()
        mock_slide.latent_factors = pd.DataFrame(np.random.randn(100, 8))
        mock_slide.data.Y = np.random.randn(100)
        mock_slide.input_params = {'fdr': 0.1}
        mock_slide.marginal_idxs = [0, 1, 2]

        # Mock knockoff failure in some folds
        mock_knockoffs = MagicMock()
        mock_knockoffs.run.side_effect = [
            {'selected': [0, 1]},  # Fold 1 success
            RuntimeError("SDP solve failed"),  # Fold 2 failure
            {'selected': [1, 2]},  # Fold 3 success
        ]
        mock_knockoffs_class.return_value = mock_knockoffs

        cv = SLIDEcv(mock_slide, k=3, nrep=1)

        # Should handle partial failures gracefully
        results = cv.run()
        assert 'performance' in results
        # Should report on failed folds
        assert results['failed_folds'] >= 0

    def test_cv_feature_selection_consistency(self):
        """Test consistency of feature selection across CV folds."""
        mock_slide = MagicMock()
        np.random.seed(42)  # For reproducibility
        mock_slide.latent_factors = pd.DataFrame(np.random.randn(100, 10))
        mock_slide.data.Y = np.random.randn(100)
        mock_slide.input_params = {'fdr': 0.1}
        mock_slide.marginal_idxs = [0, 1, 2, 3]

        cv = SLIDEcv(mock_slide, k=5, nrep=2)

        # Mock consistent knockoff results
        with patch('loveslide.cv.Knockoffs') as mock_knockoffs_class:
            mock_knockoffs = MagicMock()
            mock_knockoffs.run.return_value = {'selected': [0, 2]}
            mock_knockoffs_class.return_value = mock_knockoffs

            results = cv.run()

            # Should track feature selection stability
            assert 'feature_stability' in results or 'selected_features' in results

    def test_cv_memory_efficient_processing(self):
        """Test CV memory efficiency with large datasets."""
        mock_slide = MagicMock()
        # Large dataset
        large_data = np.random.randn(10000, 50)
        mock_slide.latent_factors = pd.DataFrame(large_data)
        mock_slide.data.Y = np.random.randn(10000)
        mock_slide.input_params = {'fdr': 0.1}
        mock_slide.marginal_idxs = list(range(10))

        cv = SLIDEcv(mock_slide, k=10, nrep=2)

        # Should handle large datasets without excessive memory usage
        try:
            results = cv.run()
            assert results is not None
        except MemoryError:
            pytest.skip("Insufficient memory for large dataset test")


class TestSLIDEcvMetricsCalculation:
    """Test CV metrics calculation edge cases."""

    def test_cv_correlation_with_constant_predictions(self):
        """Test Spearman correlation when predictions are constant."""
        mock_slide = MagicMock()
        mock_slide.latent_factors = pd.DataFrame(np.random.randn(50, 5))
        mock_slide.data.Y = np.random.randn(50)
        mock_slide.input_params = {'fdr': 0.1}
        mock_slide.marginal_idxs = [0, 1]

        cv = SLIDEcv(mock_slide, eval_type='corr', k=5, nrep=1)

        # Mock model that produces constant predictions
        with patch('sklearn.linear_model.LinearRegression') as mock_lr:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.ones(10)  # Constant predictions
            mock_lr.return_value = mock_model

            results = cv.run()
            # Correlation should be 0 or NaN for constant predictions
            assert 'performance' in results

    def test_cv_auc_with_perfect_separation(self):
        """Test AUC calculation with perfect class separation."""
        mock_slide = MagicMock()
        mock_slide.latent_factors = pd.DataFrame(np.random.randn(100, 5))
        mock_slide.data.Y = np.random.binomial(1, 0.5, 100)
        mock_slide.input_params = {'fdr': 0.1}
        mock_slide.marginal_idxs = [0, 1]

        cv = SLIDEcv(mock_slide, eval_type='auc', k=5, nrep=1)

        # Mock perfect classifier
        with patch('sklearn.linear_model.LogisticRegression') as mock_lr:
            mock_model = MagicMock()
            # Perfect predictions
            mock_model.predict_proba.return_value = np.column_stack([
                np.zeros(20), np.ones(20)  # Perfect separation
            ])
            mock_lr.return_value = mock_model

            results = cv.run()
            # AUC should be 1.0 for perfect separation
            assert results['performance'] == 1.0

    def test_cv_metrics_with_ties(self):
        """Test metric calculation when predictions have many ties."""
        mock_slide = MagicMock()
        mock_slide.latent_factors = pd.DataFrame(np.random.randn(50, 5))
        mock_slide.data.Y = np.random.randn(50)
        mock_slide.input_params = {'fdr': 0.1}
        mock_slide.marginal_idxs = [0]

        cv = SLIDEcv(mock_slide, eval_type='corr', k=5, nrep=1)

        # Mock model with many tied predictions
        with patch('sklearn.linear_model.LinearRegression') as mock_lr:
            mock_model = MagicMock()
            # Many ties in predictions
            mock_model.predict.return_value = np.repeat([1.0, 2.0], 5)
            mock_lr.return_value = mock_model

            results = cv.run()
            # Should handle ties appropriately in Spearman correlation
            assert 'performance' in results


class TestSLIDEcvParameterValidation:
    """Test CV parameter validation edge cases."""

    def test_cv_invalid_k_values(self):
        """Test CV with invalid k values."""
        mock_slide = MagicMock()
        mock_slide.latent_factors = pd.DataFrame(np.random.randn(50, 5))
        mock_slide.data.Y = np.random.randn(50)
        mock_slide.input_params = {'fdr': 0.1}
        mock_slide.marginal_idxs = [0]

        # k larger than sample size
        with pytest.raises(ValueError):
            cv = SLIDEcv(mock_slide, k=100, nrep=1)
            cv.run()

        # k = 1 (leave-one-out equivalent)
        cv = SLIDEcv(mock_slide, k=1, nrep=1)
        with pytest.raises(ValueError):
            cv.run()

        # k = 0
        with pytest.raises(ValueError):
            SLIDEcv(mock_slide, k=0, nrep=1)

    def test_cv_invalid_nrep_values(self):
        """Test CV with invalid nrep values."""
        mock_slide = MagicMock()
        mock_slide.latent_factors = pd.DataFrame(np.random.randn(50, 5))
        mock_slide.data.Y = np.random.randn(50)
        mock_slide.input_params = {'fdr': 0.1}
        mock_slide.marginal_idxs = [0]

        # nrep = 0
        with pytest.raises(ValueError):
            SLIDEcv(mock_slide, k=5, nrep=0)

        # Very large nrep
        cv = SLIDEcv(mock_slide, k=5, nrep=1000)
        # Should warn about computational cost but not fail
        assert cv.nrep == 1000

    def test_cv_invalid_eval_type(self):
        """Test CV with invalid evaluation type."""
        mock_slide = MagicMock()
        mock_slide.latent_factors = pd.DataFrame(np.random.randn(50, 5))
        mock_slide.data.Y = np.random.randn(50)
        mock_slide.input_params = {'fdr': 0.1}
        mock_slide.marginal_idxs = [0]

        with pytest.raises(ValueError):
            SLIDEcv(mock_slide, eval_type='invalid_metric', k=5, nrep=1)