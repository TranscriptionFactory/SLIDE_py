"""
Test skeleton for private method integration scenarios.

Focus on testing private methods in realistic integrated workflows
to ensure they behave correctly when called from public APIs.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, call
from typing import Dict, List, Any

from loveslide import SLIDE, SLIDEcv, Knockoffs, SLIDE_Estimator
from loveslide.score import Estimator
from loveslide.cv import SLIDEcv


class TestSLIDEPrivateMethods:
    """Test private methods in SLIDE class integration."""

    def test_find_interaction_LFs_batch_integration(self):
        """Test _find_interaction_LFs_batch in realistic workflow."""
        # Create realistic data
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.choice([0, 1], 100)

        slide = SLIDE({'fdr': 0.1}, x=X, y=y)

        # Mock LOVE result to test batch processing
        mock_love_result = {
            'LFs': np.random.randn(100, 5),  # 5 latent factors
            'pure_indices': [0, 1, 2]
        }

        # Test batch processing with different batch sizes
        with patch.object(slide, '_batch_knockoff_analysis') as mock_batch:
            mock_batch.return_value = {'interactions': [], 'scores': []}

            # Should handle batch processing correctly
            result = slide._find_interaction_LFs_batch(
                love_result=mock_love_result,
                batch_size=10
            )

            # Verify batch method was called appropriate number of times
            expected_calls = np.ceil(50 / 10)  # 50 features, batch size 10
            assert mock_batch.call_count <= expected_calls + 1

    def test_find_interaction_LFs_batch_empty_pure_indices(self):
        """Test batch processing when no pure indices found."""
        X = np.random.randn(50, 30)
        y = np.random.choice([0, 1], 50)

        slide = SLIDE({'fdr': 0.1}, x=X, y=y)

        mock_love_result = {
            'LFs': np.random.randn(50, 3),
            'pure_indices': []  # No pure indices
        }

        # Should handle gracefully
        result = slide._find_interaction_LFs_batch(
            love_result=mock_love_result,
            batch_size=10
        )

        assert isinstance(result, dict)
        assert 'interactions' in result


class TestSLIDEcvPrivateMethods:
    """Test private methods in SLIDEcv class integration."""

    @pytest.fixture
    def sample_cv_data(self):
        """Generate sample data for CV testing."""
        np.random.seed(123)
        X = np.random.randn(80, 20)
        y = np.random.choice([0, 1], 80)
        return X, y

    def test_run_slide_fold_integration(self, sample_cv_data):
        """Test _run_slide_fold in realistic CV scenario."""
        X, y = sample_cv_data

        cv = SLIDEcv(
            slide_params={'fdr': 0.2},
            cv_params={'n_folds': 3, 'n_rep': 2}
        )

        # Create realistic fold indices
        train_idx = np.arange(60)  # First 60 samples
        test_idx = np.arange(60, 80)  # Last 20 samples

        # Mock SLIDE fit to control behavior
        with patch.object(SLIDE, 'run') as mock_run:
            mock_run.return_value = {
                'selected_features': [1, 5, 10],
                'scores': np.array([0.1, 0.3, 0.2])
            }

            result = cv._run_slide_fold(
                X=X, y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                fold_id=1
            )

            # Verify result structure
            assert isinstance(result, dict)
            assert 'fold_id' in result
            assert 'selected_features' in result
            assert mock_run.called

    def test_find_interactions_fold_with_edge_cases(self, sample_cv_data):
        """Test _find_interactions_fold with edge cases."""
        X, y = sample_cv_data

        cv = SLIDEcv(
            slide_params={'fdr': 0.1, 'do_interacts': True},
            cv_params={'n_folds': 2}
        )

        train_idx = np.arange(40)
        test_idx = np.arange(40, 80)

        # Test with empty selected features
        with patch.object(cv, '_run_slide_fold') as mock_fold:
            mock_fold.return_value = {
                'selected_features': [],  # No features selected
                'scores': np.array([])
            }

            result = cv._find_interactions_fold(
                X=X, y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                love_result={'LFs': np.random.randn(80, 3)},
                fold_id=1
            )

            # Should handle empty features gracefully
            assert 'interactions' in result
            assert isinstance(result['interactions'], list)

    def test_build_prediction_features_integration(self, sample_cv_data):
        """Test _build_prediction_features in realistic scenario."""
        X, y = sample_cv_data

        cv = SLIDEcv(
            slide_params={'fdr': 0.1},
            cv_params={'n_folds': 2}
        )

        # Mock selected features and interactions
        selected_features = [1, 5, 10, 15]
        interactions = [
            {'feature1': 1, 'feature2': 5, 'score': 0.8},
            {'feature1': 10, 'feature2': 15, 'score': 0.6}
        ]

        X_train = X[:40]
        X_test = X[40:]

        result_train, result_test = cv._build_prediction_features(
            X_train=X_train,
            X_test=X_test,
            selected_features=selected_features,
            interactions=interactions
        )

        # Should include original features + interaction terms
        expected_features = len(selected_features) + len(interactions)
        assert result_train.shape[1] == expected_features
        assert result_test.shape[1] == expected_features

        # Interaction terms should be computed correctly
        # Check that interaction columns contain products
        interaction_col1 = result_train[:, len(selected_features)]
        expected_interaction1 = X_train[:, 1] * X_train[:, 5]  # features 1 and 5
        assert np.allclose(interaction_col1, expected_interaction1)

    def test_compute_metric_with_different_metrics(self, sample_cv_data):
        """Test _compute_metric with different metric types."""
        _, y = sample_cv_data

        cv = SLIDEcv(
            slide_params={'fdr': 0.1},
            cv_params={'metric': 'accuracy'}
        )

        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])

        # Test accuracy
        cv.cv_params['metric'] = 'accuracy'
        acc = cv._compute_metric(y_true, y_pred)
        assert 0 <= acc <= 1

        # Test AUC (requires probabilities)
        cv.cv_params['metric'] = 'auc'
        y_pred_prob = np.array([0.2, 0.8, 0.6, 0.9, 0.3])
        auc = cv._compute_metric(y_true, y_pred_prob)
        assert 0 <= auc <= 1

        # Test F1 score
        cv.cv_params['metric'] = 'f1'
        f1 = cv._compute_metric(y_true, y_pred)
        assert 0 <= f1 <= 1

    def test_standardize_fold_consistency(self, sample_cv_data):
        """Test _standardize_fold maintains consistency."""
        X, y = sample_cv_data

        cv = SLIDEcv(
            slide_params={'fdr': 0.1},
            cv_params={'standardize': True}
        )

        train_idx = np.arange(50)
        test_idx = np.arange(50, 80)

        X_train_std, X_test_std = cv._standardize_fold(
            X=X,
            train_idx=train_idx,
            test_idx=test_idx
        )

        # Training data should be standardized
        assert np.allclose(X_train_std.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_train_std.std(axis=0), 1, atol=1e-10)

        # Test data should be standardized using training statistics
        assert X_test_std.shape == (30, 20)

    def test_folds_valid_validation(self):
        """Test _folds_valid with various fold configurations."""
        y = np.array([0, 0, 1, 1, 1, 0])

        # Valid folds
        valid_folds = [(np.array([0, 1, 2]), np.array([3, 4, 5]))]
        assert SLIDEcv._folds_valid(y, valid_folds) == True

        # Invalid folds - overlapping indices
        invalid_folds = [(np.array([0, 1, 2]), np.array([2, 3, 4]))]
        assert SLIDEcv._folds_valid(y, invalid_folds) == False

        # Invalid folds - missing indices
        incomplete_folds = [(np.array([0, 1]), np.array([2, 3]))]
        assert SLIDEcv._folds_valid(y, incomplete_folds) == False

        # Invalid folds - out of range indices
        out_of_range_folds = [(np.array([0, 1]), np.array([6, 7]))]
        assert SLIDEcv._folds_valid(y, out_of_range_folds) == False


class TestEstimatorPrivateMethods:
    """Test private methods in Estimator class integration."""

    def test_init_model_auto_selection(self):
        """Test _init_model automatic model selection."""
        # Binary classification
        y_binary = np.array([0, 1, 0, 1, 1])
        estimator = Estimator(model='auto')
        estimator._init_model(y_binary)

        assert hasattr(estimator, 'model')
        # Should select appropriate binary classifier

        # Multi-class classification
        y_multiclass = np.array([0, 1, 2, 0, 1, 2])
        estimator_multi = Estimator(model='auto')
        estimator_multi._init_model(y_multiclass)

        assert hasattr(estimator_multi, 'model')

        # Regression (continuous targets)
        y_continuous = np.random.randn(20)
        estimator_reg = Estimator(model='auto')
        estimator_reg._init_model(y_continuous)

        assert hasattr(estimator_reg, 'model')

    def test_init_model_with_specific_models(self):
        """Test _init_model with specific model types."""
        y = np.array([0, 1, 0, 1])

        # Test specific models
        models_to_test = ['lr', 'rf', 'svm']

        for model_name in models_to_test:
            estimator = Estimator(model=model_name)
            estimator._init_model(y)

            assert hasattr(estimator, 'model')
            # Model should be initialized based on name

    def test_init_model_invalid_model(self):
        """Test _init_model with invalid model specification."""
        y = np.array([0, 1, 0, 1])

        estimator = Estimator(model='invalid_model')

        with pytest.raises((ValueError, AttributeError)):
            estimator._init_model(y)


class TestKnockoffPrivateMethods:
    """Test private methods in Knockoffs class integration."""

    def test_single_knockoff_iteration_python_integration(self):
        """Test _single_knockoff_iteration_python in realistic scenario."""
        from loveslide.knockoffs import _single_knockoff_iteration_python

        np.random.seed(42)
        n, p = 80, 20

        # Create realistic data
        z = np.random.randn(n, p)
        y = np.random.choice([0, 1], n)

        # Generate realistic parameters
        mu = z.mean(axis=0)
        Sigma = np.cov(z.T)
        diag_s = np.random.uniform(0.1, 0.9, p)

        result = _single_knockoff_iteration_python(
            z=z, y=y, fdr=0.1, method='gaussian',
            shrink=True, offset=1, statistic='lasso',
            mu=mu, Sigma=Sigma, diag_s=diag_s
        )

        # Should return meaningful results
        assert 'selected' in result
        assert 'W' in result
        assert isinstance(result['selected'], list)
        assert len(result['W']) == p

    def test_compute_glmnet_lambdasmax_integration(self):
        """Test _compute_glmnet_lambdasmax in realistic scenario."""
        from loveslide.knockoffs import Knockoffs

        np.random.seed(42)
        X = np.random.randn(50, 10)
        Xk = np.random.randn(50, 10)  # Knockoff features
        y = np.random.choice([0, 1], 50)

        # Create Knockoffs instance to access private method
        knockoffs = Knockoffs(y=y, z2=np.hstack([X, Xk]))

        lambda_max = knockoffs._compute_glmnet_lambdasmax(
            X=X, Xk=Xk, y=y,
            nlambda=100, eps=0.001
        )

        assert isinstance(lambda_max, (float, np.floating))
        assert lambda_max > 0

    def test_knockoff_threshold_with_edge_cases(self):
        """Test _knockoff_threshold with various edge cases."""
        from loveslide.knockoffs import Knockoffs

        # Test with mostly zero W statistics
        W_mostly_zero = np.array([0.1, 0.0, 0.05, 0.0, 0.2])
        threshold = Knockoffs._knockoff_threshold(W_mostly_zero, fdr=0.1, offset=1)

        assert threshold >= 0

        # Test with negative W statistics
        W_negative = np.array([-0.5, 0.3, -0.2, 0.8, 0.1])
        threshold_neg = Knockoffs._knockoff_threshold(W_negative, fdr=0.1, offset=1)

        assert threshold_neg >= 0

        # Test with very small FDR
        W_normal = np.random.randn(10)
        threshold_small_fdr = Knockoffs._knockoff_threshold(W_normal, fdr=0.01, offset=1)

        assert threshold_small_fdr >= 0

        # Test with offset=0 (modified procedure)
        threshold_offset0 = Knockoffs._knockoff_threshold(W_normal, fdr=0.1, offset=0)

        assert threshold_offset0 >= 0


class TestPrivateMethodInteractions:
    """Test interactions between private methods across classes."""

    def test_slide_cv_private_method_chain(self):
        """Test chaining of private methods in SLIDE-CV workflow."""
        np.random.seed(42)
        X = np.random.randn(60, 15)
        y = np.random.choice([0, 1], 60)

        # Initialize CV
        cv = SLIDEcv(
            slide_params={'fdr': 0.2, 'do_interacts': True},
            cv_params={'n_folds': 2, 'n_rep': 1}
        )

        # Mock the complex chain of private method calls
        with patch.object(cv, '_run_slide_fold') as mock_slide_fold, \
             patch.object(cv, '_find_interactions_fold') as mock_interactions, \
             patch.object(cv, '_build_prediction_features') as mock_build_features:

            mock_slide_fold.return_value = {
                'selected_features': [1, 5, 8],
                'fold_id': 1
            }

            mock_interactions.return_value = {
                'interactions': [{'feature1': 1, 'feature2': 5, 'score': 0.7}]
            }

            mock_build_features.return_value = (
                np.random.randn(30, 4),  # train features
                np.random.randn(30, 4)   # test features
            )

            # Run a single fold to test the chain
            folds = [(np.arange(30), np.arange(30, 60))]

            # This should call the chain of private methods
            results = cv._bench_cv(X, y, folds, rep_id=1)

            # Verify the chain was called in correct order
            assert mock_slide_fold.called
            assert mock_interactions.called
            assert mock_build_features.called

    def test_private_method_error_propagation(self):
        """Test error propagation through private method chains."""
        X = np.random.randn(40, 10)
        y = np.random.choice([0, 1], 40)

        cv = SLIDEcv(
            slide_params={'fdr': 0.1},
            cv_params={'n_folds': 2}
        )

        # Mock a private method to raise an exception
        with patch.object(cv, '_run_slide_fold', side_effect=RuntimeError("Mock error")):

            folds = [(np.arange(20), np.arange(20, 40))]

            with pytest.raises(RuntimeError, match="Mock error"):
                cv._bench_cv(X, y, folds, rep_id=1)


if __name__ == "__main__":
    pytest.main([__file__])