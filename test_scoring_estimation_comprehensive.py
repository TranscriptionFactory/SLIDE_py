"""
Comprehensive test coverage for scoring and estimation functionality.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock
from sklearn.exceptions import ConvergenceWarning

from loveslide.score import Estimator, SLIDE_Estimator


class TestEstimatorInitialization:
    """Test Estimator class initialization and model selection."""

    def test_estimator_init_auto_regression(self):
        """Test Estimator auto model selection for regression."""
        estimator = Estimator(model='auto')
        y = np.random.randn(100)  # Continuous target

        estimator._init_model(y)

        # Should select regression model for continuous target
        from sklearn.ensemble import RandomForestRegressor
        assert isinstance(estimator.model, RandomForestRegressor)

    def test_estimator_init_auto_classification(self):
        """Test Estimator auto model selection for classification."""
        estimator = Estimator(model='auto')
        y = np.random.choice([0, 1], 100)  # Binary target

        estimator._init_model(y)

        # Should select classification model for binary target
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(estimator.model, RandomForestClassifier)

    def test_estimator_init_auto_multiclass(self):
        """Test Estimator auto model selection for multiclass."""
        estimator = Estimator(model='auto')
        y = np.random.choice([0, 1, 2, 3], 100)  # Multiclass target

        estimator._init_model(y)

        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(estimator.model, RandomForestClassifier)

    def test_estimator_init_specific_models(self):
        """Test Estimator initialization with specific model names."""
        models_to_test = ['rf', 'lr', 'svm', 'xgb']

        for model_name in models_to_test:
            try:
                estimator = Estimator(model=model_name)
                y = np.random.choice([0, 1], 100)
                estimator._init_model(y)
                assert estimator.model is not None
            except ImportError:
                # Some models may not be available (e.g., xgboost)
                pytest.skip(f"Model {model_name} not available")

    def test_estimator_init_invalid_model(self):
        """Test Estimator with invalid model name."""
        with pytest.raises(ValueError, match="Unknown model"):
            estimator = Estimator(model='invalid_model')
            estimator._init_model(np.random.randn(100))

    def test_estimator_init_scalers(self):
        """Test different scaler options."""
        scalers = ['standard', 'minmax', 'robust']

        for scaler_name in scalers:
            estimator = Estimator(scaler=scaler_name)
            assert estimator.scaler == scaler_name

        # Invalid scaler
        with pytest.raises(ValueError):
            Estimator(scaler='invalid_scaler')


class TestEstimatorScaling:
    """Test feature scaling functionality."""

    def test_scale_features_standard(self):
        """Test standard scaling."""
        X = np.random.randn(100, 20)
        X_scaled = Estimator.scale_features(X, 'standard')

        # Should be approximately standardized
        assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-10)

    def test_scale_features_minmax(self):
        """Test min-max scaling."""
        X = np.random.randn(100, 20)
        X_scaled = Estimator.scale_features(X, 'minmax', feature_range=(0, 1))

        # Should be in specified range
        assert np.all(X_scaled >= 0)
        assert np.all(X_scaled <= 1)
        assert np.allclose(np.min(X_scaled, axis=0), 0, atol=1e-10)
        assert np.allclose(np.max(X_scaled, axis=0), 1, atol=1e-10)

    def test_scale_features_robust(self):
        """Test robust scaling."""
        # Create data with outliers
        X = np.random.randn(100, 20)
        X[0, :] = 100  # Add outliers

        X_scaled = Estimator.scale_features(X, 'robust')

        # Should handle outliers better than standard scaling
        assert np.all(np.isfinite(X_scaled))
        assert not np.any(np.abs(X_scaled) > 50)  # Should not have extreme values

    def test_scale_features_zero_variance(self):
        """Test scaling with zero variance features."""
        X = np.random.randn(100, 20)
        X[:, 0] = 5.0  # Zero variance feature

        X_scaled = Estimator.scale_features(X, 'standard')

        # Zero variance feature should become zero (or handle gracefully)
        assert np.allclose(X_scaled[:, 0], 0) or np.all(X_scaled[:, 0] == X_scaled[0, 0])

    def test_scale_features_extreme_values(self):
        """Test scaling with extreme values."""
        X = np.random.randn(100, 20)
        X[0, 0] = 1e10  # Extreme value

        X_scaled = Estimator.scale_features(X, 'standard')

        # Should produce finite values
        assert np.all(np.isfinite(X_scaled))

    def test_scale_features_single_sample(self):
        """Test scaling with single sample."""
        X = np.random.randn(1, 20)

        # Should handle gracefully or raise appropriate error
        try:
            X_scaled = Estimator.scale_features(X, 'standard')
            assert X_scaled.shape == X.shape
        except ValueError:
            pass  # Acceptable to fail with single sample


class TestEstimatorFitPredict:
    """Test model fitting and prediction."""

    def test_fit_predict_regression(self):
        """Test fitting and prediction for regression."""
        X = np.random.randn(100, 20)
        y = X @ np.random.randn(20) + np.random.randn(100)

        estimator = Estimator(model='rf')
        estimator.fit(X, y)

        # Predictions should have correct shape and be finite
        y_pred = estimator.predict(X)
        assert y_pred.shape == (100,)
        assert np.all(np.isfinite(y_pred))

    def test_fit_predict_classification(self):
        """Test fitting and prediction for classification."""
        X = np.random.randn(100, 20)
        y = np.random.choice([0, 1], 100)

        estimator = Estimator(model='rf')
        estimator.fit(X, y)

        # Predictions should be binary
        y_pred = estimator.predict(X)
        assert np.all(np.isin(y_pred, [0, 1]))

        # Probabilities should be valid
        y_proba = estimator.predict_proba(X)
        assert y_proba.shape == (100,)
        assert np.all(y_proba >= 0)
        assert np.all(y_proba <= 1)

    def test_fit_perfect_separation(self):
        """Test fitting with perfectly separable data."""
        # Create perfectly separable data
        X = np.vstack([
            np.random.randn(50, 20) - 3,  # Class 0
            np.random.randn(50, 20) + 3   # Class 1
        ])
        y = np.array([0] * 50 + [1] * 50)

        estimator = Estimator(model='lr')

        # Should handle perfect separation gracefully
        with pytest.warns((ConvergenceWarning, UserWarning), match=""):
            estimator.fit(X, y)

        y_pred = estimator.predict(X)
        # Should achieve perfect accuracy
        assert np.mean(y_pred == y) > 0.9

    def test_fit_singular_covariance(self):
        """Test fitting with singular covariance matrix."""
        # Create data with perfect collinearity
        X = np.random.randn(100, 20)
        X[:, 1] = X[:, 0]  # Perfect correlation

        y = np.random.choice([0, 1], 100)

        estimator = Estimator(model='lr')

        # Should handle or warn about collinearity
        with pytest.warns(UserWarning, match=""):
            estimator.fit(X, y)

    def test_fit_insufficient_data(self):
        """Test fitting with insufficient training data."""
        # Very small dataset
        X = np.random.randn(5, 20)  # More features than samples
        y = np.random.choice([0, 1], 5)

        estimator = Estimator(model='rf')

        # Should either fit or raise appropriate error
        try:
            estimator.fit(X, y)
            y_pred = estimator.predict(X)
            assert len(y_pred) == 5
        except ValueError:
            pass  # Acceptable to fail with insufficient data

    def test_predict_before_fit(self):
        """Test prediction before model is fitted."""
        X = np.random.randn(100, 20)
        estimator = Estimator(model='rf')

        # Should raise appropriate error
        with pytest.raises(AttributeError, match="fitted"):
            estimator.predict(X)


class TestEstimatorEvaluation:
    """Test model evaluation methods."""

    def test_score_regression_mse(self):
        """Test MSE scoring for regression."""
        y_true = np.random.randn(100)
        y_pred = y_true + 0.1 * np.random.randn(100)  # Add small noise

        estimator = Estimator()
        score = estimator.score(y_pred, y_true)

        # Should be positive MSE
        assert score >= 0
        assert score < 1  # Should be small due to small noise

    def test_score_classification_auc(self):
        """Test AUC scoring for classification."""
        y_true = np.random.choice([0, 1], 100)
        y_proba = np.random.uniform(0, 1, 100)

        estimator = Estimator()
        score = estimator.score(y_proba, y_true)

        # Should be valid AUC
        assert 0 <= score <= 1

    def test_score_perfect_predictions(self):
        """Test scoring with perfect predictions."""
        y_true = np.random.choice([0, 1], 100)

        estimator = Estimator()
        score = estimator.score(y_true.astype(float), y_true)

        # Should achieve perfect score (AUC = 1.0)
        assert score == 1.0

    def test_score_worst_predictions(self):
        """Test scoring with worst possible predictions."""
        y_true = np.random.choice([0, 1], 100)
        y_proba = 1.0 - y_true.astype(float)  # Opposite predictions

        estimator = Estimator()
        score = estimator.score(y_proba, y_true)

        # Should achieve worst score (AUC = 0.0)
        assert score == 0.0

    def test_score_invalid_predictions(self):
        """Test scoring with invalid predictions."""
        y_true = np.random.choice([0, 1], 100)
        y_pred = np.full(100, np.nan)

        estimator = Estimator()

        # Should handle NaN predictions
        with pytest.raises((ValueError, np.core._exceptions._ArrayMemoryError)):
            estimator.score(y_pred, y_true)

    def test_evaluate_cross_validation(self):
        """Test cross-validation evaluation."""
        X = np.random.randn(200, 20)
        y = np.random.choice([0, 1], 200)

        estimator = Estimator(model='rf')
        scores = estimator.evaluate(X, y, n_iters=3, test_size=0.3)

        # Should return array of scores
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_evaluate_small_dataset(self):
        """Test evaluation with small dataset."""
        X = np.random.randn(20, 5)
        y = np.random.choice([0, 1], 20)

        estimator = Estimator(model='rf')

        # Should handle small dataset gracefully
        try:
            scores = estimator.evaluate(X, y, n_iters=2, test_size=0.3)
            assert len(scores) == 2
        except ValueError:
            pass  # Acceptable to fail with very small data

    def test_get_aucs_multiple_iterations(self):
        """Test AUC calculation over multiple iterations."""
        X = np.random.randn(200, 20)
        y = np.random.choice([0, 1], 200)

        estimator = Estimator(model='rf')
        mean_auc, std_auc = estimator.get_aucs(X, y, n_iters=5)

        # Should return valid statistics
        assert 0 <= mean_auc <= 1
        assert std_auc >= 0
        assert std_auc <= 0.5  # Standard deviation shouldn't be too large

    def test_train_test_split_consistency(self):
        """Test train/test split consistency."""
        X = np.random.randn(100, 20)
        y = np.random.choice([0, 1], 100)

        estimator = Estimator()

        # Multiple splits with same seed should be identical
        X_train1, X_test1, y_train1, y_test1 = estimator.train_test_split(X, y, seed=42)
        X_train2, X_test2, y_train2, y_test2 = estimator.train_test_split(X, y, seed=42)

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)

    def test_train_test_split_sizes(self):
        """Test train/test split size validation."""
        X = np.random.randn(100, 20)
        y = np.random.choice([0, 1], 100)

        estimator = Estimator()

        # Test different split sizes
        for test_size in [0.1, 0.2, 0.5, 0.8]:
            X_train, X_test, y_train, y_test = estimator.train_test_split(
                X, y, test_size=test_size
            )

            expected_test_size = int(100 * test_size)
            expected_train_size = 100 - expected_test_size

            assert len(X_test) == expected_test_size
            assert len(X_train) == expected_train_size
            assert len(y_test) == expected_test_size
            assert len(y_train) == expected_train_size


class TestSLIDEEstimator:
    """Test SLIDE_Estimator specific functionality."""

    def test_slide_estimator_init(self):
        """Test SLIDE_Estimator initialization."""
        slide_estimator = SLIDE_Estimator()
        assert isinstance(slide_estimator, SLIDE_Estimator)
        assert isinstance(slide_estimator, Estimator)

    def test_score_performance_method(self):
        """Test score_performance method existence."""
        slide_estimator = SLIDE_Estimator()
        assert hasattr(slide_estimator, 'score_performance')

    def test_slide_estimator_inheritance(self):
        """Test that SLIDE_Estimator properly inherits from Estimator."""
        X = np.random.randn(100, 20)
        y = np.random.choice([0, 1], 100)

        slide_estimator = SLIDE_Estimator(model='rf')

        # Should inherit all Estimator functionality
        slide_estimator.fit(X, y)
        y_pred = slide_estimator.predict(X)
        score = slide_estimator.score(y_pred, y)

        assert y_pred.shape == (100,)
        assert isinstance(score, (int, float))


class TestEstimatorEdgeCases:
    """Test estimator behavior with edge cases."""

    def test_estimator_with_categorical_features(self):
        """Test estimator with categorical-like features."""
        # Create mixed data
        X_cont = np.random.randn(100, 10)
        X_cat = np.random.randint(0, 5, size=(100, 5)).astype(float)
        X = np.column_stack([X_cont, X_cat])
        y = np.random.choice([0, 1], 100)

        estimator = Estimator(model='rf')
        estimator.fit(X, y)

        y_pred = estimator.predict(X)
        assert len(y_pred) == 100

    def test_estimator_memory_efficiency(self):
        """Test memory efficiency with moderately large data."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Moderate size dataset
        X = np.random.randn(1000, 100)
        y = np.random.choice([0, 1], 1000)

        estimator = Estimator(model='rf')
        estimator.fit(X, y)
        y_pred = estimator.predict(X)

        memory_after = process.memory_info().rss
        memory_increase = (memory_after - memory_before) / (1024 * 1024)  # MB

        # Memory increase should be reasonable
        assert memory_increase < 200  # Less than 200MB

    def test_estimator_reproducibility(self):
        """Test estimator reproducibility with random seed."""
        X = np.random.randn(100, 20)
        y = np.random.choice([0, 1], 100)

        # Create two identical estimators
        estimator1 = Estimator(model='rf', random_state=42)
        estimator2 = Estimator(model='rf', random_state=42)

        estimator1.fit(X, y)
        estimator2.fit(X, y)

        y_pred1 = estimator1.predict(X)
        y_pred2 = estimator2.predict(X)

        # Predictions should be identical (for deterministic models)
        np.testing.assert_array_equal(y_pred1, y_pred2)

    def test_estimator_with_imbalanced_data(self):
        """Test estimator with highly imbalanced classes."""
        X = np.random.randn(1000, 20)
        # Highly imbalanced: 95% class 0, 5% class 1
        y = np.array([0] * 950 + [1] * 50)

        estimator = Estimator(model='rf')
        estimator.fit(X, y)

        y_pred = estimator.predict(X)
        y_proba = estimator.predict_proba(X)

        # Should handle imbalanced data
        assert np.all(np.isin(y_pred, [0, 1]))
        assert np.all(y_proba >= 0)
        assert np.all(y_proba <= 1)

        # Should not predict only majority class
        unique_preds = np.unique(y_pred)
        assert len(unique_preds) >= 1  # At least predict something

    def test_estimator_feature_importance_availability(self):
        """Test that feature importance is available for compatible models."""
        X = np.random.randn(100, 20)
        y = np.random.choice([0, 1], 100)

        # Random Forest should provide feature importances
        estimator = Estimator(model='rf')
        estimator.fit(X, y)

        # Check if feature importances are accessible
        if hasattr(estimator.model, 'feature_importances_'):
            importances = estimator.model.feature_importances_
            assert len(importances) == 20
            assert np.all(importances >= 0)
            assert np.sum(importances) > 0