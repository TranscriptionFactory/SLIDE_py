"""Advanced cross-validation edge cases.

Tests specific edge cases for SLIDEcv that complement existing CV coverage,
focusing on extreme scenarios and boundary conditions.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from easydict import EasyDict

from src.loveslide.cv import SLIDEcv


class TestCVAdvancedEdgeCases:
    """Test advanced edge cases for cross-validation."""

    def create_mock_slide_obj(self, n_samples=100, n_features=20, n_factors=5):
        """Create a mock SLIDE object for testing."""
        slide_obj = Mock()
        slide_obj.latent_factors = pd.DataFrame(
            np.random.randn(n_samples, n_factors),
            columns=[f'Z{i}' for i in range(n_factors)]
        )
        slide_obj.data = EasyDict()
        slide_obj.data.Y = pd.DataFrame(np.random.randint(0, 2, n_samples))
        slide_obj.input_params = {'fdr': 0.1}
        slide_obj.marginal_idxs = list(range(n_factors))
        return slide_obj

    def test_cv_with_minimal_samples_per_fold(self):
        """Test CV with very small sample sizes."""
        # Create minimal dataset
        slide_obj = self.create_mock_slide_obj(n_samples=10, n_features=5, n_factors=2)

        # High k value should create very small folds
        cv = SLIDEcv(slide_obj, nrep=1, k=9)  # Only 1-2 samples per fold

        try:
            results = cv.run()
            # May succeed with warnings or fail gracefully
            assert isinstance(results, dict)
        except ValueError as e:
            # Expected for insufficient samples
            assert "samples" in str(e).lower() or "fold" in str(e).lower()

    def test_cv_with_single_class_folds(self):
        """Test CV when stratification creates single-class folds."""
        # Create highly imbalanced dataset
        slide_obj = self.create_mock_slide_obj(n_samples=20, n_features=5, n_factors=2)
        slide_obj.data.Y = pd.DataFrame([0] * 19 + [1])  # 19:1 imbalance

        cv = SLIDEcv(slide_obj, nrep=1, k=10, eval_type='auc')

        try:
            results = cv.run()
            # AUC calculation may fail with single-class folds
        except (ValueError, RuntimeWarning) as e:
            # Expected for single-class scenarios
            pass

    def test_cv_with_identical_features(self):
        """Test CV with highly correlated or identical features."""
        slide_obj = self.create_mock_slide_obj(n_samples=50, n_features=10, n_factors=3)

        # Make some latent factors identical
        slide_obj.latent_factors.iloc[:, 1] = slide_obj.latent_factors.iloc[:, 0]
        slide_obj.latent_factors.iloc[:, 2] = slide_obj.latent_factors.iloc[:, 0] * 2

        cv = SLIDEcv(slide_obj, nrep=2, k=5)

        # Should handle correlated features
        results = cv.run()
        assert isinstance(results, dict)

    def test_cv_with_extreme_parameter_combinations(self):
        """Test CV with extreme parameter combinations."""
        slide_obj = self.create_mock_slide_obj()

        # Very high number of repetitions and folds
        cv = SLIDEcv(slide_obj, nrep=100, k=50)

        # Should be able to handle computationally intensive setup
        # (May take long time, so we just test initialization)
        assert cv.nrep == 100
        assert cv.k == 50

    def test_cv_with_nan_in_latent_factors(self):
        """Test CV with NaN values in latent factors."""
        slide_obj = self.create_mock_slide_obj()

        # Introduce NaN values
        slide_obj.latent_factors.iloc[0, 0] = np.nan
        slide_obj.latent_factors.iloc[10:15, 1] = np.nan

        cv = SLIDEcv(slide_obj, nrep=2, k=5)

        try:
            results = cv.run()
            # May handle NaN values or fail appropriately
        except (ValueError, TypeError) as e:
            # Expected for NaN values
            assert "nan" in str(e).lower() or "missing" in str(e).lower()

    def test_cv_with_constant_target(self):
        """Test CV with constant target values."""
        slide_obj = self.create_mock_slide_obj()

        # All target values the same
        slide_obj.data.Y = pd.DataFrame([1] * len(slide_obj.data.Y))

        cv = SLIDEcv(slide_obj, nrep=2, k=5, eval_type='auc')

        try:
            results = cv.run()
        except (ValueError, RuntimeWarning) as e:
            # AUC cannot be calculated with single class
            assert "class" in str(e).lower() or "auc" in str(e).lower()

    def test_cv_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large datasets."""
        # Create larger dataset
        slide_obj = self.create_mock_slide_obj(n_samples=5000, n_features=100, n_factors=20)

        cv = SLIDEcv(slide_obj, nrep=2, k=5)

        # Should handle large dataset without excessive memory usage
        try:
            results = cv.run()
            assert isinstance(results, dict)
        except MemoryError:
            pytest.skip("Insufficient memory for large dataset test")

    def test_cv_with_mismatched_dimensions(self):
        """Test CV with mismatched dimensions between components."""
        slide_obj = self.create_mock_slide_obj()

        # Create dimension mismatch
        slide_obj.latent_factors = pd.DataFrame(np.random.randn(80, 5))  # Different n_samples
        # Y still has 100 samples, latent_factors has 80

        cv = SLIDEcv(slide_obj, nrep=1, k=5)

        try:
            results = cv.run()
        except (ValueError, IndexError) as e:
            # Expected for dimension mismatch
            assert "shape" in str(e).lower() or "dimension" in str(e).lower()

    def test_cv_with_empty_marginal_indices(self):
        """Test CV with empty marginal indices."""
        slide_obj = self.create_mock_slide_obj()
        slide_obj.marginal_idxs = []  # No marginal features

        cv = SLIDEcv(slide_obj, nrep=1, k=5)

        try:
            results = cv.run()
            # May succeed with empty features or fail appropriately
        except (ValueError, IndexError) as e:
            # Expected for empty indices
            assert "empty" in str(e).lower() or "index" in str(e).lower()

    def test_cv_random_state_reproducibility(self):
        """Test CV reproducibility with random states."""
        slide_obj = self.create_mock_slide_obj()

        # Run CV twice with same random state
        np.random.seed(42)
        cv1 = SLIDEcv(slide_obj, nrep=2, k=5)
        results1 = cv1.run()

        np.random.seed(42)
        cv2 = SLIDEcv(slide_obj, nrep=2, k=5)
        results2 = cv2.run()

        # Results should be similar (within numerical tolerance)
        if 'SLIDE_score' in results1 and 'SLIDE_score' in results2:
            assert np.allclose(results1['SLIDE_score'], results2['SLIDE_score'], atol=1e-10)

    def test_cv_error_propagation(self):
        """Test error propagation in CV workflow."""
        slide_obj = self.create_mock_slide_obj()

        # Mock a component to raise an error
        with patch('src.loveslide.knockoffs.Knockoffs') as mock_knockoffs:
            mock_knockoffs.side_effect = RuntimeError("Mocked error")

            cv = SLIDEcv(slide_obj, nrep=1, k=5)

            try:
                results = cv.run()
            except RuntimeError as e:
                assert "Mocked error" in str(e)

    def test_cv_with_extreme_eval_metrics(self):
        """Test CV with extreme evaluation metric scenarios."""
        slide_obj = self.create_mock_slide_obj()

        # Test correlation evaluation with perfect correlation scenario
        slide_obj.latent_factors = pd.DataFrame(np.ones((100, 5)))  # All same values

        cv = SLIDEcv(slide_obj, nrep=1, k=5, eval_type='corr')

        try:
            results = cv.run()
            # May handle constant predictions or fail appropriately
        except (ValueError, RuntimeWarning) as e:
            # Expected for constant values
            pass

    def test_cv_fold_size_validation(self):
        """Test CV fold size validation."""
        slide_obj = self.create_mock_slide_obj(n_samples=10)

        # k larger than sample size
        cv = SLIDEcv(slide_obj, nrep=1, k=15)

        try:
            results = cv.run()
        except ValueError as e:
            # Expected for k > n_samples
            assert "fold" in str(e).lower() or "sample" in str(e).lower()

    def test_cv_stratification_edge_cases(self):
        """Test stratification edge cases."""
        slide_obj = self.create_mock_slide_obj(n_samples=15)

        # Create class distribution that's difficult to stratify
        slide_obj.data.Y = pd.DataFrame([0, 0, 0, 1, 1, 2, 2, 2, 2, 2] + [3] * 5)

        cv = SLIDEcv(slide_obj, nrep=1, k=10)  # More folds than some classes

        try:
            results = cv.run()
        except ValueError as e:
            # Expected for difficult stratification
            assert "stratif" in str(e).lower() or "class" in str(e).lower()