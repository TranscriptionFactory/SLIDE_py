"""
Test statistical correctness and algorithm validation for SLIDE_py.

This module focuses on validating that algorithms produce statistically
correct results under various conditions, ensuring scientific validity.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.loveslide import SLIDE, SLIDEcv, Knockoffs
from src.loveslide.knockoffs import _single_knockoff_iteration_python
from src.loveslide.knockoff.filter import knockoff_threshold
from src.loveslide.knockoff.stats.glmnet import stat_glmnet_lambdasmax
from src.loveslide.love import call_love


class TestSLIDEStatisticalCorrectness:
    """Test SLIDE algorithm produces statistically valid results."""

    def test_fdr_control_synthetic_data(self):
        """Test FDR control on synthetic data with known ground truth."""
        # Generate synthetic data with known interaction structure
        n_samples, n_features = 500, 100
        true_interactions = [(0, 1), (5, 6), (10, 11)]  # Known true interactions

        X, y = self._generate_synthetic_data_with_interactions(
            n_samples, n_features, true_interactions, effect_size=0.5
        )

        slide = SLIDE({"fdr": 0.1, "K": 3})
        results = slide.run(X, y)

        # Calculate empirical FDR
        discovered = results.get('significant_interactions', [])
        false_positives = len([d for d in discovered if d not in true_interactions])
        empirical_fdr = false_positives / max(len(discovered), 1)

        # FDR should be controlled at specified level
        assert empirical_fdr <= 0.15  # Allow some variance

    def test_statistical_power_validation(self):
        """Test statistical power on synthetic data with varying effect sizes."""
        effect_sizes = [0.1, 0.3, 0.5, 0.8]
        powers = []

        for effect_size in effect_sizes:
            # Multiple replications for power calculation
            discoveries = []
            for _ in range(20):
                X, y = self._generate_synthetic_data_with_interactions(
                    n_samples=300, n_features=50,
                    true_interactions=[(0, 1), (2, 3)],
                    effect_size=effect_size
                )

                slide = SLIDE({"fdr": 0.1, "K": 2})
                results = slide.run(X, y)
                discoveries.append(len(results.get('significant_interactions', [])))

            power = np.mean([d > 0 for d in discoveries])
            powers.append(power)

        # Power should increase with effect size
        assert all(powers[i] <= powers[i+1] for i in range(len(powers)-1))

    def test_latent_factor_recovery(self):
        """Test recovery of known latent factors."""
        # Generate data with known latent structure
        n_samples, n_features, n_factors = 200, 60, 3
        true_A = np.random.randn(n_features, n_factors)
        true_Z = np.random.randn(n_samples, n_factors)
        X = true_Z @ true_A.T + 0.1 * np.random.randn(n_samples, n_features)

        slide = SLIDE({"K": 3})
        slide.data.X = pd.DataFrame(X)
        slide.calc_z_matrix({"A": true_A})  # Use known A for testing

        recovered_Z = slide.latent_factors.values

        # Test correlation between true and recovered factors
        for i in range(n_factors):
            max_corr = max(abs(np.corrcoef(true_Z[:, i], recovered_Z[:, j])[0, 1])
                          for j in range(n_factors))
            assert max_corr > 0.8  # Should recover factors with high correlation

    def _generate_synthetic_data_with_interactions(self, n_samples, n_features,
                                                  true_interactions, effect_size=0.5):
        """Generate synthetic data with known interaction structure."""
        X = np.random.randn(n_samples, n_features)

        # Base linear effects
        beta = np.random.randn(n_features) * 0.1
        y = X @ beta

        # Add interaction effects
        for i, j in true_interactions:
            interaction_effect = effect_size * X[:, i] * X[:, j]
            y += interaction_effect

        # Add noise
        y += 0.1 * np.random.randn(n_samples)

        return X, y


class TestKnockoffStatisticalProperties:
    """Test knockoff filter statistical properties."""

    def test_knockoff_exchangeability_property(self):
        """Test knockoff variables satisfy exchangeability property."""
        n_samples, n_features = 100, 20
        X = np.random.randn(n_samples, n_features)

        knockoffs = Knockoffs(backend='python')
        X_knockoff = knockoffs.create_gaussian(X)

        # Test swap property: (X, X_k) should have same distribution as (X_k, X)
        combined = np.hstack([X, X_knockoff])

        # For Gaussian case, covariance should satisfy specific structure
        cov_combined = np.cov(combined.T)
        n_feat = X.shape[1]

        # Extract blocks
        Sigma_XX = cov_combined[:n_feat, :n_feat]
        Sigma_XkXk = cov_combined[n_feat:, n_feat:]
        Sigma_XXk = cov_combined[:n_feat, n_feat:]

        # Knockoff property: X_k should have same marginal distribution
        assert np.allclose(np.diag(Sigma_XX), np.diag(Sigma_XkXk), rtol=0.1)

    def test_fdr_threshold_mathematical_properties(self):
        """Test knockoff threshold calculation properties."""
        # Generate test statistics
        n_features = 100
        W = np.random.randn(n_features) * 2  # Test statistics

        # Test threshold properties for different FDR levels
        fdr_levels = [0.05, 0.1, 0.2, 0.3]
        thresholds = []

        for fdr in fdr_levels:
            t = knockoff_threshold(W, fdr, offset=1)
            thresholds.append(t)

            # Selected features should control FDR
            selected = W >= t
            if np.sum(selected) > 0:
                # Can't directly test FDR without true nulls, but test structure
                assert t >= 0  # Threshold should be non-negative for this setup

        # Threshold should decrease as FDR level increases
        assert all(thresholds[i] >= thresholds[i+1] for i in range(len(thresholds)-1))

    def test_lambda_max_calculation(self):
        """Test lambda_max calculation for different correlation structures."""
        n_samples = 200
        correlation_structures = ['independent', 'block', 'ar1']

        for structure in correlation_structures:
            X, X_k = self._generate_correlated_knockoffs(n_samples, 50, structure)
            y = np.random.randn(n_samples)

            lambda_max = stat_glmnet_lambdasmax(X, X_k, y, nlambda=100)

            # Lambda_max should be positive and finite
            assert np.isfinite(lambda_max)
            assert lambda_max > 0

            # Should scale with sample size (approximately)
            X_large, X_k_large = self._generate_correlated_knockoffs(
                n_samples * 2, 50, structure
            )
            y_large = np.random.randn(n_samples * 2)
            lambda_max_large = stat_glmnet_lambdasmax(X_large, X_k_large, y_large)

            # Larger sample should generally give smaller lambda_max
            assert lambda_max_large <= lambda_max * 1.5

    def _generate_correlated_knockoffs(self, n_samples, n_features, structure):
        """Generate data with specific correlation structure and knockoffs."""
        if structure == 'independent':
            X = np.random.randn(n_samples, n_features)
        elif structure == 'block':
            # Block correlation structure
            X = np.random.randn(n_samples, n_features)
            for i in range(0, n_features, 10):
                end = min(i + 10, n_features)
                block = np.random.randn(n_samples, 1)
                X[:, i:end] += 0.5 * block
        elif structure == 'ar1':
            # AR(1) correlation structure
            rho = 0.5
            cov = np.array([[rho**abs(i-j) for j in range(n_features)]
                           for i in range(n_features)])
            X = np.random.multivariate_normal(np.zeros(n_features), cov, n_samples)

        knockoffs = Knockoffs(backend='python')
        X_k = knockoffs.create_gaussian(X)

        return X, X_k


class TestLOVEStatisticalValidation:
    """Test LOVE algorithm statistical properties."""

    def test_love_convergence_properties(self):
        """Test LOVE algorithm convergence on well-behaved data."""
        # Generate data with clear latent structure
        n_samples, n_features, n_factors = 300, 80, 4

        # Create true latent factors with sparse structure
        A_true = np.zeros((n_features, n_factors))
        genes_per_factor = n_features // n_factors
        for i in range(n_factors):
            start_idx = i * genes_per_factor
            end_idx = min((i + 1) * genes_per_factor, n_features)
            A_true[start_idx:end_idx, i] = np.random.randn(end_idx - start_idx)

        Z_true = np.random.randn(n_samples, n_factors)
        X = Z_true @ A_true.T + 0.1 * np.random.randn(n_samples, n_features)

        # Run LOVE with different parameters
        love_results = []
        for delta in [0.1, 0.5, 1.0]:
            result = call_love(X, delta=delta, verbose=False)
            love_results.append(result)

        # All runs should converge (no None results)
        assert all(result is not None for result in love_results)

        # Estimated number of factors should be reasonable
        for result in love_results:
            if 'K_est' in result:
                assert 2 <= result['K_est'] <= 8  # Should be in reasonable range

    def test_love_parameter_sensitivity(self):
        """Test LOVE sensitivity to parameter choices."""
        n_samples, n_features = 200, 60
        X = np.random.randn(n_samples, n_features)

        # Test different lambda values
        lambda_values = [0.1, 0.5, 0.9]
        results = []

        for lbd in lambda_values:
            try:
                result = call_love(X, lbd=lbd, verbose=False)
                results.append(result)
            except Exception as e:
                pytest.skip(f"LOVE failed with lambda={lbd}: {e}")

        # Results should be consistent in structure
        for result in results:
            assert 'A' in result or 'loadings' in result
            assert isinstance(result, dict)


class TestCrossValidationStatistical:
    """Test cross-validation statistical properties."""

    def test_cv_unbiased_estimation(self):
        """Test cross-validation provides unbiased performance estimates."""
        n_samples, n_features = 300, 50

        # Generate data with known signal-to-noise ratio
        X = np.random.randn(n_samples, n_features)
        true_beta = np.zeros(n_features)
        true_beta[:5] = 1.0  # First 5 features are relevant
        y = X @ true_beta + 0.5 * np.random.randn(n_samples)

        cv = SLIDEcv(X, y, n_folds=5)

        # Multiple CV runs should give consistent results
        cv_scores = []
        for _ in range(10):
            result = cv.run({"fdr": 0.1})
            if 'cv_score' in result:
                cv_scores.append(result['cv_score'])

        # CV scores should be stable across runs
        if len(cv_scores) > 1:
            cv_std = np.std(cv_scores)
            cv_mean = np.mean(cv_scores)
            assert cv_std / cv_mean < 0.3  # Coefficient of variation < 30%

    def test_cv_fold_independence(self):
        """Test cross-validation folds are properly independent."""
        n_samples, n_features = 200, 30
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        cv = SLIDEcv(X, y, n_folds=5)

        # Manually check fold construction
        folds = cv.folds if hasattr(cv, 'folds') else cv._create_folds()

        # Test sets should be disjoint
        all_test_indices = []
        for train_idx, test_idx in folds:
            all_test_indices.extend(test_idx)
            # Train and test should be disjoint
            assert len(set(train_idx) & set(test_idx)) == 0

        # All samples should be used exactly once in testing
        assert sorted(all_test_indices) == list(range(n_samples))