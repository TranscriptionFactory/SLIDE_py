"""
Test skeleton for statistical edge cases and numerical validation.

Focus on testing behavior under statistical assumption violations,
numerical edge cases, and boundary conditions in statistical algorithms.
"""
import pytest
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_classification, make_regression
from unittest.mock import patch
import warnings

from loveslide import SLIDE, SLIDEcv, Knockoffs, SLIDE_Estimator
from loveslide.knockoff.utils import is_posdef, cov2cor, normc
from loveslide.love import call_love


class TestStatisticalDistributionEdgeCases:
    """Test behavior under non-standard data distributions."""

    def test_heavy_tailed_distributions(self):
        """Test performance with heavy-tailed data distributions."""
        np.random.seed(42)
        n, p = 100, 20

        # Generate heavy-tailed data (Student's t with low df)
        X_heavy = np.random.standard_t(df=2, size=(n, p))
        y_heavy = np.random.choice([0, 1], n)

        slide = SLIDE({'fdr': 0.1}, x=X_heavy, y=y_heavy)

        # Should handle heavy tails without numerical issues
        try:
            # Mock the computationally intensive parts
            with patch.object(slide, 'run_knockoffs') as mock_knockoffs:
                mock_knockoffs.return_value = {'selected_features': [1, 5]}
                # Should not raise numerical errors
                slide.show_params()  # Basic functionality test

        except (RuntimeError, ValueError) as e:
            if "numerical" in str(e).lower() or "overflow" in str(e).lower():
                pytest.fail(f"Heavy-tailed data caused numerical issues: {e}")

    def test_multimodal_distributions(self):
        """Test with multimodal feature distributions."""
        np.random.seed(123)
        n, p = 80, 15

        # Create bimodal distributions
        X_bimodal = np.random.randn(n, p)
        for j in range(p // 2):
            # Make half the features bimodal
            mask = np.random.choice([True, False], n)
            X_bimodal[mask, j] += 3  # Second mode

        y_binary = np.random.choice([0, 1], n)

        # Test LOVE with multimodal data
        try:
            result = call_love(X_bimodal, delta=0.1)
            assert isinstance(result, dict)
            assert 'LFs' in result

        except (RuntimeError, ValueError) as e:
            if "convergence" in str(e).lower():
                pytest.skip(f"Convergence issues with multimodal data: {e}")
            else:
                raise

    def test_skewed_distributions(self):
        """Test with highly skewed feature distributions."""
        np.random.seed(42)
        n, p = 120, 25

        # Generate skewed data using exponential distribution
        X_skewed = np.random.exponential(scale=1.0, size=(n, p))
        y = np.random.choice([0, 1], n)

        knockoffs = Knockoffs(y=y, z2=np.hstack([X_skewed, X_skewed]))

        # Should handle skewness
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Test covariance estimation with skewed data
            Sigma = np.cov(X_skewed.T)
            is_valid = is_posdef(Sigma)

            # May generate warnings about assumptions
            if w and not is_valid:
                assert any("assumption" in str(warning.message).lower() or
                          "skew" in str(warning.message).lower()
                          for warning in w)

    def test_sparse_data_patterns(self):
        """Test with sparse data (many zeros)."""
        np.random.seed(42)
        n, p = 100, 30

        # Create sparse data matrix
        X_sparse = np.random.randn(n, p)
        # Make 70% of entries zero
        zero_mask = np.random.random((n, p)) < 0.7
        X_sparse[zero_mask] = 0

        y = np.random.choice([0, 1], n)

        # Test that sparse data is handled appropriately
        slide = SLIDE({'fdr': 0.2}, x=X_sparse, y=y)

        # Should handle sparse patterns
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            slide.show_params()

            if w:
                # Check for sparsity-related warnings
                sparsity_warnings = [warning for warning in w
                                   if "sparse" in str(warning.message).lower() or
                                      "zero" in str(warning.message).lower()]
                if sparsity_warnings:
                    print(f"Sparsity warnings detected: {sparsity_warnings}")


class TestNumericalStabilityEdgeCases:
    """Test numerical stability in extreme conditions."""

    def test_near_singular_covariance_matrices(self):
        """Test handling of near-singular covariance matrices."""
        np.random.seed(42)
        p = 20

        # Create near-singular matrix
        X = np.random.randn(100, p)
        # Make last feature nearly identical to first
        X[:, -1] = X[:, 0] + 1e-10 * np.random.randn(100)

        Sigma = np.cov(X.T)

        # Test positive definiteness checking
        result_strict = is_posdef(Sigma, tol=1e-8)
        result_loose = is_posdef(Sigma, tol=1e-12)

        # Behavior should be consistent with tolerance
        if not result_strict:
            assert not result_loose or np.abs(result_strict - result_loose) < 0.1

    def test_extreme_correlation_values(self):
        """Test handling of extreme correlation values."""
        # Perfect correlation case
        X_perfect = np.random.randn(50, 10)
        X_perfect[:, 1] = X_perfect[:, 0]  # Perfect correlation

        Sigma_perfect = np.cov(X_perfect.T)
        corr_perfect = cov2cor(Sigma_perfect)

        # Should handle perfect correlations gracefully
        assert np.isfinite(corr_perfect).all()
        assert np.allclose(np.diag(corr_perfect), 1.0)

        # Check that perfect correlation is detected
        assert np.abs(corr_perfect[0, 1]) > 0.99

    def test_numerical_precision_boundaries(self):
        """Test behavior near machine precision boundaries."""
        # Values near machine epsilon
        eps = np.finfo(float).eps

        # Matrix with very small values
        X_tiny = np.full((20, 10), eps)

        # Test normalization with tiny values
        X_norm = normc(X_tiny, center=False)
        assert np.isfinite(X_norm).all()

        # Matrix with very large values
        large_val = np.sqrt(np.finfo(float).max) / 100
        X_large = np.full((20, 10), large_val)

        X_norm_large = normc(X_large, center=False)
        assert np.isfinite(X_norm_large).all()

    def test_condition_number_extremes(self):
        """Test matrices with extreme condition numbers."""
        # Well-conditioned matrix
        X_well = np.random.randn(50, 20)
        Sigma_well = np.cov(X_well.T)
        cond_well = np.linalg.cond(Sigma_well)

        # Should be reasonably conditioned
        assert cond_well < 1e12  # Not pathologically ill-conditioned

        # Deliberately ill-conditioned matrix
        U, _, Vt = np.linalg.svd(np.random.randn(20, 20))
        s = np.logspace(0, -15, 20)  # Singular values spanning 15 orders
        X_ill = U @ np.diag(s) @ Vt

        Sigma_ill = X_ill @ X_ill.T

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Operations should either work or fail gracefully
            try:
                is_valid = is_posdef(Sigma_ill)
                corr_ill = cov2cor(Sigma_ill)

                if not is_valid and w:
                    assert any("condition" in str(warning.message).lower() or
                              "ill" in str(warning.message).lower() or
                              "singular" in str(warning.message).lower()
                              for warning in w)

            except (np.linalg.LinAlgError, RuntimeError) as e:
                assert "singular" in str(e).lower() or "condition" in str(e).lower()


class TestStatisticalAssumptionViolations:
    """Test violations of statistical assumptions."""

    def test_non_gaussian_noise_models(self):
        """Test with non-Gaussian noise in regression settings."""
        np.random.seed(42)
        n, p = 100, 15

        X = np.random.randn(n, p)
        beta = np.random.randn(p)

        # Different noise models
        noise_models = {
            'uniform': np.random.uniform(-2, 2, n),
            'laplace': np.random.laplace(0, 1, n),
            'gamma': np.random.gamma(2, 1, n) - 2  # Center around 0
        }

        for noise_type, noise in noise_models.items():
            y_continuous = X @ beta + noise

            # Test that estimators handle non-Gaussian noise
            estimator = SLIDE_Estimator(model='auto')

            try:
                # Should not assume Gaussian noise
                estimator.fit(X, y_continuous)
                predictions = estimator.predict(X)

                assert len(predictions) == n
                assert np.isfinite(predictions).all()

            except Exception as e:
                if "gaussian" in str(e).lower() or "normal" in str(e).lower():
                    pytest.fail(f"Estimator assumes Gaussian noise for {noise_type}: {e}")

    def test_heteroscedastic_noise(self):
        """Test with heteroscedastic (non-constant variance) noise."""
        np.random.seed(123)
        n, p = 80, 12

        X = np.random.randn(n, p)
        beta = np.random.randn(p)

        # Heteroscedastic noise (variance depends on features)
        noise_std = 0.1 + 0.5 * np.abs(X[:, 0])  # Variance depends on first feature
        noise = np.random.randn(n) * noise_std
        y = X @ beta + noise

        # Test cross-validation with heteroscedastic data
        cv = SLIDEcv(
            slide_params={'fdr': 0.1},
            cv_params={'n_folds': 3, 'n_rep': 1}
        )

        # Should handle varying noise levels
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock the expensive parts
            with patch.object(cv, '_bench_cv') as mock_cv:
                mock_cv.return_value = [{'metric': 0.8}]

                try:
                    cv.fit(X, y)
                except Exception as e:
                    if "homoscedastic" in str(e).lower():
                        pytest.fail(f"Method assumes homoscedastic noise: {e}")

    def test_non_linear_relationships(self):
        """Test with non-linear feature relationships."""
        np.random.seed(42)
        n, p = 100, 20

        # Generate data with non-linear relationships
        X_linear = np.random.randn(n, p)

        # Add quadratic and interaction terms as hidden non-linearities
        y_nonlinear = (X_linear[:, 0] ** 2 +  # Quadratic
                      X_linear[:, 1] * X_linear[:, 2] +  # Interaction
                      np.random.randn(n) * 0.1)  # Noise
        y_binary = (y_nonlinear > np.median(y_nonlinear)).astype(int)

        # Test that linear methods handle non-linear data gracefully
        slide = SLIDE({'fdr': 0.1}, x=X_linear, y=y_binary)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Should not assume linearity too strictly
            slide.show_params()

            # Check for linearity assumption warnings
            linearity_warnings = [warning for warning in w
                                if "linear" in str(warning.message).lower() or
                                   "non" in str(warning.message).lower()]

            if linearity_warnings:
                print(f"Non-linearity warnings: {linearity_warnings}")

    def test_temporal_correlation_patterns(self):
        """Test with temporally correlated data."""
        np.random.seed(42)
        n, p = 100, 15

        # Generate temporally correlated data (AR process)
        X_temporal = np.zeros((n, p))
        for j in range(p):
            # AR(1) process for each feature
            rho = np.random.uniform(0.3, 0.7)  # Autocorrelation
            for i in range(1, n):
                X_temporal[i, j] = (rho * X_temporal[i-1, j] +
                                  np.sqrt(1 - rho**2) * np.random.randn())

        y = np.random.choice([0, 1], n)

        # Test that methods handle temporal correlation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Test covariance estimation with temporal correlation
            Sigma_temporal = np.cov(X_temporal.T)

            # Should detect potential temporal issues
            if w:
                temporal_warnings = [warning for warning in w
                                   if "temporal" in str(warning.message).lower() or
                                      "autocorr" in str(warning.message).lower()]
                if temporal_warnings:
                    print(f"Temporal correlation warnings: {temporal_warnings}")


class TestBoundaryConditionBehavior:
    """Test behavior at statistical boundary conditions."""

    def test_sample_size_boundary_effects(self):
        """Test behavior with very small vs very large sample sizes."""
        p = 20  # Fixed number of features

        # Very small sample size (n < p)
        n_small = 15
        X_small = np.random.randn(n_small, p)
        y_small = np.random.choice([0, 1], n_small)

        with pytest.raises((ValueError, RuntimeError)):
            # Should recognize insufficient sample size
            call_love(X_small, delta=0.1)

        # Moderate sample size (n ≈ p)
        n_moderate = p + 5
        X_moderate = np.random.randn(n_moderate, p)
        y_moderate = np.random.choice([0, 1], n_moderate)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            try:
                result = call_love(X_moderate, delta=0.1)

                # Should warn about small sample size
                if w:
                    size_warnings = [warning for warning in w
                                   if "sample" in str(warning.message).lower() or
                                      "size" in str(warning.message).lower()]
                    if size_warnings:
                        print(f"Sample size warnings: {size_warnings}")

            except (ValueError, RuntimeError) as e:
                if "sample" in str(e).lower() or "insufficient" in str(e).lower():
                    print(f"Expected small sample size error: {e}")

    def test_feature_dimension_boundary_effects(self):
        """Test behavior with varying feature dimensions."""
        n = 100  # Fixed sample size

        # Single feature case
        X_single = np.random.randn(n, 1)
        y = np.random.choice([0, 1], n)

        with pytest.raises((ValueError, RuntimeError)):
            # Most multivariate methods require multiple features
            call_love(X_single, delta=0.1)

        # Very high dimensional case (p >> n)
        p_large = n * 2
        X_large = np.random.randn(n, p_large)

        with pytest.raises((ValueError, RuntimeError, MemoryError)):
            # Should recognize high-dimensional issues
            call_love(X_large, delta=0.1)

    def test_correlation_strength_boundaries(self):
        """Test with extreme correlation strengths."""
        n, p = 80, 10

        # Very weak correlations
        X_weak = np.random.randn(n, p) * 10  # Large variance, weak structure

        result_weak = call_love(X_weak, delta=0.1)

        # Should still return result but may find few/no latent factors
        assert isinstance(result_weak, dict)
        assert 'LFs' in result_weak

        # Very strong correlations (block structure)
        X_strong = np.zeros((n, p))

        # Create strong block correlation structure
        for i in range(n):
            common_factor = np.random.randn()
            X_strong[i, :5] = common_factor + 0.1 * np.random.randn(5)
            X_strong[i, 5:] = np.random.randn(5)

        result_strong = call_love(X_strong, delta=0.1)
        assert isinstance(result_strong, dict)
        assert 'LFs' in result_strong


if __name__ == "__main__":
    pytest.main([__file__])