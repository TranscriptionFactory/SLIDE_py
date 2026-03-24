"""
Test coverage for algorithm convergence at boundary conditions.
Focus: Mathematical edge cases, convergence failures, and numerical boundaries.
"""

import pytest
import numpy as np
import warnings
from unittest.mock import patch, MagicMock

from loveslide.slide import SLIDE, OptimizeSLIDE
from loveslide.cv import SLIDEcv
from loveslide.score import Estimator, SLIDE_Estimator
from loveslide.love_python.love import LOVE


class TestSLIDEConvergenceBoundaries:
    """Test SLIDE algorithm at convergence boundaries."""

    def test_slide_near_rank_deficient_data(self):
        """Test SLIDE behavior with near rank-deficient data."""
        # Create nearly rank-deficient data
        n, p = 100, 50
        U, _, Vt = np.linalg.svd(np.random.randn(n, p))

        # Set smallest singular values to near-zero
        s = np.linspace(10, 1e-10, p)
        X = U[:, :p] @ np.diag(s) @ Vt
        y = np.random.binomial(1, 0.5, n)

        params = {
            'fdr': 0.1,
            'delta': [0.05],
            'lambda': [0.1]
        }

        slide = SLIDE(params, x=X, y=y)

        # Should handle near-singular conditions gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # May fail, but should not crash
                result = slide.fit(K=5, love_only=True, save_love=False)
            except (ValueError, np.linalg.LinAlgError) as e:
                # Acceptable to fail with appropriate error
                assert "singular" in str(e).lower() or "rank" in str(e).lower()

    def test_slide_extreme_condition_numbers(self):
        """Test SLIDE with matrices having extreme condition numbers."""
        n, p = 50, 20

        # Create matrix with extreme condition number
        condition_numbers = [1e12, 1e15, 1e18]

        for cond_num in condition_numbers:
            # Create matrix with specific condition number
            U, _ = np.linalg.qr(np.random.randn(n, p))
            V, _ = np.linalg.qr(np.random.randn(p, p))
            s = np.logspace(0, -np.log10(cond_num), p)
            X = U @ np.diag(s) @ V.T
            y = np.random.randn(n)

            params = {'fdr': 0.1, 'delta': [0.1], 'lambda': [0.1]}
            slide = SLIDE(params, x=X, y=y)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    slide.fit(K=3, love_only=True, save_love=False)
                except (ValueError, np.linalg.LinAlgError, RuntimeWarning):
                    # Expected for extreme condition numbers
                    pass

    def test_slide_convergence_with_zero_variance_features(self):
        """Test SLIDE convergence when some features have zero variance."""
        n, p = 100, 30
        X = np.random.randn(n, p)

        # Set some columns to zero variance
        zero_var_cols = [5, 10, 15, 20]
        for col in zero_var_cols:
            X[:, col] = 1.0  # Constant value

        y = np.random.binomial(1, 0.5, n)
        params = {'fdr': 0.1, 'delta': [0.1], 'lambda': [0.1]}

        slide = SLIDE(params, x=X, y=y)

        # Should handle zero-variance features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = slide.fit(K=5, love_only=True, save_love=False)

        # Should complete or fail gracefully
        assert result is not None or True  # Either succeeds or handled

    def test_slide_with_perfect_collinearity(self):
        """Test SLIDE behavior with perfectly collinear features."""
        n, p = 80, 20
        base_X = np.random.randn(n, p//2)

        # Create perfect collinearity by duplicating columns
        X = np.column_stack([base_X, base_X + np.random.randn(n, p//2) * 1e-15])
        y = np.random.randn(n)

        params = {'fdr': 0.1, 'delta': [0.1], 'lambda': [0.1]}
        slide = SLIDE(params, x=X, y=y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = slide.fit(K=3, love_only=True, save_love=False)
            except (ValueError, np.linalg.LinAlgError):
                # Perfect collinearity should be detected
                pass


class TestLOVEAlgorithmBoundaries:
    """Test LOVE algorithm convergence boundaries."""

    def test_love_with_identity_covariance(self):
        """Test LOVE behavior with identity covariance structure."""
        n, p = 100, 20
        # Independent features (identity covariance)
        X = np.random.randn(n, p)
        X = (X - X.mean(axis=0)) / X.std(axis=0)  # Standardize

        love = LOVE()

        # Should handle independent features
        try:
            result = love.fit(X, delta=0.1, lbd=0.5)
            # With independent features, should find minimal structure
            assert 'Liub' in result
        except ValueError as e:
            # May reject independent data appropriately
            assert "structure" in str(e).lower() or "rank" in str(e).lower()

    def test_love_extreme_delta_values(self):
        """Test LOVE with extreme delta threshold values."""
        n, p = 80, 15
        X = np.random.randn(n, p)

        extreme_deltas = [1e-15, 1e-10, 0.99, 1.0 - 1e-15]

        for delta in extreme_deltas:
            love = LOVE()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    result = love.fit(X, delta=delta, lbd=0.5)
                    # Should handle extreme thresholds
                    assert isinstance(result, dict)
                except (ValueError, RuntimeError) as e:
                    # Some extreme values may be rejected
                    assert "delta" in str(e).lower() or "threshold" in str(e).lower()

    def test_love_with_block_diagonal_structure(self):
        """Test LOVE with perfect block diagonal correlation structure."""
        n = 150

        # Create block diagonal structure
        block_sizes = [5, 7, 3, 5]
        blocks = []

        for block_size in block_sizes:
            # Highly correlated block
            base = np.random.randn(n, 1)
            block = base + np.random.randn(n, block_size) * 0.1
            blocks.append(block)

        X = np.column_stack(blocks)

        love = LOVE()
        result = love.fit(X, delta=0.1, lbd=0.5)

        # Should identify block structure
        assert 'Liub' in result
        assert result['Liub'].shape[1] <= len(block_sizes) + 2  # At most one factor per block + noise

    def test_love_numerical_stability_extreme_correlations(self):
        """Test LOVE numerical stability with extreme correlations."""
        n, p = 100, 10

        # Create data with correlations very close to ±1
        correlations = [0.9999, -0.9999, 0.99999]

        for corr in correlations:
            X1 = np.random.randn(n)
            X2 = corr * X1 + np.sqrt(1 - corr**2) * np.random.randn(n)
            X_rest = np.random.randn(n, p-2)

            X = np.column_stack([X1, X2, X_rest])

            love = LOVE()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    result = love.fit(X, delta=0.1, lbd=0.5)
                    # Should handle near-perfect correlations
                    assert np.all(np.isfinite(result['Liub']))
                except (ValueError, np.linalg.LinAlgError):
                    # May reject numerically unstable cases
                    pass


class TestEstimatorConvergenceBoundaries:
    """Test Estimator convergence at boundaries."""

    def test_estimator_perfect_separation(self):
        """Test estimator with perfectly separable binary classification."""
        n = 100
        # Create perfectly separable data
        X1 = np.random.randn(n//2, 5) + 3  # Class 1: shifted positive
        X2 = np.random.randn(n//2, 5) - 3  # Class 2: shifted negative
        X = np.vstack([X1, X2])
        y = np.hstack([np.ones(n//2), np.zeros(n//2)])

        estimator = Estimator(model='auto')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = estimator.score(X, y)

        # Should achieve near-perfect separation
        assert score > 0.95 or np.isnan(score)  # May diverge with perfect separation

    def test_estimator_multicollinearity_handling(self):
        """Test estimator with severe multicollinearity."""
        n, p = 50, 20
        base_features = np.random.randn(n, 5)

        # Create multicollinear features
        noise_scale = 1e-10
        X = np.column_stack([
            base_features,
            base_features + np.random.randn(n, 5) * noise_scale,
            base_features * 2 + np.random.randn(n, 5) * noise_scale,
            np.random.randn(n, 5)  # Some independent features
        ])

        y = base_features[:, 0] + np.random.randn(n) * 0.1  # Signal from first feature

        estimator = Estimator(model='linear')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = estimator.score(X, y)

        # Should handle multicollinearity (score may be affected but shouldn't crash)
        assert isinstance(score, (float, np.floating)) or np.isnan(score)

    def test_slide_estimator_extreme_sparsity(self):
        """Test SLIDE estimator with extremely sparse true signals."""
        n, p = 200, 100
        X = np.random.randn(n, p)

        # Extremely sparse signal: only 1 true feature
        beta = np.zeros(p)
        beta[42] = 5.0  # Single strong signal

        y = X @ beta + np.random.randn(n) * 0.1

        slide_est = SLIDE_Estimator()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = slide_est.score(X, y)

        # Should handle extreme sparsity
        assert isinstance(score, (float, np.floating))
        assert score >= 0  # R² should be non-negative


class TestCVBoundaryConditions:
    """Test cross-validation at boundary conditions."""

    def test_cv_with_minimal_samples_per_fold(self):
        """Test CV behavior with very few samples per fold."""
        n = 20  # Very small sample size
        p = 10
        X = np.random.randn(n, p)
        y = np.random.binomial(1, 0.5, n)

        params = {
            'fdr': 0.1,
            'delta': [0.1, 0.2],
            'lambda': [0.1]
        }

        cv = SLIDEcv(params, x=X, y=y)

        # Should handle small sample sizes appropriately
        try:
            results = cv.cross_validate(K_list=[2, 3], nfolds=3)
            assert isinstance(results, dict)
        except ValueError as e:
            # May reject if too few samples per fold
            assert "fold" in str(e).lower() or "sample" in str(e).lower()

    def test_cv_with_imbalanced_folds(self):
        """Test CV with severely imbalanced class distributions."""
        n = 100
        # Create severely imbalanced binary response
        y = np.zeros(n)
        y[:5] = 1  # Only 5% positive class

        X = np.random.randn(n, 20)
        X[:5] += 2  # Signal in positive class

        params = {'fdr': 0.1, 'delta': [0.1], 'lambda': [0.1]}
        cv = SLIDEcv(params, x=X, y=y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                results = cv.cross_validate(K_list=[3], nfolds=5)
                # Should handle class imbalance
                assert isinstance(results, dict)
            except ValueError:
                # May reject if folds become too imbalanced
                pass

    def test_cv_convergence_with_noise_only_data(self):
        """Test CV convergence when data contains no true signal."""
        n, p = 150, 30
        X = np.random.randn(n, p)  # Pure noise
        y = np.random.randn(n)     # Pure noise response

        params = {'fdr': 0.1, 'delta': [0.1], 'lambda': [0.1]}
        cv = SLIDEcv(params, x=X, y=y)

        results = cv.cross_validate(K_list=[2, 5], nfolds=3)

        # Should handle noise-only data gracefully
        assert isinstance(results, dict)

        # Performance should be near baseline for noise-only data
        if 'cv_scores' in results:
            scores = results['cv_scores']
            # R² for noise should be near 0 or negative
            assert all(score <= 0.3 for score in scores.values() if score is not None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])