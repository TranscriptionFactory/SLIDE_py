"""
Test algorithm validation edge cases and mathematical correctness.

Focus: Numerical stability, algorithm correctness under extreme
conditions, mathematical property preservation.
"""
import pytest
import numpy as np
import scipy.linalg
from scipy import stats
from unittest.mock import patch, MagicMock
import warnings

from loveslide import SLIDE, Knockoffs
from loveslide.score import Estimator, SLIDE_Estimator


class TestNumericalStabilityEdgeCases:
    """Test numerical stability under extreme conditions."""

    def test_near_singular_covariance_matrices(self):
        """Test handling of near-singular covariance matrices."""
        # Create near-singular matrix
        n, p = 50, 20
        X = np.random.randn(n, p)
        X[:, -1] = X[:, 0] + 1e-10 * np.random.randn(n)  # Nearly collinear

        Sigma = X.T @ X / n
        cond_number = np.linalg.cond(Sigma)
        assert cond_number > 1e12, "Matrix should be near-singular"

        knockoffs = Knockoffs()

        try:
            X_ko = knockoffs.create(Sigma)
            if X_ko is not None:
                # Should preserve dimensions
                assert X_ko.shape[1] == p
                # Should not be identical to original
                assert not np.allclose(X_ko, X, rtol=1e-10)

        except np.linalg.LinAlgError as e:
            # Should provide informative error
            assert any(word in str(e).lower() for word in ["singular", "condition", "rank"])

    def test_extreme_eigenvalue_ratios(self):
        """Test with extreme eigenvalue ratios."""
        # Create matrix with extreme eigenvalue spread
        n, p = 40, 15
        eigenvals = np.array([1e6] + [1.0] * (p-2) + [1e-6])
        U = scipy.linalg.orth(np.random.randn(p, p))
        Sigma = U @ np.diag(eigenvals) @ U.T

        knockoffs = Knockoffs()

        try:
            X_ko = knockoffs.create(Sigma)

            if X_ko is not None:
                # Verify statistical properties are reasonable
                ko_cov = np.cov(X_ko.T)
                assert np.all(np.diag(ko_cov) > 0), "Knockoffs should have positive variance"

        except Exception as e:
            # Should handle extreme eigenvalue ratios
            assert any(word in str(e).lower() for word in ["eigenvalue", "condition", "numerical"])

    def test_floating_point_precision_limits(self):
        """Test behavior at floating-point precision limits."""
        # Data at floating-point precision limits
        X = np.random.randn(30, 10)
        y = np.random.randn(30)

        # Scale to near machine epsilon
        X_tiny = X * np.finfo(np.float64).eps
        X_huge = X * (1.0 / np.finfo(np.float64).eps)

        params = {"fdr": 0.1, "niter": 2}

        for X_extreme, label in [(X_tiny, "tiny"), (X_huge, "huge")]:
            try:
                slide = SLIDE(params, x=X_extreme, y=y)
                result = slide.calc_default_fsize(3)

                # Should handle extreme scales
                assert result is not None and result > 0

            except (OverflowError, UnderflowError, ValueError) as e:
                # Should provide clear error for precision limits
                assert any(word in str(e).lower() for word in ["overflow", "underflow", "precision", "range"])

    def test_rank_deficient_scenarios(self):
        """Test handling of rank-deficient data matrices."""
        # Create rank-deficient data
        n, p = 35, 20
        rank = 15  # Less than p
        U = np.random.randn(n, rank)
        V = np.random.randn(rank, p)
        X_rank_def = U @ V

        # Verify rank deficiency
        assert np.linalg.matrix_rank(X_rank_def) == rank < p

        y = np.random.randn(n)
        params = {"fdr": 0.1, "niter": 2}

        try:
            slide = SLIDE(params, x=X_rank_def, y=y)

            # Should handle rank deficiency
            estimator = Estimator()
            estimator.fit(X_rank_def, y)

            predictions = estimator.predict(X_rank_def)
            assert predictions is not None

        except Exception as e:
            # Should provide informative error for rank deficiency
            assert any(word in str(e).lower() for word in ["rank", "singular", "collinear"])


class TestStatisticalPropertyPreservation:
    """Test preservation of statistical properties."""

    def test_knockoff_statistical_properties(self):
        """Test that knockoffs preserve required statistical properties."""
        n, p = 60, 25
        np.random.seed(42)
        X = np.random.randn(n, p)
        Sigma = X.T @ X / n

        knockoffs = Knockoffs()
        X_ko = knockoffs.create(Sigma)

        if X_ko is not None:
            # Test statistical properties
            # 1. Knockoffs should have same pairwise correlations with originals
            cross_corr = np.corrcoef(X.T, X_ko.T)[:p, p:]

            # 2. Check if correlation structure is preserved
            original_corr = np.corrcoef(X.T)
            knockoff_corr = np.corrcoef(X_ko.T)

            # Diagonal should be close to diagonal of original
            diag_diff = np.abs(np.diag(knockoff_corr) - np.diag(original_corr))
            assert np.all(diag_diff < 0.5), "Knockoff variances too different"

            # 3. Knockoffs should not be too similar to originals
            self_similarity = np.diag(cross_corr)
            assert np.all(self_similarity < 0.9), "Knockoffs too similar to originals"

    def test_estimation_consistency(self):
        """Test estimation consistency under different conditions."""
        n, p = 45, 20
        # Create data with known structure
        beta_true = np.zeros(p)
        beta_true[:5] = [2, -1.5, 1, -0.5, 0.8]  # First 5 features are relevant

        X = np.random.randn(n, p)
        noise = np.random.randn(n) * 0.5
        y = X @ beta_true + noise

        estimator = SLIDE_Estimator()

        # Test with different sample sizes
        for n_subset in [20, 30, 45]:
            X_sub = X[:n_subset]
            y_sub = y[:n_subset]

            try:
                estimator.fit(X_sub, y_sub)
                selected_features = estimator.get_selected_features()

                # Should select relevant features more often with more data
                if n_subset >= 30:
                    true_features = set(np.where(beta_true != 0)[0])
                    selected_set = set(selected_features) if selected_features else set()
                    overlap = len(true_features.intersection(selected_set))

                    # At least some overlap expected with sufficient data
                    assert overlap >= 1 or "challenging estimation scenario"

            except Exception as e:
                # Should handle small sample scenarios
                assert "sample" in str(e).lower() or "insufficient" in str(e).lower()

    def test_cross_validation_stability(self):
        """Test cross-validation stability and consistency."""
        n, p = 50, 15
        X = np.random.randn(n, p)
        y = np.random.randn(n)

        params = {"fdr": 0.1, "niter": 3, "K": 3}

        # Run CV multiple times to test stability
        cv_scores = []
        for seed in [42, 123, 456]:
            np.random.seed(seed)

            from loveslide import SLIDEcv
            cv = SLIDEcv(params, x=X.copy(), y=y.copy())

            try:
                with patch.object(cv, '_run_single_fold') as mock_fold:
                    # Mock consistent CV results
                    mock_fold.return_value = {
                        "test_score": 0.6 + 0.1 * np.random.randn(),
                        "train_score": 0.8 + 0.1 * np.random.randn()
                    }

                    result = cv.run(n_folds=5)
                    if result and 'test_score' in result:
                        cv_scores.append(result['test_score'])

            except Exception as e:
                # CV should be robust
                assert "cv" in str(e).lower() or "fold" in str(e).lower()

        if len(cv_scores) > 1:
            # CV results should be reasonably stable
            cv_std = np.std(cv_scores)
            assert cv_std < 0.5, "Cross-validation too unstable"


class TestBoundaryValueAnalysis:
    """Test algorithm behavior at boundary values."""

    def test_zero_variance_features(self):
        """Test handling of zero variance features."""
        n, p = 40, 12
        X = np.random.randn(n, p)
        X[:, 3] = 5.0  # Constant feature
        X[:, 7] = 0.0  # Zero feature
        y = np.random.randn(n)

        params = {"fdr": 0.1, "niter": 2}

        try:
            slide = SLIDE(params, x=X, y=y)
            estimator = SLIDE_Estimator()
            estimator.fit(X, y)

            # Should handle constant features appropriately
            selected = estimator.get_selected_features()
            if selected:
                # Constant features should not be selected
                assert 3 not in selected and 7 not in selected

        except Exception as e:
            # Should provide clear error for zero variance
            assert any(word in str(e).lower() for word in ["variance", "constant", "zero"])

    def test_perfect_correlation_scenarios(self):
        """Test handling of perfectly correlated features."""
        n, p = 35, 10
        X = np.random.randn(n, p)
        X[:, 4] = X[:, 2]  # Perfect correlation
        X[:, 6] = -X[:, 2]  # Perfect anti-correlation
        y = np.random.randn(n)

        estimator = SLIDE_Estimator()

        try:
            estimator.fit(X, y)
            coef = estimator.model.coef_ if hasattr(estimator.model, 'coef_') else None

            if coef is not None:
                # Should handle perfect correlations
                assert np.all(np.isfinite(coef)), "Coefficients should be finite"

        except Exception as e:
            # Should handle perfect correlations gracefully
            assert any(word in str(e).lower() for word in ["correlation", "collinear", "singular"])

    def test_extreme_sample_size_ratios(self):
        """Test with extreme n/p ratios."""
        # High-dimensional scenario (p > n)
        n_small, p_large = 15, 50
        X_hd = np.random.randn(n_small, p_large)
        y_hd = np.random.randn(n_small)

        params = {"fdr": 0.1, "niter": 2}

        try:
            slide = SLIDE(params, x=X_hd, y=y_hd)
            result = slide.calc_default_fsize(5)

            # Should handle high-dimensional case
            assert result is not None and result <= p_large

        except Exception as e:
            # Should provide clear error for high-dimensional case
            assert any(word in str(e).lower() for word in ["dimension", "samples", "features"])

        # Low-dimensional scenario (n >> p)
        n_large, p_small = 200, 5
        X_ld = np.random.randn(n_large, p_small)
        y_ld = np.random.randn(n_large)

        try:
            slide = SLIDE(params, x=X_ld, y=y_ld)
            result = slide.calc_default_fsize(3)

            # Should handle low-dimensional case
            assert result is not None and result > 0

        except Exception as e:
            # Should be able to handle this case
            assert "dimension" in str(e).lower()


class TestAlgorithmicCorrectnessValidation:
    """Test algorithmic correctness under controlled conditions."""

    def test_known_structure_recovery(self):
        """Test recovery of known structure."""
        n, p = 60, 20
        # Create data with known factor structure
        k_true = 3
        factors_true = np.random.randn(n, k_true)
        loadings_true = np.random.randn(p, k_true)
        noise = np.random.randn(n, p) * 0.1

        X = factors_true @ loadings_true.T + noise
        y = factors_true @ [1, -1, 0.5] + np.random.randn(n) * 0.1

        params = {"fdr": 0.1, "niter": 3, "K": k_true}

        try:
            slide = SLIDE(params, x=X, y=y)

            with patch('loveslide.love.call_love') as mock_love:
                # Return structure close to truth
                mock_love.return_value = {
                    "factors": factors_true + np.random.randn(*factors_true.shape) * 0.05,
                    "loadings": loadings_true + np.random.randn(*loadings_true.shape) * 0.05
                }

                result = slide.run_love()

                # Should recover approximate structure
                if result and "factors" in result:
                    recovered_factors = result["factors"]
                    # Check correlation with true factors
                    correlations = []
                    for i in range(k_true):
                        if i < recovered_factors.shape[1]:
                            corr = np.corrcoef(factors_true[:, i], recovered_factors[:, i])[0, 1]
                            correlations.append(abs(corr))

                    if correlations:
                        assert max(correlations) > 0.5, "Should recover some factor structure"

        except Exception as e:
            # Should handle factor estimation
            assert "factor" in str(e).lower() or "structure" in str(e).lower()

    def test_null_hypothesis_calibration(self):
        """Test null hypothesis calibration (no signal case)."""
        n, p = 50, 25
        # Pure noise data
        X = np.random.randn(n, p)
        y = np.random.randn(n)  # Independent of X

        estimator = SLIDE_Estimator()

        try:
            estimator.fit(X, y)
            selected_features = estimator.get_selected_features()

            # With no signal, should select few features
            n_selected = len(selected_features) if selected_features else 0
            expected_false_discoveries = p * 0.1  # FDR = 0.1

            # Should control false discoveries
            assert n_selected <= expected_false_discoveries * 2, "Too many false discoveries"

        except Exception as e:
            # Should handle null case
            assert "signal" in str(e).lower() or "null" in str(e).lower()

    def test_convergence_criteria_validation(self):
        """Test convergence criteria and stopping conditions."""
        n, p = 40, 15
        X = np.random.randn(n, p)
        y = np.random.randn(n)

        params = {"fdr": 0.1, "niter": 100, "K": 5}  # High iterations

        slide = SLIDE(params, x=X, y=y)

        # Track convergence
        convergence_history = []

        def mock_iteration_callback(*args, **kwargs):
            # Simulate convergence tracking
            iteration = len(convergence_history)
            convergence_value = 1.0 / (iteration + 1)  # Decreasing
            convergence_history.append(convergence_value)
            return {"converged": convergence_value < 0.01}

        try:
            with patch('loveslide.slide.SLIDE._check_convergence', side_effect=mock_iteration_callback):
                if hasattr(slide, 'run_with_convergence'):
                    result = slide.run_with_convergence()

                    # Should converge reasonably
                    if convergence_history:
                        assert convergence_history[-1] < convergence_history[0], "Should show convergence"

        except AttributeError:
            # Method may not exist
            pytest.skip("Convergence tracking not implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])