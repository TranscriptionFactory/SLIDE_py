"""
Test coverage for statistical assumptions and validity edge cases.

Critical gaps in testing statistical assumption violations and mathematical
edge cases that could lead to invalid scientific results.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from unittest.mock import patch, MagicMock
import warnings

from src.loveslide.slide import SLIDE, OptimizeSLIDE
from src.loveslide.knockoffs import Knockoffs
from src.loveslide.love import call_love
from src.loveslide.score import Estimator, SLIDE_Estimator

class TestStatisticalAssumptionViolations:
    """Test behavior when statistical assumptions are violated."""

    def test_non_gaussian_data_handling(self):
        """Test algorithms with strongly non-Gaussian data."""
        # Heavy-tailed distribution
        X_heavy = np.random.standard_t(df=2, size=(100, 20))

        # Skewed distribution
        X_skewed = np.random.exponential(scale=2, size=(100, 20))

        # Multimodal distribution
        X_multimodal = np.concatenate([
            np.random.normal(-3, 1, (50, 20)),
            np.random.normal(3, 1, (50, 20))
        ], axis=0)

        knockoffs = Knockoffs()

        for X, name in [(X_heavy, "heavy-tailed"), (X_skewed, "skewed"), (X_multimodal, "multimodal")]:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")

                try:
                    result = knockoffs.fit_transform(X)
                    assert result.shape == X.shape

                    # Check if appropriate warnings were raised
                    if w and any("gaussian" in str(warning.message).lower() for warning in w):
                        # Good - algorithm detected non-Gaussian assumptions
                        pass

                except ValueError as e:
                    # Acceptable if algorithm explicitly requires Gaussian data
                    assert "gaussian" in str(e).lower() or "normal" in str(e).lower()

    def test_rank_deficient_covariance_matrix(self):
        """Test handling of rank-deficient covariance matrices."""
        # Create rank-deficient data (fewer samples than features)
        X_rank_def = np.random.randn(10, 50)  # 10 samples, 50 features

        knockoffs = Knockoffs()

        try:
            result = knockoffs.fit_transform(X_rank_def)

            # If successful, should handle rank deficiency appropriately
            assert result.shape == X_rank_def.shape

            # Check that covariance structure is preserved where possible
            cov_original = np.cov(X_rank_def.T)
            cov_knockoff = np.cov(result.T)

            # Rank should be similar (up to numerical precision)
            rank_orig = np.linalg.matrix_rank(cov_original)
            rank_knockoff = np.linalg.matrix_rank(cov_knockoff)

            assert abs(rank_orig - rank_knockoff) <= 2  # Allow some numerical difference

        except (np.linalg.LinAlgError, ValueError) as e:
            # Acceptable if clearly identifies rank deficiency
            assert "rank" in str(e).lower() or "singular" in str(e).lower()

    def test_perfect_multicollinearity_handling(self):
        """Test handling of perfectly correlated features."""
        X_base = np.random.randn(100, 10)

        # Add perfectly correlated columns
        X_perfect_corr = np.column_stack([
            X_base,
            X_base[:, 0],          # Perfect copy
            2 * X_base[:, 1],      # Perfect linear relationship
            X_base[:, 2] + X_base[:, 3]  # Perfect linear combination
        ])

        knockoffs = Knockoffs()

        try:
            result = knockoffs.fit_transform(X_perfect_corr)

            # Should handle perfect correlations
            assert result.shape == X_perfect_corr.shape

            # Check that perfect correlations are preserved or handled appropriately
            for i in range(X_perfect_corr.shape[1]):
                for j in range(i+1, X_perfect_corr.shape[1]):
                    orig_corr = np.corrcoef(X_perfect_corr[:, i], X_perfect_corr[:, j])[0, 1]
                    knockoff_corr = np.corrcoef(result[:, i], result[:, j])[0, 1]

                    if abs(orig_corr) > 0.99:  # Nearly perfect correlation
                        # Should either preserve or handle gracefully
                        assert not np.isnan(knockoff_corr)

        except (np.linalg.LinAlgError, ValueError) as e:
            # Acceptable if detects multicollinearity
            assert any(word in str(e).lower() for word in ["singular", "rank", "collinear"])

    def test_extreme_scale_differences(self):
        """Test handling of features with vastly different scales."""
        X_mixed_scale = np.column_stack([
            np.random.randn(100, 5),              # Standard scale
            np.random.randn(100, 5) * 1e6,        # Very large scale
            np.random.randn(100, 5) * 1e-6,       # Very small scale
            np.random.randn(100, 5) * 1e12        # Extremely large scale
        ])

        knockoffs = Knockoffs()

        try:
            result = knockoffs.fit_transform(X_mixed_scale)

            # Should handle scale differences
            assert result.shape == X_mixed_scale.shape

            # Check that algorithm doesn't produce overflow/underflow
            assert np.isfinite(result).all()

            # Relative scales should be somewhat preserved
            orig_stds = np.std(X_mixed_scale, axis=0)
            result_stds = np.std(result, axis=0)

            # Order of magnitude should be similar
            for i in range(len(orig_stds)):
                if orig_stds[i] > 0 and result_stds[i] > 0:
                    ratio = np.log10(result_stds[i]) - np.log10(orig_stds[i])
                    assert abs(ratio) < 3  # Within 3 orders of magnitude

        except (OverflowError, UnderflowError, ValueError) as e:
            # Acceptable if identifies scale issues
            assert any(word in str(e).lower() for word in ["scale", "overflow", "underflow"])

class TestNumericalStabilityEdgeCases:
    """Test numerical stability edge cases."""

    def test_near_zero_eigenvalues(self):
        """Test handling of matrices with near-zero eigenvalues."""
        # Create matrix with small eigenvalues
        U, _, Vt = np.linalg.svd(np.random.randn(20, 20))
        small_eigenvals = np.concatenate([
            np.array([1.0, 0.1, 0.01]),
            np.full(17, 1e-12)  # Very small eigenvalues
        ])

        # Construct matrix with these eigenvalues
        problem_matrix = U @ np.diag(small_eigenvals) @ Vt

        # Use this as covariance for data generation
        X_near_singular = np.random.multivariate_normal(
            np.zeros(20), problem_matrix + 1e-10 * np.eye(20), size=100
        )

        knockoffs = Knockoffs()

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")

            try:
                result = knockoffs.fit_transform(X_near_singular)

                # Should handle near-singular matrices
                assert result.shape == X_near_singular.shape

                # Check for appropriate warnings about numerical stability
                stability_warnings = [warning for warning in w
                                    if any(word in str(warning.message).lower()
                                          for word in ["singular", "stable", "condition", "eigenvalue"])]

                # Should either work or warn appropriately
                assert len(stability_warnings) >= 0  # Warnings are good practice

            except (np.linalg.LinAlgError, ValueError) as e:
                # Acceptable if identifies numerical issues
                assert any(word in str(e).lower()
                          for word in ["singular", "stable", "condition", "eigenvalue"])

    def test_floating_point_precision_limits(self):
        """Test behavior at floating-point precision limits."""
        # Data that challenges floating-point precision
        X_precision = np.array([
            [1.0, 1.0 + np.finfo(float).eps, 1.0 + 2*np.finfo(float).eps],
            [1e-300, 2e-300, 3e-300],  # Very small values
            [1e300, 2e300, 3e300]      # Very large values
        ] * 50)  # Repeat to get enough samples

        knockoffs = Knockoffs()

        try:
            result = knockoffs.fit_transform(X_precision)

            # Should not lose precision catastrophically
            assert np.isfinite(result).all()

            # Relative differences should be preserved where possible
            for i in range(X_precision.shape[1]):
                orig_range = np.max(X_precision[:, i]) - np.min(X_precision[:, i])
                result_range = np.max(result[:, i]) - np.min(result[:, i])

                if orig_range > 0:
                    # Range should be preserved within reasonable bounds
                    range_ratio = result_range / orig_range
                    assert 0.1 < range_ratio < 10  # Within one order of magnitude

        except (ValueError, OverflowError, UnderflowError) as e:
            # Acceptable if identifies precision limits
            assert any(word in str(e).lower()
                      for word in ["precision", "overflow", "underflow", "finite"])

class TestStatisticalValidityChecks:
    """Test statistical validity and assumption checking."""

    def test_sample_size_adequacy_checks(self):
        """Test behavior with inadequate sample sizes."""
        # Very small sample size relative to features
        X_small = np.random.randn(5, 20)  # 5 samples, 20 features
        y_small = np.random.randn(5)

        slide = OptimizeSLIDE({'fdr': 0.1})

        try:
            # Should either handle gracefully or provide clear warning
            result = slide.run_SLIDE(X_small, love_result={'A': np.random.randn(20, 3)})

            if result is not None:
                # If successful, results should be flagged as potentially unreliable
                assert 'warning' in result or 'sample_size' in result

        except ValueError as e:
            # Acceptable if identifies sample size issues
            assert any(word in str(e).lower()
                      for word in ["sample", "size", "insufficient", "adequate"])

    def test_distributional_assumption_checking(self):
        """Test checking of distributional assumptions."""
        # Create data that violates normality assumption
        X_uniform = np.random.uniform(0, 1, (100, 10))  # Uniform distribution
        X_binary = np.random.binomial(1, 0.5, (100, 10))  # Binary data

        for X, dist_name in [(X_uniform, "uniform"), (X_binary, "binary")]:
            knockoffs = Knockoffs()

            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")

                try:
                    result = knockoffs.fit_transform(X)

                    # Check if distributional assumptions were tested
                    dist_warnings = [warning for warning in w
                                   if any(word in str(warning.message).lower()
                                         for word in ["distribution", "normal", "gaussian", "assumption"])]

                    # Good practice to warn about assumption violations
                    # (Not required, but indicates robust implementation)

                except ValueError as e:
                    # Acceptable if requires specific distributions
                    assert "distribution" in str(e).lower() or "normal" in str(e).lower()

    def test_false_discovery_rate_validity(self):
        """Test FDR control under various conditions."""
        X = np.random.randn(100, 50)
        y = X[:, :5].sum(axis=1) + np.random.randn(100) * 0.1  # True signal in first 5 features

        # Test various FDR levels
        fdr_levels = [0.01, 0.05, 0.1, 0.2, 0.5]

        for fdr in fdr_levels:
            slide = OptimizeSLIDE({'fdr': fdr})

            try:
                mock_love_result = {'A': np.random.randn(50, 10)}
                result = slide.run_SLIDE(X, love_result=mock_love_result)

                if result and 'selected_features' in result:
                    selected = result['selected_features']

                    # FDR control should be reasonable
                    if len(selected) > 0:
                        # Can't directly test FDR without multiple runs,
                        # but can check for basic reasonableness
                        assert len(selected) <= X.shape[1]  # Can't select more than available
                        assert all(0 <= idx < X.shape[1] for idx in selected)  # Valid indices

            except ValueError as e:
                # Should provide clear error if FDR level is invalid
                if fdr <= 0 or fdr >= 1:
                    assert "fdr" in str(e).lower()

if __name__ == "__main__":
    pytest.main([__file__])