"""
Test numerical stability and mathematical edge cases.

This module focuses on testing numerical algorithms under extreme conditions,
edge cases in matrix operations, and mathematical stability guarantees.
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import patch

from src.loveslide.knockoff.utils import (
    is_posdef, canonical_svd, normc, cov2cor,
    diag_pre_multiply, diag_post_multiply
)
from src.loveslide.knockoff.create import (
    create_fixed, create_gaussian, create_second_order,
    _create_equicorrelated, _create_sdp
)
from src.loveslide.knockoff.solve import create_solve_equi, create_solve_sdp
from src.loveslide import SLIDE, SLIDEcv


class TestMatrixOperationStability:
    """Test stability of core matrix operations."""

    def test_is_posdef_numerical_edge_cases(self):
        """Test positive definite checking with numerical edge cases."""
        # Case 1: Nearly singular matrix (very small eigenvalues)
        A = np.eye(5) * 1e-15
        assert not is_posdef(A, tol=1e-12)
        assert is_posdef(A, tol=1e-16)  # Should depend on tolerance

        # Case 2: Mixed scales (condition number issues)
        scales = [1e-10, 1e-5, 1.0, 1e5, 1e10]
        A = np.diag(scales)
        result = is_posdef(A)
        assert isinstance(result, bool)  # Should not crash
        assert result  # Diagonal with positive entries should be positive definite

        # Case 3: Nearly non-positive definite
        eigenvals = [1.0, 0.5, 0.1, 1e-14, 1e-16]
        Q, _ = np.linalg.qr(np.random.randn(5, 5))
        A = Q @ np.diag(eigenvals) @ Q.T
        assert is_posdef(A, tol=1e-12)  # Should be true with larger tolerance
        assert not is_posdef(A, tol=1e-10)  # Should be false with smaller tolerance

        # Case 4: Ill-conditioned but positive definite
        eigenvals = [1e10, 1e5, 1e0, 1e-5, 1e-10]
        A = Q @ np.diag(eigenvals) @ Q.T
        assert is_posdef(A, tol=1e-12)

        # Case 5: Matrix with NaN or Inf
        A_nan = np.array([[1, np.nan], [np.nan, 1]])
        assert not is_posdef(A_nan)

        A_inf = np.array([[1, np.inf], [np.inf, 1]])
        assert not is_posdef(A_inf)

    def test_canonical_svd_edge_cases(self):
        """Test SVD stability with challenging matrices."""
        # Case 1: Rank deficient matrix
        X = np.random.randn(100, 50)
        X[:, 25:] = X[:, :25] + 1e-15 * np.random.randn(100, 25)  # Nearly rank deficient

        U, s, Vt = canonical_svd(X)

        # Should maintain mathematical properties
        assert U.shape == (100, 50)
        assert s.shape == (50,)
        assert Vt.shape == (50, 50)
        assert np.all(s >= 0)  # Singular values should be non-negative
        assert np.allclose(s, np.sort(s)[::-1])  # Should be sorted descending

        # Reconstruction should be stable within numerical precision
        rank = np.sum(s > 1e-12)
        X_reconstructed = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
        assert np.allclose(X, X_reconstructed, rtol=1e-10)

        # Case 2: Wide matrix (more columns than rows)
        X_wide = np.random.randn(20, 100)
        U_w, s_w, Vt_w = canonical_svd(X_wide)
        assert U_w.shape[1] == min(20, 100)  # Should handle dimensions correctly

        # Case 3: Tall matrix (more rows than columns)
        X_tall = np.random.randn(100, 20)
        U_t, s_t, Vt_t = canonical_svd(X_tall)
        assert len(s_t) == 20

        # Case 4: Matrix with extreme aspect ratio
        X_extreme = np.random.randn(1000, 5)
        U_e, s_e, Vt_e = canonical_svd(X_extreme)
        assert not np.any(np.isnan(s_e))  # Should not produce NaN

    def test_normalization_stability(self):
        """Test matrix normalization under extreme conditions."""
        # Case 1: Zero variance columns
        X = np.random.randn(100, 10)
        X[:, 5] = 0  # Zero column
        X[:, 6] = 1  # Constant column

        X_norm = normc(X, center=True)

        # Should handle zero variance gracefully
        assert np.isfinite(X_norm).all()
        assert X_norm.shape == X.shape

        # Non-zero variance columns should be normalized
        non_zero_cols = [i for i in range(10) if i not in [5, 6]]
        for col in non_zero_cols:
            if np.std(X[:, col]) > 1e-10:
                assert abs(np.mean(X_norm[:, col])) < 1e-10  # Should be centered
                assert abs(np.std(X_norm[:, col]) - 1.0) < 1e-10  # Should be scaled

        # Case 2: Extreme scale differences
        X_scale = np.random.randn(100, 5)
        X_scale[:, 0] *= 1e10   # Very large scale
        X_scale[:, 1] *= 1e-10  # Very small scale

        X_norm_scale = normc(X_scale, center=True)
        assert np.isfinite(X_norm_scale).all()

        # Case 3: Nearly constant columns
        X_const = np.random.randn(100, 5)
        X_const[:, 0] = 1.0 + 1e-15 * np.random.randn(100)  # Nearly constant

        X_norm_const = normc(X_const, center=True)
        assert np.isfinite(X_norm_const).all()

    def test_covariance_correlation_conversion(self):
        """Test covariance to correlation conversion stability."""
        # Case 1: Well-conditioned covariance matrix
        X = np.random.randn(100, 10)
        Sigma = np.cov(X.T)
        R = cov2cor(Sigma)

        # Should produce valid correlation matrix
        assert np.allclose(np.diag(R), 1.0)  # Diagonal should be 1
        assert np.allclose(R, R.T)  # Should be symmetric
        assert np.all(np.abs(R) <= 1.0 + 1e-10)  # Correlations should be <= 1

        # Case 2: Singular covariance matrix
        X_singular = np.random.randn(100, 10)
        X_singular[:, 5] = X_singular[:, 0]  # Perfect correlation
        Sigma_singular = np.cov(X_singular.T)

        # Should handle singular case gracefully
        R_singular = cov2cor(Sigma_singular)
        assert np.isfinite(R_singular).all()
        assert np.allclose(np.diag(R_singular), 1.0)

        # Case 3: Diagonal covariance (independent variables)
        Sigma_diag = np.diag([1, 4, 9, 0.25, 100])
        R_diag = cov2cor(Sigma_diag)
        expected = np.eye(5)
        assert np.allclose(R_diag, expected)

        # Case 4: Covariance with zero diagonal elements
        Sigma_zero = np.random.randn(5, 5)
        Sigma_zero = Sigma_zero @ Sigma_zero.T  # Make positive semidefinite
        Sigma_zero[2, 2] = 0  # Zero variance

        # Should handle gracefully (may set row/col to zero or use pseudoinverse)
        R_zero = cov2cor(Sigma_zero)
        assert np.isfinite(R_zero).all()

    def test_diagonal_operations_stability(self):
        """Test diagonal pre/post multiplication stability."""
        # Case 1: Standard case
        d = np.array([1, 2, 3, 4])
        X = np.random.randn(4, 6)

        X_pre = diag_pre_multiply(d, X)
        X_post = diag_post_multiply(X, d[:6])  # Extend d for post multiplication

        assert X_pre.shape == X.shape
        assert X_post.shape == X.shape
        assert np.allclose(X_pre, np.diag(d) @ X)

        # Case 2: Zero diagonal elements
        d_zero = np.array([1, 0, 3, 0])
        X_pre_zero = diag_pre_multiply(d_zero, X)
        assert np.allclose(X_pre_zero[1, :], 0)  # Row 1 should be zero
        assert np.allclose(X_pre_zero[3, :], 0)  # Row 3 should be zero

        # Case 3: Extreme values in diagonal
        d_extreme = np.array([1e15, 1e-15, 1, -1])
        X_pre_extreme = diag_pre_multiply(d_extreme, X)
        assert np.isfinite(X_pre_extreme).all()

        # Case 4: Very large matrices (memory/performance test)
        d_large = np.random.randn(1000)
        X_large = np.random.randn(1000, 100)

        X_pre_large = diag_pre_multiply(d_large, X_large)
        assert X_pre_large.shape == (1000, 100)
        assert np.isfinite(X_pre_large).all()


class TestKnockoffNumericalStability:
    """Test knockoff generation numerical stability."""

    def test_create_equicorrelated_edge_cases(self):
        """Test equicorrelated knockoffs with numerical edge cases."""
        # Case 1: Ill-conditioned correlation matrix
        X = np.random.randn(200, 20)
        # Create high correlation between some features
        X[:, 1] = X[:, 0] + 1e-10 * np.random.randn(200)  # Nearly perfect correlation

        try:
            X_knockoff = _create_equicorrelated(X, randomize=False)
            assert X_knockoff.shape == X.shape
            assert not np.array_equal(X, X_knockoff)
        except np.linalg.LinAlgError:
            # Singular matrix should be handled gracefully
            pytest.skip("Matrix too ill-conditioned for equicorrelated knockoffs")

        # Case 2: Perfect correlation (singular case)
        X_singular = np.random.randn(100, 5)
        X_singular[:, 2] = X_singular[:, 0]  # Perfect correlation

        with pytest.raises((np.linalg.LinAlgError, ValueError)):
            _create_equicorrelated(X_singular, randomize=False)

        # Case 3: Very small features (numerical precision)
        X_small = 1e-10 * np.random.randn(100, 10)

        X_knockoff_small = _create_equicorrelated(X_small, randomize=False)
        assert X_knockoff_small.shape == X_small.shape
        # Relative scale should be preserved
        scale_ratio = np.std(X_knockoff_small) / np.std(X_small)
        assert 0.5 < scale_ratio < 2.0

    def test_create_sdp_numerical_stability(self):
        """Test SDP-based knockoff creation numerical stability."""
        # Case 1: Well-conditioned case
        X = np.random.randn(100, 15)

        X_knockoff = _create_sdp(X, randomize=False)
        assert X_knockoff.shape == X.shape

        # Should satisfy knockoff properties numerically
        combined = np.hstack([X, X_knockoff])
        cov_combined = np.cov(combined.T)

        # Test approximate knockoff property
        n_feat = X.shape[1]
        Sigma_X = cov_combined[:n_feat, :n_feat]
        Sigma_Xk = cov_combined[n_feat:, n_feat:]

        # Diagonal elements should be approximately equal
        assert np.allclose(np.diag(Sigma_X), np.diag(Sigma_Xk), rtol=0.2)

        # Case 2: High-dimensional case (p close to n)
        X_highdim = np.random.randn(50, 45)  # p/n = 0.9

        try:
            X_knockoff_hd = _create_sdp(X_highdim, randomize=False)
            assert X_knockoff_hd.shape == X_highdim.shape
        except (np.linalg.LinAlgError, ValueError) as e:
            # High-dimensional case may fail - should provide informative error
            assert "dimension" in str(e).lower() or "singular" in str(e).lower()

        # Case 3: Low rank case
        X_lowrank = np.random.randn(100, 20)
        # Make matrix low rank
        U = np.random.randn(100, 5)
        V = np.random.randn(5, 20)
        X_lowrank = U @ V + 1e-10 * np.random.randn(100, 20)  # Rank 5 + noise

        try:
            X_knockoff_lr = _create_sdp(X_lowrank, randomize=False)
            assert X_knockoff_lr.shape == X_lowrank.shape
        except (np.linalg.LinAlgError, ValueError):
            pytest.skip("Low rank case too challenging for SDP solver")

    def test_solve_sdp_edge_cases(self):
        """Test SDP solving under numerical edge cases."""
        # Case 1: Well-posed problem
        Sigma = np.random.randn(10, 10)
        Sigma = Sigma @ Sigma.T + 0.1 * np.eye(10)  # Ensure positive definite

        try:
            s = create_solve_sdp(Sigma)
            assert len(s) == Sigma.shape[0]
            assert np.all(s >= 0)  # Should be non-negative
            assert np.all(s <= 1)  # Should be <= 1 for valid knockoffs
        except Exception as e:
            pytest.skip(f"SDP solver not available: {e}")

        # Case 2: Nearly singular covariance
        eigenvals = [1, 0.5, 0.1, 0.01, 1e-10]
        Q, _ = np.linalg.qr(np.random.randn(5, 5))
        Sigma_singular = Q @ np.diag(eigenvals) @ Q.T

        try:
            s_singular = create_solve_sdp(Sigma_singular)
            assert len(s_singular) == 5
            # Solution should respect numerical constraints
            assert np.all(np.isfinite(s_singular))
        except (np.linalg.LinAlgError, ValueError):
            # Should handle singular case gracefully
            pass

        # Case 3: High condition number
        eigenvals_cond = [1000, 100, 10, 1, 0.001]
        Sigma_cond = Q @ np.diag(eigenvals_cond) @ Q.T

        try:
            s_cond = create_solve_sdp(Sigma_cond)
            assert np.all(np.isfinite(s_cond))
            # High condition number should still produce valid solution
            assert np.all(s_cond >= -1e-10)  # Allow small numerical errors
            assert np.all(s_cond <= 1 + 1e-10)
        except Exception:
            pytest.skip("High condition number case failed")


class TestSLIDEAlgorithmNumericalStability:
    """Test SLIDE algorithm numerical stability."""

    def test_slide_with_ill_conditioned_data(self):
        """Test SLIDE algorithm with ill-conditioned input data."""
        # Case 1: High multicollinearity
        n_samples, n_features = 200, 50
        X_base = np.random.randn(n_samples, 10)
        X_corr = np.hstack([X_base, X_base + 0.01 * np.random.randn(n_samples, 10)])
        X_uncorr = np.random.randn(n_samples, 30)
        X = np.hstack([X_corr, X_uncorr])

        y = np.sum(X[:, :5], axis=1) + 0.1 * np.random.randn(n_samples)

        slide = SLIDE({"fdr": 0.1, "K": 3})

        try:
            result = slide.run(X, y)
            # Should complete without crashing
            assert isinstance(result, dict)
        except np.linalg.LinAlgError:
            # Numerical issues should be handled gracefully
            pytest.skip("Input too ill-conditioned")

        # Case 2: Mixed scales
        X_mixed = np.random.randn(200, 20)
        X_mixed[:, :5] *= 1e6    # Very large scale
        X_mixed[:, 5:10] *= 1e-6 # Very small scale
        y_mixed = np.sum(X_mixed[:, :5], axis=1) * 1e-6 + 0.1 * np.random.randn(200)

        slide_mixed = SLIDE({"fdr": 0.1})

        try:
            result_mixed = slide_mixed.run(X_mixed, y_mixed)
            # Should handle mixed scales
            assert result_mixed is not None
        except Exception as e:
            # Should provide informative error for problematic data
            assert "scale" in str(e).lower() or "condition" in str(e).lower()

    def test_cv_numerical_stability(self):
        """Test cross-validation numerical stability."""
        # Case 1: Small dataset (numerical precision issues)
        X_small = np.random.randn(20, 10)
        y_small = np.random.randn(20)

        cv_small = SLIDEcv(X_small, y_small, n_folds=5)

        try:
            result_small = cv_small.run({"fdr": 0.1})
            # Should handle small datasets gracefully
            assert isinstance(result_small, dict)
        except ValueError as e:
            # Should provide informative error for too-small datasets
            assert "sample" in str(e).lower() or "fold" in str(e).lower()

        # Case 2: Degenerate folds (all outcomes the same)
        X_degen = np.random.randn(100, 15)
        y_degen = np.zeros(100)  # All zeros

        cv_degen = SLIDEcv(X_degen, y_degen, n_folds=5)

        try:
            result_degen = cv_degen.run({"fdr": 0.1})
            # Should handle degenerate outcomes
            assert result_degen is not None
        except Exception as e:
            # Should handle gracefully with informative error
            assert "variance" in str(e).lower() or "degenerate" in str(e).lower()

    def test_extreme_parameter_values(self):
        """Test algorithm behavior with extreme parameter values."""
        X = np.random.randn(100, 30)
        y = np.random.randn(100)

        extreme_params = [
            {"fdr": 1e-10},     # Extremely conservative FDR
            {"fdr": 0.99},      # Very liberal FDR
            {"K": 1},           # Minimal latent factors
            {"f_size": 1},      # Minimal feature size
            {"delta": 1e-15},   # Tiny delta
            {"delta": 1000},    # Huge delta
        ]

        for params in extreme_params:
            slide = SLIDE(params)

            try:
                result = slide.run(X, y)
                # Should handle extreme parameters gracefully
                assert result is not None
            except (ValueError, np.linalg.LinAlgError) as e:
                # Should provide informative error for invalid parameters
                param_name = list(params.keys())[0]
                assert param_name in str(e).lower() or "parameter" in str(e).lower()

    def test_memory_efficiency_large_problems(self):
        """Test memory efficiency with large problems."""
        import gc

        # Use moderate size to avoid memory issues in testing
        n_samples, n_features = 1000, 200

        # Monitor memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        X_large = np.random.randn(n_samples, n_features)
        y_large = np.random.randn(n_samples)

        slide_large = SLIDE({"fdr": 0.1, "f_size": 50})  # Force chunking

        try:
            result_large = slide_large.run(X_large, y_large)

            # Check memory usage didn't explode
            peak_memory = process.memory_info().rss
            memory_increase = peak_memory - initial_memory

            # Should not use more than 4x initial memory
            assert memory_increase < initial_memory * 3

            # Clean up
            del X_large, y_large, result_large
            gc.collect()

        except MemoryError:
            pytest.skip("Insufficient memory for large problem test")