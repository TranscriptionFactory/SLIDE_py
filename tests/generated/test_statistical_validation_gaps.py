"""
Test coverage for statistical validation and scientific reproducibility gaps.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from unittest.mock import patch, MagicMock
from src.loveslide.knockoff.filter import knockoff_threshold, knockoff_filter
from src.loveslide.knockoff.stats.base import compute_difference_stat, compute_signed_max_stat
from src.loveslide.love_python.love.score import Score_mat, LP_Score
from src.loveslide.love_python.love.cv import CV_delta, CV_lbd
from src.loveslide.knockoff.utils import random_problem
from src.loveslide.tools import calc_default_fsize


class TestStatisticalCorrectness:
    """Test statistical correctness and mathematical properties."""

    def test_knockoff_threshold_fdr_control(self):
        """Test that knockoff threshold provides proper FDR control."""
        # Generate test statistics with known null/non-null structure
        np.random.seed(42)
        p = 100
        n_nonnull = 20

        # Create W statistics: first 20 are signal, rest are noise
        W_signal = np.abs(np.random.randn(n_nonnull)) + 2  # Signal
        W_noise = np.random.randn(p - n_nonnull)           # Noise

        W = np.concatenate([W_signal, W_noise])
        np.random.shuffle(W)  # Randomize order

        fdr_levels = [0.05, 0.1, 0.2, 0.5]

        for target_fdr in fdr_levels:
            threshold = knockoff_threshold(W, fdr=target_fdr, offset=1)

            # Test threshold properties
            assert isinstance(threshold, (int, float))
            assert threshold >= 0

            # Count rejections
            rejections = np.sum(W >= threshold)

            # For well-separated signal, should reject some hypotheses
            if target_fdr >= 0.1:
                assert rejections > 0, f"No rejections at FDR {target_fdr}"

    def test_knockoff_filter_null_distribution(self):
        """Test knockoff filter under null hypothesis."""
        np.random.seed(123)

        # Generate pure null data (no signal)
        n, p = 100, 50
        X = np.random.randn(n, p)
        y = np.random.randn(n)  # Pure noise

        def null_statistic(X, y):
            # Statistic that should be symmetric around 0 under null
            return np.random.randn(X.shape[1])

        # Run multiple iterations to test FDR control
        fdr_target = 0.1
        n_trials = 10
        rejection_rates = []

        for trial in range(n_trials):
            np.random.seed(trial)
            selected = knockoff_filter(
                X, y, null_statistic,
                fdr=fdr_target, offset=1
            )

            rejection_rate = len(selected) / p
            rejection_rates.append(rejection_rate)

        # Under null, average rejection rate should be close to FDR
        mean_rejection_rate = np.mean(rejection_rates)
        assert mean_rejection_rate <= fdr_target + 0.05, \
            f"Rejection rate {mean_rejection_rate} exceeds FDR {fdr_target}"

    def test_difference_statistic_symmetry(self):
        """Test symmetry properties of difference statistics."""
        np.random.seed(456)

        # Create symmetric W+ and W- statistics
        p = 30
        W_plus = np.random.exponential(scale=1, size=p)
        W_minus = np.random.exponential(scale=1, size=p)

        # Test difference statistic
        W_diff = compute_difference_stat(W_plus, W_minus, swap=np.zeros(p))

        # Properties: should have both positive and negative values
        assert np.any(W_diff > 0), "No positive difference statistics"
        assert np.any(W_diff < 0), "No negative difference statistics"

        # Test with swapping
        swap = np.random.binomial(1, 0.5, p).astype(bool)
        W_diff_swap = compute_difference_stat(W_plus, W_minus, swap=swap)

        # Swapped differences should be negated for swapped indices
        expected = np.where(swap, W_minus - W_plus, W_plus - W_minus)
        assert np.allclose(W_diff_swap, expected)

    def test_signed_max_statistic_properties(self):
        """Test mathematical properties of signed max statistic."""
        np.random.seed(789)

        p = 25
        W_plus = np.random.gamma(2, 2, size=p)
        W_minus = np.random.gamma(2, 2, size=p)

        W_max = compute_signed_max_stat(W_plus, W_minus, swap=np.zeros(p))

        # Should always take the maximum in absolute value
        expected = np.where(W_plus >= W_minus, W_plus, -W_minus)
        assert np.allclose(W_max, expected)

        # Test magnitude property: |W_max| >= max(W_plus, W_minus)
        assert np.all(np.abs(W_max) >= np.maximum(W_plus, W_minus) - 1e-10)

    def test_love_score_matrix_properties(self):
        """Test mathematical properties of LOVE score matrices."""
        np.random.seed(321)

        # Create correlation matrix with known structure
        p = 20
        R = np.random.randn(p, p)
        R = R @ R.T  # Make positive semidefinite
        R = R / np.sqrt(np.outer(np.diag(R), np.diag(R)))  # Convert to correlation
        np.fill_diagonal(R, 1)  # Ensure unit diagonal

        # Test score matrix computation
        scores = Score_mat(R, q=2, exact=True)

        assert 'scores' in scores
        assert 'qvalues' in scores

        score_matrix = scores['scores']

        # Should be symmetric
        assert np.allclose(score_matrix, score_matrix.T, atol=1e-10)

        # Diagonal should be zero (self-correlation)
        assert np.allclose(np.diag(score_matrix), 0, atol=1e-10)

        # All scores should be non-negative (test statistic property)
        assert np.all(score_matrix >= -1e-10)

    def test_cv_delta_convergence(self):
        """Test cross-validation for delta parameter convergence."""
        np.random.seed(654)

        # Generate data with block structure
        n, p = 100, 30
        X = np.random.randn(n, p)

        # Add block correlation structure
        for i in range(0, p, 3):
            end = min(i + 3, p)
            X[:, i:end] = X[:, i:i+1] + 0.5 * np.random.randn(n, end-i)

        # Test CV for different delta values
        delta_grid = np.linspace(0.01, 0.5, 10)

        cv_results = CV_delta(
            X, delta_grid, diagonal=True,
            Kfolds=3, rep=2, verbose=False
        )

        assert 'optDelta' in cv_results
        assert 'lossGrid' in cv_results

        opt_delta = cv_results['optDelta']

        # Optimal delta should be in the tested range
        assert delta_grid.min() <= opt_delta <= delta_grid.max()

        # Loss should be finite and non-negative
        losses = cv_results['lossGrid']
        assert np.all(np.isfinite(losses))
        assert np.all(losses >= 0)

    def test_random_problem_statistical_properties(self):
        """Test statistical properties of generated random problems."""
        np.random.seed(987)

        # Generate multiple random problems
        n, p, k = 50, 20, 5
        n_problems = 10

        snr_values = []  # Signal-to-noise ratios

        for seed in range(n_problems):
            problem = random_problem(n=n, p=p, k=k, amplitude=3.0, seed=seed)

            X = problem['X']
            y = problem['y']
            beta = problem['beta']

            # Test problem properties
            assert X.shape == (n, p)
            assert len(y) == n
            assert len(beta) == p
            assert np.sum(beta != 0) == k

            # Calculate empirical SNR
            signal_var = np.var(X @ beta)
            noise_var = np.var(y - X @ beta)
            snr = signal_var / noise_var if noise_var > 0 else np.inf
            snr_values.append(snr)

            # Test design matrix properties
            # Columns should have approximately unit norm
            col_norms = np.linalg.norm(X, axis=0)
            assert np.allclose(col_norms, 1, atol=0.1)

            # Should be approximately centered
            col_means = np.mean(X, axis=0)
            assert np.allclose(col_means, 0, atol=0.1)

        # SNR should be reasonably consistent across problems
        snr_std = np.std(snr_values)
        assert snr_std < 2.0, f"SNR too variable: std={snr_std}"

    def test_fsize_calculation_mathematical_properties(self):
        """Test mathematical properties of default feature size calculation."""
        # Test boundary conditions and mathematical consistency

        test_cases = [
            # (n_rows, K, expected properties)
            (100, 10, lambda f: f == 10),  # Normal case: n > K, K < 100
            (100, 150, lambda f: f == 100),  # K > 100, n < K
            (10, 10, lambda f: f == 8),     # n == K, close difference
            (10, 12, lambda f: f == 10),    # n < K
            (50, 50, lambda f: f == 48),    # n == K, abs difference <= 2
            (1, 1, lambda f: f == -1),      # Minimum case
        ]

        for n_rows, K, property_check in test_cases:
            f_size = calc_default_fsize(n_rows, K)

            assert isinstance(f_size, int), f"f_size should be integer for n={n_rows}, K={K}"
            assert property_check(f_size), f"Property failed for n={n_rows}, K={K}, f_size={f_size}"

        # Test monotonicity properties where expected
        for K in [10, 50, 200]:
            f_sizes = [calc_default_fsize(n, K) for n in range(K-5, K+10)]

            # Should not decrease too rapidly
            diffs = np.diff(f_sizes)
            assert np.all(diffs >= -2), f"f_size decreases too rapidly for K={K}"


class TestNumericalPrecision:
    """Test numerical precision and stability."""

    def test_correlation_matrix_conditioning(self):
        """Test correlation matrix conditioning and numerical stability."""
        from src.loveslide.knockoff.utils import cov2cor, is_posdef

        # Test with ill-conditioned covariance matrix
        p = 10
        Sigma = np.random.randn(p, p)
        Sigma = Sigma @ Sigma.T

        # Make it ill-conditioned
        eigenvals = np.linalg.eigvals(Sigma)
        min_eigval = np.min(eigenvals)
        Sigma += (1e-12 - min_eigval) * np.eye(p)  # Nearly singular

        R = cov2cor(Sigma)

        # Should maintain mathematical properties despite conditioning
        assert np.allclose(np.diag(R), 1)  # Unit diagonal
        assert np.allclose(R, R.T, atol=1e-12)  # Symmetry

        # Should remain positive semidefinite
        R_eigenvals = np.linalg.eigvals(R)
        assert np.all(R_eigenvals >= -1e-10), "Correlation matrix not PSD"

    def test_knockoff_threshold_numerical_stability(self):
        """Test knockoff threshold computation numerical stability."""
        # Test with extreme W values
        extreme_cases = [
            np.array([1e-15, 1e-14, 1e-13]),  # Very small values
            np.array([1e15, 1e14, 1e13]),     # Very large values
            np.array([1e-10, 1e10, -1e10]),   # Mixed scales
            np.array([0, 0, 0, 0.001]),       # Mostly zeros
        ]

        for W in extreme_cases:
            for fdr in [0.05, 0.1, 0.2]:
                threshold = knockoff_threshold(W, fdr=fdr, offset=1)

                # Should return finite threshold
                assert np.isfinite(threshold)
                assert threshold >= 0

                # Should be consistent with input scale
                if np.max(W) > 0:
                    assert threshold <= np.max(W) + 1e-10

    def test_floating_point_edge_cases(self):
        """Test handling of floating point edge cases."""
        from src.loveslide.love_python.love.score import LP_Score

        # Test with edge case inputs
        edge_cases = [
            np.array([0.0, 0.0, 0.0]),     # All zeros
            np.array([np.inf, 1, 2]),      # Contains infinity
            np.array([np.nan, 1, 2]),      # Contains NaN
            np.array([1e-100, 1e-99]),     # Underflow range
            np.array([1e100, 1e99]),       # Overflow range
        ]

        for R_ij in edge_cases:
            for ind in range(len(R_ij)):
                try:
                    score = LP_Score(R_ij, ind, exact=True)

                    # If computation succeeds, result should be finite
                    if not (np.isinf(R_ij).any() or np.isnan(R_ij).any()):
                        assert np.isfinite(score), f"Non-finite score for input {R_ij}"

                except (ValueError, FloatingPointError, OverflowError):
                    # These exceptions are acceptable for edge cases
                    pass


if __name__ == "__main__":
    pytest.main([__file__])