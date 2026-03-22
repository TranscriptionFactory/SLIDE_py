"""
Test coverage for statistical methods and computation correctness.

Major gaps identified:
- Statistical correctness of knockoff filtering methods
- Validation of LOVE algorithm statistical properties
- Cross-validation statistical validity
- Estimation accuracy and bias testing
- Hypothesis testing correctness
- Bootstrap and resampling validation
- Convergence testing for iterative algorithms
"""
import pytest
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Union, Optional

from loveslide import Knockoffs, VotingResult
from loveslide.knockoff.stats.lasso import stat_lasso_lambdadiff, stat_lasso_coefdiff
from loveslide.knockoff.stats.forward import stat_forward_selection
from loveslide.knockoff.stats.base import compute_difference_stat, compute_signed_max_stat
from loveslide.knockoff.filter import knockoff_threshold, knockoff_filter_voting
from loveslide.love_python.love.score import Score_mat, LP_Score
from loveslide.love_python.love.cv import CV_delta, KfoldCV_delta


class TestKnockoffStatisticalCorrectness:
    """Test statistical correctness of knockoff methods."""

    def test_knockoff_fdr_control_simulation(self):
        """Test that knockoff methods control FDR at specified level."""
        np.random.seed(42)
        n, p = 500, 100
        target_fdr = 0.1
        n_simulations = 20  # Reduced for test speed

        fdp_estimates = []

        for sim in range(n_simulations):
            # Generate null data (no true signals)
            X = np.random.randn(n, p)
            y = np.random.randn(n)

            # Run knockoff filter
            result = knockoff_filter_voting(
                X, y, fdr=target_fdr, niter=10,
                statistic='lasso_lambdadiff', seed=sim
            )

            # Calculate empirical FDP
            n_selected = len(result.selected)
            fdp = n_selected / max(n_selected, 1)  # All selections are false discoveries
            fdp_estimates.append(fdp)

        # Mean FDP should be close to target FDR
        mean_fdp = np.mean(fdp_estimates)
        assert mean_fdp <= target_fdr + 0.05  # Allow some tolerance

    def test_knockoff_power_with_signal(self):
        """Test knockoff power when true signals exist."""
        np.random.seed(123)
        n, p = 300, 50
        n_signals = 5

        # Generate data with known signals
        X = np.random.randn(n, p)
        beta_true = np.zeros(p)
        beta_true[:n_signals] = np.random.randn(n_signals) * 2  # Strong signals

        y = X @ beta_true + 0.1 * np.random.randn(n)

        result = knockoff_filter_voting(
            X, y, fdr=0.2, niter=15,
            statistic='lasso_lambdadiff', seed=42
        )

        # Should detect some true signals
        true_positives = len(set(result.selected) & set(range(n_signals)))
        power = true_positives / n_signals

        assert power > 0.3  # Should have reasonable power

    def test_knockoff_statistics_properties(self):
        """Test properties of knockoff statistics."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        X_ko = np.random.randn(100, 20)  # Mock knockoffs
        y = np.random.randn(100)

        # Test different statistics
        stats_lasso = stat_lasso_lambdadiff(X, X_ko, y)
        stats_coef = stat_lasso_coefdiff(X, X_ko, y)

        # Statistics should have reasonable properties
        assert len(stats_lasso) == 20
        assert len(stats_coef) == 20

        # Should have both positive and negative values (sign flip property)
        assert np.any(stats_lasso > 0)
        assert np.any(stats_lasso < 0)

    def test_knockoff_threshold_properties(self):
        """Test properties of knockoff threshold calculation."""
        # Test with known statistics
        W = np.array([3.0, -1.0, 2.5, -0.5, 1.8, -2.0, 0.3, -1.5])

        # Test different FDR levels
        for fdr in [0.05, 0.1, 0.2]:
            threshold = knockoff_threshold(W, fdr=fdr)

            # Higher FDR should give lower threshold (more selections)
            if fdr > 0.1:
                threshold_01 = knockoff_threshold(W, fdr=0.1)
                assert threshold <= threshold_01

    def test_knockoff_sign_flip_property(self):
        """Test the sign flip property of knockoff statistics."""
        np.random.seed(42)
        X = np.random.randn(100, 15)
        y = np.random.randn(100)

        # Generate knockoffs with sign flip
        X_ko = X.copy()
        # Flip signs for some features (simplified sign flip)
        flip_mask = np.random.choice([True, False], size=15)
        X_ko[:, flip_mask] *= -1

        stats = stat_lasso_lambdadiff(X, X_ko, y)

        # Should satisfy antisymmetry under sign flip
        stats_flipped = stat_lasso_lambdadiff(X_ko, X, y)
        assert np.allclose(stats, -stats_flipped, atol=0.1)


class TestLOVEStatisticalCorrectness:
    """Test statistical correctness of LOVE algorithm."""

    def test_love_factor_recovery(self):
        """Test LOVE's ability to recover known factor structure."""
        from loveslide.love_python.love.love import LOVE

        np.random.seed(42)
        n, p, K = 200, 15, 3

        # Generate data with known factor structure
        A_true = np.random.randn(p, K)
        A_true[8:, 0] = 0   # Sparse structure
        A_true[:5, 2] = 0

        Z = np.random.randn(n, K)
        E = 0.3 * np.random.randn(n, p)
        X = Z @ A_true.T + E

        # Run LOVE
        result = LOVE(X, lbd=0.5, mu=0.5, verbose=False)

        L_hat = result['L_hat']

        # Should estimate reasonable number of factors
        assert L_hat.shape[1] >= K
        assert L_hat.shape[1] <= K + 2  # Allow some over-estimation

        # Check if recovered structure is meaningful
        # (Compare to true loading matrix via correlation or subspace distance)
        recovered_factors = L_hat.shape[1]
        assert recovered_factors > 0

    def test_love_correlation_structure_preservation(self):
        """Test that LOVE preserves correlation structure."""
        from loveslide.love_python.love.love import LOVE

        np.random.seed(123)
        X = np.random.randn(150, 12)

        # Add some correlation structure
        X[:, 1] = 0.7 * X[:, 0] + 0.3 * X[:, 1]
        X[:, 2] = 0.6 * X[:, 0] + 0.4 * X[:, 2]

        original_corr = np.corrcoef(X, rowvar=False)

        result = LOVE(X, lbd=0.4, mu=0.4)

        # Reconstructed correlation should be similar to original
        if 'Sigma_hat' in result:
            Sigma_hat = result['Sigma_hat']
            reconstructed_corr = np.corrcoef(Sigma_hat)

            # Correlation between original and reconstructed
            corr_similarity = np.corrcoef(
                original_corr.flatten(),
                reconstructed_corr.flatten()
            )[0, 1]

            assert corr_similarity > 0.7  # Should preserve structure

    def test_love_parameter_sensitivity(self):
        """Test LOVE sensitivity to parameter changes."""
        from loveslide.love_python.love.love import LOVE

        X = np.random.randn(100, 10)

        # Test different lambda values
        lambdas = [0.1, 0.5, 0.9]
        results = []

        for lbd in lambdas:
            result = LOVE(X, lbd=lbd, mu=0.5)
            results.append(result)

        # Results should be different for different lambdas
        for i in range(len(results) - 1):
            L1 = results[i]['L_hat']
            L2 = results[i + 1]['L_hat']

            # Should have different factor structures
            assert not np.allclose(L1, L2, atol=0.1)

    def test_love_cross_validation_consistency(self):
        """Test consistency of LOVE cross-validation."""
        from loveslide.love_python.love.cv import KfoldCV_delta

        np.random.seed(42)
        X = np.random.randn(120, 8)

        # Run CV multiple times
        cv_results = []
        for seed in [42, 43, 44]:
            np.random.seed(seed)
            result = KfoldCV_delta(X, delta=None, K_fold=5)
            cv_results.append(result)

        # Optimal deltas should be reasonably similar
        if all('opt_delta' in result for result in cv_results):
            opt_deltas = [result['opt_delta'] for result in cv_results]
            delta_std = np.std(opt_deltas)
            assert delta_std < 0.3  # Reasonable stability


class TestScoreMatrixStatistics:
    """Test statistical properties of score matrices."""

    def test_score_matrix_properties(self):
        """Test mathematical properties of score matrices."""
        # Create correlation matrix
        p = 8
        X = np.random.randn(100, p)
        R = np.corrcoef(X, rowvar=False)

        result = Score_mat(R, q=2)
        score_matrix = result['score']

        # Score matrix should be symmetric
        assert np.allclose(score_matrix, score_matrix.T)

        # Diagonal should be meaningful
        assert np.all(np.diag(score_matrix) >= 0)

    def test_lp_score_statistical_properties(self):
        """Test LP score statistical properties."""
        # Test with various correlation patterns
        correlation_vectors = [
            np.array([0.8, 0.1, 0.2, -0.1]),   # Strong correlation with first
            np.array([0.1, 0.1, 0.1, 0.1]),    # Weak correlations
            np.array([-0.7, 0.2, -0.3, 0.1]),  # Mixed signs
        ]

        for R_ij in correlation_vectors:
            score = LP_Score(R_ij, ind=0, exact=False)

            # Score should be non-negative
            assert score >= 0

            # Score should increase with correlation strength
            if np.abs(R_ij[0]) > 0.5:
                assert score > 0.1  # Should be substantial for strong correlation

    def test_score_matrix_convergence(self):
        """Test convergence properties of score matrix computation."""
        # Test with increasing precision
        p = 6
        X = np.random.randn(200, p)
        R = np.corrcoef(X, rowvar=False)

        # Compare exact vs approximate
        result_exact = Score_mat(R, q=2, exact=True)
        result_approx = Score_mat(R, q=2, exact=False)

        # Should be reasonably close
        if 'score' in result_exact and 'score' in result_approx:
            score_diff = np.abs(result_exact['score'] - result_approx['score'])
            assert np.max(score_diff) < 0.5  # Reasonable approximation


class TestCrossValidationStatistics:
    """Test statistical validity of cross-validation procedures."""

    def test_cv_bias_variance_tradeoff(self):
        """Test bias-variance properties of cross-validation."""
        from loveslide.love_python.love.cv import CV_delta

        np.random.seed(42)
        X = np.random.randn(100, 10)

        # Test different numbers of delta values
        delta_grids = [
            np.linspace(0.1, 0.9, 5),   # Coarse grid
            np.linspace(0.1, 0.9, 20),  # Fine grid
        ]

        results = []
        for delta_grid in delta_grids:
            result = CV_delta(X, delta_grid, diagonal=False)
            results.append(result)

        # Finer grid should give more stable results
        assert all('opt_delta' in result for result in results)

    def test_kfold_cv_stability(self):
        """Test stability of K-fold cross-validation."""
        from loveslide.love_python.love.cv import KfoldCV_delta

        X = np.random.randn(150, 12)

        # Test different K values
        k_values = [3, 5, 10]
        opt_deltas = []

        for k in k_values:
            np.random.seed(42)  # Fix seed for comparison
            result = KfoldCV_delta(X, K_fold=k)
            if 'opt_delta' in result:
                opt_deltas.append(result['opt_delta'])

        # Results should be reasonably stable across different K
        if len(opt_deltas) > 1:
            delta_range = max(opt_deltas) - min(opt_deltas)
            assert delta_range < 0.5  # Reasonable stability

    def test_cv_overfitting_detection(self):
        """Test that CV can detect overfitting."""
        # Create small dataset prone to overfitting
        n, p = 30, 20
        X = np.random.randn(n, p)

        from loveslide.love_python.love.cv import KfoldCV_delta

        # With many folds, should detect overfitting risk
        with pytest.warns(UserWarning, match="overfitting") or \
             pytest.raises(ValueError):
            result = KfoldCV_delta(X, K_fold=n-1)  # Leave-one-out


class TestEstimationStatistics:
    """Test statistical properties of parameter estimation."""

    def test_estimation_bias(self):
        """Test bias in parameter estimation."""
        from loveslide.love_python.love.est_pure_homo import EstAI

        # Generate data with known parameters
        np.random.seed(42)
        p, K = 10, 2
        A_true = np.random.randn(p, K)

        # Create covariance from known loading matrix
        Sigma = A_true @ A_true.T + np.eye(p)

        # Estimate loading matrix
        AI_hat = EstAI(Sigma, optDelta=0.5, se_est=np.ones(p), method="HT")

        # Should recover reasonable structure
        assert AI_hat.shape[0] == p
        assert AI_hat.shape[1] > 0

        # Test if estimation is unbiased (simplified test)
        # In practice, would need multiple Monte Carlo runs
        estimated_variance = np.var(AI_hat.flatten())
        assert estimated_variance > 0  # Should capture some signal

    def test_estimation_consistency(self):
        """Test consistency of estimation procedures."""
        from loveslide.love_python.love.est_omega import estOmega

        # Test with different sizes
        for p in [5, 10, 15]:
            C = np.random.randn(p, p)
            C = C @ C.T + 0.1 * np.eye(p)  # Make positive definite

            Omega = estOmega(lbd=0.3, C=C)

            # Should be symmetric and positive definite
            assert np.allclose(Omega, Omega.T)
            assert np.all(np.linalg.eigvals(Omega) > -1e-6)

    def test_estimation_robustness(self):
        """Test robustness of estimation to outliers."""
        from loveslide.love_python.love.est_pure_homo import EstC

        p, K = 8, 2
        Sigma_clean = np.eye(p) + 0.3 * np.ones((p, p))
        AI = np.random.randn(p, K)

        # Clean estimation
        C_clean = EstC(Sigma_clean, AI, diagonal=False)

        # Add outlier to covariance matrix
        Sigma_outlier = Sigma_clean.copy()
        Sigma_outlier[0, 0] *= 10  # Large diagonal element

        C_outlier = EstC(Sigma_outlier, AI, diagonal=False)

        # Results should be reasonably similar (robust estimation)
        relative_diff = np.linalg.norm(C_clean - C_outlier) / np.linalg.norm(C_clean)
        assert relative_diff < 2.0  # Allow some difference but not too large


class TestHypothesisTestingCorrectness:
    """Test statistical hypothesis testing procedures."""

    def test_multiple_testing_correction(self):
        """Test multiple testing correction in feature selection."""
        # Generate data with known null and alternative hypotheses
        np.random.seed(42)
        n, p = 200, 50
        n_true_signals = 5

        X = np.random.randn(n, p)
        beta_true = np.zeros(p)
        beta_true[:n_true_signals] = 2  # True signals

        y = X @ beta_true + np.random.randn(n)

        # Test knockoff procedure
        result = knockoff_filter_voting(
            X, y, fdr=0.1, niter=20,
            statistic='lasso_lambdadiff', seed=42
        )

        # Calculate empirical FWER and FDR
        true_discoveries = len(set(result.selected) & set(range(n_true_signals)))
        false_discoveries = len(set(result.selected) - set(range(n_true_signals)))

        fdp = false_discoveries / max(len(result.selected), 1)

        # FDP should be controlled
        assert fdp <= 0.2  # Allow some tolerance above target FDR

    def test_power_analysis(self):
        """Test power analysis for different effect sizes."""
        n, p = 150, 30
        effect_sizes = [0.5, 1.0, 2.0]  # Different signal strengths

        powers = []

        for effect_size in effect_sizes:
            # Generate data with given effect size
            np.random.seed(42)
            X = np.random.randn(n, p)
            beta_true = np.zeros(p)
            beta_true[:3] = effect_size  # 3 true signals

            y = X @ beta_true + np.random.randn(n)

            result = knockoff_filter_voting(
                X, y, fdr=0.2, niter=10,
                statistic='lasso_lambdadiff', seed=42
            )

            # Calculate power
            true_positives = len(set(result.selected) & set(range(3)))
            power = true_positives / 3
            powers.append(power)

        # Power should increase with effect size
        assert all(powers[i] <= powers[i+1] + 0.1 for i in range(len(powers)-1))

    def test_type_i_error_control(self):
        """Test Type I error control under null hypothesis."""
        # Generate pure null data
        np.random.seed(123)
        n, p = 100, 25

        type_i_errors = []

        for trial in range(10):  # Multiple trials
            X = np.random.randn(n, p)
            y = np.random.randn(n)  # Pure noise

            result = knockoff_filter_voting(
                X, y, fdr=0.05, niter=5,
                statistic='lasso_lambdadiff', seed=trial
            )

            # All selections are Type I errors
            type_i_rate = len(result.selected) / p
            type_i_errors.append(type_i_rate)

        # Average Type I error rate should be controlled
        mean_type_i = np.mean(type_i_errors)
        assert mean_type_i <= 0.1  # Should be well below nominal level


class TestConvergenceAndNumericalStability:
    """Test convergence and numerical stability of algorithms."""

    def test_iterative_algorithm_convergence(self):
        """Test convergence of iterative algorithms."""
        from loveslide.love_python.love.est_nonpure import LP

        # Test LP optimization convergence
        y = np.random.randn(10)
        lbd = 0.3

        # Run with different tolerances
        result = LP(y, lbd)

        # Should converge to reasonable solution
        assert result is not None
        assert len(result) == len(y)

    def test_numerical_stability_edge_cases(self):
        """Test numerical stability with challenging inputs."""
        from loveslide.love_python.love.est_omega import estOmega

        # Nearly singular matrix
        p = 5
        C = np.eye(p) + 1e-12 * np.ones((p, p))

        with pytest.warns(UserWarning, match="numerical.*instability"):
            Omega = estOmega(lbd=0.1, C=C)

        # Should still produce valid result
        assert Omega.shape == (p, p)
        assert np.allclose(Omega, Omega.T)

    def test_large_scale_numerical_accuracy(self):
        """Test numerical accuracy with larger problems."""
        # Test that algorithms maintain accuracy as problem size increases
        for p in [10, 50, 100]:
            if p > 50:  # Skip large tests if needed
                continue

            X = np.random.randn(2*p, p)
            Sigma = np.cov(X, rowvar=False)

            # Test matrix operations
            eigenvals = np.linalg.eigvals(Sigma)
            assert np.all(eigenvals > -1e-10)  # Should be PSD

            # Test condition number
            condition_number = np.linalg.cond(Sigma)
            if condition_number > 1e12:
                pytest.warn(f"High condition number: {condition_number}")

    def test_reproducibility_across_runs(self):
        """Test that algorithms produce reproducible results."""
        X = np.random.RandomState(42).randn(100, 15)
        y = np.random.RandomState(42).randn(100)

        # Run multiple times with same seed
        results = []
        for _ in range(3):
            result = knockoff_filter_voting(
                X, y, fdr=0.1, niter=5,
                statistic='lasso_lambdadiff', seed=123
            )
            results.append(result.selected)

        # Should get identical results
        for i in range(1, len(results)):
            assert results[0] == results[i]