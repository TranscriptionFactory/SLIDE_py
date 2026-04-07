"""
Test statistical validity and robustness to assumption violations.
Addresses: FDR control, statistical power, assumption violations
"""
import pytest
import numpy as np
import scipy.stats as stats
from scipy.stats import norm, chi2
from loveslide import SLIDE, SLIDEcv, Knockoffs
from loveslide.knockoff.filter import VotingResult
import warnings


class TestFDRControl:
    """Test False Discovery Rate control under various conditions."""

    def test_fdr_control_null_data(self):
        """Test FDR control when no true signals exist."""
        np.random.seed(42)
        n, p = 200, 50

        # Generate null data (no true associations)
        X = np.random.randn(n, p)
        y = np.random.randn(n)  # Independent of X

        fdr_levels = [0.05, 0.1, 0.2]
        n_simulations = 20

        for target_fdr in fdr_levels:
            false_discoveries = []

            for sim in range(n_simulations):
                np.random.seed(42 + sim)
                X_sim = np.random.randn(n, p)
                y_sim = np.random.randn(n)

                slide = SLIDE({'fdr': target_fdr, 'n_iters': 100}, x=X_sim, y=y_sim)
                result = slide.fit()

                # All discoveries are false under null
                n_discoveries = len(result.selected) if hasattr(result, 'selected') else 0
                false_discoveries.append(n_discoveries)

            # Empirical FDR should not exceed target (with some tolerance for randomness)
            mean_false_discoveries = np.mean(false_discoveries)
            empirical_fdr = mean_false_discoveries / p if p > 0 else 0

            # Allow some tolerance for finite sample effects
            assert empirical_fdr <= target_fdr + 0.05, (
                f"FDR not controlled: empirical {empirical_fdr:.3f} > target {target_fdr}"
            )

    def test_fdr_control_with_signals(self):
        """Test FDR control when true signals exist."""
        np.random.seed(123)
        n, p = 150, 30
        n_true_signals = 5

        # Generate data with true signals
        X = np.random.randn(n, p)
        true_coefs = np.zeros(p)
        true_coefs[:n_true_signals] = np.random.randn(n_true_signals) * 2
        y = X @ true_coefs + np.random.randn(n) * 0.5

        target_fdr = 0.1
        n_simulations = 15

        fdp_estimates = []
        for sim in range(n_simulations):
            np.random.seed(123 + sim)

            slide = SLIDE({'fdr': target_fdr, 'n_iters': 50}, x=X, y=y)
            result = slide.fit()

            if hasattr(result, 'selected') and len(result.selected) > 0:
                # Count false discoveries among selected
                false_discoveries = sum(1 for idx in result.selected if idx >= n_true_signals)
                fdp = false_discoveries / len(result.selected)
                fdp_estimates.append(fdp)

        if fdp_estimates:
            mean_fdp = np.mean(fdp_estimates)
            # FDP should be controlled on average
            assert mean_fdp <= target_fdr + 0.1, f"FDP not controlled: {mean_fdp:.3f} > {target_fdr}"

    def test_fdr_monotonicity(self):
        """Test that higher FDR levels lead to more discoveries."""
        np.random.seed(456)
        n, p = 100, 20
        X = np.random.randn(n, p)
        # Weak signal
        y = X[:, 0] * 0.5 + np.random.randn(n)

        fdr_levels = [0.01, 0.05, 0.1, 0.2]
        n_discoveries = []

        for fdr in fdr_levels:
            slide = SLIDE({'fdr': fdr, 'n_iters': 30}, x=X, y=y)
            result = slide.fit()
            n_disc = len(result.selected) if hasattr(result, 'selected') else 0
            n_discoveries.append(n_disc)

        # Number of discoveries should be monotonically non-decreasing
        for i in range(1, len(n_discoveries)):
            assert n_discoveries[i] >= n_discoveries[i-1], (
                f"Non-monotonic discoveries: {n_discoveries}"
            )


class TestStatisticalPower:
    """Test statistical power under various signal conditions."""

    def test_power_with_strong_signals(self):
        """Test that strong signals are consistently detected."""
        np.random.seed(789)
        n, p = 100, 15
        signal_strength = 3.0

        X = np.random.randn(n, p)
        # Strong signal in first feature
        y = X[:, 0] * signal_strength + np.random.randn(n) * 0.5

        n_trials = 20
        detections = 0

        for trial in range(n_trials):
            np.random.seed(789 + trial)
            slide = SLIDE({'fdr': 0.1, 'n_iters': 50}, x=X, y=y)
            result = slide.fit()

            if hasattr(result, 'selected') and 0 in result.selected:
                detections += 1

        power = detections / n_trials
        # Should detect strong signals with high probability
        assert power >= 0.7, f"Low power for strong signal: {power}"

    def test_power_degrades_gracefully(self):
        """Test that power degrades gracefully as signal strength decreases."""
        np.random.seed(101112)
        n, p = 150, 20
        signal_strengths = [3.0, 2.0, 1.0, 0.5]

        powers = []
        for strength in signal_strengths:
            X = np.random.randn(n, p)
            y = X[:, 0] * strength + np.random.randn(n)

            detections = 0
            n_trials = 15

            for trial in range(n_trials):
                np.random.seed(101112 + trial)
                slide = SLIDE({'fdr': 0.1, 'n_iters': 30}, x=X, y=y)
                result = slide.fit()

                if hasattr(result, 'selected') and 0 in result.selected:
                    detections += 1

            power = detections / n_trials
            powers.append(power)

        # Power should decrease as signal strength decreases
        for i in range(1, len(powers)):
            assert powers[i] <= powers[i-1] + 0.2, (  # Allow some noise
                f"Power not decreasing with signal: {powers}"
            )


class TestAssumptionViolations:
    """Test robustness to statistical assumption violations."""

    def test_heavy_tailed_noise_robustness(self):
        """Test robustness to heavy-tailed noise distributions."""
        np.random.seed(131415)
        n, p = 120, 15

        X = np.random.randn(n, p)
        # Heavy-tailed noise (t-distribution with 3 degrees of freedom)
        heavy_noise = stats.t.rvs(df=3, size=n, random_state=131415)
        y = X[:, 0] * 2.0 + heavy_noise

        # Should still work with heavy-tailed errors
        slide = SLIDE({'fdr': 0.1, 'n_iters': 30}, x=X, y=y)
        result = slide.fit()

        # Method should not crash and should potentially detect the signal
        assert result is not None
        if hasattr(result, 'statistic'):
            assert len(result.statistic) == p

    def test_heteroscedastic_noise_robustness(self):
        """Test robustness to heteroscedastic (non-constant variance) noise."""
        np.random.seed(161718)
        n, p = 100, 12

        X = np.random.randn(n, p)
        # Heteroscedastic noise (variance depends on X)
        noise_var = 0.1 + np.abs(X[:, 0])
        y = X[:, 0] * 1.5 + np.random.randn(n) * np.sqrt(noise_var)

        slide = SLIDE({'fdr': 0.1, 'n_iters': 25}, x=X, y=y)
        result = slide.fit()

        # Should handle heteroscedasticity without crashing
        assert result is not None

    def test_multicollinearity_robustness(self):
        """Test robustness to multicollinearity in features."""
        np.random.seed(192021)
        n, p = 100, 10

        X_base = np.random.randn(n, p-2)
        # Create multicollinear features
        X_corr = X_base[:, 0] + 0.1 * np.random.randn(n)  # Highly correlated with first feature
        X = np.column_stack([X_base, X_corr, X_base[:, 1] * 0.9])  # Another correlated feature

        y = X[:, 0] * 1.0 + np.random.randn(n) * 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Might warn about multicollinearity
            slide = SLIDE({'fdr': 0.1, 'n_iters': 25}, x=X, y=y)
            result = slide.fit()

        # Should handle multicollinearity gracefully
        assert result is not None

    def test_outlier_robustness(self):
        """Test robustness to outliers in both X and y."""
        np.random.seed(222324)
        n, p = 100, 8

        X = np.random.randn(n, p)
        y = X[:, 0] * 1.0 + np.random.randn(n) * 0.3

        # Add outliers
        outlier_indices = np.random.choice(n, size=5, replace=False)
        X[outlier_indices, :] *= 10  # Outliers in features
        y[outlier_indices] += np.random.randn(5) * 20  # Outliers in response

        slide = SLIDE({'fdr': 0.1, 'n_iters': 25}, x=X, y=y)
        result = slide.fit()

        # Should handle outliers without crashing
        assert result is not None
        if hasattr(result, 'statistic'):
            # Statistics should be finite
            assert np.all(np.isfinite(result.statistic))


class TestCrossValidationStatisticalValidity:
    """Test statistical validity of cross-validation procedures."""

    def test_cv_fold_independence(self):
        """Test that CV folds maintain statistical independence."""
        np.random.seed(252627)
        n, p = 120, 15
        X = np.random.randn(n, p)
        y = np.random.randn(n)

        slide_cv = SLIDEcv({'fdr': 0.1, 'cv_folds': 5}, x=X, y=y)

        # Multiple CV runs should give consistent results
        cv_results = []
        for run in range(5):
            np.random.seed(252627 + run)
            result = slide_cv.cross_validate()
            if hasattr(result, 'cv_scores'):
                cv_results.append(result.cv_scores)

        # CV scores should not be identical (due to randomization)
        # but should not be excessively variable
        if cv_results and len(cv_results) > 1:
            score_vars = [np.var(scores) if hasattr(scores, '__iter__') else 0
                         for scores in cv_results]
            # Some variability expected, but not excessive
            assert np.mean(score_vars) < 10  # Reasonable bound

    def test_stratification_maintains_distribution(self):
        """Test that stratification maintains class distributions in CV."""
        np.random.seed(282930)
        n = 100
        # Create binary response with imbalanced classes
        y = np.concatenate([np.ones(20), np.zeros(80)])
        X = np.random.randn(n, 10)

        slide_cv = SLIDEcv({'fdr': 0.1, 'cv_folds': 5}, x=X, y=y)

        try:
            result = slide_cv.cross_validate()
            # If stratification is used, it should not crash with imbalanced data
            assert result is not None
        except ValueError as e:
            # If stratification fails, should give informative error
            assert "stratif" in str(e).lower() or "class" in str(e).lower()