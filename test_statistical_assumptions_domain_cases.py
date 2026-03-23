"""
Test skeleton for statistical assumptions and data science domain-specific edge cases.

Focus on testing behavior under various statistical scenarios that may
occur in real-world data science applications.
"""
import pytest
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_classification, make_regression
from unittest.mock import patch
from typing import Dict, List, Tuple, Optional

from loveslide import SLIDE, SLIDEcv, Knockoffs, SLIDE_Estimator
from loveslide.knockoff.filter import knockoff_filter_voting
from loveslide.love import call_love
from loveslide.knockoff.utils import is_posdef, cov2cor


class TestStatisticalAssumptions:
    """Test behavior under various statistical assumption violations."""

    def test_non_gaussian_data_distributions(self):
        """Test performance with non-Gaussian data distributions."""
        n, p = 100, 20

        distributions = [
            # Heavy-tailed distributions
            {'name': 'student_t', 'generator': lambda: stats.t.rvs(df=3, size=(n, p))},
            {'name': 'laplace', 'generator': lambda: stats.laplace.rvs(size=(n, p))},

            # Skewed distributions
            {'name': 'exponential', 'generator': lambda: stats.expon.rvs(size=(n, p))},
            {'name': 'log_normal', 'generator': lambda: stats.lognorm.rvs(s=0.5, size=(n, p))},

            # Discrete/mixed distributions
            {'name': 'poisson', 'generator': lambda: stats.poisson.rvs(mu=2, size=(n, p)).astype(float)},
            {'name': 'uniform', 'generator': lambda: stats.uniform.rvs(size=(n, p))},

            # Multimodal
            {'name': 'bimodal', 'generator': lambda: np.where(
                np.random.rand(n, p) < 0.5,
                stats.norm.rvs(loc=-2, scale=0.5, size=(n, p)),
                stats.norm.rvs(loc=2, scale=0.5, size=(n, p))
            )},
        ]

        for dist in distributions:
            X = dist['generator']()
            y = np.random.randn(n)  # Keep response Gaussian for now

            # Test SLIDE behavior with different distributions
            slide = SLIDE(X, y, fdr=0.1)

            try:
                result = slide.select()

                # Basic validation - shouldn't crash
                if result is not None and hasattr(result, 'selections'):
                    assert len(result.selections) <= p

                # Distribution-specific checks
                if dist['name'] in ['student_t', 'laplace']:
                    # Heavy-tailed: might be more conservative
                    pass
                elif dist['name'] in ['exponential', 'log_normal']:
                    # Skewed: might affect correlation structure
                    pass
                elif dist['name'] == 'poisson':
                    # Discrete: might violate continuous assumptions
                    pass

            except Exception as e:
                # Some methods might not handle non-Gaussian well
                if "assumption" in str(e).lower() or "gaussian" in str(e).lower():
                    pytest.skip(f"Method doesn't support {dist['name']} distribution")
                else:
                    pytest.fail(f"Unexpected error with {dist['name']}: {e}")

    def test_correlation_structure_patterns(self):
        """Test with different correlation structures found in real data."""
        n, p = 150, 30

        correlation_patterns = [
            # Block diagonal structure
            {
                'name': 'block_diagonal',
                'generator': lambda: self._generate_block_diagonal_data(n, p, n_blocks=3)
            },

            # Autoregressive structure AR(1)
            {
                'name': 'ar1',
                'generator': lambda: self._generate_ar1_data(n, p, rho=0.7)
            },

            # Compound symmetry
            {
                'name': 'compound_symmetry',
                'generator': lambda: self._generate_compound_symmetry_data(n, p, rho=0.5)
            },

            # Toeplitz structure
            {
                'name': 'toeplitz',
                'generator': lambda: self._generate_toeplitz_data(n, p, decay=0.8)
            },

            # Factor model structure
            {
                'name': 'factor_model',
                'generator': lambda: self._generate_factor_model_data(n, p, n_factors=3)
            },

            # Sparse precision matrix
            {
                'name': 'sparse_precision',
                'generator': lambda: self._generate_sparse_precision_data(n, p, sparsity=0.9)
            },
        ]

        for pattern in correlation_patterns:
            X, true_structure = pattern['generator']()
            y = np.random.randn(n)

            # Test different methods with structured correlation
            methods_to_test = ['equicorrelated', 'sdp']

            for method in methods_to_test:
                slide = SLIDE(X, y, fdr=0.1, method=method)

                try:
                    result = slide.select()

                    if result is not None and hasattr(result, 'selections'):
                        # Validate that method respects correlation structure
                        # TODO: Add structure-specific validation
                        if pattern['name'] == 'block_diagonal':
                            # Should tend to select from same blocks
                            pass
                        elif pattern['name'] == 'ar1':
                            # Should respect temporal/sequential correlation
                            pass

                except Exception as e:
                    if "singular" in str(e).lower() or "conditioning" in str(e).lower():
                        # Some structures might cause numerical issues
                        continue
                    else:
                        pytest.fail(f"Error with {pattern['name']}, method {method}: {e}")

    def test_missing_data_patterns(self):
        """Test behavior with different missing data patterns."""
        n, p = 200, 25
        X_complete = np.random.randn(n, p)
        y = np.random.randn(n)

        missing_patterns = [
            # Missing Completely At Random (MCAR)
            {
                'name': 'MCAR',
                'missing_rate': 0.1,
                'generator': lambda X: self._create_mcar_missing(X, 0.1)
            },

            # Missing At Random (MAR) - depends on observed variables
            {
                'name': 'MAR',
                'missing_rate': 0.15,
                'generator': lambda X: self._create_mar_missing(X, 0.15)
            },

            # Missing Not At Random (MNAR) - depends on missing values themselves
            {
                'name': 'MNAR',
                'missing_rate': 0.1,
                'generator': lambda X: self._create_mnar_missing(X, 0.1)
            },

            # Monotone missing pattern
            {
                'name': 'monotone',
                'missing_rate': 0.2,
                'generator': lambda X: self._create_monotone_missing(X, 0.2)
            },
        ]

        for pattern in missing_patterns:
            X_missing = pattern['generator'](X_complete.copy())

            # Test different imputation strategies
            imputation_methods = ['complete_case', 'mean_impute', 'median_impute']

            for impute_method in imputation_methods:
                try:
                    X_processed = self._handle_missing_data(X_missing, method=impute_method)

                    slide = SLIDE(X_processed, y, fdr=0.1)
                    result = slide.select()

                    # Validate that missing data handling didn't break the analysis
                    if result is not None and hasattr(result, 'selections'):
                        assert len(result.selections) <= X_processed.shape[1]

                        # Pattern-specific checks
                        if pattern['name'] == 'MCAR' and impute_method == 'complete_case':
                            # Complete case analysis should be unbiased for MCAR
                            pass

                except Exception as e:
                    if "insufficient" in str(e).lower() and impute_method == 'complete_case':
                        # Complete case might not have enough data
                        continue
                    else:
                        pytest.fail(f"Error with {pattern['name']}, {impute_method}: {e}")

    def test_collinearity_and_multicollinearity(self):
        """Test behavior under various collinearity scenarios."""
        n = 100

        collinearity_scenarios = [
            # Perfect collinearity
            {
                'name': 'perfect_collinear',
                'generator': lambda: self._create_perfect_collinearity(n, base_p=10, n_collinear=2)
            },

            # High multicollinearity
            {
                'name': 'high_multicollinear',
                'generator': lambda: self._create_high_multicollinearity(n, p=15, condition_number=1e6)
            },

            # Moderate multicollinearity
            {
                'name': 'moderate_multicollinear',
                'generator': lambda: self._create_moderate_multicollinearity(n, p=20, min_eigenvalue=1e-3)
            },

            # Near-collinear variables
            {
                'name': 'near_collinear',
                'generator': lambda: self._create_near_collinearity(n, p=12, correlation=0.99)
            },
        ]

        for scenario in collinearity_scenarios:
            X, collinear_info = scenario['generator']()
            y = np.random.randn(n)

            # Test how different methods handle collinearity
            methods = ['equicorrelated', 'sdp']

            for method in methods:
                slide = SLIDE(X, y, fdr=0.1, method=method)

                try:
                    result = slide.select()

                    # Validate collinearity handling
                    if result is not None and hasattr(result, 'selections'):
                        if scenario['name'] == 'perfect_collinear':
                            # Should not select all collinear variables
                            # TODO: Check against known collinear pairs
                            pass

                except Exception as e:
                    if "singular" in str(e).lower() or "collinear" in str(e).lower():
                        # Expected for some collinearity scenarios
                        continue
                    else:
                        pytest.fail(f"Unexpected error with {scenario['name']}, {method}: {e}")

    def test_outliers_and_leverage_points(self):
        """Test robustness to outliers and high-leverage points."""
        n, p = 100, 15

        # Generate clean base data
        X_clean = np.random.randn(n, p)
        y_clean = np.random.randn(n)

        outlier_scenarios = [
            # Vertical outliers (in y)
            {
                'name': 'y_outliers',
                'generator': lambda: self._add_y_outliers(X_clean, y_clean, n_outliers=5, magnitude=5.0)
            },

            # Leverage points (extreme X values)
            {
                'name': 'leverage_points',
                'generator': lambda: self._add_leverage_points(X_clean, y_clean, n_points=3, magnitude=4.0)
            },

            # Influential points (both extreme X and y)
            {
                'name': 'influential_points',
                'generator': lambda: self._add_influential_points(X_clean, y_clean, n_points=3, x_mag=3.0, y_mag=4.0)
            },

            # Clustered outliers
            {
                'name': 'clustered_outliers',
                'generator': lambda: self._add_clustered_outliers(X_clean, y_clean, cluster_size=5, magnitude=3.0)
            },
        ]

        for scenario in outlier_scenarios:
            X_outliers, y_outliers, outlier_indices = scenario['generator']()

            slide = SLIDE(X_outliers, y_outliers, fdr=0.1)

            try:
                result = slide.select()

                # Compare with clean data analysis
                slide_clean = SLIDE(X_clean, y_clean, fdr=0.1)
                result_clean = slide_clean.select()

                if result is not None and result_clean is not None:
                    if hasattr(result, 'selections') and hasattr(result_clean, 'selections'):
                        # Calculate selection stability
                        jaccard_similarity = self._jaccard_similarity(
                            result.selections, result_clean.selections
                        )

                        # Results should be somewhat robust to outliers
                        if jaccard_similarity < 0.3:  # Less than 30% overlap might indicate instability
                            # TODO: Decide if this is a failure or expected behavior
                            print(f"Low stability with {scenario['name']}: Jaccard = {jaccard_similarity:.3f}")

            except Exception as e:
                if "numerical" in str(e).lower() or "conditioning" in str(e).lower():
                    # Outliers might cause numerical issues
                    continue
                else:
                    pytest.fail(f"Error with {scenario['name']}: {e}")

    # Helper methods for generating test data
    def _generate_block_diagonal_data(self, n: int, p: int, n_blocks: int) -> Tuple[np.ndarray, Dict]:
        """Generate data with block diagonal correlation structure."""
        block_size = p // n_blocks
        blocks = []

        for i in range(n_blocks):
            # Generate correlated block
            block_corr = np.random.rand() * 0.7 + 0.2  # Correlation between 0.2 and 0.9
            block_data = np.random.randn(n, block_size)

            # Add within-block correlation
            for j in range(1, block_size):
                block_data[:, j] = block_corr * block_data[:, 0] + np.sqrt(1 - block_corr**2) * block_data[:, j]

            blocks.append(block_data)

        # Handle remaining columns
        remaining = p - n_blocks * block_size
        if remaining > 0:
            blocks.append(np.random.randn(n, remaining))

        X = np.hstack(blocks)
        return X, {'n_blocks': n_blocks, 'block_size': block_size}

    def _generate_ar1_data(self, n: int, p: int, rho: float) -> Tuple[np.ndarray, Dict]:
        """Generate data with AR(1) correlation structure."""
        # Create AR(1) correlation matrix
        corr_matrix = np.array([[rho**abs(i-j) for j in range(p)] for i in range(p)])

        # Generate data
        X = np.random.multivariate_normal(np.zeros(p), corr_matrix, n)
        return X, {'rho': rho, 'structure': 'AR1'}

    def _generate_compound_symmetry_data(self, n: int, p: int, rho: float) -> Tuple[np.ndarray, Dict]:
        """Generate data with compound symmetry correlation."""
        corr_matrix = (1 - rho) * np.eye(p) + rho * np.ones((p, p))
        X = np.random.multivariate_normal(np.zeros(p), corr_matrix, n)
        return X, {'rho': rho, 'structure': 'compound_symmetry'}

    def _generate_toeplitz_data(self, n: int, p: int, decay: float) -> Tuple[np.ndarray, Dict]:
        """Generate data with Toeplitz correlation structure."""
        corr_matrix = np.array([[decay**abs(i-j) for j in range(p)] for i in range(p)])
        X = np.random.multivariate_normal(np.zeros(p), corr_matrix, n)
        return X, {'decay': decay, 'structure': 'toeplitz'}

    def _generate_factor_model_data(self, n: int, p: int, n_factors: int) -> Tuple[np.ndarray, Dict]:
        """Generate data from a factor model."""
        # Generate factors
        factors = np.random.randn(n, n_factors)

        # Generate loadings
        loadings = np.random.randn(p, n_factors)

        # Generate specific factors (noise)
        specific_factors = np.random.randn(n, p) * 0.5

        # Combine
        X = factors @ loadings.T + specific_factors
        return X, {'n_factors': n_factors, 'structure': 'factor_model'}

    def _generate_sparse_precision_data(self, n: int, p: int, sparsity: float) -> Tuple[np.ndarray, Dict]:
        """Generate data with sparse precision matrix."""
        # Create sparse precision matrix
        precision = np.eye(p)

        # Add random sparse connections
        n_connections = int((1 - sparsity) * p * (p - 1) / 2)
        for _ in range(n_connections):
            i, j = np.random.choice(p, 2, replace=False)
            value = np.random.randn() * 0.3
            precision[i, j] = precision[j, i] = value

        # Ensure positive definiteness
        precision += (abs(np.linalg.eigvals(precision).min()) + 0.1) * np.eye(p)

        # Generate data
        cov_matrix = np.linalg.inv(precision)
        X = np.random.multivariate_normal(np.zeros(p), cov_matrix, n)
        return X, {'sparsity': sparsity, 'structure': 'sparse_precision'}

    # Helper methods for missing data
    def _create_mcar_missing(self, X: np.ndarray, rate: float) -> np.ndarray:
        """Create Missing Completely At Random pattern."""
        X_missing = X.copy()
        mask = np.random.rand(*X.shape) < rate
        X_missing[mask] = np.nan
        return X_missing

    def _create_mar_missing(self, X: np.ndarray, rate: float) -> np.ndarray:
        """Create Missing At Random pattern."""
        X_missing = X.copy()
        n, p = X.shape

        # Missing depends on first column
        first_col_high = X[:, 0] > np.median(X[:, 0])

        for j in range(1, p):
            # Higher probability of missing if first column is high
            prob_missing = np.where(first_col_high, rate * 1.5, rate * 0.5)
            mask = np.random.rand(n) < prob_missing
            X_missing[mask, j] = np.nan

        return X_missing

    def _create_mnar_missing(self, X: np.ndarray, rate: float) -> np.ndarray:
        """Create Missing Not At Random pattern."""
        X_missing = X.copy()

        # Missing depends on the value itself (high values more likely to be missing)
        for j in range(X.shape[1]):
            high_values = X[:, j] > np.percentile(X[:, j], 75)
            prob_missing = np.where(high_values, rate * 2, rate * 0.5)
            mask = np.random.rand(X.shape[0]) < prob_missing
            X_missing[mask, j] = np.nan

        return X_missing

    def _create_monotone_missing(self, X: np.ndarray, rate: float) -> np.ndarray:
        """Create monotone missing pattern."""
        X_missing = X.copy()
        n, p = X.shape

        # Progressive dropout
        for j in range(p):
            dropout_rate = rate * (j + 1) / p
            n_dropout = int(dropout_rate * n)
            dropout_indices = np.random.choice(n, n_dropout, replace=False)
            X_missing[dropout_indices, j:] = np.nan

        return X_missing

    def _handle_missing_data(self, X: np.ndarray, method: str) -> np.ndarray:
        """Handle missing data with specified method."""
        if method == 'complete_case':
            # Remove rows with any missing values
            complete_rows = ~np.isnan(X).any(axis=1)
            return X[complete_rows]

        elif method == 'mean_impute':
            X_imputed = X.copy()
            for j in range(X.shape[1]):
                mask = np.isnan(X[:, j])
                if mask.any():
                    X_imputed[mask, j] = np.nanmean(X[:, j])
            return X_imputed

        elif method == 'median_impute':
            X_imputed = X.copy()
            for j in range(X.shape[1]):
                mask = np.isnan(X[:, j])
                if mask.any():
                    X_imputed[mask, j] = np.nanmedian(X[:, j])
            return X_imputed

        else:
            raise ValueError(f"Unknown imputation method: {method}")

    # Helper methods for collinearity
    def _create_perfect_collinearity(self, n: int, base_p: int, n_collinear: int) -> Tuple[np.ndarray, Dict]:
        """Create data with perfect collinearity."""
        X_base = np.random.randn(n, base_p)

        # Add perfectly collinear columns
        collinear_cols = []
        for i in range(n_collinear):
            # Linear combination of existing columns
            weights = np.random.randn(base_p)
            collinear_col = X_base @ weights
            collinear_cols.append(collinear_col.reshape(-1, 1))

        X = np.hstack([X_base] + collinear_cols)
        return X, {'base_p': base_p, 'n_collinear': n_collinear}

    def _create_high_multicollinearity(self, n: int, p: int, condition_number: float) -> Tuple[np.ndarray, Dict]:
        """Create data with high multicollinearity."""
        # Generate random matrix
        A = np.random.randn(p, p)
        U, _, Vt = np.linalg.svd(A)

        # Create singular values with desired condition number
        s = np.logspace(0, -np.log10(condition_number), p)
        cov_matrix = U @ np.diag(s) @ Vt

        X = np.random.multivariate_normal(np.zeros(p), cov_matrix, n)
        return X, {'condition_number': condition_number}

    def _create_moderate_multicollinearity(self, n: int, p: int, min_eigenvalue: float) -> Tuple[np.ndarray, Dict]:
        """Create data with moderate multicollinearity."""
        # Create correlation matrix with small minimum eigenvalue
        A = np.random.randn(p, p)
        corr_matrix = A @ A.T
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

        # Adjust eigenvalues
        eigenvalues[eigenvalues < min_eigenvalue] = min_eigenvalue
        corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Normalize to correlation matrix
        corr_matrix = cov2cor(corr_matrix)

        X = np.random.multivariate_normal(np.zeros(p), corr_matrix, n)
        return X, {'min_eigenvalue': min_eigenvalue}

    def _create_near_collinearity(self, n: int, p: int, correlation: float) -> Tuple[np.ndarray, Dict]:
        """Create data with near-collinear variables."""
        X = np.random.randn(n, p)

        # Make some pairs near-collinear
        n_pairs = p // 3
        for i in range(n_pairs):
            j = 2 * i + 1
            if j < p:
                # Make column j highly correlated with column j-1
                noise = np.sqrt(1 - correlation**2) * np.random.randn(n)
                X[:, j] = correlation * X[:, j-1] + noise

        return X, {'correlation': correlation, 'n_pairs': n_pairs}

    # Helper methods for outliers
    def _add_y_outliers(self, X: np.ndarray, y: np.ndarray, n_outliers: int, magnitude: float) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Add vertical outliers to y."""
        X_out, y_out = X.copy(), y.copy()
        outlier_indices = np.random.choice(len(y), n_outliers, replace=False)

        for idx in outlier_indices:
            y_out[idx] += magnitude * np.random.choice([-1, 1]) * np.std(y)

        return X_out, y_out, outlier_indices.tolist()

    def _add_leverage_points(self, X: np.ndarray, y: np.ndarray, n_points: int, magnitude: float) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Add high-leverage points."""
        X_out, y_out = X.copy(), y.copy()
        point_indices = np.random.choice(len(X), n_points, replace=False)

        for idx in point_indices:
            # Make extreme in X space
            X_out[idx] *= magnitude

        return X_out, y_out, point_indices.tolist()

    def _add_influential_points(self, X: np.ndarray, y: np.ndarray, n_points: int, x_mag: float, y_mag: float) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Add influential points (extreme in both X and y)."""
        X_out, y_out = X.copy(), y.copy()
        point_indices = np.random.choice(len(X), n_points, replace=False)

        for idx in point_indices:
            X_out[idx] *= x_mag
            y_out[idx] += y_mag * np.random.choice([-1, 1]) * np.std(y)

        return X_out, y_out, point_indices.tolist()

    def _add_clustered_outliers(self, X: np.ndarray, y: np.ndarray, cluster_size: int, magnitude: float) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Add a cluster of outliers."""
        X_out, y_out = X.copy(), y.copy()
        cluster_center = np.random.choice(len(X))
        cluster_indices = np.random.choice(len(X), cluster_size, replace=False)

        # Create outlier cluster around a central point
        outlier_direction = np.random.randn(X.shape[1])
        outlier_direction /= np.linalg.norm(outlier_direction)

        for idx in cluster_indices:
            X_out[idx] += magnitude * outlier_direction * np.std(X, axis=0)
            y_out[idx] += magnitude * np.random.choice([-1, 1]) * np.std(y)

        return X_out, y_out, cluster_indices.tolist()

    def _jaccard_similarity(self, set1: List, set2: List) -> float:
        """Calculate Jaccard similarity between two sets."""
        s1, s2 = set(set1), set(set2)
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        return intersection / union if union > 0 else 1.0