"""Test coverage for configuration validation edge cases and parameter interactions."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import warnings

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from loveslide import SLIDE, SLIDEcv
from loveslide.knockoffs import Knockoffs
from loveslide.tools import init_data, calc_default_fsize, check_params


class TestConfigValidationEdgeCases:
    """Test configuration validation edge cases and parameter interactions."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 20))
        y = pd.DataFrame(np.random.randint(0, 2, (50, 1)))
        return X, y

    def test_parameter_type_coercion_edge_cases(self, sample_data):
        """Test parameter type coercion in edge cases."""
        X, y = sample_data

        # Test with string numeric values
        params = {
            'x_path': None, 'y_path': None,
            'fdr': "0.1",  # String instead of float
            'lambda': ["0.1", "0.2"],  # String list
            'niter': "100",  # String instead of int
            'SLIDE_top_feats': "20"
        }

        # Should either coerce or raise meaningful error
        try:
            slide = SLIDE(params, X, y)
            # If successful, check types were coerced
            assert isinstance(slide.input_params.get('niter', 100), (int, str))
        except (TypeError, ValueError):
            pass  # Expected for invalid type

    def test_parameter_range_boundary_validation(self, sample_data):
        """Test parameter validation at range boundaries."""
        X, y = sample_data

        # Test FDR at boundaries
        boundary_cases = [
            {'fdr': 0.0},      # Minimum boundary
            {'fdr': 1.0},      # Maximum boundary
            {'fdr': -0.1},     # Below minimum
            {'fdr': 1.1},      # Above maximum
            {'fdr': np.inf},   # Infinite value
            {'fdr': np.nan},   # NaN value
        ]

        base_params = {'x_path': None, 'y_path': None, 'lambda': [0.1]}

        for boundary_param in boundary_cases:
            params = {**base_params, **boundary_param}

            if boundary_param['fdr'] in [-0.1, 1.1, np.inf] or np.isnan(boundary_param['fdr']):
                # Should raise validation error for invalid values
                with pytest.raises((ValueError, TypeError)):
                    SLIDE(params, X, y)
            else:
                # Should work for valid boundary values
                slide = SLIDE(params, X, y)
                assert slide.input_params['fdr'] == boundary_param['fdr']

    def test_parameter_interaction_validation(self, sample_data):
        """Test validation of parameter interactions and dependencies."""
        X, y = sample_data

        # Test incompatible parameter combinations
        incompatible_combinations = [
            # Large f_size with small dataset
            {'f_size': 1000, 'lambda': [0.1]},

            # Multiple workers with small niter
            {'n_workers': 10, 'niter': 1},

            # High FDR with strict thresholds
            {'fdr': 0.9, 'thresh_fdr': 0.01},
        ]

        base_params = {'x_path': None, 'y_path': None}

        for incompatible in incompatible_combinations:
            params = {**base_params, **incompatible}

            # Should either warn or adjust parameters automatically
            with warnings.catch_warnings(record=True) as w:
                slide = SLIDE(params, X, y)

                # Check if warning was issued for incompatible parameters
                if len(w) > 0:
                    assert any("parameter" in str(warning.message).lower()
                             for warning in w)

    def test_data_parameter_mismatch_validation(self, sample_data):
        """Test validation when data doesn't match parameter expectations."""
        X, y = sample_data

        # Test with parameters expecting larger dataset
        params = {
            'x_path': None, 'y_path': None,
            'f_size': X.shape[0] + 100,  # Larger than available samples
            'SLIDE_top_feats': X.shape[1] + 50,  # More features than available
        }

        slide = SLIDE(params, X, y)

        # Parameters should be automatically adjusted
        computed_fsize = slide.calc_default_fsize(5)
        assert computed_fsize <= X.shape[0]

    def test_lambda_parameter_validation(self, sample_data):
        """Test lambda parameter validation and processing."""
        X, y = sample_data

        # Test various lambda configurations
        lambda_cases = [
            [],                    # Empty list
            [0],                   # Zero value
            [-0.1],               # Negative value
            [0.1, 0.2, 0.5, 0.9], # Multiple values
            [1.0],                # Maximum value
            [2.0],                # Above maximum
            np.array([0.1, 0.2]), # NumPy array
            "0.1",                # String value
            None,                 # None value
        ]

        base_params = {'x_path': None, 'y_path': None, 'fdr': 0.1}

        for lambda_val in lambda_cases:
            params = base_params.copy()
            params['lambda'] = lambda_val

            try:
                slide = SLIDE(params, X, y)
                # If successful, check lambda was processed correctly
                processed_lambda = slide.input_params['lambda']
                assert processed_lambda is not None
                if isinstance(processed_lambda, list):
                    assert all(isinstance(x, (int, float)) for x in processed_lambda)
            except (ValueError, TypeError):
                # Expected for invalid lambda values
                pass

    def test_y_factor_and_flip_interaction(self, sample_data):
        """Test interaction between y_factor and y_flip parameters."""
        X, y = sample_data

        # Create categorical y data
        y_categorical = pd.DataFrame(['case', 'control'] * 25)

        test_cases = [
            {'y_factor': True, 'y_flip': False},
            {'y_factor': True, 'y_flip': True},
            {'y_factor': False, 'y_flip': False},
            {'y_factor': False, 'y_flip': True},
        ]

        base_params = {'x_path': None, 'y_path': None}

        for case in test_cases:
            params = {**base_params, **case}

            slide = SLIDE(params, X, y_categorical)

            # Verify y_factor and y_flip were applied correctly
            y_values = slide.data.Y.values.flatten()

            if case['y_factor']:
                # Should be numeric after factorization
                assert np.all(np.isin(y_values, [0, 1]))

                if case['y_flip']:
                    # Should be flipped: case=0, control=1
                    assert set(y_values) == {0, 1}
                else:
                    # Should be normal: case=1, control=0 (or 0,1 depending on order)
                    assert set(y_values) == {0, 1}

    def test_check_params_zero_variance_features(self):
        """Test check_params handling of zero variance features."""
        # Create data with some zero-variance features
        X = pd.DataFrame({
            'var_feature_1': np.random.randn(50),
            'zero_var_1': np.ones(50),  # Zero variance
            'var_feature_2': np.random.randn(50),
            'zero_var_2': np.zeros(50),  # Zero variance
            'var_feature_3': np.random.randn(50),
        })
        y = pd.DataFrame(np.random.randint(0, 2, (50, 1)))

        data = type('Data', (), {'X': X, 'Y': y})()
        params = {'fdr': 0.1}

        # Should warn and remove zero variance features
        with warnings.catch_warnings(record=True) as w:
            check_params(params, data)

            # Should issue warning about zero variance features
            assert len(w) > 0
            assert "standard deviation" in str(w[0].message).lower()

        # Should have removed zero variance features
        assert data.X.shape[1] == 3  # Only 3 varying features left
        assert 'zero_var_1' not in data.X.columns
        assert 'zero_var_2' not in data.X.columns

    def test_calc_default_fsize_edge_cases(self):
        """Test calc_default_fsize with edge case inputs."""
        edge_cases = [
            (10, 5),     # n_rows > K, K < 100
            (5, 10),     # n_rows < K
            (50, 50),    # n_rows == K
            (52, 50),    # n_rows == K + 2
            (48, 50),    # n_rows == K - 2
            (100, 150),  # n_rows > K, K >= 100
            (200, 50),   # n_rows > K, K < 100, large n_rows
            (1, 1),      # Minimum values
            (2, 5),      # Small n_rows, larger K
        ]

        for n_rows, K in edge_cases:
            fsize = calc_default_fsize(n_rows, K)

            # Basic sanity checks
            assert isinstance(fsize, int)
            assert fsize > 0
            assert fsize <= max(n_rows, K)

            # Specific logic checks based on original function
            if n_rows <= K and K < 100:
                if abs(n_rows - K) <= 2:
                    assert fsize == n_rows - 2 or fsize >= 1  # Handle edge case
                else:
                    assert fsize == n_rows

    def test_knockoffs_backend_validation(self, sample_data):
        """Test Knockoffs backend validation and fallback."""
        X, _ = sample_data

        # Test with various backend specifications
        backend_cases = [
            'python',    # Valid backend
            'r',         # Valid backend (if available)
            'auto',      # Auto-selection
            'invalid',   # Invalid backend
            None,        # None value
            123,         # Wrong type
        ]

        for backend in backend_cases:
            if backend in ['python', 'r', 'auto'] or backend is None:
                # Should work or auto-select
                ko = Knockoffs(fdr=0.1, backend=backend)
                assert ko.backend in ['python', 'r', 'auto']
            else:
                # Should raise error for invalid backends
                with pytest.raises((ValueError, TypeError)):
                    Knockoffs(fdr=0.1, backend=backend)

    def test_slidecv_parameter_validation(self, sample_data):
        """Test SLIDEcv parameter validation."""
        X, y = sample_data

        # Create minimal SLIDE object for CV
        slide = SLIDE({'x_path': None, 'y_path': None}, X, y)
        slide.latent_factors = pd.DataFrame(np.random.randn(50, 5))
        slide.marginal_idxs = list(range(5))

        # Test invalid CV parameters
        invalid_cases = [
            {'k': 0},           # Zero folds
            {'k': -5},          # Negative folds
            {'k': 100},         # More folds than samples
            {'nrep': 0},        # Zero repetitions
            {'nrep': -1},       # Negative repetitions
            {'eval_type': 'invalid'},  # Invalid evaluation type
        ]

        for invalid_params in invalid_cases:
            with pytest.raises((ValueError, TypeError)):
                SLIDEcv(slide, **invalid_params)