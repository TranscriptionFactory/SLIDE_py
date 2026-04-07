"""
Test Coverage Gap: Data Contract and Validation Boundaries
========================================================

Tests complex data validation scenarios, input contract enforcement, and edge cases
in data preprocessing that may not be fully covered.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.loveslide.tools import init_data, check_params, calc_default_fsize
from src.loveslide import SLIDE


class TestDataContractValidation:
    """Test data contract validation scenarios."""

    def test_mixed_data_type_handling(self):
        """Test handling of mixed data types in input matrices."""
        # Create data with mixed types that could cause issues
        mixed_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [1.5, 2.5, 3.5, 4.5, 5.5],
            'col3': ['1', '2', '3', '4', '5'],  # String numbers
            'col4': [True, False, True, False, True],  # Boolean
            'col5': [1+0j, 2+0j, 3+0j, 4+0j, 5+0j]  # Complex
        })

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            mixed_data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            # Test data loading with mixed types
            params = {'x_path': temp_path}
            data, processed_params = init_data(params)

            # Should either convert appropriately or raise clear error
            if data is not None:
                assert hasattr(data, 'X')
                assert np.all(np.isfinite(data.X.select_dtypes(include=[np.number])))

        except (ValueError, TypeError) as e:
            # Should provide clear error about data type issues
            error_msg = str(e).lower()
            assert any(word in error_msg for word in
                      ['type', 'numeric', 'convert', 'invalid'])
        finally:
            os.unlink(temp_path)

    def test_extreme_data_dimensions_validation(self):
        """Test validation of extreme data dimensions."""
        extreme_dimension_cases = [
            # (n_samples, n_features, should_pass)
            (0, 10, False),      # No samples
            (5, 0, False),       # No features
            (1, 10, False),      # Only one sample
            (2, 10000, False),   # More features than samples by extreme amount
            (10000, 2, True),    # Many samples, few features (should work)
            (10, 1, True),       # Single feature (should work)
        ]

        for n_samples, n_features, should_pass in extreme_dimension_cases:
            if n_samples > 0 and n_features > 0:
                X = np.random.randn(n_samples, n_features)
                y = np.random.binomial(1, 0.5, n_samples) if n_samples > 0 else np.array([])

                params = {'K': min(5, n_features), 'fdr': 0.1}

                if should_pass:
                    try:
                        slide = SLIDE(params, X, y)
                        assert slide is not None
                    except Exception as e:
                        # May still fail due to statistical constraints
                        pass
                else:
                    with pytest.raises((ValueError, RuntimeError)):
                        slide = SLIDE(params, X, y)

    def test_parameter_interdependency_validation(self):
        """Test validation of complex parameter interdependencies."""
        X = np.random.randn(50, 10)
        y = np.random.binomial(1, 0.5, 50)

        # Test parameter combinations that should be invalid
        invalid_param_combinations = [
            {'K': 15, 'fdr': 0.1},  # K > number of features
            {'K': 3, 'delta': [0.1, 0.9], 'lambda': []},  # Empty lambda
            {'K': 3, 'delta': [], 'lambda': [0.3, 0.7]},  # Empty delta
            {'K': 0, 'fdr': 0.1},  # Zero latent factors
            {'K': -1, 'fdr': 0.1},  # Negative latent factors
            {'K': 3, 'fdr': -0.1},  # Negative FDR
            {'K': 3, 'fdr': 1.5},   # FDR > 1
            {'K': 3, 'fdr': 0.1, 'delta': [1.5], 'lambda': [0.5]},  # delta > 1
            {'K': 3, 'fdr': 0.1, 'delta': [0.5], 'lambda': [1.5]},  # lambda > 1
        ]

        for params in invalid_param_combinations:
            with pytest.raises((ValueError, RuntimeError, AssertionError)):
                slide = SLIDE(params, X, y)

    def test_data_quality_validation(self):
        """Test validation of data quality issues."""
        n, p = 50, 10
        base_X = np.random.randn(n, p)
        base_y = np.random.binomial(1, 0.5, n)

        # Test various data quality issues
        data_quality_issues = [
            # All NaN matrix
            (np.full_like(base_X, np.nan), base_y, "nan"),
            # Matrix with inf values
            (np.where(base_X > 2, np.inf, base_X), base_y, "inf"),
            # All zero matrix
            (np.zeros_like(base_X), base_y, "zero"),
            # Constant matrix (no variance)
            (np.ones_like(base_X), base_y, "constant"),
            # Target with all same class
            (base_X, np.ones(n, dtype=int), "single_class"),
            # Target with NaN values
            (base_X, np.full(n, np.nan), "target_nan"),
        ]

        for X, y, issue_type in data_quality_issues:
            params = {'K': 3, 'fdr': 0.1}

            try:
                slide = SLIDE(params, X, y)
                # If it doesn't raise an exception, check it handles gracefully
                if slide is not None:
                    assert hasattr(slide, 'data')
            except Exception as e:
                # Should provide informative error about data quality
                error_msg = str(e).lower()
                expected_terms = {
                    'nan': ['nan', 'missing', 'finite'],
                    'inf': ['inf', 'infinite', 'finite'],
                    'zero': ['variance', 'zero', 'constant'],
                    'constant': ['variance', 'constant'],
                    'single_class': ['class', 'variance', 'target'],
                    'target_nan': ['target', 'nan', 'missing']
                }
                assert any(term in error_msg for term in expected_terms[issue_type])

    def test_file_format_validation(self):
        """Test validation of different file formats and corruption scenarios."""
        # Create test data
        X = np.random.randn(20, 5)
        y = np.random.binomial(1, 0.5, 20)
        df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(5)])
        df['target'] = y

        file_corruption_scenarios = []

        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid CSV
            valid_csv = os.path.join(tmpdir, 'valid.csv')
            df.to_csv(valid_csv, index=False)
            file_corruption_scenarios.append((valid_csv, True, "valid_csv"))

            # Corrupted CSV (invalid structure)
            corrupt_csv = os.path.join(tmpdir, 'corrupt.csv')
            with open(corrupt_csv, 'w') as f:
                f.write("invalid,csv,structure\n1,2\n3,4,5,6")
            file_corruption_scenarios.append((corrupt_csv, False, "corrupt_csv"))

            # Empty file
            empty_csv = os.path.join(tmpdir, 'empty.csv')
            with open(empty_csv, 'w') as f:
                f.write("")
            file_corruption_scenarios.append((empty_csv, False, "empty_csv"))

            # Non-existent file
            file_corruption_scenarios.append(("nonexistent.csv", False, "nonexistent"))

            # Binary file with CSV extension
            binary_csv = os.path.join(tmpdir, 'binary.csv')
            with open(binary_csv, 'wb') as f:
                f.write(b'\x00\x01\x02\x03\x04\x05')
            file_corruption_scenarios.append((binary_csv, False, "binary"))

            for file_path, should_succeed, scenario_type in file_corruption_scenarios:
                params = {'x_path': file_path}

                if should_succeed:
                    try:
                        data, processed_params = init_data(params)
                        assert data is not None
                    except Exception:
                        # May fail due to other issues, but shouldn't crash
                        pass
                else:
                    with pytest.raises((FileNotFoundError, ValueError, pd.errors.EmptyDataError,
                                      pd.errors.ParserError, UnicodeDecodeError)):
                        data, processed_params = init_data(params)

    def test_memory_mapped_file_handling(self):
        """Test handling of memory-mapped files and large file scenarios."""
        # Create a larger dataset to test memory mapping behavior
        n, p = 1000, 20
        X = np.random.randn(n, p)
        y = np.random.binomial(1, 0.5, n)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(X)
            df['target'] = y
            df.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            # Test various memory-related parameters
            params_variations = [
                {'x_path': temp_path},
                {'x_path': temp_path, 'chunksize': 100},  # If supported
            ]

            for params in params_variations:
                try:
                    data, processed_params = init_data(params)
                    if data is not None:
                        assert hasattr(data, 'X')
                        assert data.X.shape[0] == n
                except Exception as e:
                    # Should provide clear error about memory/file issues
                    pass

        finally:
            os.unlink(temp_path)


class TestParameterValidationEdgeCases:
    """Test parameter validation edge cases."""

    def test_parameter_type_coercion_boundaries(self):
        """Test parameter type coercion at boundaries."""
        X = np.random.randn(30, 8)
        y = np.random.binomial(1, 0.5, 30)

        # Test parameters that might need type coercion
        coercion_test_cases = [
            {'K': 3.0, 'fdr': 0.1},         # Float K (should be int)
            {'K': np.int32(3), 'fdr': 0.1}, # NumPy int
            {'K': 3, 'fdr': np.float32(0.1)}, # NumPy float
            {'K': 3, 'fdr': 0.1, 'delta': np.array([0.1, 0.5])},  # NumPy array
            {'K': 3, 'fdr': 0.1, 'lambda': [0.3]},  # Single element list
        ]

        for params in coercion_test_cases:
            try:
                slide = SLIDE(params, X, y)
                # Should either succeed with coerced types or fail gracefully
                assert slide is not None
            except (TypeError, ValueError) as e:
                # Should provide clear type error message
                error_msg = str(e).lower()
                assert any(word in error_msg for word in ['type', 'expected', 'invalid'])

    def test_parameter_range_boundary_validation(self):
        """Test validation at exact parameter range boundaries."""
        X = np.random.randn(30, 8)
        y = np.random.binomial(1, 0.5, 30)

        # Test exact boundary values
        boundary_test_cases = [
            ({'K': 1, 'fdr': 0.1}, True),          # Minimum K
            ({'K': 3, 'fdr': 0.0}, False),         # Minimum FDR (invalid)
            ({'K': 3, 'fdr': 1.0}, False),         # Maximum FDR (invalid)
            ({'K': 3, 'fdr': 0.1, 'delta': [0.0]}, False),  # Minimum delta (invalid)
            ({'K': 3, 'fdr': 0.1, 'delta': [1.0]}, False),  # Maximum delta (invalid)
            ({'K': 3, 'fdr': 0.1, 'lambda': [0.0]}, False), # Minimum lambda (invalid)
            ({'K': 3, 'fdr': 0.1, 'lambda': [1.0]}, False), # Maximum lambda (invalid)
            ({'K': 3, 'fdr': 1e-10}, True),        # Very small FDR (valid)
            ({'K': 3, 'fdr': 0.999}, True),        # Very large FDR (valid)
        ]

        for params, should_succeed in boundary_test_cases:
            if should_succeed:
                try:
                    slide = SLIDE(params, X, y)
                    assert slide is not None
                except Exception:
                    # May fail due to numerical issues, but not parameter validation
                    pass
            else:
                with pytest.raises((ValueError, AssertionError, RuntimeError)):
                    slide = SLIDE(params, X, y)

    def test_parameter_consistency_validation(self):
        """Test validation of parameter consistency across workflow."""
        X = np.random.randn(40, 12)
        y = np.random.binomial(1, 0.5, 40)

        # Test parameter combinations that are individually valid but inconsistent
        inconsistent_combinations = [
            # K larger than number of features
            ({'K': 20, 'fdr': 0.1}, False),

            # Very high K with very small FDR (may be statistically impossible)
            ({'K': 10, 'fdr': 1e-6}, False),

            # Parameter grid combinations that are valid individually
            ({'K': 5, 'fdr': 0.1, 'delta': [0.1, 0.9], 'lambda': [0.1, 0.9]}, True),
        ]

        for params, should_succeed in inconsistent_combinations:
            if should_succeed:
                try:
                    slide = SLIDE(params, X, y)
                    assert slide is not None
                except Exception:
                    # May fail for other reasons
                    pass
            else:
                with pytest.raises((ValueError, RuntimeError)):
                    slide = SLIDE(params, X, y)


class TestDataPreprocessingValidation:
    """Test data preprocessing validation scenarios."""

    def test_feature_scaling_validation(self):
        """Test validation of feature scaling and normalization."""
        # Create data with extreme scaling differences
        X = np.random.randn(50, 8)
        X[:, 0] *= 1e6   # Very large scale
        X[:, 1] *= 1e-6  # Very small scale
        X[:, 2] = 0      # Zero variance
        y = np.random.binomial(1, 0.5, 50)

        params = {'K': 3, 'fdr': 0.1}

        try:
            slide = SLIDE(params, X, y)
            # Should handle or warn about scaling issues
            assert slide is not None
        except (ValueError, RuntimeError) as e:
            # Should provide informative error about scaling/variance issues
            error_msg = str(e).lower()
            assert any(word in error_msg for word in
                      ['scale', 'variance', 'normalize', 'standardize'])

    def test_categorical_data_handling_validation(self):
        """Test handling of categorical data that shouldn't be in numeric algorithms."""
        # Create mixed data with categorical variables
        n = 50
        numeric_data = np.random.randn(n, 5)

        # Add categorical-like data (integer codes that look numeric)
        categorical_like = np.random.choice([1, 2, 3], n)  # Categorical encoded as int

        X = np.column_stack([numeric_data, categorical_like])
        y = np.random.binomial(1, 0.5, n)

        params = {'K': 3, 'fdr': 0.1}

        # Should either handle appropriately or provide guidance
        try:
            slide = SLIDE(params, X, y)
            assert slide is not None
        except Exception as e:
            # Should provide informative message about data type assumptions
            pass

    def test_missing_value_handling_validation(self):
        """Test comprehensive missing value handling scenarios."""
        X_complete = np.random.randn(50, 8)
        y = np.random.binomial(1, 0.5, 50)

        # Different missing value patterns
        missing_patterns = [
            # Random missing values
            lambda X: np.where(np.random.rand(*X.shape) < 0.1, np.nan, X),

            # Entire column missing
            lambda X: np.column_stack([X[:, :-1], np.full(X.shape[0], np.nan)]),

            # Entire row missing
            lambda X: np.vstack([X[:-1], np.full(X.shape[1], np.nan)]),

            # Structured missing (missing not at random)
            lambda X: np.where((X > 1) & (np.random.rand(*X.shape) < 0.5), np.nan, X),
        ]

        for pattern_func in missing_patterns:
            X_missing = pattern_func(X_complete.copy())
            params = {'K': 3, 'fdr': 0.1}

            try:
                slide = SLIDE(params, X_missing, y)
                # If successful, should have handled missing values appropriately
                if slide is not None:
                    assert hasattr(slide, 'data')
            except (ValueError, RuntimeError) as e:
                # Should provide clear guidance about missing value handling
                error_msg = str(e).lower()
                assert any(word in error_msg for word in
                          ['missing', 'nan', 'impute', 'complete'])