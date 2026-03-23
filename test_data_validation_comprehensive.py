"""
Comprehensive data validation edge case testing for SLIDE_py.

This module tests data validation scenarios that might lead to silent failures
or incorrect behavior if not properly handled.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch
from pathlib import Path

from src.loveslide import SLIDE, OptimizeSLIDE, SLIDEcv, Knockoffs
from src.loveslide.tools import init_data, check_params
from src.loveslide.love import call_love
from src.loveslide.score import Estimator


class TestInputDataValidation:
    """Test validation of input data in various formats and conditions."""

    def test_mixed_data_types_in_features(self):
        """Test handling of mixed data types in feature matrix."""
        # Create CSV with mixed types
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("num_col,str_col,bool_col\n")
            f.write("1.5,hello,True\n")
            f.write("2.3,world,False\n")
            f.write("3.1,test,True\n")
            mixed_file = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("y\n1\n2\n3\n")
            y_file = f.name

        input_params = {'x_path': mixed_file, 'y_path': y_file}

        try:
            with pytest.raises((ValueError, TypeError)):
                init_data(input_params)
        finally:
            os.unlink(mixed_file)
            os.unlink(y_file)

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters in data."""
        # Create CSV with unicode characters
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write("col1,col2,col3\n")
            f.write("1.0,2.5,3.7\n")
            f.write("4.2,Σ,6.1\n")  # Unicode character in data
            f.write("7.8,8.9,∞\n")  # Infinity symbol
            unicode_file = f.name

        input_params = {'x_path': unicode_file}

        try:
            with pytest.raises((ValueError, TypeError)):
                data, params = init_data(input_params)
        finally:
            os.unlink(unicode_file)

    def test_scientific_notation_edge_cases(self):
        """Test handling of scientific notation in various formats."""
        # Create CSV with various scientific notation formats
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2,col3,col4\n")
            f.write("1e10,1E-10,1.5e+20,2.3E-15\n")
            f.write("3.14159e0,2.718e1,1.414e2,6.022e23\n")
            f.write("inf,-inf,1.23e-100,9.99e99\n")
            sci_file = f.name

        input_params = {'x_path': sci_file}

        try:
            data, params = init_data(input_params)
            X = data['X']

            # Should convert scientific notation properly
            assert X[0, 0] == 1e10
            assert X[0, 1] == 1e-10
            assert np.isinf(X[2, 0])  # inf
            assert np.isinf(X[2, 1])  # -inf

        finally:
            os.unlink(sci_file)

    def test_missing_value_patterns(self):
        """Test various missing value patterns and representations."""
        missing_patterns = [
            "col1,col2,col3\n1,2,3\n,5,6\n7,,9\n",  # Empty cells
            "col1,col2,col3\n1,2,3\nNA,5,6\n7,NULL,9\n",  # NA, NULL
            "col1,col2,col3\n1,2,3\nnan,5,6\n7,NaN,9\n",  # nan, NaN
            "col1,col2,col3\n1,2,3\n#N/A,5,6\n7,#NULL!,9\n",  # Excel formats
            "col1,col2,col3\n1,2,3\n.,5,6\n7,-,9\n",  # Other missing indicators
        ]

        for i, pattern in enumerate(missing_patterns):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write(pattern)
                missing_file = f.name

            input_params = {'x_path': missing_file}

            try:
                data, params = init_data(input_params)
                X = data['X']

                # Should handle missing values (either drop or impute)
                assert not np.any(np.isnan(X)) or X.shape[0] < 3

            except ValueError:
                # Acceptable to raise error for some missing value patterns
                pass
            finally:
                os.unlink(missing_file)

    def test_inconsistent_row_lengths(self):
        """Test handling of rows with inconsistent lengths."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2,col3\n")
            f.write("1,2,3\n")
            f.write("4,5\n")  # Missing column
            f.write("7,8,9,10\n")  # Extra column
            inconsistent_file = f.name

        input_params = {'x_path': inconsistent_file}

        try:
            with pytest.raises((ValueError, pd.errors.ParserError)):
                init_data(input_params)
        finally:
            os.unlink(inconsistent_file)

    def test_extreme_value_ranges(self):
        """Test handling of extreme numerical values."""
        extreme_values = [
            [1e308, 1e-308, 1.7976931348623157e+308],  # Near float64 limits
            [2.2250738585072014e-308, -1e308, 0.0],     # Tiny and large values
            [np.finfo(float).max, np.finfo(float).min, np.finfo(float).eps],  # Float limits
        ]

        for values in extreme_values:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("col1,col2,col3\n")
                f.write(f"{values[0]},{values[1]},{values[2]}\n")
                f.write("1,2,3\n")  # Normal values
                extreme_file = f.name

            input_params = {'x_path': extreme_file}

            try:
                data, params = init_data(input_params)
                X = data['X']

                # Should handle extreme values without overflow/underflow
                assert not np.any(np.isinf(X))
                assert not np.any(np.isnan(X))
                assert X.shape[1] == 3

            finally:
                os.unlink(extreme_file)

    def test_duplicate_column_names(self):
        """Test handling of duplicate column names."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col1,col2\n")  # Duplicate col1
            f.write("1,2,3\n")
            f.write("4,5,6\n")
            dup_file = f.name

        input_params = {'x_path': dup_file}

        try:
            data, params = init_data(input_params)
            X = data['X']

            # Should handle duplicates (rename or error)
            assert X.shape[1] == 3

        except ValueError:
            # Acceptable to raise error for duplicate columns
            pass
        finally:
            os.unlink(dup_file)

    def test_whitespace_and_formatting_issues(self):
        """Test handling of whitespace and formatting inconsistencies."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(" col1 , col2,col3 \n")  # Spaces in header
            f.write(" 1 ,2, 3\n")  # Leading/trailing spaces
            f.write("4,  5  ,6\n")  # Multiple spaces
            f.write("\t7\t,8,9\n")  # Tabs mixed with commas
            whitespace_file = f.name

        input_params = {'x_path': whitespace_file}

        try:
            data, params = init_data(input_params)
            X = data['X']

            # Should clean up whitespace properly
            assert X.shape == (3, 3)
            assert not np.any(np.isnan(X))

        finally:
            os.unlink(whitespace_file)


class TestMatrixValidation:
    """Test validation of matrix properties and constraints."""

    def test_rank_deficient_input_matrices(self):
        """Test handling of rank-deficient input matrices."""
        # Create rank-deficient matrix (all columns linearly dependent)
        X_rank_def = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])  # Rank 1
        y = np.random.randn(3)

        knockoffs = Knockoffs(X_rank_def)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = knockoffs.filter(y=y, model='equi')

        # Should handle rank deficiency gracefully
        assert hasattr(result, 'selected')

    def test_ill_conditioned_matrices(self):
        """Test handling of ill-conditioned matrices."""
        # Create ill-conditioned matrix
        condition_numbers = [1e12, 1e15, 1e18]

        for cond_num in condition_numbers:
            # Create matrix with specified condition number
            U = np.random.randn(50, 10)
            s = np.logspace(-np.log10(cond_num), 0, 10)
            V = np.random.randn(10, 10)
            X = U @ np.diag(s) @ V

            y = np.random.randn(50)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                knockoffs = Knockoffs(X)
                result = knockoffs.filter(y=y, model='equi')

            # Should handle ill-conditioning with warnings
            assert hasattr(result, 'selected')

    def test_sparse_matrices_dense_conversion(self):
        """Test handling when sparse matrices are provided as dense."""
        # Create matrix with many zeros (sparse-like)
        X = np.zeros((100, 50))
        X[:10, :5] = np.random.randn(10, 5)  # Only small portion has values

        y = np.random.randn(100)

        knockoffs = Knockoffs(X)
        result = knockoffs.filter(y=y, model='equi')

        # Should handle sparse-like data
        assert hasattr(result, 'selected')

    def test_matrices_with_perfect_correlations(self):
        """Test handling of matrices with perfect correlations."""
        # Create matrix with perfectly correlated features
        base = np.random.randn(100, 1)
        X = np.hstack([base, base * 2, base * -1, np.random.randn(100, 2)])

        y = np.random.randn(100)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            knockoffs = Knockoffs(X)
            result = knockoffs.filter(y=y, model='equi')

        # Should detect and handle perfect correlations
        assert hasattr(result, 'selected')

    def test_constant_features(self):
        """Test handling of constant (zero variance) features."""
        X = np.random.randn(80, 10)
        X[:, 0] = 5.0  # Constant feature
        X[:, 5] = -2.3  # Another constant feature

        y = np.random.randn(80)

        knockoffs = Knockoffs(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = knockoffs.filter(y=y, model='equi')

        # Should handle constant features
        assert hasattr(result, 'selected')

    def test_matrices_with_outliers(self):
        """Test handling of matrices with extreme outliers."""
        X = np.random.randn(100, 20)

        # Add extreme outliers
        X[0, 0] = 1000  # Large positive outlier
        X[1, 1] = -1000  # Large negative outlier
        X[2, :] = 500   # Entire outlier row

        y = np.random.randn(100)

        knockoffs = Knockoffs(X)
        result = knockoffs.filter(y=y, model='equi')

        # Should be robust to outliers
        assert hasattr(result, 'selected')


class TestParameterValidation:
    """Test parameter validation edge cases."""

    def test_invalid_parameter_combinations(self):
        """Test detection of invalid parameter combinations."""
        X = np.random.randn(50, 15)
        y = np.random.randn(50)

        knockoffs = Knockoffs(X)

        # Invalid FDR values
        with pytest.raises(ValueError):
            knockoffs.filter(y=y, fdr=-0.1)

        with pytest.raises(ValueError):
            knockoffs.filter(y=y, fdr=1.5)

        # Invalid model specification
        with pytest.raises(ValueError):
            knockoffs.filter(y=y, model='invalid_model')

    def test_parameter_type_validation(self):
        """Test parameter type validation."""
        X = np.random.randn(30, 10)

        # Wrong X type
        with pytest.raises(TypeError):
            Knockoffs("not_a_matrix")

        with pytest.raises(TypeError):
            Knockoffs([1, 2, 3, 4])

        # Wrong y type in filtering
        knockoffs = Knockoffs(X)
        with pytest.raises(TypeError):
            knockoffs.filter(y="not_a_vector")

    def test_dimension_consistency_validation(self):
        """Test dimension consistency validation."""
        X = np.random.randn(100, 20)
        y_wrong = np.random.randn(50)  # Wrong length

        knockoffs = Knockoffs(X)

        with pytest.raises(ValueError):
            knockoffs.filter(y=y_wrong)

    def test_parameter_range_validation(self):
        """Test parameter range validation."""
        input_params = {}

        # Invalid fold numbers
        with pytest.raises(ValueError):
            check_params({'n_folds': 0}, {})

        with pytest.raises(ValueError):
            check_params({'n_folds': -5}, {})

        # Invalid feature sizes
        with pytest.raises(ValueError):
            check_params({'fsize': 0}, {'X': np.random.randn(100, 50)})


class TestLOVEDataValidation:
    """Test LOVE algorithm data validation."""

    def test_love_parameter_validation(self):
        """Test LOVE parameter validation."""
        X = np.random.randn(60, 15)

        # Invalid lambda values
        with pytest.raises(ValueError):
            call_love(X, lbd=-0.5)

        with pytest.raises(ValueError):
            call_love(X, lbd=1.5)  # > 1

        # Invalid mu values
        with pytest.raises(ValueError):
            call_love(X, mu=-0.1)

        with pytest.raises(ValueError):
            call_love(X, mu=2.0)  # > 1

        # Invalid thresh_fdr values
        with pytest.raises(ValueError):
            call_love(X, thresh_fdr=-0.05)

        with pytest.raises(ValueError):
            call_love(X, thresh_fdr=1.2)  # > 1

    def test_love_matrix_requirements(self):
        """Test LOVE matrix requirements."""
        # Too few samples
        X_small = np.random.randn(5, 20)
        with pytest.warns(UserWarning):
            call_love(X_small)

        # More features than samples
        X_wide = np.random.randn(10, 50)
        with pytest.warns(UserWarning):
            call_love(X_wide)

    def test_love_degenerate_cases(self):
        """Test LOVE with degenerate input cases."""
        # All identical rows
        X_identical = np.ones((30, 10))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = call_love(X_identical)

        assert 'pure_indices' in result


class TestEstimatorDataValidation:
    """Test estimator data validation."""

    def test_estimator_input_validation(self):
        """Test estimator input validation."""
        estimator = Estimator(model_type='sklearn_linear')

        # Non-numeric input
        with pytest.raises((TypeError, ValueError)):
            estimator.fit("not_numeric", [1, 2, 3])

        # Mismatched dimensions
        X = np.random.randn(50, 10)
        y_wrong = np.random.randn(30)

        with pytest.raises(ValueError):
            estimator.fit(X, y_wrong)

    def test_estimator_prediction_validation(self):
        """Test estimator prediction validation."""
        estimator = Estimator(model_type='sklearn_linear')
        X_train = np.random.randn(40, 8)
        y_train = np.random.randn(40)

        estimator.fit(X_train, y_train)

        # Wrong feature dimensions
        X_test_wrong = np.random.randn(20, 5)
        with pytest.raises(ValueError):
            estimator.predict(X_test_wrong)


if __name__ == "__main__":
    pytest.main([__file__])