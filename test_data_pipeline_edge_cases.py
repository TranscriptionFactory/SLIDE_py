"""
Test coverage for data pipeline edge cases and boundary conditions.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, mock_open
from src.loveslide.tools import init_data, show_params, check_params


class TestDataPipelineEdgeCases:
    """Test data pipeline edge cases not covered in existing tests."""

    def test_init_data_corrupted_csv(self):
        """Test handling of corrupted CSV files."""
        corrupted_csv_content = "col1,col2,col3\n1,2\ninvalid,data,\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(corrupted_csv_content)
            f.flush()

            try:
                params = {'x_path': f.name, 'y_path': f.name}
                with pytest.raises(pd.errors.ParserError):
                    init_data(params)
            finally:
                os.unlink(f.name)

    def test_init_data_mixed_dtypes_coercion(self):
        """Test handling of mixed data types that need coercion."""
        # Create data with mixed types that pandas might misinterpret
        X_data = pd.DataFrame({
            'numeric_as_str': ['1.5', '2.0', 'inf'],
            'bool_as_str': ['True', 'False', 'True'],
            'mixed_numeric': [1, '2.5', None]
        })

        params = {'y_factor': False, 'y_flip': False}
        data, _ = init_data(params, x=X_data, y=pd.Series([0, 1, 0]))

        # Should handle mixed types gracefully
        assert data.X is not None
        assert data.Y is not None

    def test_init_data_unicode_encoding_issues(self):
        """Test handling of different character encodings."""
        # Create file with non-ASCII characters
        unicode_data = "ñame,vålue\në,1\nñ,2\n"

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         suffix='.csv', delete=False) as f:
            f.write(unicode_data)
            f.flush()

            try:
                params = {'x_path': f.name}
                y_data = pd.Series([0, 1])
                data, _ = init_data(params, y=y_data)

                # Should handle Unicode characters
                assert not data.X.empty
            finally:
                os.unlink(f.name)

    def test_init_data_extremely_large_row_names(self):
        """Test with pathologically long row/column names."""
        long_name = 'x' * 1000  # 1000 character column name
        X_data = pd.DataFrame({long_name: [1, 2, 3]})

        params = {}
        data, _ = init_data(params, x=X_data, y=pd.Series([0, 1, 0]))

        assert long_name in data.X.columns

    def test_init_data_scientific_notation_edge_cases(self):
        """Test handling of extreme scientific notation values."""
        X_data = pd.DataFrame({
            'tiny': [1e-300, 1e-310, 1e-320],
            'huge': [1e300, 1e310, 1e320],
            'inf': [np.inf, -np.inf, np.nan]
        })

        params = {}
        data, _ = init_data(params, x=X_data, y=pd.Series([0, 1, 0]))

        # Should preserve extreme values
        assert np.isinf(data.X['inf'].iloc[0])
        assert np.isinf(data.X['inf'].iloc[1])


class TestDataConsistencyValidation:
    """Test data consistency validation not covered elsewhere."""

    def test_y_factor_with_non_categorical_data(self):
        """Test y_factor=True with continuous response."""
        y_continuous = pd.Series([1.5, 2.7, 3.1, 1.5, 2.7])
        X_data = pd.DataFrame(np.random.randn(5, 3))

        params = {'y_factor': True}
        data, _ = init_data(params, x=X_data, y=y_continuous)

        # Should convert to integer categories
        assert data.Y.dtype.kind == 'i'
        assert len(np.unique(data.Y)) == 3  # Three unique values

    def test_y_flip_edge_cases(self):
        """Test y_flip with non-binary data."""
        y_multiclass = pd.Series([0, 1, 2, 0, 1, 2])
        X_data = pd.DataFrame(np.random.randn(6, 3))

        params = {'y_flip': True, 'y_factor': False}
        data, _ = init_data(params, x=X_data, y=y_multiclass)

        # For multiclass, flip formula is 1-y, so 0->1, 1->0, 2->-1
        expected = 1 - y_multiclass
        assert np.array_equal(data.Y, expected)

    def test_parameter_boundary_validation(self):
        """Test parameter validation at boundaries."""
        X_data = pd.DataFrame(np.random.randn(100, 10))
        y_data = pd.Series(np.random.binomial(1, 0.5, 100))

        # Test extreme parameter values
        extreme_params = {
            'delta': [1e-10, 1.0],  # Very small and maximum
            'lambda': [1e-10, 1.0],
            'fdr': 1e-10,  # Extremely strict
            'thresh_fdr': 1.0,  # Maximum threshold
            'spec': 0.0,  # Minimum specificity
            'niter': 1,  # Minimum iterations
        }

        data, processed_params = init_data(extreme_params, x=X_data, y=y_data)

        # Should accept extreme but valid parameters
        assert processed_params['delta'] == [1e-10, 1.0]
        assert processed_params['fdr'] == 1e-10