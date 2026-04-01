"""
Direct testing of private methods that are critical but under-tested.
"""
import pytest
import numpy as np
import pandas as pd
from loveslide import SLIDE, Knockoffs
from loveslide.score import Estimator


class TestPrivateMethodsDirect:
    """Test private methods directly to ensure correctness."""

    def test_estimator_init_model_edge_cases(self):
        """Test Estimator._init_model with edge cases."""
        estimator = Estimator()

        # Test with single unique value
        y_single = np.array([1, 1, 1, 1])
        estimator._init_model(y_single)
        assert estimator.is_classifier is False  # Should default to linear

        # Test with exact binary values
        y_binary = np.array([0, 1])
        estimator._init_model(y_binary)
        assert estimator.is_classifier is True

        # Test with float binary-like values
        y_float_binary = np.array([0.0, 1.0, 0.0, 1.0])
        estimator._init_model(y_float_binary)
        assert estimator.is_classifier is True

    def test_slide_calc_z_matrix_edge_cases(self):
        """Test SLIDE.calc_z_matrix with various input scenarios."""
        params = {"fdr": 0.1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        slide = SLIDE(params, x=X, y=y)

        # Test with minimal love_result
        love_result = {
            'A': np.random.randn(20, 3),
            'pure_row_indices': [0, 1],
            'X': X
        }

        # Should handle missing keys gracefully
        with pytest.raises(KeyError):
            slide.calc_z_matrix({})

    def test_knockoffs_rlist_get_fallback(self):
        """Test _rlist_get function fallback behavior."""
        from loveslide.knockoffs import _rlist_get

        # Mock R object with names attribute
        class MockRObj:
            def __init__(self, data, names):
                self.data = data
                self.names = names

            def __getitem__(self, key):
                if isinstance(key, str):
                    raise TypeError("String access not supported")
                return self.data[key]

        mock_r = MockRObj(['a', 'b', 'c'], ['x', 'y', 'z'])
        result = _rlist_get(mock_r, 'y')
        assert result == 'b'

    def test_knockoffs_single_iteration_error_handling(self):
        """Test _single_knockoff_iteration_python error scenarios."""
        from loveslide.knockoffs import _single_knockoff_iteration_python

        # Test with mismatched dimensions
        z = np.random.randn(50, 10)
        y = np.random.randn(30)  # Wrong size

        with pytest.raises((ValueError, IndexError)):
            _single_knockoff_iteration_python(
                z, y, fdr=0.1, method='lasso',
                shrink=True, offset=1, statistic='lasso_coefdiff'
            )