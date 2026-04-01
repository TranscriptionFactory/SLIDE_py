"""Test coverage for unguarded private method access patterns."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from loveslide import SLIDE, SLIDEcv
from loveslide.knockoffs import Knockoffs, _rlist_get, _create_second_order_r
from loveslide.love import _convert_r_pure_ind
from loveslide.tools import init_data, calc_default_fsize


class TestPrivateMethodAccess:
    """Test direct access to private methods with invalid states."""

    def test_rlist_get_with_invalid_r_objects(self):
        """Test _rlist_get with corrupted R objects."""
        # Test with None R object
        with pytest.raises((AttributeError, ValueError)):
            _rlist_get(None, "test_attribute")

        # Test with malformed R object
        mock_robj = Mock()
        mock_robj.rx2 = Mock(side_effect=AttributeError("Invalid R object"))
        with pytest.raises(AttributeError):
            _rlist_get(mock_robj, "nonexistent")

    def test_convert_r_pure_ind_edge_cases(self):
        """Test R result conversion with edge case inputs."""
        # Test with empty R list
        with pytest.raises((ValueError, IndexError)):
            _convert_r_pure_ind([])

        # Test with malformed R list structure
        with pytest.raises((KeyError, AttributeError)):
            malformed_list = [{"wrong_key": [1, 2, 3]}]
            _convert_r_pure_ind(malformed_list)

    def test_create_second_order_r_memory_boundaries(self):
        """Test R knockoff creation at memory boundaries."""
        # Test with extremely large matrix (should gracefully fail)
        large_matrix = np.random.randn(10000, 5000)
        with pytest.raises((MemoryError, RuntimeError)):
            _create_second_order_r(large_matrix)

        # Test with singular matrix
        singular_matrix = np.ones((100, 100))
        with pytest.warns(UserWarning):
            result = _create_second_order_r(singular_matrix)
            # Should either fail gracefully or return valid knockoffs
            assert result is None or isinstance(result, np.ndarray)

    def test_slide_internal_state_corruption(self):
        """Test SLIDE behavior when internal state is corrupted."""
        input_params = {
            'x_path': None, 'y_path': None,
            'fdr': 0.1, 'lambda': [0.1]
        }
        X = pd.DataFrame(np.random.randn(50, 20))
        y = pd.DataFrame(np.random.randint(0, 2, (50, 1)))

        slide = SLIDE(input_params, X, y)

        # Corrupt internal data structures
        slide.data.X = None
        with pytest.raises((AttributeError, ValueError)):
            slide.show_params()

        # Corrupt input parameters
        slide.input_params = None
        with pytest.raises((AttributeError, TypeError)):
            slide.calc_default_fsize(5)

    def test_calc_default_fsize_boundary_conditions(self):
        """Test calc_default_fsize with boundary mathematical conditions."""
        # Test negative inputs
        assert calc_default_fsize(-10, 5) >= 0
        assert calc_default_fsize(10, -5) >= 0

        # Test zero inputs
        assert calc_default_fsize(0, 5) >= 0
        assert calc_default_fsize(10, 0) >= 0

        # Test extreme values
        assert calc_default_fsize(np.iinfo(np.int32).max, 100) > 0
        assert calc_default_fsize(100, np.iinfo(np.int32).max) > 0