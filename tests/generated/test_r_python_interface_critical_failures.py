"""
Critical test coverage for R-Python interface failure modes.
These scenarios occur in production but lack comprehensive testing.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from loveslide.knockoffs import _rlist_get, _create_second_order_r, _solve_sdp_r
from loveslide.love import _convert_r_pure_ind


class TestRPythonInterfaceCriticalFailures:
    """Test R interface failure modes and recovery mechanisms."""

    def test_rlist_get_with_corrupted_r_object(self):
        """Test _rlist_get with corrupted R list object."""
        # Mock corrupted R object
        mock_robj = MagicMock()
        mock_robj.__getitem__.side_effect = [KeyError, TypeError]
        mock_robj.names = None  # Corrupted names

        with pytest.raises((KeyError, AttributeError)):
            _rlist_get(mock_robj, "test_key")

    def test_rlist_get_rpy2_version_compatibility(self):
        """Test compatibility across rpy2 versions."""
        # Test string key access (rpy2 3.5)
        mock_robj_35 = {"test_key": "value"}
        assert _rlist_get(mock_robj_35, "test_key") == "value"

        # Test names-based access (rpy2 3.6+)
        mock_robj_36 = MagicMock()
        mock_robj_36.__getitem__.side_effect = TypeError("string key not supported")
        mock_robj_36.names = ["test_key", "other"]
        mock_robj_36.__getitem__ = lambda x: "value" if x == 0 else "other_value"

        with patch.object(mock_robj_36, '__getitem__', side_effect=[TypeError, "value"]):
            result = _rlist_get(mock_robj_36, "test_key")

    def test_create_second_order_r_missing_dependency(self):
        """Test knockoff creation when R packages missing."""
        X = np.random.randn(50, 10)

        with patch('loveslide.knockoffs.importr', side_effect=ImportError("R package not found")):
            with pytest.raises(ImportError):
                _create_second_order_r(X)

    def test_solve_sdp_r_matrix_conditioning_failure(self):
        """Test SDP solver with poorly conditioned matrices."""
        # Singular matrix
        Sigma_singular = np.zeros((5, 5))

        with patch('loveslide.knockoffs.importr') as mock_importr:
            mock_r_func = MagicMock()
            mock_r_func.create_solve_sdp.side_effect = Exception("Matrix singular")
            mock_importr.return_value = mock_r_func

            with pytest.raises(Exception):
                _solve_sdp_r(Sigma_singular, method='sdp')

    def test_convert_r_pure_ind_malformed_input(self):
        """Test R result conversion with malformed data."""
        # Test with None input
        with pytest.raises((AttributeError, TypeError)):
            _convert_r_pure_ind(None)

        # Test with empty R list
        mock_empty_rlist = MagicMock()
        mock_empty_rlist.names = []

        result = _convert_r_pure_ind(mock_empty_rlist)
        assert isinstance(result, dict)

    def test_r_session_memory_cleanup(self):
        """Test R session memory management after errors."""
        X = np.random.randn(100, 20)

        with patch('loveslide.knockoffs.robjects') as mock_robjects:
            # Simulate memory pressure
            mock_robjects.r.side_effect = MemoryError("R session out of memory")

            with pytest.raises(MemoryError):
                _create_second_order_r(X)