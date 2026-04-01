"""
Comprehensive LOVE algorithm integration testing.
Tests R-Python interface, error handling, and edge cases.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from loveslide.love import call_love, call_love_r, _convert_r_pure_ind


class TestLOVERPythonInterface:
    """Test LOVE R-Python interface edge cases."""

    @patch('rpy2.robjects.r')
    def test_call_love_r_missing_dependencies(self, mock_r):
        """Test call_love_r when R dependencies are missing."""
        # Mock R dependency check failure
        mock_r.side_effect = ImportError("R package 'LOVE' not found")

        X = np.random.randn(100, 50)

        with pytest.raises(ImportError):
            call_love_r(X, lbd=0.5)

    @patch('rpy2.robjects.r')
    def test_call_love_r_memory_error(self, mock_r):
        """Test call_love_r with memory exhaustion."""
        # Very large matrix that might cause R memory issues
        X_large = np.random.randn(10000, 5000)

        mock_r.side_effect = MemoryError("R memory exhausted")

        with pytest.raises(MemoryError):
            call_love_r(X_large, lbd=0.5)

    @patch('rpy2.robjects.r')
    def test_call_love_r_convergence_failure(self, mock_r):
        """Test call_love_r when algorithm fails to converge."""
        # Mock R function that returns non-convergent result
        mock_result = MagicMock()
        mock_result.names = ['A', 'convergence', 'pure_ind']
        mock_result[0] = np.random.randn(50, 10)  # A matrix
        mock_result[1] = [False]  # No convergence
        mock_result[2] = []  # No pure indices

        mock_r.return_value = mock_result

        X = np.random.randn(100, 50)

        result = call_love_r(X, lbd=0.5)
        # Should handle non-convergence gracefully
        assert result['convergence'] == False
        assert len(result['pure_ind']) == 0

    def test_convert_r_pure_ind_edge_cases(self):
        """Test _convert_r_pure_ind with edge cases."""
        # Empty R list
        empty_r_list = MagicMock()
        empty_r_list.names = []

        result = _convert_r_pure_ind(empty_r_list)
        assert result == []

        # R list with NULL elements
        null_r_list = MagicMock()
        null_r_list.names = ['factor1', 'factor2']
        null_r_list[0] = None  # NULL in R
        null_r_list[1] = [1, 2, 3]

        result = _convert_r_pure_ind(null_r_list)
        assert len(result) == 2
        assert result[0] == []  # NULL converted to empty list
        assert result[1] == [1, 2, 3]

    @patch('rpy2.robjects.numpy2ri.activate')
    def test_call_love_r_conversion_errors(self, mock_activate):
        """Test call_love_r with numpy-R conversion errors."""
        mock_activate.side_effect = Exception("Conversion failed")

        X = np.random.randn(50, 20)

        with pytest.raises(Exception):
            call_love_r(X, lbd=0.5)


class TestLOVEPythonImplementation:
    """Test pure Python LOVE implementation edge cases."""

    def test_call_love_singular_matrix(self):
        """Test call_love with singular covariance matrix."""
        # Create matrix with perfect multicollinearity
        X = np.random.randn(100, 20)
        X[:, 1] = X[:, 0] * 2  # Perfect correlation
        X[:, 2] = X[:, 0] + X[:, 1]  # Linear combination

        try:
            result = call_love(X, lbd=0.5, mu=0.5)
            # Should handle singularity gracefully
            assert 'A' in result
            assert 'convergence' in result
        except np.linalg.LinAlgError:
            # Acceptable to fail on singular matrix
            pass

    def test_call_love_extreme_parameters(self):
        """Test call_love with extreme parameter values."""
        X = np.random.randn(100, 50)

        # Lambda at boundaries
        result_0 = call_love(X, lbd=0.0, mu=0.5)
        assert 'A' in result_0

        result_1 = call_love(X, lbd=1.0, mu=0.5)
        assert 'A' in result_1

        # Mu at boundaries
        result_mu_0 = call_love(X, lbd=0.5, mu=0.0)
        assert 'A' in result_mu_0

        result_mu_1 = call_love(X, lbd=0.5, mu=1.0)
        assert 'A' in result_mu_1

        # Very small thresh_fdr
        result_small_fdr = call_love(X, thresh_fdr=1e-10)
        assert 'A' in result_small_fdr

    def test_call_love_non_convergence_scenarios(self):
        """Test call_love scenarios that might not converge."""
        # Very noisy data
        X_noisy = np.random.randn(50, 100) * 1000

        result = call_love(X_noisy, lbd=0.5, mu=0.5, verbose=False)
        # Should terminate even if not converged
        assert 'convergence' in result
        assert isinstance(result['convergence'], bool)

        # Data with extreme variance differences
        X_mixed = np.random.randn(100, 20)
        X_mixed[:, :10] *= 1e-6  # Very small variance
        X_mixed[:, 10:] *= 1e6   # Very large variance

        result = call_love(X_mixed, lbd=0.5, mu=0.5)
        assert 'A' in result

    def test_call_love_memory_constraints(self):
        """Test call_love with memory-constrained scenarios."""
        # Large matrix
        try:
            X_large = np.random.randn(1000, 1000)
            result = call_love(X_large, lbd=0.5, mu=0.5, verbose=False)
            assert 'A' in result
        except MemoryError:
            # Expected for very large matrices
            pytest.skip("Insufficient memory for large matrix test")

    def test_call_love_numerical_precision_limits(self):
        """Test call_love at numerical precision limits."""
        # Data near machine precision
        X_tiny = np.random.randn(100, 20) * 1e-15

        try:
            result = call_love(X_tiny, lbd=0.5, mu=0.5)
            assert 'A' in result
            # Check for numerical instability
            assert not np.any(np.isnan(result['A']))
            assert not np.any(np.isinf(result['A']))
        except np.linalg.LinAlgError:
            # Acceptable to fail at precision limits
            pass


class TestLOVEParameterValidation:
    """Test LOVE parameter validation edge cases."""

    def test_call_love_invalid_dimensions(self):
        """Test call_love with invalid data dimensions."""
        # Single column
        X_single = np.random.randn(100, 1)

        try:
            result = call_love(X_single, lbd=0.5)
            # Should handle single column appropriately
            assert result['A'].shape[1] <= 1
        except ValueError:
            # May reject single-column data
            pass

        # More columns than rows (p >> n scenario)
        X_wide = np.random.randn(20, 100)

        result = call_love(X_wide, lbd=0.5, mu=0.5)
        assert 'A' in result
        # Should handle high-dimensional case

    def test_call_love_invalid_parameter_combinations(self):
        """Test call_love with invalid parameter combinations."""
        X = np.random.randn(100, 50)

        # Lambda > 1
        with pytest.raises(ValueError):
            call_love(X, lbd=1.5, mu=0.5)

        # Lambda < 0
        with pytest.raises(ValueError):
            call_love(X, lbd=-0.5, mu=0.5)

        # Mu > 1
        with pytest.raises(ValueError):
            call_love(X, lbd=0.5, mu=1.5)

        # Mu < 0
        with pytest.raises(ValueError):
            call_love(X, lbd=0.5, mu=-0.5)

        # Invalid thresh_fdr
        with pytest.raises(ValueError):
            call_love(X, thresh_fdr=-0.1)

        with pytest.raises(ValueError):
            call_love(X, thresh_fdr=1.5)

    def test_call_love_non_numeric_data(self):
        """Test call_love with non-numeric data."""
        # String data
        X_str = np.array([['a', 'b'], ['c', 'd']])

        with pytest.raises(TypeError):
            call_love(X_str, lbd=0.5)

        # Mixed numeric/non-numeric
        X_mixed = np.array([[1, 'a'], [2, 'b']], dtype=object)

        with pytest.raises(TypeError):
            call_love(X_mixed, lbd=0.5)

        # Complex numbers
        X_complex = np.random.randn(50, 20) + 1j * np.random.randn(50, 20)

        try:
            result = call_love(X_complex, lbd=0.5)
            # May handle complex data or reject it
            assert 'A' in result
        except TypeError:
            # Acceptable to reject complex data
            pass


class TestLOVEIntegrationWithSLIDE:
    """Test LOVE integration within SLIDE workflows."""

    def test_love_slide_integration_failure_recovery(self):
        """Test SLIDE recovery when LOVE call fails."""
        from loveslide import OptimizeSLIDE

        params = {"fdr": 0.1, "niter": 2}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        opt_slide = OptimizeSLIDE(params, x=X, y=y)

        # Mock LOVE failure
        with patch('loveslide.love.call_love') as mock_love:
            mock_love.side_effect = RuntimeError("LOVE algorithm failed")

            with pytest.raises(RuntimeError):
                opt_slide.optimize_SLIDE()

    def test_love_result_validation_in_slide(self):
        """Test SLIDE validation of LOVE results."""
        from loveslide import SLIDE

        params = {"fdr": 0.1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        slide = SLIDE(params, x=X, y=y)

        # Invalid LOVE result - missing A matrix
        invalid_result = {"convergence": True}

        with patch('pickle.load', return_value=invalid_result):
            slide.load_love("fake_path.pkl")
            # Should handle invalid result gracefully
            assert not hasattr(slide, 'A') or slide.A is None

        # LOVE result with wrong dimensions
        wrong_dim_result = {
            "A": np.random.randn(10, 5),  # Wrong number of rows
            "convergence": True
        }

        with patch('pickle.load', return_value=wrong_dim_result):
            slide.load_love("fake_path.pkl")
            # Should detect dimension mismatch
            assert not hasattr(slide, 'A') or slide.A is None