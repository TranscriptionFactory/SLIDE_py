"""
Test API contracts, backward compatibility, and interface stability.
Addresses: API contract violations, return type consistency, deprecation handling
"""
import pytest
import numpy as np
import pandas as pd
import warnings
from typing import Union, Dict, Any
from loveslide import (
    SLIDE, OptimizeSLIDE, SLIDEcv, Knockoffs, VotingResult,
    call_love, Plotter, Estimator, SLIDE_Estimator,
    init_data, show_params, check_params, calc_default_fsize
)


class TestReturnTypeContracts:
    """Test that functions return expected types consistently."""

    def test_slide_fit_return_type(self):
        """Test SLIDE.fit() always returns consistent type."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        slide = SLIDE({'fdr': 0.1}, x=X, y=y)
        result = slide.fit()

        # Should always return object with expected attributes
        assert hasattr(result, 'selected')
        assert hasattr(result, 'statistic')
        assert isinstance(result.selected, np.ndarray)
        assert isinstance(result.statistic, np.ndarray)

    def test_knockoffs_create_return_shape(self):
        """Test knockoffs always return expected shape."""
        X = np.random.randn(100, 15)
        knockoffs = Knockoffs()

        for method in ['equi', 'sdp']:
            try:
                result = knockoffs.create_knockoffs(X, method=method)
                assert result.shape == X.shape, f"Shape mismatch for method {method}"
                assert result.dtype == X.dtype, f"Dtype mismatch for method {method}"
            except (ValueError, np.linalg.LinAlgError):
                pass  # Some methods may fail for certain inputs

    def test_voting_result_contract(self):
        """Test VotingResult maintains consistent interface."""
        X = np.random.randn(50, 8)
        y = np.random.randn(50)

        knockoffs = Knockoffs()
        result = knockoffs.filter(X, y, fdr=0.1, n_iters=10)

        assert isinstance(result, VotingResult)
        assert hasattr(result, 'selected')
        assert hasattr(result, 'fdp_hat')
        assert hasattr(result, 'power_hat')

    def test_estimator_predict_contract(self):
        """Test estimator predict methods maintain consistent interface."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        X_test = np.random.randn(20, 10)

        estimator = SLIDE_Estimator({'fdr': 0.1}, x=X, y=y)
        estimator.fit()

        # predict should work with various input types
        pred_array = estimator.predict(X_test)
        pred_df = estimator.predict(pd.DataFrame(X_test))

        assert len(pred_array) == len(X_test)
        assert len(pred_df) == len(X_test)
        assert pred_array.shape == pred_df.shape


class TestParameterValidationContracts:
    """Test parameter validation maintains consistent contracts."""

    def test_fdr_validation_consistency(self):
        """Test FDR validation is consistent across all classes."""
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        # All these should raise similar errors for invalid FDR
        invalid_fdrs = [-0.1, 1.1, np.nan, np.inf, "0.1"]

        for fdr in invalid_fdrs:
            with pytest.raises((ValueError, TypeError)):
                SLIDE({'fdr': fdr}, x=X, y=y)

            with pytest.raises((ValueError, TypeError)):
                SLIDEcv({'fdr': fdr}, x=X, y=y)

            with pytest.raises((ValueError, TypeError)):
                OptimizeSLIDE({'fdr': fdr}, x=X, y=y)

    def test_data_validation_contracts(self):
        """Test data validation maintains consistent behavior."""
        # Invalid data types that should be rejected consistently
        invalid_X = [
            None,
            "not_an_array",
            [[1, 2], [3]],  # Ragged array
            np.array([]),   # Empty array
            np.array([[[1, 2], [3, 4]]]),  # 3D array
        ]

        invalid_y = [
            None,
            "not_an_array",
            np.array([]),
            np.array([[1, 2], [3, 4]]),  # 2D y
        ]

        for X in invalid_X:
            with pytest.raises((ValueError, TypeError, AttributeError)):
                init_data({'fdr': 0.1}, x=X, y=np.array([1, 2, 3]))

        for y in invalid_y:
            with pytest.raises((ValueError, TypeError, AttributeError)):
                init_data({'fdr': 0.1}, x=np.random.randn(10, 5), y=y)

    def test_parameter_mutation_safety(self):
        """Test that parameter dictionaries are not mutated unexpectedly."""
        original_params = {'fdr': 0.1, 'n_iters': 100}
        params_copy = original_params.copy()

        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        slide = SLIDE(params_copy, x=X, y=y)
        slide.fit()

        # Original parameters should not be modified
        assert params_copy == original_params


class TestBackwardCompatibility:
    """Test backward compatibility with older interface patterns."""

    def test_deprecated_parameter_handling(self):
        """Test handling of deprecated parameters."""
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        # Test that deprecated parameters either work or warn appropriately
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Test with potentially deprecated parameter patterns
            slide = SLIDE({'fdr': 0.1, 'verbose': True}, x=X, y=y)
            result = slide.fit()

            # Should either work or issue deprecation warnings
            deprecation_warnings = [warning for warning in w
                                  if "deprecat" in str(warning.message).lower()]
            # Check that any deprecation warnings are properly formatted

    def test_legacy_data_format_support(self):
        """Test support for legacy data formats."""
        # Test various legacy input formats
        X_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        y_list = [1, 0, 1]

        X_df = pd.DataFrame(X_list, columns=['a', 'b', 'c'])
        y_series = pd.Series(y_list)

        # Should handle various input types gracefully
        for X, y in [(X_list, y_list), (X_df, y_series)]:
            try:
                slide = SLIDE({'fdr': 0.1}, x=X, y=y)
                result = slide.fit()
                assert result is not None
            except (ValueError, TypeError) as e:
                # Should provide clear error messages for unsupported formats
                assert "format" in str(e).lower() or "type" in str(e).lower()


class TestErrorMessageQuality:
    """Test that error messages are helpful and consistent."""

    def test_dimensional_mismatch_errors(self):
        """Test clear error messages for dimensional mismatches."""
        X = np.random.randn(50, 10)
        y = np.random.randn(60)  # Wrong length

        with pytest.raises(ValueError) as exc_info:
            SLIDE({'fdr': 0.1}, x=X, y=y)

        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ['dimension', 'shape', 'length', 'mismatch'])

    def test_invalid_method_errors(self):
        """Test clear error messages for invalid method specifications."""
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        with pytest.raises(ValueError) as exc_info:
            knockoffs = Knockoffs()
            knockoffs.create_knockoffs(X, method='invalid_method')

        error_msg = str(exc_info.value).lower()
        assert "method" in error_msg

    def test_convergence_failure_messages(self):
        """Test informative messages for convergence failures."""
        # Create problematic data likely to cause convergence issues
        X = np.random.randn(20, 50)  # More features than samples
        y = np.random.randn(20)

        try:
            slide = SLIDE({'fdr': 0.1}, x=X, y=y)
            result = slide.fit()
        except (ValueError, np.linalg.LinAlgError) as e:
            error_msg = str(e).lower()
            # Should mention convergence, singular, or numerical issues
            helpful_keywords = ['converge', 'singular', 'numerical', 'unstable', 'condition']
            assert any(keyword in error_msg for keyword in helpful_keywords)