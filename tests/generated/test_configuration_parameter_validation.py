"""
Test coverage gaps for configuration and parameter validation.

Critical gaps in parameter validation that could lead to silent failures,
incorrect results, or security vulnerabilities.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.loveslide.tools import init_data, check_params, calc_default_fsize
from src.loveslide.slide import SLIDE, OptimizeSLIDE
from src.loveslide.knockoffs import Knockoffs


class TestParameterBoundaryValidation:
    """Test parameter validation at boundary conditions."""

    def test_extreme_parameter_values(self):
        """Test handling of extreme parameter values."""
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 100)

        # Test extreme values
        extreme_params = [
            {'delta': [1e-15]},  # Extremely small
            {'delta': [0.99999]},  # Near maximum
            {'lambda': [1e-10]},  # Extremely small regularization
            {'lambda': [1e10]},   # Extremely large regularization
            {'fdr': 1e-10},       # Extremely small FDR
            {'fdr': 0.999},       # Very large FDR
            {'niter': 1},         # Minimum iterations
            {'niter': 100000},    # Excessive iterations
        ]

        for params in extreme_params:
            base_params = {'x_path': None, 'y_path': None}
            base_params.update(params)

            # Should either work or raise informative error
            try:
                slide = SLIDE(base_params, x=X, y=y)
                result = slide.run()
                # If it succeeds, should produce valid results
                assert hasattr(result, 'marginal_idxs')
            except ValueError as e:
                # Should provide clear error message
                assert len(str(e)) > 10, "Error message too brief"
                assert any(key in str(e).lower() for key in params.keys()), "Error doesn't mention problematic parameter"

    def test_parameter_type_validation(self):
        """Test validation of parameter types."""
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 100)

        invalid_type_params = [
            {'delta': "invalid"},     # String instead of list/float
            {'lambda': None},         # None instead of list
            {'fdr': "0.1"},          # String instead of float
            {'niter': 10.5},         # Float instead of int
            {'n_workers': -1},       # Negative workers
            {'do_interacts': "yes"}, # String instead of bool
        ]

        for params in invalid_type_params:
            base_params = {'x_path': None, 'y_path': None}
            base_params.update(params)

            with pytest.raises(TypeError, match="invalid type|expected.*got"):
                slide = SLIDE(base_params, x=X, y=y)

    def test_parameter_range_validation(self):
        """Test validation of parameter ranges."""
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 100)

        out_of_range_params = [
            {'fdr': -0.1},           # Negative FDR
            {'fdr': 1.1},            # FDR > 1
            {'thresh_fdr': -0.1},    # Negative threshold
            {'thresh_fdr': 1.5},     # Threshold > 1
            {'n_workers': 0},        # Zero workers
            {'niter': 0},            # Zero iterations
            {'spec': -0.1},          # Negative specificity
            {'spec': 1.1},           # Specificity > 1
        ]

        for params in out_of_range_params:
            base_params = {'x_path': None, 'y_path': None}
            base_params.update(params)

            with pytest.raises(ValueError, match="out of range|invalid range|must be"):
                slide = SLIDE(base_params, x=X, y=y)


class TestDataValidationEdgeCases:
    """Test data validation edge cases."""

    def test_mismatched_dimensions(self):
        """Test handling of mismatched data dimensions."""
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 80)  # Wrong number of samples

        params = {'x_path': None, 'y_path': None}

        with pytest.raises(ValueError, match="dimension mismatch|shape.*mismatch"):
            slide = SLIDE(params, x=X, y=y)

    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_cases = [
            (np.array([]).reshape(0, 10), np.array([])),  # Empty arrays
            (np.random.randn(10, 0), np.random.randint(0, 2, 10)),  # Zero features
            (None, None),  # Null data
        ]

        for X, y in empty_cases:
            params = {'x_path': None, 'y_path': None}

            with pytest.raises(ValueError, match="empty|no data|insufficient data"):
                slide = SLIDE(params, x=X, y=y)

    def test_data_type_conversion(self):
        """Test automatic data type conversion and validation."""
        # Test various input types
        data_types = [
            np.random.randn(100, 50).astype(np.float32),  # float32
            np.random.randn(100, 50).astype(np.float64),  # float64
            (np.random.randn(100, 50) * 100).astype(int), # int
        ]

        for X in data_types:
            y = np.random.randint(0, 2, 100)
            params = {'x_path': None, 'y_path': None}

            # Should handle type conversion gracefully
            slide = SLIDE(params, x=X, y=y)
            result = slide.run()
            assert hasattr(result, 'marginal_idxs')

    def test_missing_value_detection(self):
        """Test detection and handling of missing values."""
        X = np.random.randn(100, 50)
        X[10, 5] = np.nan  # Inject NaN
        X[20, 10] = np.inf  # Inject inf
        y = np.random.randint(0, 2, 100)

        params = {'x_path': None, 'y_path': None}

        with pytest.raises(ValueError, match="missing values|NaN|infinite"):
            slide = SLIDE(params, x=X, y=y)


class TestConfigurationFileValidation:
    """Test configuration file validation."""

    def test_malformed_config_files(self):
        """Test handling of malformed configuration files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write malformed CSV
            f.write("col1,col2\n1,2,3\n4,5")  # Inconsistent columns
            malformed_file = f.name

        try:
            params = {'x_path': malformed_file, 'y_path': None}
            y = np.random.randint(0, 2, 100)

            with pytest.raises((pd.errors.ParserError, ValueError)):
                data, _ = init_data(params, y=y)
        finally:
            os.unlink(malformed_file)

    def test_file_permission_handling(self):
        """Test handling of file permission issues."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Create file and remove read permissions
            f.write(b"test,data\n1,2\n3,4")
            restricted_file = f.name

        try:
            os.chmod(restricted_file, 0o000)  # Remove all permissions

            params = {'x_path': restricted_file, 'y_path': None}
            y = np.random.randint(0, 2, 2)

            with pytest.raises(PermissionError):
                data, _ = init_data(params, y=y)
        finally:
            os.chmod(restricted_file, 0o644)  # Restore permissions
            os.unlink(restricted_file)

    def test_nonexistent_file_handling(self):
        """Test handling of nonexistent files."""
        params = {'x_path': '/nonexistent/path/file.csv', 'y_path': None}
        y = np.random.randint(0, 2, 100)

        with pytest.raises(FileNotFoundError):
            data, _ = init_data(params, y=y)


class TestParameterInteractionValidation:
    """Test validation of parameter interactions."""

    def test_conflicting_parameter_combinations(self):
        """Test detection of conflicting parameter combinations."""
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 100)

        conflicting_combinations = [
            {'pure_homo': True, 'pure_hetero': True},  # Conflicting purity assumptions
            {'fdr': 0.01, 'thresh_fdr': 0.005},       # Threshold < FDR
            {'n_workers': 8, 'serial_only': True},    # Parallel with serial flag
            {'do_interacts': False, 'interaction_only': True},  # Conflicting interaction flags
        ]

        for params in conflicting_combinations:
            base_params = {'x_path': None, 'y_path': None}
            base_params.update(params)

            with pytest.raises(ValueError, match="conflicting|incompatible|inconsistent"):
                slide = SLIDE(base_params, x=X, y=y)

    def test_parameter_dependency_validation(self):
        """Test validation of parameter dependencies."""
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 100)

        dependency_violations = [
            {'cv_folds': 10},  # CV folds without CV enabled
            {'knockoff_method': 'sdp'},  # Method without knockoffs enabled
            {'parallel_knockoffs': True, 'n_workers': 1},  # Parallel with 1 worker
        ]

        for params in dependency_violations:
            base_params = {'x_path': None, 'y_path': None}
            base_params.update(params)

            with pytest.warns(UserWarning, match="dependency|requires|needs"):
                slide = SLIDE(base_params, x=X, y=y)


class TestSecurityValidation:
    """Test security-related parameter validation."""

    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
        ]

        for path in malicious_paths:
            params = {'x_path': path, 'y_path': None, 'out_path': path}
            y = np.random.randint(0, 2, 100)

            # Should either sanitize path or raise security error
            with pytest.raises((ValueError, PermissionError, FileNotFoundError),
                             match="invalid path|security|not allowed"):
                data, _ = init_data(params, y=y)

    def test_command_injection_prevention(self):
        """Test prevention of command injection through parameters."""
        injection_attempts = [
            "; rm -rf /",
            "$(rm -rf /)",
            "`rm -rf /`",
            "| rm -rf /",
            "& rm -rf /",
        ]

        for injection in injection_attempts:
            params = {
                'x_path': None, 'y_path': None,
                'out_path': f"/tmp/output{injection}",
                'custom_command': injection,
            }
            X = np.random.randn(10, 5)
            y = np.random.randint(0, 2, 10)

            # Should sanitize or reject malicious input
            with pytest.raises(ValueError, match="invalid|unsafe|malicious"):
                slide = SLIDE(params, x=X, y=y)

    def test_resource_limit_validation(self):
        """Test validation of resource limits."""
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 100)

        excessive_resources = [
            {'n_workers': 1000},     # Too many workers
            {'max_memory': '1TB'},   # Excessive memory
            {'timeout': 86400*365},  # 1 year timeout
        ]

        for params in excessive_resources:
            base_params = {'x_path': None, 'y_path': None}
            base_params.update(params)

            with pytest.raises(ValueError, match="excessive|too large|unreasonable"):
                slide = SLIDE(base_params, x=X, y=y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])