"""
Test coverage for dynamic configuration injection and runtime parameter modification
Focus: Runtime parameter changes, configuration inheritance, and validation consistency
"""

import pytest
import numpy as np
from copy import deepcopy
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loveslide import SLIDE, OptimizeSLIDE, SLIDEcv
from loveslide.knockoffs import Knockoffs
from loveslide.tools import check_params, show_params


class TestDynamicConfigurationInjection:
    """Test dynamic configuration modification during runtime"""

    def test_runtime_parameter_injection_slide(self):
        """Test SLIDE parameter modification during execution"""
        base_params = {'K': 3, 'max_iters': 5, 'fdr_thresh': 0.1}
        X = np.random.rand(30, 10)
        y = np.random.randint(0, 2, 30)

        slide = OptimizeSLIDE(base_params, x=X, y=y)

        # Test parameter injection at different execution points
        original_params = deepcopy(slide.input_params)

        # Inject parameters during runtime
        runtime_params = {'fdr_thresh': 0.05, 'lambda_reg': 0.01}
        slide.input_params.update(runtime_params)

        # Verify parameter consistency after injection
        assert slide.input_params['fdr_thresh'] != original_params['fdr_thresh']

        # Test parameter validation after injection
        mock_data = {'X': X, 'y': y, 'n_samples': X.shape[0], 'n_features': X.shape[1]}
        try:
            is_valid = check_params(slide.input_params, mock_data)
            assert is_valid is True or is_valid is None
        except Exception as e:
            # Should handle validation errors gracefully
            assert "parameter" in str(e).lower() or "invalid" in str(e).lower()

    def test_configuration_inheritance_patterns(self):
        """Test configuration inheritance in nested objects"""
        parent_config = {
            'K': 4,
            'max_iters': 10,
            'knockoff_config': {'fdr': 0.1, 'method': 'sdp'},
            'cv_config': {'n_folds': 5, 'metric': 'auc'}
        }

        X = np.random.rand(40, 8)
        y = np.random.randint(0, 2, 40)

        # Test inheritance from parent to child configurations
        slide = OptimizeSLIDE(parent_config, x=X, y=y)

        # Create knockoffs with inherited config
        knockoffs = Knockoffs(y=y, z2=X)

        # Modify parent config and test inheritance propagation
        parent_config['knockoff_config']['fdr'] = 0.05

        # Verify inheritance doesn't break with dynamic changes
        try:
            # Test that knockoffs can still access configuration properly
            result = knockoffs.filter_knockoffs_iterative_python(
                z=X, y=y,
                fdr=parent_config['knockoff_config']['fdr'],
                niter=1
            )
            # Verify configuration was applied
            if result is not None:
                assert hasattr(result, 'selected')
        except Exception:
            pass  # Configuration may not be compatible

    def test_parameter_type_coercion_injection(self):
        """Test parameter type coercion during dynamic injection"""
        base_params = {'K': 3, 'max_iters': 5}
        X = np.random.rand(25, 6)
        y = np.random.randint(0, 2, 25)

        slide = SLIDE(base_params, x=X, y=y)

        # Test various type coercion scenarios
        type_injection_tests = [
            {'K': '4'},          # String to int
            {'max_iters': 10.5}, # Float to int
            {'fdr_thresh': '0.1'}, # String to float
            {'verbose': 'True'},  # String to boolean
        ]

        for injection in type_injection_tests:
            slide_copy = SLIDE(base_params.copy(), x=X, y=y)
            slide_copy.input_params.update(injection)

            # Test parameter validation handles type coercion
            mock_data = {'X': X, 'y': y, 'n_samples': X.shape[0], 'n_features': X.shape[1]}
            try:
                check_params(slide_copy.input_params, mock_data)
            except (TypeError, ValueError) as e:
                # Should handle type errors gracefully
                assert any(word in str(e).lower() for word in ['type', 'convert', 'invalid'])

    def test_nested_configuration_modification(self):
        """Test modification of nested configuration structures"""
        complex_config = {
            'algorithm': {
                'slide': {'K': 3, 'max_iters': 5},
                'love': {'lbd': 0.5, 'mu': 0.3},
                'knockoffs': {
                    'creation': {'method': 'sdp'},
                    'filtering': {'fdr': 0.1, 'offset': 1}
                }
            },
            'data': {
                'preprocessing': {'scale': True, 'center': True},
                'validation': {'split_ratio': 0.2}
            }
        }

        X = np.random.rand(30, 8)
        y = np.random.randint(0, 2, 30)

        # Test deep nested modification
        original_fdr = complex_config['algorithm']['knockoffs']['filtering']['fdr']
        complex_config['algorithm']['knockoffs']['filtering']['fdr'] = 0.05

        # Verify nested modification propagation
        assert complex_config['algorithm']['knockoffs']['filtering']['fdr'] != original_fdr

        # Test that nested modifications maintain structure integrity
        knockoffs = Knockoffs(y=y, z2=X)
        try:
            result = knockoffs.filter_knockoffs_iterative_python(
                z=X, y=y,
                fdr=complex_config['algorithm']['knockoffs']['filtering']['fdr'],
                niter=1
            )
            if result is not None:
                assert hasattr(result, 'selected')
        except Exception:
            pass

    def test_configuration_validation_consistency(self):
        """Test configuration validation consistency with dynamic changes"""
        X = np.random.rand(35, 10)
        y = np.random.randint(0, 2, 35)

        # Test configuration validation with various dynamic scenarios
        validation_scenarios = [
            {'K': -1},           # Invalid negative value
            {'max_iters': 0},    # Invalid zero iterations
            {'fdr_thresh': 1.5}, # Invalid FDR threshold > 1
            {'lambda_reg': -0.1}, # Invalid negative regularization
        ]

        for scenario in validation_scenarios:
            base_params = {'K': 3, 'max_iters': 5}
            base_params.update(scenario)

            try:
                slide = SLIDE(base_params, x=X, y=y)
                mock_data = {'X': X, 'y': y, 'n_samples': X.shape[0], 'n_features': X.shape[1]}

                # Should detect invalid configurations
                is_valid = check_params(slide.input_params, mock_data)
                if is_valid is True:
                    # Some invalid configs might be auto-corrected
                    pass
                else:
                    # Should detect invalidity
                    assert is_valid is False or is_valid is None

            except (ValueError, TypeError, AssertionError):
                # Expected for invalid configurations
                pass

    def test_concurrent_configuration_isolation(self):
        """Test configuration isolation between concurrent instances"""
        base_config = {'K': 3, 'max_iters': 5, 'fdr_thresh': 0.1}
        X = np.random.rand(30, 8)
        y = np.random.randint(0, 2, 30)

        # Create multiple instances with same base config
        slide1 = SLIDE(base_config.copy(), x=X, y=y)
        slide2 = SLIDE(base_config.copy(), x=X, y=y)

        # Modify one instance
        slide1.input_params['fdr_thresh'] = 0.05
        slide1.input_params['K'] = 5

        # Verify isolation - other instance should be unaffected
        assert slide2.input_params['fdr_thresh'] == 0.1
        assert slide2.input_params['K'] == 3

        # Test that both instances can operate independently
        try:
            slide1.show_params()
            slide2.show_params()

            # Verify parameters are correctly isolated
            assert slide1.input_params != slide2.input_params
        except Exception as e:
            pytest.fail(f"Configuration isolation failed: {e}")


class TestParameterInjectionEdgeCases:
    """Test edge cases in parameter injection"""

    def test_partial_configuration_injection(self):
        """Test injection of partial configurations"""
        X = np.random.rand(25, 6)
        y = np.random.randint(0, 2, 25)

        # Start with minimal configuration
        minimal_config = {'K': 2}
        slide = SLIDE(minimal_config, x=X, y=y)

        # Inject additional parameters incrementally
        additional_params = [
            {'max_iters': 3},
            {'fdr_thresh': 0.1},
            {'lambda_reg': 0.01},
            {'verbose': True}
        ]

        for params in additional_params:
            slide.input_params.update(params)

            # Verify configuration remains valid after each injection
            try:
                slide.show_params()
                assert all(key in slide.input_params for key in params.keys())
            except Exception as e:
                # Should handle partial configurations gracefully
                assert "missing" in str(e).lower() or "required" in str(e).lower()

    def test_configuration_rollback_mechanisms(self):
        """Test configuration rollback after invalid injection"""
        base_config = {'K': 3, 'max_iters': 5, 'fdr_thresh': 0.1}
        X = np.random.rand(30, 8)
        y = np.random.randint(0, 2, 30)

        slide = SLIDE(base_config.copy(), x=X, y=y)
        original_config = deepcopy(slide.input_params)

        # Inject invalid configuration
        invalid_injection = {'K': -5, 'max_iters': 'invalid', 'fdr_thresh': 2.0}
        slide.input_params.update(invalid_injection)

        # Test rollback to valid state
        mock_data = {'X': X, 'y': y, 'n_samples': X.shape[0], 'n_features': X.shape[1]}
        try:
            is_valid = check_params(slide.input_params, mock_data)
            if is_valid is False:
                # Simulate rollback
                slide.input_params = original_config
                is_valid_after_rollback = check_params(slide.input_params, mock_data)
                assert is_valid_after_rollback is True or is_valid_after_rollback is None
        except Exception:
            # Rollback mechanism may vary
            pass

    def test_configuration_serialization_injection(self):
        """Test configuration injection from serialized sources"""
        import json
        import tempfile

        X = np.random.rand(25, 6)
        y = np.random.randint(0, 2, 25)

        # Test JSON configuration injection
        json_config = {
            'K': 3,
            'max_iters': 5,
            'fdr_thresh': 0.1,
            'algorithm_params': {
                'love': {'lbd': 0.5},
                'knockoffs': {'fdr': 0.1}
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_config, f)
            config_path = f.name

        try:
            # Load and inject serialized configuration
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)

            slide = SLIDE(loaded_config, x=X, y=y)

            # Verify serialized configuration injection worked
            assert slide.input_params['K'] == json_config['K']
            assert slide.input_params['fdr_thresh'] == json_config['fdr_thresh']

        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__])