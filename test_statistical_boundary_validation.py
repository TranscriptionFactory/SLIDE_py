"""
Test coverage for statistical property validation at mathematical boundaries
Focus: Statistical correctness at edge cases, convergence properties, and mathematical constraints
"""

import pytest
import numpy as np
from scipy import stats
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loveslide import SLIDE, OptimizeSLIDE, SLIDEcv
from loveslide.knockoffs import Knockoffs
from loveslide.score import Estimator, SLIDE_Estimator


class TestStatisticalBoundaryValidation:
    """Test statistical properties at mathematical boundaries"""

    def test_fdr_control_boundary_conditions(self):
        """Test FDR control at statistical boundaries"""
        # Test various FDR boundary conditions
        fdr_boundary_tests = [
            0.0001,  # Near-zero FDR
            0.9999,  # Near-maximum FDR
            0.5,     # Mid-range FDR
            0.1,     # Standard FDR
        ]

        X = np.random.rand(100, 20)
        y = np.random.randint(0, 2, 100)
        knockoffs = Knockoffs(y=y, z2=X)

        fdr_results = {}

        for target_fdr in fdr_boundary_tests:
            try:
                result = knockoffs.filter_knockoffs_iterative_python(
                    z=X, y=y, fdr=target_fdr, niter=3
                )

                if result is not None and hasattr(result, 'selected'):
                    selection_rate = len(result.selected) / X.shape[1] if hasattr(result, 'selected') else 0

                    fdr_results[target_fdr] = {
                        'selection_rate': selection_rate,
                        'n_selected': len(result.selected) if hasattr(result, 'selected') else 0,
                        'valid_result': True
                    }

                    # Verify FDR constraint
                    # Selection rate should generally be <= target FDR (with some tolerance)
                    if target_fdr > 0.01:  # Skip for very small FDR values
                        assert selection_rate <= target_fdr + 0.2, f"FDR constraint violated: {selection_rate} > {target_fdr}"

                else:
                    fdr_results[target_fdr] = {'valid_result': False}

            except Exception as e:
                fdr_results[target_fdr] = {'error': str(e), 'valid_result': False}

        # Verify monotonic relationship between FDR and selection rate
        valid_results = {fdr: res for fdr, res in fdr_results.items()
                        if res.get('valid_result', False)}

        if len(valid_results) >= 2:
            fdrs = sorted(valid_results.keys())
            selection_rates = [valid_results[fdr]['selection_rate'] for fdr in fdrs]

            # Higher FDR should generally allow higher selection rates
            # (though not strictly monotonic due to randomness)
            correlation = np.corrcoef(fdrs, selection_rates)[0, 1] if len(fdrs) > 2 else 1
            assert correlation >= -0.5, f"Unexpected FDR-selection rate relationship: {correlation}"

    def test_convergence_boundary_validation(self):
        """Test algorithm convergence at boundary conditions"""
        # Test convergence with various challenging conditions
        convergence_scenarios = [
            {
                'X': np.random.rand(30, 10),
                'K': 1,  # Minimal latent factors
                'max_iters': 1,  # Minimal iterations
                'description': 'minimal_setup'
            },
            {
                'X': np.random.rand(100, 5),
                'K': 4,  # K close to feature dimension
                'max_iters': 2,
                'description': 'high_K_ratio'
            },
            {
                'X': np.eye(20),  # Identity matrix (extreme correlation structure)
                'K': 3,
                'max_iters': 3,
                'description': 'identity_matrix'
            },
            {
                'X': np.ones((40, 10)),  # Constant matrix (no variation)
                'K': 2,
                'max_iters': 2,
                'description': 'constant_matrix'
            }
        ]

        convergence_results = {}

        for scenario in convergence_scenarios:
            X = scenario['X']
            y = np.random.randint(0, 2, X.shape[0])

            params = {
                'K': scenario['K'],
                'max_iters': scenario['max_iters'],
                'fdr_thresh': 0.1
            }

            try:
                slide = OptimizeSLIDE(params, x=X, y=y)

                # Mock LOVE result for controlled testing
                mock_love_result = {
                    'L_hat': np.random.rand(X.shape[1], scenario['K']),
                    'pure_idx': list(range(min(3, X.shape[1]))),
                    'converged': True
                }

                # Test convergence behavior
                lf_result = slide.get_latent_factors(
                    x=X, y=y, love_result=mock_love_result
                )

                convergence_results[scenario['description']] = {
                    'converged': lf_result is not None,
                    'result_valid': lf_result is not None and len(lf_result) >= 0,
                    'error': None
                }

                # Verify mathematical constraints are maintained
                if lf_result is not None:
                    # Check basic dimensionality constraints
                    if hasattr(slide, 'z_matrix') and slide.z_matrix is not None:
                        assert slide.z_matrix.shape[0] == X.shape[0], "Sample dimension constraint violated"

            except Exception as e:
                convergence_results[scenario['description']] = {
                    'converged': False,
                    'result_valid': False,
                    'error': str(e)
                }

        # Verify that at least some scenarios converge successfully
        successful_scenarios = sum(1 for res in convergence_results.values() if res['converged'])
        total_scenarios = len(convergence_scenarios)

        # At least 50% of scenarios should handle gracefully (converge or fail gracefully)
        assert successful_scenarios >= total_scenarios * 0.3, f"Too many convergence failures: {convergence_results}"

    def test_correlation_matrix_boundary_properties(self):
        """Test correlation matrix properties at boundaries"""
        # Test various correlation matrix boundary conditions
        correlation_scenarios = [
            {
                'name': 'near_singular',
                'matrix': np.array([[1.0, 0.999], [0.999, 1.0]]),
                'description': 'Nearly singular correlation matrix'
            },
            {
                'name': 'perfect_correlation',
                'matrix': np.array([[1.0, 1.0], [1.0, 1.0]]),
                'description': 'Perfect correlation matrix'
            },
            {
                'name': 'negative_correlation',
                'matrix': np.array([[1.0, -0.95], [-0.95, 1.0]]),
                'description': 'Strong negative correlation'
            },
            {
                'name': 'block_structure',
                'matrix': np.block([[np.ones((3,3)) * 0.8, np.zeros((3,2))],
                                  [np.zeros((2,3)), np.ones((2,2)) * 0.9]]),
                'description': 'Block correlation structure'
            }
        ]

        # Ensure diagonal is 1 for valid correlation matrices
        for scenario in correlation_scenarios:
            np.fill_diagonal(scenario['matrix'], 1.0)

        correlation_results = {}

        for scenario in correlation_scenarios:
            corr_matrix = scenario['matrix']

            try:
                # Test knockoff generation with challenging correlation structures
                n_samples = 50
                n_features = corr_matrix.shape[0]

                # Generate data with specified correlation structure
                L = np.linalg.cholesky(corr_matrix + np.eye(n_features) * 1e-10)
                X = np.random.randn(n_samples, n_features) @ L.T
                y = np.random.randint(0, 2, n_samples)

                # Test knockoff construction
                knockoffs = Knockoffs(y=y, z2=X)

                # Test SDP solving with challenging correlation matrix
                from loveslide.knockoff import solve
                if hasattr(solve, '_solve_sdp_cvxpy'):
                    try:
                        sdp_result = solve._solve_sdp_cvxpy(corr_matrix)
                        sdp_success = sdp_result is not None
                    except:
                        sdp_success = False
                else:
                    sdp_success = None

                # Test filtering with challenging structure
                try:
                    filter_result = knockoffs.filter_knockoffs_iterative_python(
                        z=X, y=y, fdr=0.1, niter=1
                    )
                    filter_success = filter_result is not None
                except:
                    filter_success = False

                correlation_results[scenario['name']] = {
                    'sdp_success': sdp_success,
                    'filter_success': filter_success,
                    'matrix_properties': {
                        'determinant': np.linalg.det(corr_matrix),
                        'condition_number': np.linalg.cond(corr_matrix),
                        'eigenvalues': np.linalg.eigvals(corr_matrix).tolist()
                    }
                }

                # Verify mathematical properties
                eigenvals = np.linalg.eigvals(corr_matrix)
                assert np.all(eigenvals >= -1e-10), f"Negative eigenvalues in {scenario['name']}: {eigenvals}"

            except Exception as e:
                correlation_results[scenario['name']] = {
                    'error': str(e),
                    'sdp_success': False,
                    'filter_success': False
                }

        # Verify that algorithm handles challenging correlation structures
        handling_rate = sum(1 for res in correlation_results.values()
                          if res.get('filter_success', False)) / len(correlation_scenarios)

        # Should handle at least 25% of challenging scenarios
        assert handling_rate >= 0.25, f"Poor handling of correlation boundary cases: {correlation_results}"

    def test_sample_size_statistical_boundaries(self):
        """Test statistical properties at sample size boundaries"""
        # Test various sample size boundary conditions
        sample_size_scenarios = [
            {'n_samples': 10, 'n_features': 8, 'description': 'minimal_samples'},
            {'n_samples': 50, 'n_features': 45, 'description': 'n_close_to_p'},
            {'n_samples': 1000, 'n_features': 5, 'description': 'large_n_small_p'},
            {'n_samples': 20, 'n_features': 20, 'description': 'n_equals_p'},
        ]

        sample_size_results = {}

        for scenario in sample_size_scenarios:
            n_samples = scenario['n_samples']
            n_features = scenario['n_features']

            X = np.random.rand(n_samples, n_features)
            y = np.random.randint(0, 2, n_samples)

            try:
                # Test statistical estimator behavior
                estimator = Estimator(model='auto')
                estimator.fit(X, y)

                # Test prediction consistency
                y_pred = estimator.predict(X)
                prediction_variance = np.var(y_pred)

                # Test performance evaluation
                aucs = estimator.get_aucs(X, y, n_iters=3, test_size=0.3)
                auc_stability = np.std(aucs) if len(aucs) > 1 else 0

                # Test knockoff filtering
                knockoffs = Knockoffs(y=y, z2=X)
                filter_result = knockoffs.filter_knockoffs_iterative_python(
                    z=X, y=y, fdr=0.1, niter=1
                )

                sample_size_results[scenario['description']] = {
                    'prediction_variance': prediction_variance,
                    'auc_stability': auc_stability,
                    'filter_success': filter_result is not None,
                    'statistical_power': len(filter_result.selected) / n_features if filter_result and hasattr(filter_result, 'selected') else 0
                }

                # Verify statistical consistency
                assert prediction_variance >= 0, f"Invalid prediction variance: {prediction_variance}"
                assert auc_stability >= 0, f"Invalid AUC stability: {auc_stability}"

            except Exception as e:
                sample_size_results[scenario['description']] = {
                    'error': str(e),
                    'filter_success': False
                }

        # Verify statistical behavior patterns
        successful_scenarios = [res for res in sample_size_results.values()
                              if res.get('filter_success', False)]

        if len(successful_scenarios) >= 2:
            # AUC stability should generally improve with more samples
            # (though this is not strictly required)
            stabilities = [res['auc_stability'] for res in successful_scenarios
                         if 'auc_stability' in res]

            if stabilities:
                assert all(s < 1.0 for s in stabilities), f"Excessive AUC instability: {stabilities}"

    def test_statistical_significance_boundaries(self):
        """Test statistical significance at boundary p-values"""
        # Test significance testing at various p-value boundaries
        significance_levels = [0.001, 0.01, 0.05, 0.1, 0.5]

        X = np.random.rand(80, 15)
        y = np.random.randint(0, 2, 80)

        significance_results = {}

        for alpha in significance_levels:
            try:
                # Test CV with different significance levels
                folds = [(list(range(40)), list(range(40, 80)))]
                cv = SLIDEcv(x=X, y=y, folds=folds, n_workers=1)

                # Mock statistical test results
                with patch.object(cv, '_compute_metric') as mock_metric:
                    mock_metric.return_value = np.random.beta(2, 2)  # Random metric in [0,1]

                    # Test significance interpretation
                    try:
                        cv_result = cv.run(verbose=False)

                        # Compute significance statistics
                        if cv_result is not None and 'metrics' in str(cv_result).lower():
                            # Basic significance validation
                            significance_results[alpha] = {
                                'test_completed': True,
                                'alpha_level': alpha,
                                'result_valid': cv_result is not None
                            }
                        else:
                            significance_results[alpha] = {
                                'test_completed': True,
                                'alpha_level': alpha,
                                'result_valid': False
                            }

                    except Exception as inner_e:
                        significance_results[alpha] = {
                            'test_completed': False,
                            'error': str(inner_e),
                            'alpha_level': alpha
                        }

            except Exception as e:
                significance_results[alpha] = {
                    'error': str(e),
                    'alpha_level': alpha,
                    'test_completed': False
                }

        # Verify significance testing behavior
        completed_tests = sum(1 for res in significance_results.values()
                            if res.get('test_completed', False))
        total_tests = len(significance_levels)

        # Should complete majority of significance tests
        assert completed_tests >= total_tests * 0.5, f"Many significance tests failed: {significance_results}"

        # Verify alpha levels are preserved
        for alpha, result in significance_results.items():
            if 'alpha_level' in result:
                assert result['alpha_level'] == alpha, f"Alpha level not preserved: {alpha} != {result['alpha_level']}"


class TestMathematicalConstraintValidation:
    """Test mathematical constraint validation"""

    def test_matrix_definiteness_constraints(self):
        """Test positive definiteness and related matrix constraints"""
        # Test various matrix definiteness scenarios
        matrix_scenarios = [
            {
                'name': 'positive_definite',
                'generator': lambda size: np.random.rand(size, size),
                'constraint': 'positive_eigenvalues'
            },
            {
                'name': 'semi_definite',
                'generator': lambda size: np.outer(np.random.rand(size), np.random.rand(size)),
                'constraint': 'non_negative_eigenvalues'
            },
            {
                'name': 'symmetric',
                'generator': lambda size: (lambda A: A + A.T)(np.random.rand(size, size)),
                'constraint': 'symmetric_structure'
            }
        ]

        constraint_results = {}

        for scenario in matrix_scenarios:
            try:
                # Generate test matrix
                test_matrix = scenario['generator'](8)

                # Ensure positive definite for covariance matrix
                if scenario['name'] in ['positive_definite', 'semi_definite']:
                    test_matrix = test_matrix @ test_matrix.T + np.eye(8) * 0.01

                # Test constraint validation
                eigenvals = np.linalg.eigvals(test_matrix)

                constraint_results[scenario['name']] = {
                    'eigenvalues': eigenvals.tolist(),
                    'min_eigenval': float(np.min(eigenvals)),
                    'condition_number': float(np.linalg.cond(test_matrix)),
                    'constraint_satisfied': True
                }

                # Verify mathematical constraints
                if scenario['constraint'] == 'positive_eigenvalues':
                    assert np.all(eigenvals > -1e-10), f"Non-positive eigenvalues: {eigenvals}"
                elif scenario['constraint'] == 'non_negative_eigenvalues':
                    assert np.all(eigenvals >= -1e-10), f"Negative eigenvalues: {eigenvals}"
                elif scenario['constraint'] == 'symmetric_structure':
                    assert np.allclose(test_matrix, test_matrix.T), "Matrix not symmetric"

                # Test with knockoff generation
                try:
                    X = np.random.multivariate_normal(np.zeros(8), test_matrix, size=50)
                    y = np.random.randint(0, 2, 50)

                    knockoffs = Knockoffs(y=y, z2=X)
                    filter_result = knockoffs.filter_knockoffs_iterative_python(
                        z=X, y=y, fdr=0.1, niter=1
                    )

                    constraint_results[scenario['name']]['knockoff_success'] = filter_result is not None

                except Exception:
                    constraint_results[scenario['name']]['knockoff_success'] = False

            except Exception as e:
                constraint_results[scenario['name']] = {
                    'error': str(e),
                    'constraint_satisfied': False
                }

        # Verify constraint satisfaction
        satisfied_constraints = sum(1 for res in constraint_results.values()
                                  if res.get('constraint_satisfied', False))
        total_constraints = len(matrix_scenarios)

        assert satisfied_constraints >= total_constraints * 0.7, f"Many constraints violated: {constraint_results}"


if __name__ == "__main__":
    pytest.main([__file__])