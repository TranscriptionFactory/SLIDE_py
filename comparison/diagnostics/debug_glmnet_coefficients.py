#!/usr/bin/env python3
"""
Debug script to compare coefficient magnitudes between:
1. knockoff-filter's vendored Fortran glmnet (ElasticNet)
2. sklearn's lasso_path
3. R's glmnet (via rpy2)

This will identify why the Fortran glmnet produces compressed W-statistics.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import lasso_path
import warnings

# Add knockoff-filter to path
KNOCKOFF_FILTER_PATH = "/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter"
if KNOCKOFF_FILTER_PATH not in sys.path:
    sys.path.insert(0, KNOCKOFF_FILTER_PATH)


def compare_coefficient_paths(X, y, nlambda=500):
    """Compare coefficient paths from Fortran glmnet vs sklearn lasso_path."""
    n, p = X.shape

    # Standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    y_centered = y - y.mean()

    # Lambda sequence (same as knockoff-filter)
    lambda_max = np.max(np.abs(X_std.T @ y_centered)) / n
    lambda_min = lambda_max / 2000
    k = np.arange(nlambda) / nlambda
    lambdas = lambda_max * (lambda_min / lambda_max) ** k

    print(f"Lambda sequence: max={lambda_max:.6f}, min={lambda_min:.8f}")
    print(f"First 5 lambdas: {lambdas[:5]}")
    print(f"Last 5 lambdas: {lambdas[-5:]}")
    print()

    results = {}

    # 1. Fortran glmnet (from knockoff-filter)
    try:
        from knockoff._vendor.glmnet import ElasticNet
        print("=== Fortran ElasticNet ===")

        model = ElasticNet(
            alpha=1.0,  # Pure lasso
            n_lambda=nlambda,
            lambda_path=lambdas,
            standardize=False,  # Already standardized
            fit_intercept=False,  # y already centered
            n_splits=0,  # No CV
            tol=1e-7,
            max_iter=100000,
        )
        model.fit(X_std, y_centered)

        fortran_lambdas = model.lambda_path_
        fortran_coefs = model.coef_path_  # (p, n_lambda)

        print(f"Output lambda path shape: {fortran_lambdas.shape}")
        print(f"Coef path shape: {fortran_coefs.shape}")
        print(f"Lambda path matches input: {np.allclose(fortran_lambdas, lambdas)}")
        print(f"First 5 output lambdas: {fortran_lambdas[:5]}")

        # Coefficient statistics
        max_coef = np.max(np.abs(fortran_coefs))
        min_nonzero = np.min(np.abs(fortran_coefs[fortran_coefs != 0])) if np.any(fortran_coefs != 0) else 0
        n_nonzero_per_lambda = np.sum(np.abs(fortran_coefs) > 0, axis=0)

        print(f"Max |coefficient|: {max_coef:.6f}")
        print(f"Min nonzero |coefficient|: {min_nonzero:.10f}")
        print(f"Nonzero coeffs at lambda[0]: {n_nonzero_per_lambda[0]}")
        print(f"Nonzero coeffs at lambda[100]: {n_nonzero_per_lambda[100]}")
        print(f"Nonzero coeffs at lambda[-1]: {n_nonzero_per_lambda[-1]}")

        results['fortran'] = {
            'lambdas': fortran_lambdas,
            'coefs': fortran_coefs,
            'max_coef': max_coef,
            'min_nonzero': min_nonzero,
        }
        print()

    except Exception as e:
        print(f"Fortran glmnet failed: {e}")
        results['fortran'] = None

    # 2. sklearn lasso_path
    print("=== sklearn lasso_path ===")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sklearn_lambdas, sklearn_coefs, _ = lasso_path(
                X_std, y_centered, alphas=lambdas, max_iter=10000
            )

        print(f"Output lambda path shape: {sklearn_lambdas.shape}")
        print(f"Coef path shape: {sklearn_coefs.shape}")
        print(f"Lambda path matches input: {np.allclose(sklearn_lambdas, lambdas)}")
        print(f"First 5 output lambdas: {sklearn_lambdas[:5]}")

        max_coef = np.max(np.abs(sklearn_coefs))
        min_nonzero = np.min(np.abs(sklearn_coefs[sklearn_coefs != 0])) if np.any(sklearn_coefs != 0) else 0
        n_nonzero_per_lambda = np.sum(np.abs(sklearn_coefs) > 0, axis=0)

        print(f"Max |coefficient|: {max_coef:.6f}")
        print(f"Min nonzero |coefficient|: {min_nonzero:.10f}")
        print(f"Nonzero coeffs at lambda[0]: {n_nonzero_per_lambda[0]}")
        print(f"Nonzero coeffs at lambda[100]: {n_nonzero_per_lambda[100]}")
        print(f"Nonzero coeffs at lambda[-1]: {n_nonzero_per_lambda[-1]}")

        results['sklearn'] = {
            'lambdas': sklearn_lambdas,
            'coefs': sklearn_coefs,
            'max_coef': max_coef,
            'min_nonzero': min_nonzero,
        }
        print()

    except Exception as e:
        print(f"sklearn lasso_path failed: {e}")
        results['sklearn'] = None

    # 3. R glmnet (if available)
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()

        print("=== R glmnet ===")
        glmnet_r = importr('glmnet')

        # Convert to R
        X_r = robjects.r['as.matrix'](X_std)
        y_r = robjects.FloatVector(y_centered)
        lambda_r = robjects.FloatVector(lambdas)

        # Fit glmnet
        fit = glmnet_r.glmnet(
            x=X_r, y=y_r,
            alpha=1.0,  # Lasso
            lambda_=lambda_r,
            standardize=False,
            intercept=False,
            thresh=1e-7,
            maxit=100000
        )

        # Extract coefficients
        r_lambdas = np.array(fit.rx2('lambda'))
        # Coefficients are stored in sparse format, convert to dense
        beta = fit.rx2('beta')
        r_coefs = np.array(robjects.r['as.matrix'](beta))  # (p, n_lambda)

        print(f"Output lambda path shape: {r_lambdas.shape}")
        print(f"Coef path shape: {r_coefs.shape}")
        print(f"First 5 output lambdas: {r_lambdas[:5]}")

        max_coef = np.max(np.abs(r_coefs))
        min_nonzero = np.min(np.abs(r_coefs[r_coefs != 0])) if np.any(r_coefs != 0) else 0
        n_nonzero_per_lambda = np.sum(np.abs(r_coefs) > 0, axis=0)

        print(f"Max |coefficient|: {max_coef:.6f}")
        print(f"Min nonzero |coefficient|: {min_nonzero:.10f}")
        print(f"Nonzero coeffs at lambda[0]: {n_nonzero_per_lambda[0]}")
        print(f"Nonzero coeffs at lambda[100]: {n_nonzero_per_lambda[100] if len(n_nonzero_per_lambda) > 100 else 'N/A'}")
        print(f"Nonzero coeffs at lambda[-1]: {n_nonzero_per_lambda[-1]}")

        results['r'] = {
            'lambdas': r_lambdas,
            'coefs': r_coefs,
            'max_coef': max_coef,
            'min_nonzero': min_nonzero,
        }

        numpy2ri.deactivate()
        print()

    except Exception as e:
        print(f"R glmnet failed: {e}")
        results['r'] = None

    # Compare coefficient paths
    print("=" * 60)
    print("COEFFICIENT PATH COMPARISON")
    print("=" * 60)

    if results.get('fortran') and results.get('sklearn'):
        f_coefs = results['fortran']['coefs']
        s_coefs = results['sklearn']['coefs']

        # Find common shape
        n_lambda_common = min(f_coefs.shape[1], s_coefs.shape[1])

        print(f"\nComparing first {n_lambda_common} lambda points:")

        # Compare at specific lambda indices
        for idx in [0, 50, 100, 200, 300, 400, n_lambda_common-1]:
            if idx >= n_lambda_common:
                continue
            f_col = f_coefs[:, idx]
            s_col = s_coefs[:, idx]

            f_nonzero = np.sum(np.abs(f_col) > 0)
            s_nonzero = np.sum(np.abs(s_col) > 0)
            f_nonzero_1e10 = np.sum(np.abs(f_col) > 1e-10)
            s_nonzero_1e10 = np.sum(np.abs(s_col) > 1e-10)

            # Correlation where both nonzero
            mask = (np.abs(f_col) > 1e-10) & (np.abs(s_col) > 1e-10)
            if np.sum(mask) > 1:
                corr = np.corrcoef(f_col[mask], s_col[mask])[0, 1]
                ratio = np.median(np.abs(s_col[mask]) / (np.abs(f_col[mask]) + 1e-15))
            else:
                corr = np.nan
                ratio = np.nan

            print(f"\nLambda index {idx} (λ={lambdas[idx]:.6f}):")
            print(f"  Fortran nonzero (>0): {f_nonzero}, (>1e-10): {f_nonzero_1e10}")
            print(f"  sklearn nonzero (>0): {s_nonzero}, (>1e-10): {s_nonzero_1e10}")
            print(f"  Correlation (where both >1e-10): {corr:.4f}")
            print(f"  Median ratio sklearn/fortran: {ratio:.4f}")

            # Show max coefficient for this lambda
            print(f"  Fortran max |coef|: {np.max(np.abs(f_col)):.6f}")
            print(f"  sklearn max |coef|: {np.max(np.abs(s_col)):.6f}")

    # Entry time comparison
    print("\n" + "=" * 60)
    print("ENTRY TIME COMPARISON (first nonzero detection)")
    print("=" * 60)

    def compute_entry_times(coefs, lambdas, threshold=0):
        """Compute entry time for each variable."""
        p = coefs.shape[0]
        entry_times = np.zeros(p)
        for j in range(p):
            nonzero_mask = np.abs(coefs[j, :]) > threshold
            if np.any(nonzero_mask):
                first_idx = np.argmax(nonzero_mask)
                entry_times[j] = lambdas[first_idx]
            else:
                entry_times[j] = 0
        return entry_times

    if results.get('fortran') and results.get('sklearn'):
        # Using threshold=0 (knockoff-filter's method)
        f_entry_0 = compute_entry_times(results['fortran']['coefs'], lambdas, threshold=0)
        s_entry_0 = compute_entry_times(results['sklearn']['coefs'], lambdas, threshold=0)

        # Using threshold=1e-10 (SLIDE's method)
        f_entry_1e10 = compute_entry_times(results['fortran']['coefs'], lambdas, threshold=1e-10)
        s_entry_1e10 = compute_entry_times(results['sklearn']['coefs'], lambdas, threshold=1e-10)

        print("\nWith threshold=0 (knockoff-filter's method):")
        print(f"  Fortran: {np.sum(f_entry_0 > 0)} variables have nonzero entry time")
        print(f"  sklearn: {np.sum(s_entry_0 > 0)} variables have nonzero entry time")
        print(f"  Correlation: {np.corrcoef(f_entry_0, s_entry_0)[0,1]:.4f}")
        print(f"  Fortran max entry time: {np.max(f_entry_0):.4f}")
        print(f"  sklearn max entry time: {np.max(s_entry_0):.4f}")

        print("\nWith threshold=1e-10 (SLIDE's method):")
        print(f"  Fortran: {np.sum(f_entry_1e10 > 0)} variables have nonzero entry time")
        print(f"  sklearn: {np.sum(s_entry_1e10 > 0)} variables have nonzero entry time")
        print(f"  Correlation: {np.corrcoef(f_entry_1e10, s_entry_1e10)[0,1]:.4f}")
        print(f"  Fortran max entry time: {np.max(f_entry_1e10):.4f}")
        print(f"  sklearn max entry time: {np.max(s_entry_1e10):.4f}")

        # Show variables with biggest differences
        diff_0 = np.abs(f_entry_0 - s_entry_0)
        top_diff_idx = np.argsort(diff_0)[-5:][::-1]
        print("\nTop 5 variables with largest entry time differences (threshold=0):")
        for idx in top_diff_idx:
            print(f"  Var {idx}: Fortran={f_entry_0[idx]:.4f}, sklearn={s_entry_0[idx]:.4f}, diff={diff_0[idx]:.4f}")

    return results


def main():
    # Load data
    z_path = "/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/output_comparison/20260119_225506_rstyle/0.1_0.5_out/z_matrix.csv"
    y_path = "/ix/djishnu/Aaron/0_for_others/Crystal/SLIDE/Scleroderma_Control/Scleroderma_Control_y.csv"

    print("Loading data...")
    Z_df = pd.read_csv(z_path, index_col=0)
    X = Z_df.values.astype(np.float64)

    y_df = pd.read_csv(y_path, index_col=0)
    y = y_df.values.flatten().astype(np.float64)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print()

    results = compare_coefficient_paths(X, y)

    return results


if __name__ == '__main__':
    main()
