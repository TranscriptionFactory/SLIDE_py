#!/usr/bin/env python3
"""
Test R's glmnet behavior to understand standardization and lambda sequence.
"""

import numpy as np
import pandas as pd

def test_r_glmnet():
    """Test R's glmnet standardization behavior."""
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr

    numpy2ri.activate()

    # Load test data
    z_path = "/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/output_comparison/20260119_225506_rstyle/0.1_0.5_out/z_matrix.csv"
    y_path = "/ix/djishnu/Aaron/0_for_others/Crystal/SLIDE/Scleroderma_Control/Scleroderma_Control_y.csv"

    Z_df = pd.read_csv(z_path, index_col=0)
    X = Z_df.values.astype(np.float64)
    y_df = pd.read_csv(y_path, index_col=0)
    y = y_df.values.flatten().astype(np.float64)

    n, p = X.shape
    print(f"Data: n={n}, p={p}")

    # Test 1: R glmnet with standardize=TRUE (default)
    print("\n" + "="*70)
    print("TEST 1: R glmnet with standardize=TRUE (default)")
    print("="*70)

    glmnet_r = importr('glmnet')

    X_r = robjects.r['as.matrix'](X)
    y_r = robjects.FloatVector(y)

    fit_std = glmnet_r.glmnet(x=X_r, y=y_r, alpha=1.0, standardize=True)

    lambda_std = np.array(fit_std.rx2('lambda'))
    beta_std = np.array(robjects.r['as.matrix'](fit_std.rx2('beta')))

    print(f"Lambda range: [{lambda_std.min():.6f}, {lambda_std.max():.6f}]")
    print(f"Number of lambdas: {len(lambda_std)}")
    print(f"Beta shape: {beta_std.shape}")
    print(f"Max |beta| at first lambda: {np.max(np.abs(beta_std[:, 0])):.6f}")
    print(f"Max |beta| at last lambda: {np.max(np.abs(beta_std[:, -1])):.6f}")

    # Compute entry times
    entry_times_std = np.zeros(p)
    for j in range(p):
        nonzero = np.where(np.abs(beta_std[j, :]) > 0)[0]
        entry_times_std[j] = lambda_std[nonzero[0]] * n if len(nonzero) > 0 else 0

    print(f"Entry times range: [{entry_times_std.min():.4f}, {entry_times_std.max():.4f}]")
    print(f"Variable 55 entry time: {entry_times_std[55]:.4f}")

    # Test 2: R glmnet with standardize=FALSE
    print("\n" + "="*70)
    print("TEST 2: R glmnet with standardize=FALSE")
    print("="*70)

    fit_nostd = glmnet_r.glmnet(x=X_r, y=y_r, alpha=1.0, standardize=False)

    lambda_nostd = np.array(fit_nostd.rx2('lambda'))
    beta_nostd = np.array(robjects.r['as.matrix'](fit_nostd.rx2('beta')))

    print(f"Lambda range: [{lambda_nostd.min():.6f}, {lambda_nostd.max():.6f}]")
    print(f"Number of lambdas: {len(lambda_nostd)}")
    print(f"Max |beta| at first lambda: {np.max(np.abs(beta_nostd[:, 0])):.6f}")
    print(f"Max |beta| at last lambda: {np.max(np.abs(beta_nostd[:, -1])):.6f}")

    # Compute entry times
    entry_times_nostd = np.zeros(p)
    for j in range(p):
        nonzero = np.where(np.abs(beta_nostd[j, :]) > 0)[0]
        entry_times_nostd[j] = lambda_nostd[nonzero[0]] * n if len(nonzero) > 0 else 0

    print(f"Entry times range: [{entry_times_nostd.min():.4f}, {entry_times_nostd.max():.4f}]")
    print(f"Variable 55 entry time: {entry_times_nostd[55]:.4f}")

    # Test 3: Check what lambda_max R computes internally
    print("\n" + "="*70)
    print("TEST 3: Lambda max computation")
    print("="*70)

    # R's lambda_max for standardize=TRUE
    # Formula: max(|X_std^T @ y_centered|) / n
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    y_centered = y - y.mean()

    lambda_max_manual_std = np.max(np.abs(X_std.T @ y_centered)) / n
    print(f"Manual lambda_max (standardized X): {lambda_max_manual_std:.6f}")
    print(f"R's actual lambda_max (standardize=TRUE): {lambda_std[0]:.6f}")
    print(f"Ratio: {lambda_std[0] / lambda_max_manual_std:.4f}")

    # R's lambda_max for standardize=FALSE
    lambda_max_manual_nostd = np.max(np.abs(X.T @ y_centered)) / n
    print(f"\nManual lambda_max (unstandardized X): {lambda_max_manual_nostd:.6f}")
    print(f"R's actual lambda_max (standardize=FALSE): {lambda_nostd[0]:.6f}")
    print(f"Ratio: {lambda_nostd[0] / lambda_max_manual_nostd:.4f}")

    # Test 4: Compare entry time correlations
    print("\n" + "="*70)
    print("TEST 4: Entry time comparison")
    print("="*70)

    corr = np.corrcoef(entry_times_std, entry_times_nostd)[0, 1]
    print(f"Correlation between standardize=TRUE and FALSE entry times: {corr:.4f}")

    # Which variables differ most?
    diff = np.abs(entry_times_std - entry_times_nostd)
    top_diff = np.argsort(diff)[-5:][::-1]
    print("\nTop 5 variables with largest entry time differences:")
    for idx in top_diff:
        print(f"  Var {idx}: std={entry_times_std[idx]:.4f}, nostd={entry_times_nostd[idx]:.4f}, diff={diff[idx]:.4f}")

    numpy2ri.deactivate()

    return {
        'entry_times_std': entry_times_std,
        'entry_times_nostd': entry_times_nostd,
        'lambda_std': lambda_std,
        'lambda_nostd': lambda_nostd,
    }


if __name__ == '__main__':
    test_r_glmnet()
