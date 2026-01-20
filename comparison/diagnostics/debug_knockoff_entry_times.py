#!/usr/bin/env python3
"""
Debug knockoff entry times - compare how variables enter in the augmented [X, Xk] matrix
for different glmnet implementations.

This will identify why knockoff_filter produces compressed W-statistics.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import lasso_path
import warnings

# Add knockoff-filter to path
KNOCKOFF_FILTER_PATH = "/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter"
if KNOCKOFF_FILTER_PATH not in sys.path:
    sys.path.insert(0, KNOCKOFF_FILTER_PATH)

# Add project src to path
PROJECT_ROOT = "/ix/djishnu/Aaron/1_general_use/SLIDE_py"
sys.path.insert(0, f"{PROJECT_ROOT}/src")

np.random.seed(42)


def compute_entry_times_comparison(X, y, nlambda=500):
    """
    Compare entry times in augmented [X, Xk] matrix for:
    1. knockoff-filter's stat_glmnet_lambdasmax (uses Fortran glmnet)
    2. custom implementation (uses sklearn lasso_path)
    """
    from knockpy.knockoffs import GaussianSampler
    from knockoff.stats import stat_glmnet_lambdasmax
    from knockoff.stats.glmnet import _lasso_max_lambda_glmnet, HAS_GLMNET
    from knockoff.stats.base import swap_columns, correct_for_swap, compute_signed_max_stat

    n, p = X.shape
    print(f"Data: n={n}, p={p}")
    print(f"Fortran glmnet available: {HAS_GLMNET}")
    print()

    # Generate knockoffs using knockpy (same for both methods)
    mu = np.mean(X, axis=0)
    Sigma = np.cov(X, rowvar=False)
    sampler = GaussianSampler(X=X, mu=mu, Sigma=Sigma, method='sdp')
    Xk = sampler.sample_knockoffs()
    print(f"Generated knockoffs: Xk shape = {Xk.shape}")
    print()

    # =========================================================================
    # Method 1: knockoff-filter's stat_glmnet_lambdasmax
    # =========================================================================
    print("=" * 70)
    print("METHOD 1: knockoff-filter stat_glmnet_lambdasmax (Fortran glmnet)")
    print("=" * 70)

    # Use the same random state for swapping
    np.random.seed(42)
    W_kf = stat_glmnet_lambdasmax(X, Xk, y.flatten())
    print(f"W-statistics: min={W_kf.min():.4f}, max={W_kf.max():.4f}")
    print(f"Top 5 W values: {np.sort(W_kf)[-5:]}")
    print(f"Variable with max W: {np.argmax(W_kf)} (W={W_kf.max():.4f})")
    print()

    # Now manually trace what happens inside
    np.random.seed(42)  # Reset to get same swap
    X_swap, Xk_swap, swap = swap_columns(X, Xk)
    X_combined = np.hstack([X_swap, Xk_swap])

    print("Inside stat_glmnet_lambdasmax:")
    print(f"  Swap pattern (first 10): {swap[:10]}")
    print(f"  X_combined shape: {X_combined.shape}")

    # Call the internal function to get entry times
    Z_kf = _lasso_max_lambda_glmnet(X_combined, y.flatten(), nlambda=nlambda)
    print(f"  Raw Z (entry times) range: [{Z_kf.min():.4f}, {Z_kf.max():.4f}]")
    print(f"  Z for original vars (Z[:p]): [{Z_kf[:p].min():.4f}, {Z_kf[:p].max():.4f}]")
    print(f"  Z for knockoffs (Z[p:]): [{Z_kf[p:].min():.4f}, {Z_kf[p:].max():.4f}]")

    # Compute W manually from Z
    W_manual = compute_signed_max_stat(Z_kf, p)
    W_manual = correct_for_swap(W_manual, swap)
    print(f"  W from manual computation: min={W_manual.min():.4f}, max={W_manual.max():.4f}")
    print()

    # =========================================================================
    # Method 2: custom glmnet (sklearn lasso_path, no standardization)
    # =========================================================================
    print("=" * 70)
    print("METHOD 2: custom glmnet (sklearn lasso_path, no standardization)")
    print("=" * 70)

    np.random.seed(42)  # Reset to get same swap
    swap2 = np.random.binomial(1, 0.5, size=p)
    X_swap2 = X * (1 - swap2) + Xk * swap2
    Xk_swap2 = X * swap2 + Xk * (1 - swap2)
    X_full = np.hstack([X_swap2, Xk_swap2])

    print(f"Swap pattern (first 10): {swap2[:10]}")
    print(f"Swaps match: {np.array_equal(swap, swap2)}")
    print(f"X_full shape: {X_full.shape}")

    # Lambda sequence (NO standardization)
    lambda_max = np.max(np.abs(X_full.T @ y.flatten())) / n
    lambda_min = lambda_max * 0.0005
    lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), nlambda)
    print(f"Lambda range: [{lambda_min:.6f}, {lambda_max:.6f}]")

    # Lasso path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, coef_path, _ = lasso_path(X_full, y.flatten(), alphas=lambdas, max_iter=10000)

    # Entry times
    Z_custom = np.zeros(2 * p)
    for j in range(2 * p):
        nonzero = np.where(np.abs(coef_path[j, :]) > 1e-10)[0]
        Z_custom[j] = lambdas[nonzero[0]] * n if len(nonzero) > 0 else 0

    print(f"Raw Z (entry times) range: [{Z_custom.min():.4f}, {Z_custom.max():.4f}]")
    print(f"Z for original vars (Z[:p]): [{Z_custom[:p].min():.4f}, {Z_custom[:p].max():.4f}]")
    print(f"Z for knockoffs (Z[p:]): [{Z_custom[p:].min():.4f}, {Z_custom[p:].max():.4f}]")

    # W statistics
    Z_orig = Z_custom[:p]
    Z_knock = Z_custom[p:]
    W_custom = np.maximum(Z_orig, Z_knock) * np.sign(Z_orig - Z_knock)
    W_custom = W_custom * (1 - 2 * swap2)

    print(f"W-statistics: min={W_custom.min():.4f}, max={W_custom.max():.4f}")
    print(f"Top 5 W values: {np.sort(W_custom)[-5:]}")
    print(f"Variable with max W: {np.argmax(W_custom)} (W={W_custom.max():.4f})")
    print()

    # =========================================================================
    # Method 3: knockoff-filter's _lasso_max_lambda_glmnet on UNSTANDARDIZED data
    # =========================================================================
    print("=" * 70)
    print("METHOD 3: Fortran glmnet on UNSTANDARDIZED data (like custom)")
    print("=" * 70)

    try:
        from knockoff._vendor.glmnet import ElasticNet

        np.random.seed(42)
        # Use same X_full as method 2 (no standardization)
        lambda_max3 = np.max(np.abs(X_full.T @ y.flatten())) / n
        lambda_min3 = lambda_max3 * 0.0005
        lambdas3 = np.logspace(np.log10(lambda_max3), np.log10(lambda_min3), nlambda)

        model = ElasticNet(
            alpha=1.0,
            n_lambda=nlambda,
            lambda_path=lambdas3,
            standardize=False,
            fit_intercept=False,
            n_splits=0,
            tol=1e-7,
            max_iter=100000,
        )
        # Don't center y
        model.fit(X_full, y.flatten())

        fortran_lambdas = model.lambda_path_
        fortran_coefs = model.coef_path_

        # Entry times
        Z_fortran = np.zeros(2 * p)
        for j in range(2 * p):
            nonzero = np.where(np.abs(fortran_coefs[j, :]) > 1e-10)[0]
            Z_fortran[j] = fortran_lambdas[nonzero[0]] * n if len(nonzero) > 0 else 0

        print(f"Lambda range: [{lambdas3.min():.6f}, {lambdas3.max():.6f}]")
        print(f"Raw Z (entry times) range: [{Z_fortran.min():.4f}, {Z_fortran.max():.4f}]")
        print(f"Z for original vars: [{Z_fortran[:p].min():.4f}, {Z_fortran[:p].max():.4f}]")
        print(f"Z for knockoffs: [{Z_fortran[p:].min():.4f}, {Z_fortran[p:].max():.4f}]")

        # W statistics
        Z_orig3 = Z_fortran[:p]
        Z_knock3 = Z_fortran[p:]
        W_fortran = np.maximum(Z_orig3, Z_knock3) * np.sign(Z_orig3 - Z_knock3)
        W_fortran = W_fortran * (1 - 2 * swap2)

        print(f"W-statistics: min={W_fortran.min():.4f}, max={W_fortran.max():.4f}")
        print(f"Top 5 W values: {np.sort(W_fortran)[-5:]}")
        print(f"Variable with max W: {np.argmax(W_fortran)} (W={W_fortran.max():.4f})")

    except Exception as e:
        print(f"Method 3 failed: {e}")
        Z_fortran = None
        W_fortran = None

    print()

    # =========================================================================
    # Compare the three methods
    # =========================================================================
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print("Lambda max values:")
    print(f"  Method 1 (knockoff-filter, standardized): uses internal lambda computation")
    print(f"  Method 2 (custom sklearn, unstandardized): {lambda_max:.6f}")
    print(f"  Method 3 (Fortran, unstandardized): {lambda_max3:.6f}")
    print()
    print("W-statistic ranges:")
    print(f"  Method 1 (knockoff-filter): [{W_kf.min():.4f}, {W_kf.max():.4f}]")
    print(f"  Method 2 (custom sklearn):  [{W_custom.min():.4f}, {W_custom.max():.4f}]")
    if W_fortran is not None:
        print(f"  Method 3 (Fortran unstand): [{W_fortran.min():.4f}, {W_fortran.max():.4f}]")
    print()
    print("Variable with max W:")
    print(f"  Method 1: var {np.argmax(W_kf)} (W={W_kf.max():.4f})")
    print(f"  Method 2: var {np.argmax(W_custom)} (W={W_custom.max():.4f})")
    if W_fortran is not None:
        print(f"  Method 3: var {np.argmax(W_fortran)} (W={W_fortran.max():.4f})")
    print()

    # Check correlation
    print("Correlations:")
    print(f"  Method 1 vs Method 2: {np.corrcoef(W_kf, W_custom)[0,1]:.4f}")
    if W_fortran is not None:
        print(f"  Method 2 vs Method 3: {np.corrcoef(W_custom, W_fortran)[0,1]:.4f}")
        print(f"  Method 1 vs Method 3: {np.corrcoef(W_kf, W_fortran)[0,1]:.4f}")
    print()

    # =========================================================================
    # Debug variable 55 specifically (the one R identifies)
    # =========================================================================
    print("=" * 70)
    print("VARIABLE 55 ANALYSIS (the variable R identifies as significant)")
    print("=" * 70)
    var_idx = 55
    print()
    print(f"Method 1 (knockoff-filter, standardized):")
    print(f"  Z[{var_idx}] (original): {Z_kf[var_idx]:.4f}")
    print(f"  Z[{var_idx}+p] (knockoff): {Z_kf[var_idx + p]:.4f}")
    print(f"  swap[{var_idx}]: {swap[var_idx]}")
    print(f"  W[{var_idx}]: {W_kf[var_idx]:.4f}")
    print()
    print(f"Method 2 (custom sklearn, unstandardized):")
    print(f"  Z[{var_idx}] (original): {Z_custom[var_idx]:.4f}")
    print(f"  Z[{var_idx}+p] (knockoff): {Z_custom[var_idx + p]:.4f}")
    print(f"  swap[{var_idx}]: {swap2[var_idx]}")
    print(f"  W[{var_idx}]: {W_custom[var_idx]:.4f}")
    print()
    if Z_fortran is not None:
        print(f"Method 3 (Fortran unstandardized):")
        print(f"  Z[{var_idx}] (original): {Z_fortran[var_idx]:.4f}")
        print(f"  Z[{var_idx}+p] (knockoff): {Z_fortran[var_idx + p]:.4f}")
        print(f"  W[{var_idx}]: {W_fortran[var_idx]:.4f}")

    return {
        'W_knockoff_filter': W_kf,
        'W_custom': W_custom,
        'W_fortran_unstd': W_fortran,
        'Z_knockoff_filter': Z_kf,
        'Z_custom': Z_custom,
        'Z_fortran_unstd': Z_fortran,
    }


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

    results = compute_entry_times_comparison(X, y)
    return results


if __name__ == '__main__':
    main()
