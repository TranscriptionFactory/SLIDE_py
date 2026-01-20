#!/usr/bin/env python3
"""
Compare R's scale() vs Python's StandardScaler standardization.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def test_standardization():
    """Compare R scale() vs Python StandardScaler."""
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    # Load test data
    z_path = "/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/output_comparison/20260119_225506_rstyle/0.1_0.5_out/z_matrix.csv"
    Z_df = pd.read_csv(z_path, index_col=0)
    X = Z_df.values.astype(np.float64)

    n, p = X.shape
    print(f"Data: n={n}, p={p}")

    # Python StandardScaler
    scaler = StandardScaler()  # Default: with_mean=True, with_std=True
    X_py = scaler.fit_transform(X)

    print("\n=== Python StandardScaler ===")
    print(f"Mean: {scaler.mean_[:5]}")
    print(f"Std (scale_): {scaler.scale_[:5]}")
    print(f"Var (var_): {scaler.var_[:5]}")
    print(f"X_py[0, :5]: {X_py[0, :5]}")
    print(f"X_py column 55 mean: {X_py[:, 55].mean():.10f}")
    print(f"X_py column 55 std: {X_py[:, 55].std():.10f}")

    # Python manual with ddof=1 (like R)
    X_means = np.mean(X, axis=0)
    X_stds_ddof1 = np.std(X, axis=0, ddof=1)  # R's default: sample std
    X_py_manual = (X - X_means) / X_stds_ddof1

    print("\n=== Python manual (ddof=1, like R) ===")
    print(f"Mean: {X_means[:5]}")
    print(f"Std (ddof=1): {X_stds_ddof1[:5]}")
    print(f"X_py_manual[0, :5]: {X_py_manual[0, :5]}")
    print(f"X_py_manual column 55 mean: {X_py_manual[:, 55].mean():.10f}")
    print(f"X_py_manual column 55 std: {X_py_manual[:, 55].std():.10f}")

    # R scale()
    X_r = robjects.r['as.matrix'](X)
    X_r_scaled = robjects.r['scale'](X_r)
    X_r_np = np.array(X_r_scaled)

    print("\n=== R scale() ===")
    print(f"X_r_scaled[0, :5]: {X_r_np[0, :5]}")
    print(f"X_r_scaled column 55 mean: {X_r_np[:, 55].mean():.10f}")
    print(f"X_r_scaled column 55 std: {X_r_np[:, 55].std():.10f}")

    # Compare
    print("\n=== Comparison ===")
    print(f"Python StandardScaler vs R scale() max diff: {np.max(np.abs(X_py - X_r_np)):.10f}")
    print(f"Python manual (ddof=1) vs R scale() max diff: {np.max(np.abs(X_py_manual - X_r_np)):.10f}")
    print(f"Python StandardScaler vs manual (ddof=1) max diff: {np.max(np.abs(X_py - X_py_manual)):.10f}")

    # Check std calculation
    print("\n=== Std calculation details ===")
    col = 55
    print(f"Column {col}:")
    print(f"  Raw values[:5]: {X[:5, col]}")
    print(f"  Python std (ddof=0, StandardScaler default): {np.std(X[:, col], ddof=0):.10f}")
    print(f"  Python std (ddof=1, R default): {np.std(X[:, col], ddof=1):.10f}")
    print(f"  R attr scale: {np.array(robjects.r['attr'](X_r_scaled, 'scaled:scale'))[col]:.10f}")

    numpy2ri.deactivate()


if __name__ == '__main__':
    test_standardization()
