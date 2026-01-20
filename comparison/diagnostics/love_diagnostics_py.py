#!/usr/bin/env python3
"""
LOVE Diagnostic Script - Python Implementation

Runs Python LOVE with fixed delta (no CV) and saves all intermediate values
for comparison with R implementation.

Usage:
    python love_diagnostics_py.py --data_path /path/to/data --delta 0.05 --output_dir ./diagnostics_output
"""

import numpy as np
import pandas as pd
import os
import argparse
import pickle
from pathlib import Path


def run_love_diagnostics(
    X: np.ndarray,
    delta: float = 0.05,
    thresh_fdr: float = 0.2,
    output_dir: str = "./love_diagnostics",
    **kwargs
):
    """
    Run LOVE with diagnostic output at each step.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n x p)
    delta : float
        Fixed delta value (no CV)
    thresh_fdr : float
        FDR threshold for correlation matrix
    output_dir : str
        Directory to save diagnostic outputs
    """
    from loveslide.love_pkg.love.utilities import (
        thresh_sigma, corr_to_p, _benjamini_hochberg
    )
    from loveslide.love_pkg.love.est_pure_homo import (
        EstAI, EstC, FindRowMax, FindPureNode, FindSignPureNode, RecoverAI
    )

    os.makedirs(output_dir, exist_ok=True)

    n, p = X.shape
    print(f"Data shape: n={n}, p={p}")

    # Step 1: Preprocessing (center and scale)
    print("\n=== Step 1: Preprocessing ===")
    X_centered = X - np.mean(X, axis=0)
    X_std = np.std(X, axis=0, ddof=1)
    X_std[X_std == 0] = 1
    X_scaled = X_centered / X_std

    np.savetxt(os.path.join(output_dir, "01_X_scaled.csv"), X_scaled, delimiter=",")
    print(f"X_scaled: min={X_scaled.min():.6f}, max={X_scaled.max():.6f}, mean={X_scaled.mean():.6f}")

    # Step 2: Compute correlation matrix
    print("\n=== Step 2: Correlation Matrix ===")
    R_hat = np.corrcoef(X_scaled, rowvar=False)
    Sigma = np.cov(X_scaled, rowvar=False)

    np.savetxt(os.path.join(output_dir, "02_R_hat_raw.csv"), R_hat, delimiter=",")
    np.savetxt(os.path.join(output_dir, "02_Sigma_raw.csv"), Sigma, delimiter=",")
    print(f"R_hat: min={R_hat.min():.6f}, max={R_hat.max():.6f}")
    print(f"Sigma: min={Sigma.min():.6f}, max={Sigma.max():.6f}")

    # Step 3: FDR thresholding
    print("\n=== Step 3: FDR Thresholding ===")

    # Compute p-values
    p_values = corr_to_p(R_hat, n)
    np.savetxt(os.path.join(output_dir, "03_pvalues_raw.csv"), p_values, delimiter=",")

    # Benjamini-Hochberg correction
    p_flat = p_values.flatten()
    p_adjusted = _benjamini_hochberg(p_flat)
    p_adjusted_mat = p_adjusted.reshape(R_hat.shape)
    np.savetxt(os.path.join(output_dir, "03_pvalues_adjusted.csv"), p_adjusted_mat, delimiter=",")

    # Apply threshold
    kept_entries = (p_adjusted_mat <= thresh_fdr).astype(float)
    R_thresh = R_hat * kept_entries
    Sigma_thresh = Sigma * kept_entries  # Note: Python LOVE applies to Sigma too

    np.savetxt(os.path.join(output_dir, "03_kept_entries.csv"), kept_entries, delimiter=",")
    np.savetxt(os.path.join(output_dir, "03_R_hat_thresholded.csv"), R_thresh, delimiter=",")
    np.savetxt(os.path.join(output_dir, "03_Sigma_thresholded.csv"), Sigma_thresh, delimiter=",")

    n_kept = np.sum(kept_entries) - p  # Exclude diagonal
    n_total = p * (p - 1)
    print(f"Kept {n_kept}/{n_total} off-diagonal entries ({100*n_kept/n_total:.1f}%)")

    # Step 4: Scale delta
    print("\n=== Step 4: Delta Scaling ===")
    se_est = np.std(X_scaled, axis=0, ddof=1)
    delta_scaled = delta * np.sqrt(np.log(max(p, n)) / n)

    np.savetxt(os.path.join(output_dir, "04_se_est.csv"), se_est, delimiter=",")
    print(f"delta_raw={delta}, delta_scaled={delta_scaled:.6f}")
    print(f"se_est: min={se_est.min():.6f}, max={se_est.max():.6f}, mean={se_est.mean():.6f}")

    with open(os.path.join(output_dir, "04_delta_values.txt"), "w") as f:
        f.write(f"delta_raw={delta}\n")
        f.write(f"delta_scaled={delta_scaled}\n")
        f.write(f"log(max(p,n))/n = {np.log(max(p, n)) / n}\n")
        f.write(f"sqrt(log(max(p,n))/n) = {np.sqrt(np.log(max(p, n)) / n)}\n")

    # Step 5: Find row maxima
    print("\n=== Step 5: Find Row Maxima ===")
    off_Sigma = np.abs(Sigma_thresh.copy())
    np.fill_diagonal(off_Sigma, 0)

    result_Ms = FindRowMax(off_Sigma)
    Ms = result_Ms['M']
    arg_Ms = result_Ms['arg_M']

    np.savetxt(os.path.join(output_dir, "05_off_Sigma.csv"), off_Sigma, delimiter=",")
    np.savetxt(os.path.join(output_dir, "05_row_max_values.csv"), Ms, delimiter=",")
    np.savetxt(os.path.join(output_dir, "05_row_max_indices.csv"), arg_Ms, delimiter=",", fmt="%d")
    print(f"Ms: min={Ms.min():.6f}, max={Ms.max():.6f}")

    # Step 6: Find pure nodes
    print("\n=== Step 6: Find Pure Nodes ===")
    merge = False  # Use union (matching R SLIDE)
    resultPure = FindPureNode(off_Sigma, delta_scaled, Ms, arg_Ms, se_est, merge)
    estPureIndices = resultPure['pureInd']
    estPureVec = resultPure['pureVec']

    with open(os.path.join(output_dir, "06_pure_indices.txt"), "w") as f:
        f.write(f"Number of groups: {len(estPureIndices)}\n")
        f.write(f"Total pure variables: {len(estPureVec)}\n\n")
        for i, group in enumerate(estPureIndices):
            f.write(f"Group {i}: {sorted(group)}\n")
        f.write(f"\nAll pure indices: {sorted(estPureVec)}\n")

    if len(estPureVec) > 0:
        np.savetxt(os.path.join(output_dir, "06_pure_vec.csv"),
                   np.array(sorted(estPureVec)), delimiter=",", fmt="%d")

    print(f"Found {len(estPureIndices)} groups with {len(estPureVec)} pure variables")

    # Step 7: Find sign partition
    print("\n=== Step 7: Sign Partition ===")
    estSignPureIndices = FindSignPureNode(estPureIndices, Sigma_thresh)

    with open(os.path.join(output_dir, "07_sign_partition.txt"), "w") as f:
        for i, group in enumerate(estSignPureIndices):
            f.write(f"Group {i}: pos={group['pos']}, neg={group['neg']}\n")

    # Step 8: Recover AI matrix
    print("\n=== Step 8: Recover AI Matrix ===")
    AI = RecoverAI(estSignPureIndices, p)
    K = AI.shape[1]

    np.savetxt(os.path.join(output_dir, "08_AI_matrix.csv"), AI, delimiter=",")
    print(f"AI shape: {AI.shape} (K={K})")

    # Step 9: Estimate C (covariance of Z)
    print("\n=== Step 9: Estimate C Matrix ===")
    C_hat = EstC(Sigma_thresh, AI, diagonal=False)

    np.savetxt(os.path.join(output_dir, "09_C_hat.csv"), C_hat, delimiter=",")
    print(f"C_hat shape: {C_hat.shape}")
    print(f"C_hat diagonal: {np.diag(C_hat)}")

    # Step 10: Estimate Gamma (error variance)
    print("\n=== Step 10: Estimate Gamma ===")
    Gamma_hat = np.zeros(p)
    I_hat_list = list(estPureVec)
    if len(I_hat_list) > 0:
        diag_Sigma_I = np.diag(Sigma_thresh[np.ix_(I_hat_list, I_hat_list)])
        A_I = AI[I_hat_list, :]
        diag_ACA = np.diag(A_I @ C_hat @ A_I.T)
        Gamma_hat[I_hat_list] = diag_Sigma_I - diag_ACA

    Gamma_hat[Gamma_hat < 0] = 0

    np.savetxt(os.path.join(output_dir, "10_Gamma_hat.csv"), Gamma_hat, delimiter=",")
    print(f"Gamma_hat: min={Gamma_hat.min():.6f}, max={Gamma_hat.max():.6f}")

    # Save summary
    print("\n=== Summary ===")
    summary = {
        'n': n,
        'p': p,
        'delta_raw': delta,
        'delta_scaled': delta_scaled,
        'thresh_fdr': thresh_fdr,
        'K': K,
        'n_pure_variables': len(estPureVec),
        'n_groups': len(estPureIndices),
        'pure_indices': [list(g) for g in estPureIndices],
        'pure_vec': list(estPureVec) if len(estPureVec) > 0 else [],
    }

    with open(os.path.join(output_dir, "summary.pkl"), "wb") as f:
        pickle.dump(summary, f)

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"K (number of latent factors): {K}")
    print(f"Pure variables: {len(estPureVec)}")
    print(f"Groups: {len(estPureIndices)}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="LOVE Diagnostic Script - Python")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data CSV (samples x features)")
    parser.add_argument("--delta", type=float, default=0.05,
                        help="Fixed delta value (no CV)")
    parser.add_argument("--thresh_fdr", type=float, default=0.2,
                        help="FDR threshold")
    parser.add_argument("--output_dir", type=str, default="./love_diagnostics_py",
                        help="Output directory")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path, index_col=0)
    X = df.values
    print(f"Loaded data: {X.shape}")

    # Run diagnostics
    run_love_diagnostics(
        X=X,
        delta=args.delta,
        thresh_fdr=args.thresh_fdr,
        output_dir=args.output_dir
    )

    print(f"\nDiagnostic outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
