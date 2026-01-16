#!/usr/bin/env python3
"""
Step-by-step comparison of Python LOVE vs R LOVE intermediate results.

This script loads R intermediate results (saved by save_r_intermediate.R)
and compares them with Python calculations at each step to identify
the first point of divergence.

Usage:
    python compare_step_by_step.py <data_file> <r_output_dir> [--mode hetero|homo] [--fixed-delta VALUE]
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Add the love_pkg to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "loveslide"))

from love_pkg.love.score import Score_mat
from love_pkg.love.est_pure_hetero import Est_Pure, Est_BI_C, Re_Est_Pure
from love_pkg.love.est_pure_homo import EstAI, EstC, FindSignPureNode


def load_r_csv(filepath: Path) -> np.ndarray:
    """Load R CSV output as numpy array."""
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath)
    return df.values


def load_r_single_value(filepath: Path) -> float:
    """Load single value from R CSV."""
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath)
    return df.iloc[0, 0]


def compare_matrices(py_mat: np.ndarray, r_mat: np.ndarray, name: str,
                     tol: float = 1e-10) -> dict:
    """Compare Python and R matrices."""
    if r_mat is None:
        return {"status": "SKIP", "message": f"R {name} not found"}
    if py_mat is None:
        return {"status": "SKIP", "message": f"Python {name} not computed"}

    if py_mat.shape != r_mat.shape:
        return {
            "status": "FAIL",
            "message": f"Shape mismatch: Python {py_mat.shape} vs R {r_mat.shape}"
        }

    # Handle NaN values
    py_nan = np.isnan(py_mat)
    r_nan = np.isnan(r_mat)

    if not np.array_equal(py_nan, r_nan):
        return {
            "status": "FAIL",
            "message": f"NaN pattern mismatch: Python has {np.sum(py_nan)} NaNs, R has {np.sum(r_nan)} NaNs"
        }

    # Compare non-NaN values
    mask = ~py_nan
    if np.sum(mask) == 0:
        return {"status": "PASS", "message": "All NaN (empty comparison)"}

    diff = np.abs(py_mat[mask] - r_mat[mask])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    if max_diff <= tol:
        return {
            "status": "PASS",
            "message": f"Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}"
        }
    else:
        # Find location of max difference
        if len(py_mat.shape) == 2:
            max_idx = np.unravel_index(np.argmax(np.abs(py_mat - r_mat) * mask.astype(float)), py_mat.shape)
            return {
                "status": "FAIL",
                "message": f"Max diff: {max_diff:.2e} at {max_idx}, Mean diff: {mean_diff:.2e}"
            }
        else:
            max_idx = np.argmax(np.abs(py_mat - r_mat) * mask.astype(float))
            return {
                "status": "FAIL",
                "message": f"Max diff: {max_diff:.2e} at idx {max_idx}, Mean diff: {mean_diff:.2e}"
            }


def compare_scalars(py_val, r_val, name: str, tol: float = 1e-10) -> dict:
    """Compare Python and R scalar values."""
    if r_val is None:
        return {"status": "SKIP", "message": f"R {name} not found"}
    if py_val is None:
        return {"status": "SKIP", "message": f"Python {name} not computed"}

    diff = abs(py_val - r_val)
    if diff <= tol:
        return {"status": "PASS", "message": f"Diff: {diff:.2e}"}
    else:
        return {"status": "FAIL", "message": f"Python={py_val}, R={r_val}, Diff={diff:.2e}"}


def compare_lists(py_list, r_list, name: str) -> dict:
    """Compare Python and R lists/arrays of indices."""
    if r_list is None:
        return {"status": "SKIP", "message": f"R {name} not found"}
    if py_list is None:
        return {"status": "SKIP", "message": f"Python {name} not computed"}

    py_set = set(py_list)
    r_set = set(r_list)

    if py_set == r_set:
        return {"status": "PASS", "message": f"Both have {len(py_set)} elements"}
    else:
        only_py = py_set - r_set
        only_r = r_set - py_set
        return {
            "status": "FAIL",
            "message": f"Py only: {len(only_py)}, R only: {len(only_r)}, Common: {len(py_set & r_set)}"
        }


def print_result(step: str, result: dict, verbose: bool = True):
    """Print comparison result with color coding."""
    status = result["status"]
    if status == "PASS":
        color = "\033[92m"  # Green
        symbol = "✓"
    elif status == "FAIL":
        color = "\033[91m"  # Red
        symbol = "✗"
    else:
        color = "\033[93m"  # Yellow
        symbol = "○"

    reset = "\033[0m"
    print(f"{color}[{symbol}]{reset} {step}: {result['message']}")
    return status == "PASS"


def run_hetero_comparison(data_file: str, r_dir: Path, fixed_delta: float = None,
                          seed: int = 42, verbose: bool = True):
    """Run step-by-step comparison for heterogeneous case."""

    print("\n" + "="*80)
    print("HETEROGENEOUS CASE (pure_homo=False) STEP-BY-STEP COMPARISON")
    print("="*80)

    np.random.seed(seed)

    all_passed = True
    first_failure = None

    # Step 0: Load data
    print("\n--- Step 0: Load data ---")
    X_df = pd.read_csv(data_file, index_col=0)
    X = X_df.values
    n, p = X.shape
    print(f"Data dimensions: {n} samples x {p} features")

    r_X = load_r_csv(r_dir / "step0_X_input.csv")
    if r_X is not None:
        result = compare_matrices(X, r_X[:, 1:] if r_X.shape[1] == p + 1 else r_X, "X_input")
        if not print_result("X_input", result):
            all_passed = False
            if first_failure is None:
                first_failure = "Step 0: X_input"

    # Step 1: Center data
    print("\n--- Step 1: Center data ---")
    X_centered = X - np.mean(X, axis=0)

    r_X_centered = load_r_csv(r_dir / "step1_X_centered.csv")
    if r_X_centered is not None:
        # R CSV may have rownames
        r_vals = r_X_centered[:, 1:] if r_X_centered.shape[1] == p + 1 else r_X_centered
        result = compare_matrices(X_centered, r_vals, "X_centered")
        if not print_result("X_centered", result):
            all_passed = False
            if first_failure is None:
                first_failure = "Step 1: X_centered"

    # Step 2: Correlation and covariance matrices
    print("\n--- Step 2: Correlation and covariance ---")
    R_hat = np.corrcoef(X_centered, rowvar=False)
    Sigma = np.cov(X_centered, rowvar=False)

    r_R = load_r_csv(r_dir / "step2_R_corr.csv")
    r_Sigma = load_r_csv(r_dir / "step2_Sigma_cov.csv")

    result = compare_matrices(R_hat, r_R, "R_corr")
    if not print_result("R_corr", result):
        all_passed = False
        if first_failure is None:
            first_failure = "Step 2: R_corr"

    result = compare_matrices(Sigma, r_Sigma, "Sigma_cov")
    if not print_result("Sigma_cov", result):
        all_passed = False
        if first_failure is None:
            first_failure = "Step 2: Sigma_cov"

    # Step 3: Score matrix
    print("\n--- Step 3: Score matrix ---")
    score_res = Score_mat(R_hat, q=2, exact=False)
    score_mat = score_res['score']
    moments_mat = score_res['moments']

    r_score = load_r_csv(r_dir / "step3_score_mat.csv")
    r_moments = load_r_csv(r_dir / "step3_moments_M.csv")

    result = compare_matrices(score_mat, r_score, "score_mat")
    if not print_result("score_mat", result):
        all_passed = False
        if first_failure is None:
            first_failure = "Step 3: score_mat"

    result = compare_matrices(moments_mat, r_moments, "moments_M")
    if not print_result("moments_M", result):
        all_passed = False
        if first_failure is None:
            first_failure = "Step 3: moments_M"

    # Step 4: Delta selection
    print("\n--- Step 4: Delta selection ---")
    if fixed_delta is not None:
        delta_min = fixed_delta
        print(f"Using fixed delta: {fixed_delta}")
    else:
        r_delta_min = load_r_single_value(r_dir / "step4_delta_min.csv")
        if r_delta_min is not None:
            delta_min = r_delta_min
            print(f"Using R delta_min: {delta_min}")
        else:
            # Compute our own
            max_pure = 1.0 if n > p else 0.8
            row_mins = np.nanmin(score_mat, axis=1)
            delta_max = np.quantile(row_mins[np.isfinite(row_mins)], max_pure)
            delta_min_val = np.nanmin(score_mat)
            delta_grid = np.linspace(delta_max, delta_min_val, 50)
            delta_min = delta_grid[len(delta_grid) // 2]
            print(f"Computed delta_min: {delta_min}")

    # Step 5: Est_Pure
    print("\n--- Step 5: Est_Pure ---")
    pure_res = Est_Pure(score_mat, delta_min)
    K_init = pure_res['K']
    I_init = pure_res['I']
    I_part_init = pure_res['I_part']

    print(f"Python: K={K_init}, |I|={len(I_init)}")

    r_K = load_r_single_value(r_dir / "step5_K_init.csv")
    r_I_df = pd.read_csv(r_dir / "step5_I_init.csv") if (r_dir / "step5_I_init.csv").exists() else None

    if r_K is not None:
        r_K = int(r_K)
        print(f"R: K={r_K}")
        result = compare_scalars(K_init, r_K, "K_init")
        if not print_result("K_init", result):
            all_passed = False
            if first_failure is None:
                first_failure = "Step 5: K_init"

    if r_I_df is not None:
        # R uses 1-based indexing
        r_I = r_I_df['I'].values - 1  # Convert to 0-based
        result = compare_lists(list(I_init), list(r_I), "I_init")
        if not print_result("I_init", result):
            all_passed = False
            if first_failure is None:
                first_failure = "Step 5: I_init"

    # Load R I_part for later use
    r_I_part = None
    r_I_part_df = pd.read_csv(r_dir / "step5_I_part_init.csv") if (r_dir / "step5_I_part_init.csv").exists() else None
    if r_I_part_df is not None:
        r_I_part = []
        for g in r_I_part_df['group'].unique():
            indices = r_I_part_df[r_I_part_df['group'] == g]['idx'].values - 1  # 0-based
            r_I_part.append(list(indices))

    # Step 6: Est_BI_C (first call)
    if K_init >= 2:
        print("\n--- Step 6: Est_BI_C (first and only call) ---")

        # Use R's I_part if available (to match exactly)
        use_I_part = r_I_part if r_I_part is not None else I_part_init
        use_I = np.array(sorted(set(idx for part in use_I_part for idx in part)))

        BI_C_res = Est_BI_C(moments_mat, R_hat, use_I_part, use_I)

        B_hat = BI_C_res['B']
        C_hat = BI_C_res['C']
        B_left_inv = BI_C_res['B_left_inv']
        Gamma = BI_C_res['Gamma']

        r_B = load_r_csv(r_dir / "step6_B_hat.csv")
        r_C = load_r_csv(r_dir / "step6_C_hat.csv")
        r_B_left_inv = load_r_csv(r_dir / "step6_B_left_inv.csv")
        r_Gamma_df = pd.read_csv(r_dir / "step6_Gamma.csv") if (r_dir / "step6_Gamma.csv").exists() else None
        r_Gamma = r_Gamma_df['Gamma'].values if r_Gamma_df is not None else None

        result = compare_matrices(B_hat, r_B, "B_hat")
        if not print_result("B_hat", result):
            all_passed = False
            if first_failure is None:
                first_failure = "Step 6: B_hat"

        result = compare_matrices(C_hat, r_C, "C_hat")
        if not print_result("C_hat", result):
            all_passed = False
            if first_failure is None:
                first_failure = "Step 6: C_hat"

        result = compare_matrices(B_left_inv, r_B_left_inv, "B_left_inv")
        if not print_result("B_left_inv", result):
            all_passed = False
            if first_failure is None:
                first_failure = "Step 6: B_left_inv"

        if r_Gamma is not None:
            result = compare_matrices(Gamma, r_Gamma, "Gamma")
            if not print_result("Gamma", result):
                all_passed = False
                if first_failure is None:
                    first_failure = "Step 6: Gamma"

        # Step 7: Re_Est_Pure
        print("\n--- Step 7: Re_Est_Pure ---")
        est_I_updated = Re_Est_Pure(X_centered, Sigma, moments_mat, use_I_part, Gamma)
        est_I_set_updated = np.array([idx for part in est_I_updated for idx in part])
        K_updated = len(est_I_updated)

        print(f"Python: K_updated={K_updated}, |I|={len(est_I_set_updated)}")

        r_K_updated = load_r_single_value(r_dir / "step7_K_updated.csv")
        if r_K_updated is not None:
            r_K_updated = int(r_K_updated)
            print(f"R: K_updated={r_K_updated}")
            result = compare_scalars(K_updated, r_K_updated, "K_updated")
            if not print_result("K_updated", result):
                all_passed = False
                if first_failure is None:
                    first_failure = "Step 7: K_updated"

        # Step 8: Final A matrix
        print("\n--- Step 8: Final A matrix ---")
        D_Sigma = np.diag(Sigma)
        B_scaled = np.sqrt(D_Sigma)[:, np.newaxis] * B_hat
        D_B = np.max(np.abs(B_scaled), axis=0)
        A_hat = B_scaled / D_B
        C_final = D_B[:, np.newaxis] * C_hat

        r_A = load_r_csv(r_dir / "step8_A_hat.csv")
        r_C_final = load_r_csv(r_dir / "step8_C_final.csv")

        result = compare_matrices(A_hat, r_A, "A_hat", tol=1e-8)
        if not print_result("A_hat", result):
            all_passed = False
            if first_failure is None:
                first_failure = "Step 8: A_hat"

        result = compare_matrices(C_final, r_C_final, "C_final", tol=1e-8)
        if not print_result("C_final", result):
            all_passed = False
            if first_failure is None:
                first_failure = "Step 8: C_final"

        # Step 9: Final Gamma
        print("\n--- Step 9: Final Gamma ---")
        Gamma_final = Gamma * D_Sigma
        Gamma_final[Gamma_final < 0] = 0

        r_Gamma_final_df = pd.read_csv(r_dir / "step9_Gamma_final.csv") if (r_dir / "step9_Gamma_final.csv").exists() else None
        r_Gamma_final = r_Gamma_final_df['Gamma'].values if r_Gamma_final_df is not None else None

        if r_Gamma_final is not None:
            result = compare_matrices(Gamma_final, r_Gamma_final, "Gamma_final", tol=1e-8)
            if not print_result("Gamma_final", result):
                all_passed = False
                if first_failure is None:
                    first_failure = "Step 9: Gamma_final"

    else:
        print(f"\nK={K_init} < 2, skipping Est_BI_C steps")

    # Summary
    print("\n" + "="*80)
    if all_passed:
        print("\033[92m✓ ALL STEPS PASSED\033[0m")
    else:
        print(f"\033[91m✗ FIRST FAILURE: {first_failure}\033[0m")
    print("="*80)

    return all_passed, first_failure


def main():
    parser = argparse.ArgumentParser(description="Step-by-step LOVE comparison")
    parser.add_argument("data_file", help="Input data CSV file")
    parser.add_argument("r_output_dir", help="Directory containing R intermediate outputs")
    parser.add_argument("--mode", default="hetero", choices=["hetero", "homo"],
                        help="Comparison mode (default: hetero)")
    parser.add_argument("--fixed-delta", type=float, default=None,
                        help="Use fixed delta value")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    r_dir = Path(args.r_output_dir)
    if not r_dir.exists():
        print(f"Error: R output directory not found: {r_dir}")
        sys.exit(1)

    if args.mode == "hetero":
        passed, failure = run_hetero_comparison(
            args.data_file, r_dir,
            fixed_delta=args.fixed_delta,
            seed=args.seed,
            verbose=args.verbose
        )
    else:
        print("Homogeneous comparison not yet implemented")
        sys.exit(1)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
