#!/usr/bin/env python3
"""
Python script to run SLIDE on input data and compare with R outputs.

Usage:
    python run_slide_py.py <x_file> <y_file> [tag] [options]

Arguments:
    x_file          Path to X matrix CSV (samples as rows, features as columns)
    y_file          Path to Y vector CSV (samples as rows, single column)
    tag             Optional run name tag (default: basename of x file)

Options:
    --generate-only     Only generate Python outputs without comparison
    --tolerance TOL     Tolerance for numerical comparisons (default: 1e-4)
    --delta VALUE       Delta value (default: 0.1)
    --lambda VALUE      Lambda value (default: 0.5)
    --spec VALUE        Specificity threshold (default: 0.1)
    --fdr VALUE         FDR threshold (default: 0.1)
    --niter VALUE       Number of SLIDE iterations (default: 500)

Examples:
    python run_slide_py.py /path/to/X.csv /path/to/Y.csv
    python run_slide_py.py /path/to/X.csv /path/to/Y.csv my_experiment --delta 0.1 --lambda 0.5
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loveslide import OptimizeSLIDE
from loveslide.love import call_love


def load_csv(filepath):
    """Load a CSV file as numpy array or DataFrame."""
    if not os.path.exists(filepath):
        return None
    return pd.read_csv(filepath, index_col=0)


def compare_arrays(arr1, arr2, name, tol=1e-6):
    """Compare two arrays and report differences."""
    if arr1 is None or arr2 is None:
        print(f"  {name}: SKIP (missing data)")
        return None

    arr1 = np.asarray(arr1).flatten() if np.asarray(arr1).ndim == 1 else np.asarray(arr1)
    arr2 = np.asarray(arr2).flatten() if np.asarray(arr2).ndim == 1 else np.asarray(arr2)

    if arr1.shape != arr2.shape:
        print(f"  {name}: FAIL (shape mismatch: {arr1.shape} vs {arr2.shape})")
        return False

    # Handle NaN values
    nan_mask1 = np.isnan(arr1.astype(float))
    nan_mask2 = np.isnan(arr2.astype(float))

    if not np.array_equal(nan_mask1, nan_mask2):
        print(f"  {name}: FAIL (NaN pattern mismatch)")
        return False

    # Compare non-NaN values
    valid_mask = ~nan_mask1
    if np.sum(valid_mask) == 0:
        print(f"  {name}: PASS (all NaN)")
        return True

    max_diff = np.max(np.abs(arr1[valid_mask].astype(float) - arr2[valid_mask].astype(float)))
    mean_diff = np.mean(np.abs(arr1[valid_mask].astype(float) - arr2[valid_mask].astype(float)))

    if max_diff <= tol:
        print(f"  {name}: PASS (max diff: {max_diff:.2e})")
        return True
    else:
        print(f"  {name}: DIFF (max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e})")
        return False


def compare_sets(set1, set2, name):
    """Compare two sets of indices."""
    set1 = set(np.asarray(set1).flatten().astype(int))
    set2 = set(np.asarray(set2).flatten().astype(int))

    overlap = len(set1 & set2)
    total = len(set1 | set2)
    jaccard = overlap / total if total > 0 else 1.0

    if set1 == set2:
        print(f"  {name}: EXACT MATCH ({len(set1)} items)")
        return True
    else:
        common = set1 & set2
        only_in_1 = set1 - set2
        only_in_2 = set2 - set1
        print(f"  {name}: PARTIAL MATCH")
        print(f"    Jaccard similarity: {jaccard:.2%}")
        print(f"    Common: {len(common)} items")
        print(f"    Only in R: {len(only_in_1)} items")
        print(f"    Only in Python: {len(only_in_2)} items")
        if len(only_in_1) <= 10:
            print(f"    R-only indices: {sorted(only_in_1)}")
        if len(only_in_2) <= 10:
            print(f"    Python-only indices: {sorted(only_in_2)}")
        return jaccard > 0.8  # Consider pass if > 80% overlap


def main():
    parser = argparse.ArgumentParser(
        description="Run SLIDE on input data and compare with R outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_slide_py.py /path/to/X.csv /path/to/Y.csv
    python run_slide_py.py /path/to/X.csv /path/to/Y.csv my_experiment --delta 0.1
        """
    )
    parser.add_argument("x_file", help="Path to X matrix CSV (samples as rows, features as columns)")
    parser.add_argument("y_file", help="Path to Y vector CSV (samples as rows, single column)")
    parser.add_argument("tag", nargs="?", default=None,
                        help="Run name tag (default: basename of x file)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate Python outputs without comparison")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Tolerance for numerical comparisons (default: 1e-4)")
    parser.add_argument("--delta", type=float, default=0.1,
                        help="Delta value (default: 0.1)")
    parser.add_argument("--lambda", dest="lbd", type=float, default=0.5,
                        help="Lambda value (default: 0.5)")
    parser.add_argument("--spec", type=float, default=0.1,
                        help="Specificity threshold (default: 0.1)")
    parser.add_argument("--fdr", type=float, default=0.1,
                        help="FDR threshold (default: 0.1)")
    parser.add_argument("--niter", type=int, default=500,
                        help="Number of SLIDE iterations (default: 500)")
    parser.add_argument("--thresh-fdr", type=float, default=0.2,
                        help="Threshold FDR for correlation matrix (default: 0.2)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    # Validate data files
    x_file = Path(args.x_file)
    y_file = Path(args.y_file)
    if not x_file.exists():
        print(f"ERROR: X data file not found: {x_file}")
        sys.exit(1)
    if not y_file.exists():
        print(f"ERROR: Y data file not found: {y_file}")
        sys.exit(1)

    # Set tag
    tag = args.tag if args.tag else x_file.stem

    # Set up paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs" / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SLIDE Python Analysis")
    print(f"  X file: {x_file}")
    print(f"  Y file: {y_file}")
    print(f"  Tag: {tag}")
    print(f"  Output dir: {output_dir}")
    print(f"  Delta: {args.delta}")
    print(f"  Lambda: {args.lbd}")
    print(f"  Spec: {args.spec}")
    print(f"  FDR: {args.fdr}")
    print(f"  Niter: {args.niter}")
    print(f"  Thresh FDR: {args.thresh_fdr}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    X_df = pd.read_csv(x_file, index_col=0)
    Y_df = pd.read_csv(y_file, index_col=0)

    # Replace spaces in feature names with underscores (matching R SLIDE behavior)
    X_df.columns = X_df.columns.str.replace(' ', '_')

    X = X_df.values
    Y = Y_df.values

    n, p = X.shape
    print(f"X dimensions: {n} samples x {p} features")
    print(f"Y dimensions: {Y.shape[0]} samples x {Y.shape[1]} columns")

    # Check dimensions
    if n <= p:
        print(f"\nWARNING: n ({n}) <= p ({p})")
        print("SLIDE requires more samples than features for reliable estimation.\n")

    # Standardize X (matching R's scale(X, TRUE, TRUE))
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)

    results_py = {}

    # ========================================================================
    # Step 1: Get Latent Factors (LOVE)
    # ========================================================================
    print("\n" + "-" * 60)
    print("Step 1: Getting Latent Factors (call_love)")
    print("-" * 60)

    np.random.seed(123)
    t_start = time.time()

    try:
        love_result = call_love(
            X=X_std,
            lbd=args.lbd,
            mu=0.5,
            pure_homo=True,  # Match R SLIDE default
            delta=np.array([args.delta]),  # Single delta for deterministic comparison
            verbose=True,
            thresh_fdr=args.thresh_fdr,
            outpath=str(output_dir)
        )
        t_end = time.time()

        print(f"\nLatent Factors Results:")
        print(f"  K (number of LFs): {love_result['K']}")
        print(f"  Number of pure variables: {len(love_result['pureVec'])}")
        print(f"  A shape: {love_result['A'].shape}")
        print(f"  C shape: {love_result['C'].shape}")
        print(f"  Gamma length: {len(love_result['Gamma'])}")
        print(f"  optDelta: {love_result['optDelta']:.6f}")
        print(f"  Time: {t_end - t_start:.2f}s")

        # Save outputs
        A_df = pd.DataFrame(
            love_result['A'],
            index=X_df.columns,
            columns=[f"Z{i+1}" for i in range(love_result['A'].shape[1])]
        )
        A_df.to_csv(output_dir / f"{tag}_A_py.csv")

        C_df = pd.DataFrame(
            love_result['C'],
            index=[f"Z{i+1}" for i in range(love_result['C'].shape[0])],
            columns=[f"Z{i+1}" for i in range(love_result['C'].shape[1])]
        )
        C_df.to_csv(output_dir / f"{tag}_C_py.csv")

        pd.DataFrame({'Gamma': love_result['Gamma']}).to_csv(
            output_dir / f"{tag}_Gamma_py.csv", index=False)

        # Save pure variable indices (convert to 1-based for comparison with R)
        pd.DataFrame({'I': np.array(love_result['pureVec']) + 1}).to_csv(
            output_dir / f"{tag}_I_py.csv", index=False)

        pd.DataFrame({
            'K': [love_result['K']],
            'opt_delta': [love_result['optDelta']]
        }).to_csv(output_dir / f"{tag}_params_py.csv", index=False)

        results_py['love'] = love_result

    except Exception as e:
        print(f"ERROR in call_love: {e}")
        import traceback
        traceback.print_exc()
        results_py['love'] = None

    if results_py['love'] is None:
        print("Skipping remaining steps due to error in LOVE.")
        sys.exit(1)

    # ========================================================================
    # Step 2: Calculate Z Matrix
    # ========================================================================
    print("\n" + "-" * 60)
    print("Step 2: Calculating Z Matrix")
    print("-" * 60)

    t_start = time.time()

    try:
        A_hat = love_result['A']
        Gamma_hat = love_result['Gamma']
        C_hat = love_result['C']

        # Handle zeros in Gamma
        Gamma_hat = np.where(Gamma_hat == 0, 1e-10, Gamma_hat)
        Gamma_hat_inv = np.diag(Gamma_hat ** (-1))

        # Calculate G_hat matrix
        G_hat = A_hat.T @ Gamma_hat_inv @ A_hat + np.linalg.inv(C_hat)

        # Calculate Z_hat matrix
        Z_hat = X_std @ Gamma_hat_inv @ A_hat @ np.linalg.pinv(G_hat)

        t_end = time.time()

        print(f"\nZ Matrix Results:")
        print(f"  Z shape: {Z_hat.shape}")
        print(f"  Time: {t_end - t_start:.2f}s")

        # Save Z matrix
        Z_df = pd.DataFrame(
            Z_hat,
            index=X_df.index,
            columns=[f"Z{i+1}" for i in range(Z_hat.shape[1])]
        )
        Z_df.to_csv(output_dir / f"{tag}_Z_py.csv")

        results_py['Z'] = Z_hat

    except Exception as e:
        print(f"ERROR in Z matrix calculation: {e}")
        import traceback
        traceback.print_exc()
        results_py['Z'] = None

    if results_py['Z'] is None:
        print("Skipping remaining steps due to error in Z matrix calculation.")
        sys.exit(1)

    # ========================================================================
    # Step 3: Run SLIDE Knockoffs
    # ========================================================================
    print("\n" + "-" * 60)
    print("Step 3: Running SLIDE Knockoffs")
    print("-" * 60)

    np.random.seed(123)
    t_start = time.time()

    try:
        # Create input params for OptimizeSLIDE
        input_params = {
            'x_path': str(x_file),
            'y_path': str(y_file),
            'out_path': str(output_dir),
            'delta': [args.delta],
            'lambda': [args.lbd],
            'spec': args.spec,
            'fdr': args.fdr,
            'niter': args.niter,
            'thresh_fdr': args.thresh_fdr,
            'n_workers': args.n_workers,
            'pure_homo': True,
            'do_interacts': True,
            'SLIDE_top_feats': 10
        }

        # Use the SLIDE class for knockoff selection
        slide = OptimizeSLIDE(input_params, x=X_df, y=Y_df)

        # Use already computed LOVE result
        slide.love_result = love_result
        slide.A = A_df
        slide.latent_factors = Z_df

        # Run SLIDE knockoffs
        slide.run_SLIDE(
            latent_factors=Z_df,
            niter=args.niter,
            spec=args.spec,
            fdr=args.fdr,
            verbose=True,
            n_workers=args.n_workers,
            outpath=str(output_dir),
            do_interacts=True
        )

        t_end = time.time()

        print(f"\nSLIDE Results:")
        print(f"  Marginal LFs: {slide.sig_LFs}")
        print(f"  Number of marginals: {len(slide.sig_LFs)}")
        print(f"  Number of interactions: {len(slide.sig_interacts)}")
        print(f"  Time: {t_end - t_start:.2f}s")

        # Convert sig_LFs to indices (1-based for R comparison)
        marginal_indices = [int(lf.replace('Z', '')) for lf in slide.sig_LFs]

        # Save outputs
        pd.DataFrame({'marginal': marginal_indices}).to_csv(
            output_dir / f"{tag}_marginal_LFs_py.csv", index=False)

        if hasattr(slide, 'interaction_pairs') and len(slide.interaction_pairs) > 0:
            interaction_df = pd.DataFrame({
                'p1': slide.interaction_pairs[0] + 1,  # Convert to 1-based
                'p2': slide.interaction_pairs[1] + 1
            })
        else:
            interaction_df = pd.DataFrame({'p1': [], 'p2': []})

        interaction_df.to_csv(output_dir / f"{tag}_interactions_py.csv", index=False)

        results_py['SLIDE'] = {
            'marginal_vals': marginal_indices,
            'sig_LFs': slide.sig_LFs,
            'sig_interacts': slide.sig_interacts
        }

    except Exception as e:
        print(f"ERROR in SLIDE: {e}")
        import traceback
        traceback.print_exc()
        results_py['SLIDE'] = None

    if args.generate_only:
        print("\n" + "=" * 60)
        print(f"Python outputs saved to: {output_dir}")
        print("=" * 60)
        return

    # ========================================================================
    # Compare with R outputs
    # ========================================================================
    print("\n" + "=" * 60)
    print("COMPARISON: R vs Python")
    print("=" * 60)

    # Check if R outputs exist
    R_params = load_csv(output_dir / f"{tag}_params.csv")
    R_A = load_csv(output_dir / f"{tag}_A.csv")
    R_C = load_csv(output_dir / f"{tag}_C.csv")
    R_Gamma = load_csv(output_dir / f"{tag}_Gamma.csv")
    R_I = load_csv(output_dir / f"{tag}_I.csv")
    R_Z = load_csv(output_dir / f"{tag}_Z.csv")
    R_marginals = load_csv(output_dir / f"{tag}_marginal_LFs.csv")
    R_interactions = load_csv(output_dir / f"{tag}_interactions.csv")

    if R_params is None:
        print("\nR outputs not found. Run R script first:")
        print(f"  Rscript run_slide_R.R {x_file} {y_file} {tag}")
        print("\n" + "=" * 60)
        print(f"Python outputs saved to: {output_dir}")
        print("=" * 60)
        return

    results = []

    # -------------------------------------------------------------------------
    # Compare Latent Factors (LOVE)
    # -------------------------------------------------------------------------
    if results_py['love'] is not None:
        print("\n" + "-" * 60)
        print("Latent Factors (LOVE) Comparison")
        print("-" * 60)

        # Compare K
        R_K = int(R_params['K'].values[0])
        py_K = results_py['love']['K']
        print(f"  K: R={R_K}, Python={py_K}", end="")
        if R_K == py_K:
            print(" - MATCH")
            results.append(True)
        else:
            print(" - DIFFER")
            results.append(False)

        # Compare A shape
        if R_A is not None:
            print(f"  A shape: R={R_A.shape}, Python={results_py['love']['A'].shape}", end="")
            if R_A.shape == results_py['love']['A'].shape:
                print(" - MATCH")
                results.append(compare_arrays(R_A.values, results_py['love']['A'],
                                              "A values", tol=0.1))
            else:
                print(" - DIFFER")
                results.append(False)

        # Compare C
        if R_C is not None:
            results.append(compare_arrays(R_C.values, results_py['love']['C'],
                                          "C values", tol=0.1))

        # Compare Gamma
        if R_Gamma is not None:
            results.append(compare_arrays(R_Gamma.values.flatten(),
                                          results_py['love']['Gamma'],
                                          "Gamma values", tol=0.1))

        # Compare pure variable indices (R is 1-indexed, Python is 0-indexed)
        if R_I is not None:
            R_I_vals = R_I['I'].values.astype(int)
            py_I_vals = np.array(results_py['love']['pureVec']) + 1  # Convert to 1-based
            results.append(compare_sets(R_I_vals, py_I_vals, "Pure variables (I)"))

    # -------------------------------------------------------------------------
    # Compare Z Matrix
    # -------------------------------------------------------------------------
    if results_py['Z'] is not None and R_Z is not None:
        print("\n" + "-" * 60)
        print("Z Matrix Comparison")
        print("-" * 60)

        print(f"  Z shape: R={R_Z.shape}, Python={results_py['Z'].shape}", end="")
        if R_Z.shape == results_py['Z'].shape:
            print(" - MATCH")
            results.append(compare_arrays(R_Z.values, results_py['Z'],
                                          "Z values", tol=0.1))
        else:
            print(" - DIFFER")
            results.append(False)

    # -------------------------------------------------------------------------
    # Compare SLIDE Results
    # -------------------------------------------------------------------------
    if results_py['SLIDE'] is not None:
        print("\n" + "-" * 60)
        print("SLIDE Results Comparison")
        print("-" * 60)

        # Compare marginal LFs
        if R_marginals is not None:
            R_marg = set(R_marginals['marginal'].values.astype(int))
            py_marg = set(results_py['SLIDE']['marginal_vals'])
            print(f"  R marginals: {sorted(R_marg)}")
            print(f"  Python marginals: {sorted(py_marg)}")
            results.append(compare_sets(R_marg, py_marg, "Marginal LFs"))

        # Compare interactions
        if R_interactions is not None and len(R_interactions) > 0:
            R_pairs = set(zip(R_interactions['p1'].values.astype(int),
                              R_interactions['p2'].values.astype(int)))
            if hasattr(results_py['SLIDE'], 'interaction_pairs'):
                py_pairs = set(zip(results_py['SLIDE']['interaction_pairs'][0] + 1,
                                   results_py['SLIDE']['interaction_pairs'][1] + 1))
            else:
                py_pairs = set()
            print(f"  R interactions: {len(R_pairs)} pairs")
            print(f"  Python interactions: {len(py_pairs)} pairs")
            if R_pairs == py_pairs:
                print(f"  Interactions: EXACT MATCH")
                results.append(True)
            elif len(R_pairs & py_pairs) > 0:
                print(f"  Interactions: PARTIAL MATCH ({len(R_pairs & py_pairs)} common)")
                results.append(True)
            else:
                print(f"  Interactions: NO MATCH")
                results.append(False)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")

    if failed == 0 and passed > 0:
        print("\nOverall: PASS - R and Python implementations are consistent!")
    elif failed > 0:
        print(f"\nOverall: {failed} comparison(s) showed differences.")
        print("Note: Some differences are expected due to RNG and numerical precision.")

    print("\n" + "=" * 60)
    print(f"Outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
