#!/usr/bin/env python3
"""
Python script to compare SLIDE outputs between R and Python implementations.

This script compares pre-generated outputs from both R and Python SLIDE runs.
Run the R script first (run_slide_R.R), then run Python (run_slide_py.py),
and finally use this script to generate a detailed comparison report.

Usage:
    python compare_outputs.py <tag> [options]

Arguments:
    tag             Run name tag (directory name under outputs/)

Options:
    --tolerance TOL     Tolerance for numerical comparisons (default: 1e-4)
    --output-file FILE  Write comparison results to file (default: stdout)
    --detailed          Show detailed element-wise differences

Examples:
    python compare_outputs.py my_experiment
    python compare_outputs.py my_experiment --tolerance 0.01 --detailed
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


class SLIDEComparison:
    """Class to compare R and Python SLIDE outputs."""

    def __init__(self, output_dir, tolerance=1e-4, detailed=False):
        self.output_dir = Path(output_dir)
        self.tolerance = tolerance
        self.detailed = detailed
        self.results = {}

    def load_csv(self, filepath):
        """Load a CSV file as DataFrame."""
        if not os.path.exists(filepath):
            return None
        return pd.read_csv(filepath, index_col=0)

    def load_csv_no_index(self, filepath):
        """Load a CSV file as DataFrame without index column."""
        if not os.path.exists(filepath):
            return None
        return pd.read_csv(filepath)

    def compare_scalars(self, r_val, py_val, name, tol=None):
        """Compare two scalar values."""
        if tol is None:
            tol = self.tolerance

        if r_val is None or py_val is None:
            print(f"  {name}: SKIP (missing data)")
            return None

        diff = abs(float(r_val) - float(py_val))
        if diff <= tol:
            print(f"  {name}: PASS (R={r_val:.6f}, Python={py_val:.6f}, diff={diff:.2e})")
            return True
        else:
            print(f"  {name}: FAIL (R={r_val:.6f}, Python={py_val:.6f}, diff={diff:.2e})")
            return False

    def compare_arrays(self, arr1, arr2, name, tol=None):
        """Compare two arrays and report differences."""
        if tol is None:
            tol = self.tolerance

        if arr1 is None or arr2 is None:
            print(f"  {name}: SKIP (missing data)")
            return None

        arr1 = np.asarray(arr1).astype(float)
        arr2 = np.asarray(arr2).astype(float)

        if arr1.shape != arr2.shape:
            print(f"  {name}: FAIL (shape mismatch: R={arr1.shape} vs Python={arr2.shape})")
            return False

        # Handle NaN values
        nan_mask1 = np.isnan(arr1)
        nan_mask2 = np.isnan(arr2)

        if not np.array_equal(nan_mask1, nan_mask2):
            print(f"  {name}: FAIL (NaN pattern mismatch)")
            return False

        # Compare non-NaN values
        valid_mask = ~nan_mask1
        if np.sum(valid_mask) == 0:
            print(f"  {name}: PASS (all NaN)")
            return True

        max_diff = np.max(np.abs(arr1[valid_mask] - arr2[valid_mask]))
        mean_diff = np.mean(np.abs(arr1[valid_mask] - arr2[valid_mask]))
        corr = np.corrcoef(arr1[valid_mask].flatten(), arr2[valid_mask].flatten())[0, 1]

        if max_diff <= tol:
            print(f"  {name}: PASS (max_diff={max_diff:.2e}, corr={corr:.6f})")
            return True
        else:
            print(f"  {name}: DIFF (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, corr={corr:.6f})")
            if self.detailed and arr1.ndim <= 2:
                self._show_diff_details(arr1, arr2, name)
            return False

    def _show_diff_details(self, arr1, arr2, name):
        """Show detailed element-wise differences."""
        diff = np.abs(arr1 - arr2)
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"    Max diff at index {max_idx}: R={arr1[max_idx]:.6f}, Python={arr2[max_idx]:.6f}")

        # Show distribution of differences
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            print(f"    {p}th percentile diff: {np.percentile(diff, p):.2e}")

    def compare_sets(self, set1, set2, name, index_offset=0):
        """Compare two sets of indices."""
        if set1 is None or set2 is None:
            print(f"  {name}: SKIP (missing data)")
            return None

        set1 = set(np.asarray(set1).flatten().astype(int) + index_offset)
        set2 = set(np.asarray(set2).flatten().astype(int))

        overlap = len(set1 & set2)
        total = len(set1 | set2)
        jaccard = overlap / total if total > 0 else 1.0

        if set1 == set2:
            print(f"  {name}: PASS (exact match, {len(set1)} items)")
            return True
        else:
            common = set1 & set2
            only_in_r = set1 - set2
            only_in_py = set2 - set1
            print(f"  {name}: PARTIAL (Jaccard={jaccard:.2%})")
            print(f"    Common: {len(common)} items")
            print(f"    Only in R: {len(only_in_r)} items")
            print(f"    Only in Python: {len(only_in_py)} items")
            if self.detailed:
                if len(only_in_r) <= 20:
                    print(f"    R-only: {sorted(only_in_r)}")
                if len(only_in_py) <= 20:
                    print(f"    Python-only: {sorted(only_in_py)}")
            return jaccard > 0.8

    def run_comparison(self, tag):
        """Run all comparison tests."""
        print("=" * 70)
        print(f"SLIDE R vs Python Comparison")
        print(f"Tag: {tag}")
        print(f"Output directory: {self.output_dir}")
        print(f"Tolerance: {self.tolerance}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        # Check if output directory exists
        if not self.output_dir.exists():
            print(f"\nERROR: Output directory not found: {self.output_dir}")
            return False

        # Run all comparison tests
        self.results['params'] = self.compare_params(tag)
        self.results['A_matrix'] = self.compare_A_matrix(tag)
        self.results['C_matrix'] = self.compare_C_matrix(tag)
        self.results['Gamma'] = self.compare_Gamma(tag)
        self.results['pure_vars'] = self.compare_pure_vars(tag)
        self.results['Z_matrix'] = self.compare_Z_matrix(tag)
        self.results['marginals'] = self.compare_marginals(tag)
        self.results['interactions'] = self.compare_interactions(tag)

        self.print_summary()
        return all(v is not False for v in self.results.values() if v is not None)

    def compare_params(self, tag):
        """Compare K and other parameters."""
        print("\n" + "-" * 70)
        print("1. Parameters (K, delta, lambda)")
        print("-" * 70)

        R_params = self.load_csv_no_index(self.output_dir / f"{tag}_params.csv")
        py_params = self.load_csv_no_index(self.output_dir / f"{tag}_params_py.csv")

        if R_params is None or py_params is None:
            print("  SKIP (missing parameter files)")
            return None

        results = []

        # Compare K
        R_K = int(R_params['K'].values[0])
        py_K = int(py_params['K'].values[0])
        print(f"  K: R={R_K}, Python={py_K}", end="")
        if R_K == py_K:
            print(" - MATCH")
            results.append(True)
        else:
            print(" - DIFFER")
            results.append(False)

        # Compare opt_delta
        R_delta = R_params['opt_delta'].values[0]
        py_delta = py_params['opt_delta'].values[0]
        results.append(self.compare_scalars(R_delta, py_delta, "opt_delta", tol=0.1))

        return all(r for r in results if r is not None) if results else None

    def compare_A_matrix(self, tag):
        """Compare A (loading) matrix."""
        print("\n" + "-" * 70)
        print("2. A Matrix (Loading/Membership Matrix)")
        print("-" * 70)

        R_A = self.load_csv(self.output_dir / f"{tag}_A.csv")
        py_A = self.load_csv(self.output_dir / f"{tag}_A_py.csv")

        if R_A is None or py_A is None:
            print("  SKIP (missing A matrix files)")
            return None

        print(f"  R shape: {R_A.shape}, Python shape: {py_A.shape}")

        if R_A.shape != py_A.shape:
            print("  Shape mismatch - cannot compare element-wise")
            return False

        # Compare absolute values (sign/permutation may differ)
        return self.compare_arrays(np.abs(R_A.values), np.abs(py_A.values),
                                   "A matrix (absolute)", tol=0.1)

    def compare_C_matrix(self, tag):
        """Compare C (covariance) matrix."""
        print("\n" + "-" * 70)
        print("3. C Matrix (Latent Factor Covariance)")
        print("-" * 70)

        R_C = self.load_csv(self.output_dir / f"{tag}_C.csv")
        py_C = self.load_csv(self.output_dir / f"{tag}_C_py.csv")

        if R_C is None or py_C is None:
            print("  SKIP (missing C matrix files)")
            return None

        return self.compare_arrays(R_C.values, py_C.values, "C matrix", tol=0.1)

    def compare_Gamma(self, tag):
        """Compare Gamma (error variance) vector."""
        print("\n" + "-" * 70)
        print("4. Gamma (Error Variance)")
        print("-" * 70)

        R_Gamma = self.load_csv_no_index(self.output_dir / f"{tag}_Gamma.csv")
        py_Gamma = self.load_csv_no_index(self.output_dir / f"{tag}_Gamma_py.csv")

        if R_Gamma is None or py_Gamma is None:
            print("  SKIP (missing Gamma files)")
            return None

        return self.compare_arrays(R_Gamma.values.flatten(),
                                   py_Gamma.values.flatten(),
                                   "Gamma", tol=0.1)

    def compare_pure_vars(self, tag):
        """Compare pure variable indices."""
        print("\n" + "-" * 70)
        print("5. Pure Variable Indices (I)")
        print("-" * 70)

        R_I = self.load_csv_no_index(self.output_dir / f"{tag}_I.csv")
        py_I = self.load_csv_no_index(self.output_dir / f"{tag}_I_py.csv")

        if R_I is None or py_I is None:
            print("  SKIP (missing pure variable index files)")
            return None

        # Both should be 1-indexed now (Python saves as 1-based)
        return self.compare_sets(R_I['I'].values, py_I['I'].values,
                                 "Pure variables", index_offset=0)

    def compare_Z_matrix(self, tag):
        """Compare Z (latent factor) matrix."""
        print("\n" + "-" * 70)
        print("6. Z Matrix (Latent Factors)")
        print("-" * 70)

        R_Z = self.load_csv(self.output_dir / f"{tag}_Z.csv")
        py_Z = self.load_csv(self.output_dir / f"{tag}_Z_py.csv")

        if R_Z is None or py_Z is None:
            print("  SKIP (missing Z matrix files)")
            return None

        print(f"  R shape: {R_Z.shape}, Python shape: {py_Z.shape}")

        if R_Z.shape != py_Z.shape:
            print("  Shape mismatch - cannot compare element-wise")
            return False

        return self.compare_arrays(R_Z.values, py_Z.values, "Z matrix", tol=0.1)

    def compare_marginals(self, tag):
        """Compare marginal (standalone) latent factors."""
        print("\n" + "-" * 70)
        print("7. Marginal LFs (Standalone Latent Factors)")
        print("-" * 70)

        R_marg = self.load_csv_no_index(self.output_dir / f"{tag}_marginal_LFs.csv")
        py_marg = self.load_csv_no_index(self.output_dir / f"{tag}_marginal_LFs_py.csv")

        if R_marg is None or py_marg is None:
            print("  SKIP (missing marginal LF files)")
            return None

        R_vals = R_marg['marginal'].values if 'marginal' in R_marg.columns else R_marg.values.flatten()
        py_vals = py_marg['marginal'].values if 'marginal' in py_marg.columns else py_marg.values.flatten()

        print(f"  R marginals: {sorted(R_vals.astype(int))}")
        print(f"  Python marginals: {sorted(py_vals.astype(int))}")

        return self.compare_sets(R_vals, py_vals, "Marginal LFs")

    def compare_interactions(self, tag):
        """Compare interaction pairs."""
        print("\n" + "-" * 70)
        print("8. Interaction Pairs")
        print("-" * 70)

        R_int = self.load_csv_no_index(self.output_dir / f"{tag}_interactions.csv")
        py_int = self.load_csv_no_index(self.output_dir / f"{tag}_interactions_py.csv")

        if R_int is None or py_int is None:
            print("  SKIP (missing interaction files)")
            return None

        # Handle empty dataframes
        if len(R_int) == 0 and len(py_int) == 0:
            print("  PASS (both empty)")
            return True

        if len(R_int) == 0 or len(py_int) == 0:
            print(f"  DIFFER (R has {len(R_int)}, Python has {len(py_int)} interactions)")
            return False

        # Convert to sets of tuples for comparison
        R_pairs = set(zip(R_int['p1'].values.astype(int),
                          R_int['p2'].values.astype(int)))
        py_pairs = set(zip(py_int['p1'].values.astype(int),
                           py_int['p2'].values.astype(int)))

        print(f"  R interactions: {len(R_pairs)} pairs")
        print(f"  Python interactions: {len(py_pairs)} pairs")

        if R_pairs == py_pairs:
            print("  PASS (exact match)")
            if self.detailed:
                for p1, p2 in sorted(R_pairs):
                    print(f"    Z{p1} - Z{p2}")
            return True
        else:
            common = R_pairs & py_pairs
            only_r = R_pairs - py_pairs
            only_py = py_pairs - R_pairs
            jaccard = len(common) / len(R_pairs | py_pairs) if R_pairs | py_pairs else 0

            print(f"  PARTIAL (Jaccard={jaccard:.2%})")
            print(f"    Common: {len(common)}")
            print(f"    Only in R: {len(only_r)}")
            print(f"    Only in Python: {len(only_py)}")

            if self.detailed:
                if only_r:
                    print(f"    R-only pairs: {sorted(only_r)[:10]}")
                if only_py:
                    print(f"    Python-only pairs: {sorted(only_py)[:10]}")

            return jaccard > 0.8

    def print_summary(self):
        """Print summary of all test results."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        total = 0
        passed = 0
        failed = 0
        skipped = 0

        for name, result in self.results.items():
            total += 1
            if result is True:
                status = "PASS"
                passed += 1
            elif result is False:
                status = "FAIL"
                failed += 1
            else:
                status = "SKIP"
                skipped += 1
            print(f"  {name}: {status}")

        print("-" * 70)
        print(f"Total: {total}, Passed: {passed}, Failed: {failed}, Skipped: {skipped}")

        if failed == 0 and passed > 0:
            print("\nOverall: PASS - R and Python implementations are consistent!")
        elif failed > 0:
            print(f"\nOverall: {failed} comparison(s) FAILED - please investigate.")
            print("\nNote: Some differences may be expected due to:")
            print("  - Random number generator differences")
            print("  - Numerical precision differences")
            print("  - LP solver implementation differences")
            print("  - Knockoff procedure randomness")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SLIDE R and Python outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tag", help="Run name tag (directory under outputs/)")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Tolerance for numerical comparisons (default: 1e-4)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Write comparison results to file")
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed element-wise differences")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs" / args.tag

    # Redirect output if requested
    if args.output_file:
        original_stdout = sys.stdout
        sys.stdout = open(args.output_file, 'w')

    comparator = SLIDEComparison(output_dir, tolerance=args.tolerance,
                                  detailed=args.detailed)
    success = comparator.run_comparison(args.tag)

    if args.output_file:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Comparison results written to: {args.output_file}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
