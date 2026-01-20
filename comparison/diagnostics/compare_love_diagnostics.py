#!/usr/bin/env python3
"""
Compare LOVE Diagnostic Outputs from R and Python

Analyzes intermediate values from both implementations to identify divergence points.

Usage:
    python compare_love_diagnostics.py --py_dir ./love_diagnostics_py --r_dir ./love_diagnostics_r
"""

import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path


def load_csv(filepath: str) -> np.ndarray:
    """Load CSV file, handling both R and Python formats.

    R saves CSVs with headers (column names), while Python np.savetxt saves
    without headers. This function detects and handles both formats.
    """
    try:
        # First, try to detect if the file has a header row
        # by checking if the first row contains numeric-looking values
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()

        # Check if first line looks like all numbers (no header)
        first_vals = first_line.split(',')
        try:
            # If we can convert all values to float, there's no header
            [float(v) for v in first_vals[:10]]  # Check first 10 values
            has_header = False
        except ValueError:
            # First row contains non-numeric values (header)
            has_header = True

        if has_header:
            df = pd.read_csv(filepath, index_col=None, header=0)
            # Check if first column looks like row names (all unique, possibly strings)
            if df.iloc[:, 0].dtype == object or df.columns[0] in ['', 'V1', 'X']:
                df = pd.read_csv(filepath, index_col=0, header=0)
            return df.values
        else:
            # No header - load directly with numpy
            return np.loadtxt(filepath, delimiter=',')
    except Exception as e:
        # Fallback
        return np.loadtxt(filepath, delimiter=',')


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, name: str,
                   rtol: float = 1e-5, atol: float = 1e-8) -> dict:
    """Compare two arrays and return comparison statistics."""
    result = {
        'name': name,
        'shape_match': arr1.shape == arr2.shape,
        'py_shape': arr1.shape,
        'r_shape': arr2.shape,
    }

    if not result['shape_match']:
        result['status'] = 'SHAPE MISMATCH'
        return result

    # Flatten for comparison
    a1 = arr1.flatten()
    a2 = arr2.flatten()

    # Basic stats
    result['py_min'] = float(np.min(a1))
    result['py_max'] = float(np.max(a1))
    result['py_mean'] = float(np.mean(a1))
    result['r_min'] = float(np.min(a2))
    result['r_max'] = float(np.max(a2))
    result['r_mean'] = float(np.mean(a2))

    # Difference stats
    diff = a1 - a2
    result['diff_min'] = float(np.min(diff))
    result['diff_max'] = float(np.max(diff))
    result['diff_mean'] = float(np.mean(np.abs(diff)))
    result['diff_std'] = float(np.std(diff))

    # Check if close
    result['allclose'] = bool(np.allclose(a1, a2, rtol=rtol, atol=atol))

    # Correlation
    if np.std(a1) > 0 and np.std(a2) > 0:
        result['correlation'] = float(np.corrcoef(a1, a2)[0, 1])
    else:
        result['correlation'] = np.nan

    # Number of exact matches
    result['n_exact_match'] = int(np.sum(a1 == a2))
    result['n_total'] = len(a1)
    result['pct_exact_match'] = 100 * result['n_exact_match'] / result['n_total']

    # Status
    if result['allclose']:
        result['status'] = 'PASS'
    elif result['correlation'] > 0.99:
        result['status'] = 'CLOSE (corr > 0.99)'
    elif result['correlation'] > 0.95:
        result['status'] = 'SIMILAR (corr > 0.95)'
    else:
        result['status'] = 'DIFFERENT'

    return result


def compare_index_arrays(arr1: np.ndarray, arr2: np.ndarray, name: str) -> dict:
    """Compare two index arrays (e.g., pure variable indices)."""
    result = {
        'name': name,
        'py_count': len(arr1),
        'r_count': len(arr2),
    }

    # Convert to sets for comparison
    set1 = set(arr1.flatten().astype(int))
    set2 = set(arr2.flatten().astype(int))

    # Account for R's 1-indexing
    set2_0indexed = {x - 1 for x in set2}

    result['intersection_r_as_is'] = len(set1 & set2)
    result['intersection_r_0indexed'] = len(set1 & set2_0indexed)

    # Check if R indices are 1-indexed
    if result['intersection_r_0indexed'] > result['intersection_r_as_is']:
        # R is 1-indexed, convert
        set2 = set2_0indexed
        result['r_indexing'] = '1-indexed (converted)'
    else:
        result['r_indexing'] = '0-indexed (or unknown)'

    result['py_only'] = sorted(set1 - set2)
    result['r_only'] = sorted(set2 - set1)
    result['common'] = sorted(set1 & set2)

    result['jaccard'] = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0

    if set1 == set2:
        result['status'] = 'MATCH'
    elif result['jaccard'] > 0.8:
        result['status'] = 'SIMILAR (Jaccard > 0.8)'
    else:
        result['status'] = 'DIFFERENT'

    return result


def run_comparison(py_dir: str, r_dir: str, output_file: str = None):
    """Run comparison between Python and R diagnostic outputs."""

    comparisons = []
    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(msg)

    log("=" * 80)
    log("LOVE Implementation Comparison: Python vs R")
    log("=" * 80)
    log(f"\nPython output dir: {py_dir}")
    log(f"R output dir: {r_dir}")

    # List of files to compare
    files_to_compare = [
        ("01_X_scaled.csv", "matrix", "Scaled Input Data"),
        ("02_R_hat_raw.csv", "matrix", "Raw Correlation Matrix"),
        ("02_Sigma_raw.csv", "matrix", "Raw Covariance Matrix"),
        ("03_pvalues_raw.csv", "matrix", "P-values (raw)"),
        ("03_pvalues_adjusted.csv", "matrix", "P-values (BH adjusted)"),
        ("03_kept_entries.csv", "matrix", "FDR Kept Entries Mask"),
        ("03_R_hat_thresholded.csv", "matrix", "Thresholded Correlation Matrix"),
        ("03_Sigma_thresholded.csv", "matrix", "Thresholded Covariance Matrix"),
        ("04_se_est.csv", "vector", "Standard Error Estimates"),
        ("05_off_Sigma.csv", "matrix", "Off-diagonal |Sigma|"),
        ("05_row_max_values.csv", "vector", "Row Maximum Values"),
        ("05_row_max_indices.csv", "indices", "Row Maximum Indices"),
        ("06_pure_vec.csv", "indices", "Pure Variable Indices"),
        ("08_AI_matrix.csv", "matrix", "AI Loading Matrix"),
        ("09_C_hat.csv", "matrix", "C_hat (Covariance of Z)"),
        ("10_Gamma_hat.csv", "vector", "Gamma_hat (Error Variance)"),
    ]

    log("\n" + "=" * 80)
    log("Step-by-Step Comparison")
    log("=" * 80)

    first_divergence = None

    for filename, dtype, description in files_to_compare:
        py_path = os.path.join(py_dir, filename)
        r_path = os.path.join(r_dir, filename)

        log(f"\n--- {description} ({filename}) ---")

        if not os.path.exists(py_path):
            log(f"  [SKIP] Python file not found")
            continue
        if not os.path.exists(r_path):
            log(f"  [SKIP] R file not found")
            continue

        try:
            py_arr = load_csv(py_path)
            r_arr = load_csv(r_path)

            if dtype == "indices":
                result = compare_index_arrays(py_arr, r_arr, description)
                log(f"  Python count: {result['py_count']}, R count: {result['r_count']}")
                log(f"  R indexing: {result['r_indexing']}")
                log(f"  Common: {len(result['common'])}, Jaccard: {result['jaccard']:.4f}")
                if result['py_only']:
                    log(f"  Python only: {result['py_only'][:10]}{'...' if len(result['py_only']) > 10 else ''}")
                if result['r_only']:
                    log(f"  R only: {result['r_only'][:10]}{'...' if len(result['r_only']) > 10 else ''}")
            else:
                result = compare_arrays(py_arr, r_arr, description)
                log(f"  Shape: Python {result['py_shape']}, R {result['r_shape']}")
                if result['shape_match']:
                    log(f"  Python: min={result['py_min']:.6f}, max={result['py_max']:.6f}, mean={result['py_mean']:.6f}")
                    log(f"  R:      min={result['r_min']:.6f}, max={result['r_max']:.6f}, mean={result['r_mean']:.6f}")
                    log(f"  Diff:   min={result['diff_min']:.2e}, max={result['diff_max']:.2e}, mean_abs={result['diff_mean']:.2e}")
                    log(f"  Correlation: {result['correlation']:.6f}")

            log(f"  STATUS: {result['status']}")
            comparisons.append(result)

            # Track first divergence
            if first_divergence is None and result['status'] not in ['PASS', 'MATCH']:
                first_divergence = (filename, description, result)

        except Exception as e:
            log(f"  [ERROR] {str(e)}")
            continue

    # Summary
    log("\n" + "=" * 80)
    log("SUMMARY")
    log("=" * 80)

    pass_count = sum(1 for c in comparisons if c['status'] in ['PASS', 'MATCH'])
    close_count = sum(1 for c in comparisons if 'CLOSE' in c['status'] or 'SIMILAR' in c['status'])
    fail_count = len(comparisons) - pass_count - close_count

    log(f"\nTotal comparisons: {len(comparisons)}")
    log(f"  PASS/MATCH: {pass_count}")
    log(f"  CLOSE/SIMILAR: {close_count}")
    log(f"  DIFFERENT: {fail_count}")

    if first_divergence:
        filename, description, result = first_divergence
        log(f"\n** First divergence point: {description} ({filename}) **")
        log(f"   Status: {result['status']}")

    # Print comparison table
    log("\n" + "-" * 80)
    log(f"{'Step':<40} {'Status':<20} {'Detail':<20}")
    log("-" * 80)
    for comp in comparisons:
        detail = ""
        if 'correlation' in comp and not np.isnan(comp.get('correlation', np.nan)):
            detail = f"corr={comp['correlation']:.4f}"
        elif 'jaccard' in comp:
            detail = f"jaccard={comp['jaccard']:.4f}"
        log(f"{comp['name']:<40} {comp['status']:<20} {detail:<20}")

    # Save report
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        log(f"\nReport saved to: {output_file}")

    return comparisons, first_divergence


def main():
    parser = argparse.ArgumentParser(description="Compare LOVE Diagnostic Outputs")
    parser.add_argument("--py_dir", type=str, required=True,
                        help="Python diagnostic output directory")
    parser.add_argument("--r_dir", type=str, required=True,
                        help="R diagnostic output directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output report file path")

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.py_dir), "comparison_report.txt")

    comparisons, first_divergence = run_comparison(
        py_dir=args.py_dir,
        r_dir=args.r_dir,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
