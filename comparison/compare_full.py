#!/usr/bin/env python3
"""
Full SLIDE Comparison Pipeline

Chains together:
1. Numerical validation (compare_outputs.py) - element-wise matrix comparisons
2. Latent factor analysis (compare_latent_factors.py) - semantic LF matching

Usage:
    python compare_full.py <output_path> [options]

Arguments:
    output_path     Path containing implementation subdirectories

Options:
    --tolerance TOL     Tolerance for numerical comparisons (default: 1e-4)
    --detailed          Show detailed per-element and per-LF comparisons
    --param COMBO       Only compare specific parameter combination
    --output FILE       Write report to file (default: <output_path>/full_comparison_report.txt)
    --no-file           Only print to stdout, don't write file
    --impl-path NAME=PATH   Override implementation path (can be repeated)
                            e.g., --impl-path R_native=/path/to/previous/R_native

Examples:
    # Basic usage
    python compare_full.py /path/to/outputs

    # Reuse R_native from a previous run
    python compare_full.py /path/to/new_outputs \\
        --impl-path R_native=/path/to/old_outputs/R_native

    # Multiple overrides
    python compare_full.py /path/to/outputs \\
        --impl-path R_native=/old/R_native \\
        --impl-path Py_rLOVE_rKO=/other/Py_rLOVE_rKO


    /ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python compare_full.py \
      /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/output_comparison/2026-01-16_15-59-27/SSc_binary_comparison \
      --impl-path R_native=/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/output_comparison/2026-01-16_10-41-07/SSc_binary_comparison/R_native
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Import from sibling modules
from compare_latent_factors import (
    DEFAULT_TASKS,
    LatentFactorComparator,
    LatentFactorLoader,
    MetricsExtractor,
    ReportGenerator,
    ReportSummary,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Numerical Comparison (from compare_outputs.py)
# =============================================================================

class NumericalComparator:
    """Element-wise numerical comparison between implementation outputs."""

    def __init__(self, tolerance: float = 1e-4, detailed: bool = False, **kwargs):
        self.tolerance = tolerance
        self.detailed = detailed

    def load_csv(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load CSV with index column."""
        if not filepath.exists():
            return None
        return pd.read_csv(filepath, index_col=0)

    def load_csv_no_index(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load CSV without index column."""
        if not filepath.exists():
            return None
        return pd.read_csv(filepath)

    def compare_arrays(self, arr1: np.ndarray, arr2: np.ndarray,
                       name: str, tol: Optional[float] = None) -> dict:
        """Compare two arrays and return metrics."""
        if tol is None:
            tol = self.tolerance

        result = {'name': name, 'status': 'SKIP', 'details': {}}

        if arr1 is None or arr2 is None:
            result['details']['reason'] = 'missing data'
            return result

        arr1 = np.asarray(arr1).astype(float)
        arr2 = np.asarray(arr2).astype(float)

        if arr1.shape != arr2.shape:
            result['status'] = 'FAIL'
            result['details'] = {
                'reason': 'shape mismatch',
                'shape1': arr1.shape,
                'shape2': arr2.shape,
            }
            return result

        # Handle NaN values
        nan_mask1 = np.isnan(arr1)
        nan_mask2 = np.isnan(arr2)

        if not np.array_equal(nan_mask1, nan_mask2):
            result['status'] = 'FAIL'
            result['details']['reason'] = 'NaN pattern mismatch'
            return result

        valid_mask = ~nan_mask1
        if np.sum(valid_mask) == 0:
            result['status'] = 'PASS'
            result['details']['reason'] = 'all NaN'
            return result

        # Compute metrics
        diff = np.abs(arr1[valid_mask] - arr2[valid_mask])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        flat1 = arr1[valid_mask].flatten()
        flat2 = arr2[valid_mask].flatten()
        corr = np.corrcoef(flat1, flat2)[0, 1] if len(flat1) > 1 else 1.0

        result['details'] = {
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'correlation': float(corr),
            'n_elements': int(np.sum(valid_mask)),
        }

        if max_diff <= tol:
            result['status'] = 'PASS'
        else:
            result['status'] = 'DIFF'

        return result

    def compare_sets(self, set1, set2, name: str) -> dict:
        """Compare two sets of indices."""
        result = {'name': name, 'status': 'SKIP', 'details': {}}

        if set1 is None or set2 is None:
            result['details']['reason'] = 'missing data'
            return result

        set1 = set(np.asarray(set1).flatten().astype(int))
        set2 = set(np.asarray(set2).flatten().astype(int))

        overlap = len(set1 & set2)
        total = len(set1 | set2)
        jaccard = overlap / total if total > 0 else 1.0

        result['details'] = {
            'set1_size': len(set1),
            'set2_size': len(set2),
            'overlap': overlap,
            'jaccard': float(jaccard),
        }

        if set1 == set2:
            result['status'] = 'PASS'
        elif jaccard > 0.8:
            result['status'] = 'PARTIAL'
        else:
            result['status'] = 'FAIL'

        return result

    def compare_param_dirs(self, path1: Path, path2: Path,
                           name1: str, name2: str) -> dict:
        """Compare outputs in two parameter directories."""
        results = {
            'path1': str(path1),
            'path2': str(path2),
            'name1': name1,
            'name2': name2,
            'comparisons': [],
        }

        # Determine file suffixes based on implementation names
        # R uses no suffix, Python uses _py suffix in compare_outputs.py convention
        # But in multi-impl setup, each has its own directory

        # A matrix comparison
        A1 = self.load_csv(path1 / 'A.csv')
        A2 = self.load_csv(path2 / 'A.csv')
        if A1 is not None and A2 is not None:
            results['comparisons'].append(
                self.compare_arrays(np.abs(A1.values), np.abs(A2.values),
                                    'A matrix (absolute)', tol=0.1)
            )

        # Z matrix comparison
        Z1 = self.load_csv(path1 / 'z_matrix.csv')
        Z2 = self.load_csv(path2 / 'z_matrix.csv')
        if Z1 is not None and Z2 is not None:
            # Normalize column names
            Z1.columns = [c.strip('"') for c in Z1.columns]
            Z2.columns = [c.strip('"') for c in Z2.columns]
            results['comparisons'].append(
                self.compare_arrays(Z1.values, Z2.values, 'Z matrix', tol=0.1)
            )

        # C matrix comparison
        C1 = self.load_csv(path1 / 'C.csv')
        C2 = self.load_csv(path2 / 'C.csv')
        if C1 is not None and C2 is not None:
            results['comparisons'].append(
                self.compare_arrays(C1.values, C2.values, 'C matrix', tol=0.1)
            )

        # Gamma comparison
        G1 = self.load_csv_no_index(path1 / 'Gamma.csv')
        G2 = self.load_csv_no_index(path2 / 'Gamma.csv')
        if G1 is not None and G2 is not None:
            results['comparisons'].append(
                self.compare_arrays(G1.values.flatten(), G2.values.flatten(),
                                    'Gamma', tol=0.1)
            )

        # Pure variable indices
        I1 = self.load_csv_no_index(path1 / 'I.csv')
        I2 = self.load_csv_no_index(path2 / 'I.csv')
        if I1 is not None and I2 is not None:
            col1 = 'I' if 'I' in I1.columns else I1.columns[0]
            col2 = 'I' if 'I' in I2.columns else I2.columns[0]
            results['comparisons'].append(
                self.compare_sets(I1[col1].values, I2[col2].values, 'Pure variables (I)')
            )

        return results

    def format_comparison_report(self, results: dict) -> str:
        """Format numerical comparison results."""
        lines = []
        lines.append(f"\n  {results['name1']} vs {results['name2']}")
        lines.append("  " + "-" * 50)

        n_pass = 0
        n_fail = 0
        n_skip = 0

        for comp in results['comparisons']:
            status = comp['status']
            name = comp['name']
            details = comp['details']

            if status == 'PASS':
                n_pass += 1
                if 'correlation' in details:
                    lines.append(f"    {name}: PASS (corr={details['correlation']:.4f})")
                elif 'jaccard' in details:
                    lines.append(f"    {name}: PASS (exact match, {details['set1_size']} items)")
                else:
                    lines.append(f"    {name}: PASS")
            elif status == 'DIFF':
                n_fail += 1
                lines.append(f"    {name}: DIFF (max={details['max_diff']:.2e}, "
                           f"corr={details['correlation']:.4f})")
            elif status == 'PARTIAL':
                n_pass += 1  # Count as pass if >80% overlap
                lines.append(f"    {name}: PARTIAL (Jaccard={details['jaccard']:.2%})")
            elif status == 'FAIL':
                n_fail += 1
                reason = details.get('reason', 'unknown')
                lines.append(f"    {name}: FAIL ({reason})")
            else:
                n_skip += 1
                lines.append(f"    {name}: SKIP ({details.get('reason', 'missing data')})")

        lines.append(f"  Summary: {n_pass} pass, {n_fail} fail, {n_skip} skip")

        return '\n'.join(lines)


# =============================================================================
# Full Comparison Pipeline
# =============================================================================

class FullComparisonPipeline:
    """Chain numerical validation and latent factor analysis."""

    def __init__(self, output_path: Path, tolerance: float = 1e-4,
                 detailed: bool = False, path_overrides: Optional[dict] = None,
                 **kwargs):
        self.output_path = Path(output_path)
        self.tolerance = tolerance
        self.detailed = detailed
        self.path_overrides = path_overrides or {}

        # Initialize components
        self.numerical = NumericalComparator(tolerance=tolerance, detailed=detailed)
        self.lf_loader = LatentFactorLoader()
        self.lf_comparator = LatentFactorComparator(detailed=detailed)
        self.metrics_extractor = MetricsExtractor(script_dir=Path(__file__).parent)
        self.summary = ReportSummary()

    def discover_implementations(self) -> dict:
        """Discover available implementations.

        Path overrides take precedence over auto-discovery. This allows
        reusing outputs from previous runs (e.g., R_native which is slow
        but deterministic).
        """
        available = {}
        known_patterns = [
            "R_native", "R_outputs",
            "Py_rLOVE_rKO", "Py_rLOVE_knockpy", "Py_rLOVE_pyKO",
            "Py_pyLOVE_rKO", "Py_pyLOVE_knockpy", "Py_pyLOVE_pyKO",
        ]

        # First, apply path overrides
        for name, override_path in self.path_overrides.items():
            override_path = Path(override_path)
            if override_path.is_dir():
                available[name] = override_path
                logger.info(f"  Using override for {name}: {override_path}")
            else:
                logger.warning(f"  Override path not found for {name}: {override_path}")

        # Then discover from output_path (skip if already overridden)
        for pattern in known_patterns:
            if pattern in available:
                continue  # Already overridden
            impl_path = self.output_path / pattern
            if impl_path.is_dir():
                available[pattern] = impl_path

        # Also discover any other directories with _out subdirs
        for child in self.output_path.iterdir():
            if child.is_dir() and child.name not in available:
                if any(d.name.endswith('_out') for d in child.iterdir() if d.is_dir()):
                    available[child.name] = child

        return available

    def find_param_dir(self, base_path: Path, combo: str) -> Optional[Path]:
        """Find parameter directory with flexible naming."""
        import re

        direct = base_path / combo
        if direct.is_dir():
            return direct

        # Try integer <-> decimal conversion
        alt_combo = re.sub(r'_(\d+)_out$', r'_\1.0_out', combo)
        alt_path = base_path / alt_combo
        if alt_path.is_dir():
            return alt_path

        alt_combo = re.sub(r'_(\d+)\.0_out$', r'_\1_out', combo)
        alt_path = base_path / alt_combo
        if alt_path.is_dir():
            return alt_path

        return None

    def print_header(self):
        """Print report header."""
        logger.info("=" * 70)
        logger.info("SLIDE Full Comparison Report")
        logger.info("=" * 70)
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Tolerance: {self.tolerance}")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.path_overrides:
            logger.info("")
            logger.info("Path overrides:")
            for name, path in self.path_overrides.items():
                logger.info(f"  {name}: {path}")
        logger.info("=" * 70)

    def run_numerical_comparison(self, param_filter: Optional[str] = None):
        """Run numerical (element-wise) comparisons."""
        available = self.discover_implementations()

        if len(available) < 2:
            logger.info("\nInsufficient implementations for numerical comparison")
            return

        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 1: Numerical Validation")
        logger.info("=" * 70)
        logger.info("Comparing matrices element-wise between implementations")

        # Get reference (prefer R_native)
        ref_name = 'R_native' if 'R_native' in available else list(available.keys())[0]
        ref_path = available[ref_name]

        # Get parameter combinations
        combos = sorted([d.name for d in ref_path.iterdir()
                        if d.is_dir() and d.name.endswith('_out')])

        if param_filter:
            found = self.find_param_dir(ref_path, param_filter)
            combos = [found.name] if found else []

        for combo in combos:
            logger.info("")
            logger.info("-" * 70)
            logger.info(f"Parameter: {combo}")
            logger.info("-" * 70)

            ref_param_path = self.find_param_dir(ref_path, combo)
            if ref_param_path is None:
                continue

            # Compare reference to each other implementation
            for impl_name, impl_path in available.items():
                if impl_name == ref_name:
                    continue

                impl_param_path = self.find_param_dir(impl_path, combo)
                if impl_param_path is None:
                    logger.info(f"\n  {ref_name} vs {impl_name}: SKIP (no matching param dir)")
                    continue

                results = self.numerical.compare_param_dirs(
                    ref_param_path, impl_param_path, ref_name, impl_name
                )
                logger.info(self.numerical.format_comparison_report(results))

    def run_lf_comparison(self, param_filter: Optional[str] = None):
        """Run latent factor content comparison."""
        available = self.discover_implementations()

        if len(available) < 2:
            logger.info("\nInsufficient implementations for LF comparison")
            return

        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 2: Latent Factor Content Analysis")
        logger.info("=" * 70)
        logger.info("Matching LFs by feature overlap and correlation")

        # Get reference
        ref_name = list(available.keys())[0]
        ref_path = available[ref_name]

        # Get parameter combinations
        combos = sorted([d.name for d in ref_path.iterdir()
                        if d.is_dir() and d.name.endswith('_out')])

        if param_filter:
            found = self.find_param_dir(ref_path, param_filter)
            combos = [found.name] if found else []

        for combo in combos:
            logger.info("")
            logger.info("-" * 70)
            logger.info(f"Parameter: {combo}")
            logger.info("-" * 70)

            # Load all implementations
            impl_data = {}
            for name, base_path in available.items():
                param_path = self.find_param_dir(base_path, combo)
                if param_path is None:
                    continue

                # Determine if R or Python based on files
                is_r = name.startswith('R_')
                if not is_r and list(param_path.glob('feature_list_Z*.txt')) and not (param_path / 'sig_LFs.txt').exists():
                    is_r = True

                if is_r:
                    data = self.lf_loader.load_r_outputs(param_path)
                else:
                    data = self.lf_loader.load_python_outputs(param_path)

                if data:
                    impl_data[name] = data

            if len(impl_data) < 2:
                logger.info(f"  Insufficient data for comparison (found {len(impl_data)})")
                continue

            # LF count summary
            logger.info("\n  LF Count Summary:")
            for name, data in impl_data.items():
                n_lfs = len(data.get('sig_LFs', []))
                logger.info(f"    {name:25s}: {n_lfs} significant LFs")
                self.summary.update_lf_count(name, combo, n_lfs)

            # Pairwise comparisons
            impl_names = list(impl_data.keys())
            for i, name1 in enumerate(impl_names):
                for name2 in impl_names[i+1:]:
                    results = self.lf_comparator.compare_implementations(
                        impl_data[name1], impl_data[name2], name1, name2
                    )
                    logger.info(self.lf_comparator.format_report(results))

                    self.summary.update_lf_comparison(
                        name1, name2,
                        results['mean_jaccard'], results['mean_z_corr'],
                        n_matched=len(results['feature_matches']['matches']),
                        lf_count1=results['lf_count1'],
                        lf_count2=results['lf_count2']
                    )

    def print_summary(self):
        """Print combined summary."""
        self.summary.finalize()

        logger.info("")
        logger.info("=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)

        logger.info(f"\nTotal pairwise comparisons: {self.summary.total_comparisons}")

        # LF counts
        avg_lf = self.summary.get_avg_lf_counts()
        if avg_lf:
            logger.info("\nAverage LF Counts:")
            for impl, count in sorted(avg_lf.items(), key=lambda x: -x[1]):
                logger.info(f"  {impl:25s}: {count:.1f}")

        # Similarity ranking
        rankings = self.summary.get_similarity_ranking()
        if rankings:
            logger.info("\nSimilarity Ranking (Jaccard + Z-corr):")
            logger.info(f"  {'Pair':<45} {'Jaccard':>8} {'Z-corr':>8}")
            logger.info(f"  {'-'*45} {'-'*8} {'-'*8}")
            for impl1, impl2, jaccard, z_corr, _ in rankings[:10]:
                pair = f"{impl1} <-> {impl2}"
                j_str = f"{jaccard:.3f}" if jaccard > 0 else "-"
                z_str = f"{z_corr:.3f}" if z_corr > 0 else "-"
                logger.info(f"  {pair:<45} {j_str:>8} {z_str:>8}")

        if self.summary.avg_jaccard > 0:
            logger.info(f"\nOverall avg Jaccard: {self.summary.avg_jaccard:.3f}")
        if self.summary.avg_z_corr > 0:
            logger.info(f"Overall avg Z-corr: {self.summary.avg_z_corr:.3f}")

        if self.summary.best_lf_match:
            impl1, impl2, jaccard, z_corr = self.summary.best_lf_match
            logger.info(f"\nBest match: {impl1} <-> {impl2}")
            logger.info(f"  Jaccard={jaccard:.3f}, Z-corr={z_corr:.3f}")

    def run(self, param_filter: Optional[str] = None):
        """Run the full comparison pipeline."""
        self.print_header()
        self.run_numerical_comparison(param_filter)
        self.run_lf_comparison(param_filter)
        self.print_summary()

        logger.info("")
        logger.info("=" * 70)
        logger.info("Full comparison complete")
        logger.info("=" * 70)


# =============================================================================
# Main
# =============================================================================

def setup_file_logging(output_file: Path):
    """Add file handler for report output."""
    file_handler = logging.FileHandler(output_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    return file_handler


def parse_impl_path(value: str) -> tuple:
    """Parse NAME=PATH argument into (name, path) tuple."""
    if '=' not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid format: '{value}'. Expected NAME=PATH (e.g., R_native=/path/to/dir)"
        )
    name, path = value.split('=', 1)
    return name.strip(), Path(path.strip())


def main():
    parser = argparse.ArgumentParser(
        description='Full SLIDE Comparison Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('output_path', type=Path,
                        help='Path containing implementation subdirectories')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Tolerance for numerical comparisons (default: 1e-4)')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed comparisons')
    parser.add_argument('--param', type=str, default=None,
                        help='Only compare specific parameter combination')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output file (default: <output_path>/full_comparison_report.txt)')
    parser.add_argument('--no-file', action='store_true',
                        help='Do not write to file')
    parser.add_argument('--impl-path', type=parse_impl_path, action='append',
                        dest='impl_paths', metavar='NAME=PATH',
                        help='Override implementation path (can be repeated). '
                             'e.g., --impl-path R_native=/path/to/previous/R_native')

    args = parser.parse_args()

    if not args.output_path.is_dir():
        logger.error(f"Output path not found: {args.output_path}")
        return 1

    # Build path_overrides dict from --impl-path arguments
    path_overrides = {}
    if args.impl_paths:
        for name, path in args.impl_paths:
            path_overrides[name] = path

    # Set up file output
    file_handler = None
    output_file = None
    if not args.no_file:
        output_file = args.output or (args.output_path / 'full_comparison_report.txt')
        file_handler = setup_file_logging(output_file)

    pipeline = FullComparisonPipeline(
        output_path=args.output_path,
        tolerance=args.tolerance,
        detailed=args.detailed,
        path_overrides=path_overrides,
    )

    pipeline.run(param_filter=args.param)

    if file_handler:
        file_handler.close()
        logger.removeHandler(file_handler)
        print(f"\nReport saved to: {output_file}")

    return 0


if __name__ == '__main__':
    exit(main())
