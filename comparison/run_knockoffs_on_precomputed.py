#!/usr/bin/env python3
"""
Run knockoff filters on pre-computed LOVE Z matrices.

This script runs knockoffs on existing LOVE results from the old directory format:
    <love_dir>/
        params.yaml
        0.05_0.1_out/z_matrix.csv
        0.05_0.5_out/z_matrix.csv
        ...

Usage:
    python run_knockoffs_on_precomputed.py \\
        --love-dir /path/to/R_native \\
        --backend R_native \\
        --output-dir results/

Available backends:
    - R_native: R knockoff package (requires rpy2)
    - R_knockoffs_py_sklearn: R knockoffs + Python sklearn stats (requires rpy2)
    - knockoff_filter_sklearn: Python knockoff-filter with sklearn (pure Python)
    - knockoff_filter: Python knockoff-filter with Fortran glmnet
    - custom_glmnet: SLIDE's custom glmnet implementation




sbatch --array=0-2 submit_knockoffs_precomputed.sh \
    /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/output_comparison/2026-01-17_04-55-31/SSc_binary_comparison/R_native

# Compare results
/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python run_knockoffs_on_precomputed.py compare \
    output_knockoffs/SSc_binary_comparison/R_native/R_native/*.json \
    output_knockoffs/SSc_binary_comparison/R_native/R_knockoffs_py_sklearn/*.json \
    output_knockoffs/SSc_binary_comparison/R_native/knockoff_filter_sklearn/*.json \
    -o output_knockoffs/SSc_binary_comparison/R_native/targeted_knockoff_comparison_final.txt


"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add source directory to path for development imports
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd
import yaml

# Import from bundled loveslide.knockoff package
try:
    from loveslide.knockoff.stats import stat_glmnet_lambdasmax
    from loveslide.knockoff.filter import knockoff_threshold
    from loveslide.knockoff.create import create_gaussian
    from loveslide.knockoff.solve import create_solve_sdp, create_solve_asdp, create_solve_equi
    from loveslide.knockoff.utils import is_posdef
    KNOCKOFF_AVAILABLE = True
except ImportError:
    KNOCKOFF_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Knockoff Backends
# =============================================================================

def compute_knockoffs_r(Z: np.ndarray, y: np.ndarray, fdr: float = 0.1,
                        seed: int = 42, **kwargs) -> Optional[Dict]:
    """R native knockoff implementation."""
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()
        pandas2ri.activate()

        robjects.r(f'set.seed({seed})')
        knockoff_r = importr('knockoff')

        Z_r = robjects.r['as.matrix'](pd.DataFrame(Z))
        y_r = robjects.FloatVector(y.flatten())

        result = knockoff_r.knockoff_filter(
            X=Z_r, y=y_r,
            knockoffs=knockoff_r.create_second_order,
            statistic=knockoff_r.stat_glmnet_lambdasmax,
            offset=0, fdr=fdr
        )

        selected = np.array(result.rx2('selected')) - 1  # R is 1-indexed
        W = np.array(result.rx2('statistic'))
        threshold = float(np.array(result.rx2('threshold'))[0])

        numpy2ri.deactivate()
        pandas2ri.deactivate()

        return {'W': W, 'threshold': threshold, 'selected': selected, 'backend': 'R_native'}

    except Exception as e:
        logger.error(f"R knockoff failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_knockoffs_r_knockoffs_py_stats(Z: np.ndarray, y: np.ndarray, fdr: float = 0.1,
                                           seed: int = 42, use_sklearn: bool = True,
                                           **kwargs) -> Optional[Dict]:
    """Hybrid: R knockoff generation + Python glmnet statistics.

    This isolates the knockoff generation from the statistic computation
    to identify which component causes divergence from R_native.
    """
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.packages import importr
        from loveslide.knockoff.stats import stat_glmnet_lambdasmax
        from loveslide.knockoff.filter import knockoff_threshold

        numpy2ri.activate()
        pandas2ri.activate()

        # Use R to generate knockoffs (same as R_native)
        robjects.r(f'set.seed({seed})')
        knockoff_r = importr('knockoff')
        Z_r = robjects.r['as.matrix'](pd.DataFrame(Z))

        # Generate knockoffs using R's create.second_order
        Zk_r = knockoff_r.create_second_order(Z_r)
        Zk = np.array(Zk_r)

        numpy2ri.deactivate()
        pandas2ri.deactivate()

        # Use Python to compute W statistics
        np.random.seed(seed)
        W = stat_glmnet_lambdasmax(Z, Zk, y.flatten(), use_sklearn=use_sklearn)
        threshold = knockoff_threshold(W, fdr=fdr, offset=0)

        if threshold < np.inf:
            selected = np.where(W >= threshold)[0]
        else:
            selected = np.array([], dtype=int)

        backend_name = 'R_knockoffs_py_stats_sklearn' if use_sklearn else 'R_knockoffs_py_stats'
        return {'W': W, 'threshold': threshold, 'selected': selected, 'backend': backend_name}

    except Exception as e:
        logger.error(f"R knockoffs + Python stats failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_knockoffs_knockoff_filter(Z: np.ndarray, y: np.ndarray, fdr: float = 0.1,
                                      method: str = 'asdp', use_sklearn: bool = False,
                                      seed: int = 42, **kwargs) -> Optional[Dict]:
    """knockoff-filter Python implementation.

    Uses ASDP by default to match R's create.second_order default behavior.
    """
    np.random.seed(seed)

    try:
        from loveslide.knockoff.stats import stat_glmnet_lambdasmax
        from loveslide.knockoff.solve import create_solve_asdp, create_solve_sdp, create_solve_equi
        from loveslide.knockoff.create import create_gaussian
        from loveslide.knockoff.filter import knockoff_threshold
        from loveslide.knockoff.utils import is_posdef
    except ImportError as e:
        logger.error(f"loveslide.knockoff not available: {e}")
        return None

    n, p = Z.shape
    Sigma = np.cov(Z, rowvar=False)
    if Sigma.ndim == 0:
        Sigma = np.array([[Sigma]])

    if not is_posdef(Sigma):
        min_eig = np.min(np.linalg.eigvalsh(Sigma))
        if min_eig < 1e-10:
            Sigma = Sigma + (1e-10 - min_eig) * np.eye(p)

    # Use method matching R's create.second_order default (asdp)
    mu = np.mean(Z, axis=0)
    # Let create_gaussian compute diag_s internally using the specified method
    Zk = create_gaussian(Z, mu, Sigma, method=method)
    W = stat_glmnet_lambdasmax(Z, Zk, y.flatten(), use_sklearn=use_sklearn)
    threshold = knockoff_threshold(W, fdr=fdr, offset=0)

    if threshold < np.inf:
        selected = np.where(W >= threshold)[0]
    else:
        selected = np.array([], dtype=int)

    backend_name = 'knockoff_filter_sklearn' if use_sklearn else 'knockoff_filter'
    return {'W': W, 'threshold': threshold, 'selected': selected, 'backend': backend_name}


def compute_knockoffs_custom_glmnet(Z: np.ndarray, y: np.ndarray, fdr: float = 0.1,
                                    method: str = 'sdp', seed: int = 42,
                                    **kwargs) -> Optional[Dict]:
    """SLIDE's custom glmnet implementation using sklearn lasso_path.

    Uses bundled knockoff-filter for knockoff generation.
    """
    from sklearn.linear_model import lasso_path
    np.random.seed(seed)

    try:
        from loveslide.knockoff.create import create_gaussian
        from loveslide.knockoff.solve import create_solve_sdp, create_solve_asdp, create_solve_equi
        from loveslide.knockoff.utils import is_posdef
    except ImportError as e:
        logger.error(f"loveslide.knockoff not available: {e}")
        return None

    n, p = Z.shape

    mu = np.mean(Z, axis=0)
    Sigma = np.cov(Z, rowvar=False)
    if Sigma.ndim == 0:
        Sigma = np.array([[Sigma]])

    # Ensure positive-definiteness
    if not is_posdef(Sigma):
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(Z)
        Sigma = lw.covariance_

    # Generate knockoffs using bundled knockoff-filter
    if method == 'equi':
        diag_s = create_solve_equi(Sigma)
    elif method in ('asdp', 'sdp'):
        # Use SDP (ASDP falls back to SDP for small p)
        if p <= 500:
            diag_s = create_solve_sdp(Sigma)
        else:
            diag_s = create_solve_asdp(Sigma)
    else:
        diag_s = create_solve_sdp(Sigma)

    Zk = create_gaussian(Z, mu, Sigma, method=method, diag_s=diag_s)

    y_flat = y.flatten()
    nlambda, eps = 500, 0.0005

    swap = np.random.binomial(1, 0.5, size=p)
    Z_swap = Z * (1 - swap) + Zk * swap
    Zk_swap = Z * swap + Zk * (1 - swap)
    Z_full = np.hstack([Z_swap, Zk_swap])

    lambda_max = np.max(np.abs(Z_full.T @ y_flat)) / n
    lambda_min = lambda_max * eps
    lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), nlambda)

    try:
        _, coef_path, _ = lasso_path(Z_full, y_flat, alphas=lambdas, max_iter=10000)
    except Exception:
        _, coef_path, _ = lasso_path(Z_full, y_flat, n_alphas=nlambda, max_iter=10000)
        lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), coef_path.shape[1])

    Z_entry = np.zeros(p)
    Zk_entry = np.zeros(p)

    for j in range(p):
        nonzero = np.where(np.abs(coef_path[j, :]) > 1e-10)[0]
        Z_entry[j] = lambdas[nonzero[0]] * n if len(nonzero) > 0 else 0
        nonzero_k = np.where(np.abs(coef_path[p + j, :]) > 1e-10)[0]
        Zk_entry[j] = lambdas[nonzero_k[0]] * n if len(nonzero_k) > 0 else 0

    W = np.maximum(Z_entry, Zk_entry) * np.sign(Z_entry - Zk_entry)
    W = W * (1 - 2 * swap)

    W_abs = np.abs(W)
    candidates = np.sort(np.concatenate([[0], W_abs]))

    threshold = np.inf
    for t in candidates:
        numerator = np.sum(W <= -t)
        denominator = max(1, np.sum(W >= t))
        if numerator / denominator <= fdr:
            threshold = t
            break

    if threshold < np.inf:
        selected = np.where(W >= threshold)[0]
    else:
        selected = np.array([], dtype=int)

    return {'W': W, 'threshold': threshold, 'selected': selected, 'backend': 'custom_glmnet'}


# =============================================================================
# Backend Registry
# =============================================================================
# RECOMMENDED for comparison:
#   1. R_native              - Pure R baseline (reference)
#   2. R_knockoffs_py_sklearn - R knockoffs + Python stats (isolates glmnet diff)
#   3. knockoff_filter_sklearn - Pure Python (statistically equivalent to R)
#
# Comparison logic:
#   R_native vs R_knockoffs_py_sklearn     → glmnet implementation difference
#   R_knockoffs_py_sklearn vs knockoff_filter_sklearn → knockoff generation difference
#   R_native vs knockoff_filter_sklearn    → total R vs Python difference
# =============================================================================

BACKENDS = {
    # --- Primary backends (recommended) ---
    'R_native': compute_knockoffs_r,
    'R_knockoffs_py_sklearn': lambda Z, y, **kw: compute_knockoffs_r_knockoffs_py_stats(Z, y, use_sklearn=True, **kw),
    'knockoff_filter_sklearn': lambda Z, y, **kw: compute_knockoffs_knockoff_filter(Z, y, use_sklearn=True, **kw),

    # --- Secondary backends (for detailed analysis) ---
    'R_knockoffs_py_stats': lambda Z, y, **kw: compute_knockoffs_r_knockoffs_py_stats(Z, y, use_sklearn=False, **kw),
    'knockoff_filter': lambda Z, y, **kw: compute_knockoffs_knockoff_filter(Z, y, use_sklearn=False, **kw),

    # --- Custom SLIDE implementation ---
    'custom_glmnet': compute_knockoffs_custom_glmnet,
}


# =============================================================================
# Data Loading
# =============================================================================

def discover_parameter_dirs(love_dir: Path) -> List[Tuple[float, float, Path]]:
    """Discover parameter directories from old format (e.g., 0.05_0.1_out)."""
    pattern = re.compile(r'^(\d+\.?\d*)_(\d+\.?\d*)_out$')
    params = []

    for d in love_dir.iterdir():
        if d.is_dir():
            match = pattern.match(d.name)
            if match:
                delta = float(match.group(1))
                lam = float(match.group(2))
                params.append((delta, lam, d))

    params.sort(key=lambda x: (x[0], x[1]))
    return params


def load_z_matrix(z_path: Path) -> np.ndarray:
    """Load Z matrix from CSV file."""
    df = pd.read_csv(z_path, index_col=0)
    return df.values


def load_y_from_config(love_dir: Path) -> np.ndarray:
    """Load y vector from path specified in params.yaml."""
    params_path = love_dir / 'params.yaml'
    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found in {love_dir}")

    with open(params_path) as f:
        config = yaml.safe_load(f)

    y_path = config['y_path']
    y_df = pd.read_csv(y_path, index_col=0)

    # Handle factor encoding if needed
    y = y_df.iloc[:, 0].values
    if y.dtype == object or config.get('y_factor', False):
        unique_vals = np.unique(y)
        if len(unique_vals) == 2:
            y = (y == unique_vals[1]).astype(float)
        else:
            # Try numeric conversion
            y = y.astype(float)

    return y


# =============================================================================
# Main Processing
# =============================================================================

def run_knockoffs(love_dir: str, backend: str, output_dir: str,
                  fdr: float = 0.1, seed: int = 42, params_filter: Optional[List[str]] = None):
    """Run knockoffs on pre-computed LOVE results."""
    love_dir = Path(love_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running knockoffs on: {love_dir}")
    logger.info(f"Backend: {backend}")
    logger.info(f"Output: {output_dir}")

    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Available: {list(BACKENDS.keys())}")

    compute_fn = BACKENDS[backend]

    # Discover parameter directories
    param_dirs = discover_parameter_dirs(love_dir)
    if not param_dirs:
        logger.error(f"No parameter directories found in {love_dir}")
        return

    logger.info(f"Found {len(param_dirs)} parameter sets")

    # Filter if specified
    if params_filter:
        param_dirs = [(d, l, p) for d, l, p in param_dirs
                      if f"{d}_{l}" in params_filter or f"d{d}_l{l}" in params_filter]
        logger.info(f"Filtered to {len(param_dirs)} parameter sets")

    # Load y
    y = load_y_from_config(love_dir)
    logger.info(f"Loaded y: shape={y.shape}, unique={np.unique(y)}")

    # Load params for FDR
    params_path = love_dir / 'params.yaml'
    if params_path.exists():
        with open(params_path) as f:
            config = yaml.safe_load(f)
        fdr = config.get('fdr', fdr)
        logger.info(f"Using FDR={fdr} from config")

    results = []

    for delta, lam, param_dir in param_dirs:
        param_str = f"d{delta}_l{lam}"
        z_path = param_dir / 'z_matrix.csv'

        if not z_path.exists():
            logger.warning(f"  {param_str}: z_matrix.csv not found, skipping")
            continue

        logger.info(f"  {param_str}: Loading Z...")
        Z = load_z_matrix(z_path)
        logger.info(f"    Z shape: {Z.shape}")

        # Validate dimensions
        if Z.shape[0] != len(y):
            logger.error(f"    Shape mismatch: Z has {Z.shape[0]} samples, y has {len(y)}")
            continue

        logger.info(f"    Running {backend}...")
        try:
            result = compute_fn(Z, y, fdr=fdr, seed=seed)

            if result:
                result_clean = {
                    'W': result['W'].tolist(),
                    'threshold': float(result['threshold']) if result['threshold'] < np.inf else 'inf',
                    'selected': result['selected'].tolist(),
                    'n_selected': len(result['selected'])
                }
                results.append({
                    'params': {'delta': delta, 'lambda': lam, 'fdr': fdr},
                    'z_shape': list(Z.shape),
                    'param_dir': str(param_dir),
                    'result': result_clean
                })
                logger.info(f"    Selected: {len(result['selected'])} variables")
            else:
                logger.warning(f"    No result returned")

        except Exception as e:
            logger.error(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = output_dir / f"knockoff_results_{backend}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'love_dir': str(love_dir),
            'backend': backend,
            'fdr': fdr,
            'seed': seed,
            'n_param_sets': len(param_dirs),
            'results': results
        }, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for r in results:
        p = r['params']
        logger.info(f"  d{p['delta']}_l{p['lambda']}: {r['result']['n_selected']} selected "
                    f"(threshold={r['result']['threshold']})")

    return results


# =============================================================================
# Comparison
# =============================================================================

def compare_results(result_files: List[str], output_file: Optional[str] = None,
                    reference_backend: str = 'R_native'):
    """Compare knockoff results across multiple backends.

    Parameters
    ----------
    result_files : list of str
        Paths to knockoff_results_*.json files
    output_file : str, optional
        Path to write comparison report
    reference_backend : str
        Backend to use as reference for comparisons (default: R_native)
    """
    # Load all results
    all_data = {}
    for fpath in result_files:
        with open(fpath) as f:
            data = json.load(f)
        backend = data['backend']
        all_data[backend] = data
        logger.info(f"Loaded {backend} from {fpath}")

    backends = list(all_data.keys())
    logger.info(f"Comparing {len(backends)} backends: {backends}")

    if reference_backend not in backends:
        reference_backend = backends[0]
        logger.warning(f"Reference backend not found, using {reference_backend}")

    # Reorganize by parameter set
    param_comparison = {}

    for backend, data in all_data.items():
        for r in data['results']:
            param_key = f"d{r['params']['delta']}_l{r['params']['lambda']}"

            if param_key not in param_comparison:
                param_comparison[param_key] = {
                    'params': r['params'],
                    'backends': {}
                }

            param_comparison[param_key]['backends'][backend] = r['result']

    # Compute comparison metrics
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("KNOCKOFF BACKEND COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Backends: {', '.join(backends)}")
    report_lines.append(f"Reference: {reference_backend}")
    report_lines.append(f"Parameter sets: {len(param_comparison)}")
    report_lines.append("")

    summary_rows = []

    for param_key in sorted(param_comparison.keys()):
        pc = param_comparison[param_key]
        report_lines.append("-" * 80)
        report_lines.append(f"Parameter set: {param_key}")
        report_lines.append("-" * 80)

        available_backends = list(pc['backends'].keys())

        # Get W statistics
        W_dict = {}
        for b in available_backends:
            W_dict[b] = np.array(pc['backends'][b]['W'])

        # Selection sets
        selected_dict = {}
        for b in available_backends:
            selected_dict[b] = set(pc['backends'][b]['selected'])

        # Print selection counts
        report_lines.append("\nSelection counts:")
        for b in available_backends:
            n_sel = len(selected_dict[b])
            report_lines.append(f"  {b}: {n_sel} variables")

        # W statistic correlations with reference
        if reference_backend in available_backends:
            report_lines.append(f"\nW correlation with {reference_backend}:")
            W_ref = W_dict[reference_backend]
            for b in available_backends:
                if b == reference_backend:
                    continue
                with np.errstate(divide='ignore', invalid='ignore'):
                    corr = np.corrcoef(W_ref, W_dict[b])[0, 1]
                corr_str = f"{corr:.4f}" if not np.isnan(corr) else "N/A"
                report_lines.append(f"  {b}: {corr_str}")

        # Selection agreement (Jaccard index)
        if reference_backend in available_backends:
            report_lines.append(f"\nSelection agreement with {reference_backend}:")
            ref_selected = selected_dict[reference_backend]

            for b in available_backends:
                if b == reference_backend:
                    continue
                other_selected = selected_dict[b]
                union = len(ref_selected | other_selected)
                intersection = len(ref_selected & other_selected)
                jaccard = intersection / union if union > 0 else 1.0

                only_ref = ref_selected - other_selected
                only_other = other_selected - ref_selected

                report_lines.append(f"  {b}:")
                report_lines.append(f"    Jaccard index: {jaccard:.4f}")
                report_lines.append(f"    Intersection: {intersection}")
                report_lines.append(f"    Only in {reference_backend}: {len(only_ref)}")
                report_lines.append(f"    Only in {b}: {len(only_other)}")

                # Add to summary
                summary_rows.append({
                    'param': param_key,
                    'backend': b,
                    'n_selected': len(other_selected),
                    'ref_n_selected': len(ref_selected),
                    'jaccard': jaccard,
                    'intersection': intersection,
                    'W_corr': np.corrcoef(W_ref, W_dict[b])[0, 1] if len(W_ref) > 1 else np.nan
                })

        report_lines.append("")

    # Summary table
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY TABLE")
    report_lines.append("=" * 80)

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        report_lines.append(df_summary.to_string(index=False))

        # Average metrics by backend
        report_lines.append("\n" + "-" * 40)
        report_lines.append("Average metrics by backend:")
        report_lines.append("-" * 40)
        avg_metrics = df_summary.groupby('backend').agg({
            'jaccard': 'mean',
            'W_corr': 'mean',
            'n_selected': 'mean'
        }).round(4)
        report_lines.append(avg_metrics.to_string())

    report = "\n".join(report_lines)
    print(report)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        logger.info(f"\nReport saved to: {output_file}")

        # Also save summary as CSV
        if summary_rows:
            csv_file = Path(output_file).with_suffix('.csv')
            pd.DataFrame(summary_rows).to_csv(csv_file, index=False)
            logger.info(f"Summary CSV saved to: {csv_file}")

    return param_comparison


def main():
    parser = argparse.ArgumentParser(
        description='Run knockoffs on pre-computed LOVE Z matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run subcommand
    run_parser = subparsers.add_parser('run', help='Run knockoffs on pre-computed LOVE')
    run_parser.add_argument('--love-dir', required=True,
                            help='Directory with pre-computed LOVE results')
    run_parser.add_argument('--backend', required=True, choices=list(BACKENDS.keys()),
                            help='Knockoff backend to use')
    run_parser.add_argument('--output-dir', required=True,
                            help='Output directory for results')
    run_parser.add_argument('--fdr', type=float, default=0.1,
                            help='FDR threshold (default: 0.1)')
    run_parser.add_argument('--seed', type=int, default=42,
                            help='Random seed (default: 42)')
    run_parser.add_argument('--params', nargs='+',
                            help='Specific parameters to run (e.g., "0.05_0.1")')

    # Compare subcommand
    cmp_parser = subparsers.add_parser('compare', help='Compare knockoff results')
    cmp_parser.add_argument('result_files', nargs='+',
                            help='knockoff_results_*.json files to compare')
    cmp_parser.add_argument('--output', '-o',
                            help='Output file for comparison report')
    cmp_parser.add_argument('--reference', default='R_native',
                            help='Reference backend for comparisons (default: R_native)')

    args = parser.parse_args()

    if args.command == 'run':
        run_knockoffs(
            love_dir=args.love_dir,
            backend=args.backend,
            output_dir=args.output_dir,
            fdr=args.fdr,
            seed=args.seed,
            params_filter=args.params
        )
    elif args.command == 'compare':
        compare_results(
            result_files=args.result_files,
            output_file=args.output,
            reference_backend=args.reference
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
