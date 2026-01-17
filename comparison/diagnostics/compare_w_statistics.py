#!/usr/bin/env python3
"""
Compare W statistics between knockpy and R knockoff implementations.

This script diagnoses why knockpy produces different W statistics than R's
knockoff package, leading to fewer significant LF selections.

Usage:
    python compare_w_statistics.py [--data-path PATH] [--output-dir DIR]

Example:
    python compare_w_statistics.py \
        --data-path /path/to/z_matrix.csv \
        --y-path /path/to/y.csv \
        --output-dir ./diagnostics_output
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project source to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from knockpy import KnockoffFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_knockpy_w_statistics(X, y, fstat='lsm', method='mvr', shrinkage=None, seed=42, **kwargs):
    """
    Compute W statistics using knockpy.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n x p)
    y : np.ndarray
        Response vector (n,)
    fstat : str
        Feature statistic method: 'lsm', 'lasso', 'lcd', 'ols'
    method : str
        Knockoff construction method: 'mvr', 'sdp', 'equicorrelated'
    shrinkage : str or None
        Covariance shrinkage: 'ledoitwolf' or None
    seed : int
        Random seed

    Returns
    -------
    dict with keys:
        - W: W statistics array
        - threshold: knockpy computed threshold
        - rejections: boolean mask of rejections
        - kfilter: the KnockoffFilter object for inspection
    """
    np.random.seed(seed)

    kfilter = KnockoffFilter(
        ksampler='gaussian',
        fstat=fstat,
        knockoff_kwargs={'method': method}
    )

    rejections = kfilter.forward(
        X=X,
        y=y.flatten(),
        fdr=0.2,  # Standard FDR for comparison
        shrinkage=shrinkage
    )

    return {
        'W': kfilter.W,
        'threshold': kfilter.threshold,
        'rejections': rejections,
        'kfilter': kfilter
    }


def compute_custom_threshold(W, fdr, offset=0):
    """
    Compute knockoff threshold using the custom implementation from knockoffs.py.

    This matches the Knockoffs._knockoff_threshold() method.
    """
    W_abs = np.abs(W)
    candidates = np.sort(W_abs[W_abs > 0])

    threshold = np.inf
    for t in candidates:
        numerator = offset + np.sum(W <= -t)
        denominator = max(1, np.sum(W >= t))
        if numerator / denominator <= fdr:
            threshold = t
            break
    return threshold


def run_fstat_comparison(X, y, seed=42):
    """
    Test all knockpy fstat options and compare results.

    Returns dict mapping fstat name to results dict.
    """
    fstat_options = ['lsm', 'lasso', 'lcd', 'ols']
    results = {}

    for fstat in fstat_options:
        logger.info(f"Testing fstat='{fstat}'...")
        try:
            result = compute_knockpy_w_statistics(X, y, fstat=fstat, seed=seed)
            results[fstat] = result

            n_selected = np.sum(result['rejections'])
            logger.info(f"  fstat='{fstat}': {n_selected} rejections, "
                       f"threshold={result['threshold']:.4f}, "
                       f"W range=[{np.min(result['W']):.4f}, {np.max(result['W']):.4f}]")
        except Exception as e:
            logger.warning(f"  fstat='{fstat}' failed: {e}")
            results[fstat] = {'error': str(e)}

    return results


def run_method_comparison(X, y, fstat='lsm', seed=42):
    """
    Test different knockoff construction methods.
    """
    methods = ['mvr', 'sdp', 'equicorrelated']
    results = {}

    for method in methods:
        logger.info(f"Testing method='{method}'...")
        try:
            result = compute_knockpy_w_statistics(X, y, fstat=fstat, method=method, seed=seed)
            results[method] = result

            n_selected = np.sum(result['rejections'])
            logger.info(f"  method='{method}': {n_selected} rejections, "
                       f"threshold={result['threshold']:.4f}")
        except Exception as e:
            logger.warning(f"  method='{method}' failed: {e}")
            results[method] = {'error': str(e)}

    return results


def run_threshold_comparison(W, fdr=0.2):
    """
    Compare different threshold computation approaches.
    """
    # Custom threshold with offset=0 (original knockoff)
    custom_offset0 = compute_custom_threshold(W, fdr, offset=0)

    # Custom threshold with offset=1 (knockoff+)
    custom_offset1 = compute_custom_threshold(W, fdr, offset=1)

    # knockpy's built-in threshold (already computed when we ran forward())
    # We need to recompute using knockpy's method
    from knockpy.knockoff_filter import data_dependent_threshhold
    knockpy_threshold = data_dependent_threshhold(W, fdr=fdr, offset=1)

    return {
        'custom_offset0': custom_offset0,
        'custom_offset1': custom_offset1,
        'knockpy_builtin': knockpy_threshold
    }


def visualize_w_statistics(results, output_dir, prefix='w_stats'):
    """
    Create visualization plots for W statistics comparison.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: W statistics histogram comparison across fstat methods
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    fstat_names = ['lsm', 'lasso', 'lcd', 'ols']
    for i, fstat in enumerate(fstat_names):
        ax = axes[i]
        if fstat in results.get('fstat_comparison', {}) and 'W' in results['fstat_comparison'][fstat]:
            W = results['fstat_comparison'][fstat]['W']
            ax.hist(W, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', label='W=0')
            threshold = results['fstat_comparison'][fstat]['threshold']
            if threshold < np.inf:
                ax.axvline(x=threshold, color='green', linestyle='--', label=f'threshold={threshold:.3f}')
            ax.set_title(f"fstat='{fstat}'")
            ax.set_xlabel('W statistic')
            ax.set_ylabel('Count')
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"'{fstat}' not available", ha='center', va='center')
            ax.set_title(f"fstat='{fstat}'")

    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_fstat_comparison.png', dpi=150)
    plt.close()

    # Plot 2: Threshold comparison
    if 'threshold_comparison' in results:
        fig, ax = plt.subplots(figsize=(8, 6))
        thresholds = results['threshold_comparison']
        names = list(thresholds.keys())
        values = [thresholds[n] if thresholds[n] < np.inf else 0 for n in names]
        colors = ['red' if v == 0 else 'blue' for v in values]

        bars = ax.bar(names, values, color=colors, alpha=0.7)
        ax.set_ylabel('Threshold value')
        ax.set_title('Threshold Comparison (red = inf/no selections)')
        ax.tick_params(axis='x', rotation=45)

        for bar, val, name in zip(bars, values, names):
            if val == 0 and thresholds[name] == np.inf:
                ax.text(bar.get_x() + bar.get_width()/2, 0.01, 'inf',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / f'{prefix}_threshold_comparison.png', dpi=150)
        plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


def print_summary(results):
    """Print a summary of the diagnostic results."""
    print("\n" + "="*80)
    print("W STATISTIC DIAGNOSTIC SUMMARY")
    print("="*80)

    if 'fstat_comparison' in results:
        print("\n--- Feature Statistic (fstat) Comparison ---")
        for fstat, res in results['fstat_comparison'].items():
            if 'error' in res:
                print(f"  {fstat}: ERROR - {res['error']}")
            else:
                W = res['W']
                n_pos = np.sum(W > 0)
                n_neg = np.sum(W < 0)
                n_reject = np.sum(res['rejections'])
                print(f"  {fstat}:")
                print(f"    W range: [{W.min():.4f}, {W.max():.4f}]")
                print(f"    Positive W: {n_pos}, Negative W: {n_neg}")
                print(f"    Threshold: {res['threshold']:.4f}")
                print(f"    Rejections: {n_reject}")

    if 'method_comparison' in results:
        print("\n--- Knockoff Construction Method Comparison ---")
        for method, res in results['method_comparison'].items():
            if 'error' in res:
                print(f"  {method}: ERROR - {res['error']}")
            else:
                n_reject = np.sum(res['rejections'])
                print(f"  {method}: {n_reject} rejections, threshold={res['threshold']:.4f}")

    if 'threshold_comparison' in results:
        print("\n--- Threshold Computation Comparison ---")
        for name, val in results['threshold_comparison'].items():
            print(f"  {name}: {val:.4f}" if val < np.inf else f"  {name}: inf")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare knockpy W statistics with R")
    parser.add_argument('--data-path', default=None,
                        help='Path to Z matrix CSV (samples x features)')
    parser.add_argument('--y-path', default=None,
                        help='Path to Y vector CSV')
    parser.add_argument('--output-dir', default='./diagnostics_output',
                        help='Output directory for plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use-synthetic', action='store_true',
                        help='Use synthetic test data instead of loading from files')

    args = parser.parse_args()

    # Load or generate data
    if args.use_synthetic or (args.data_path is None and args.y_path is None):
        logger.info("Using synthetic test data with known signal...")
        np.random.seed(args.seed)
        n, p = 200, 50
        X = np.random.randn(n, p)

        # Strong signal in first 5 features
        beta = np.zeros(p)
        beta[0] = 3.0
        beta[1] = 2.5
        beta[2] = 2.0
        beta[3] = 1.5
        beta[4] = 1.0

        y = X @ beta + 0.5 * np.random.randn(n)

        logger.info(f"Synthetic data: n={n}, p={p}")
        logger.info(f"True non-zero features: [0, 1, 2, 3, 4] with betas {list(beta[:5])}")
    else:
        logger.info(f"Loading data from {args.data_path}...")
        X_df = pd.read_csv(args.data_path, index_col=0)
        X = X_df.values

        y_df = pd.read_csv(args.y_path, index_col=0)
        y = y_df.values.flatten()

        logger.info(f"Loaded data: X shape={X.shape}, y shape={y.shape}")

    # Standardize X (as done in knockoffs.py)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    results = {}

    # Run fstat comparison
    logger.info("\n=== Testing different fstat options ===")
    results['fstat_comparison'] = run_fstat_comparison(X, y, seed=args.seed)

    # Run method comparison with best fstat
    logger.info("\n=== Testing different knockoff construction methods ===")
    results['method_comparison'] = run_method_comparison(X, y, fstat='lsm', seed=args.seed)

    # Run threshold comparison using lsm W statistics
    if 'lsm' in results['fstat_comparison'] and 'W' in results['fstat_comparison']['lsm']:
        logger.info("\n=== Comparing threshold computations ===")
        W_lsm = results['fstat_comparison']['lsm']['W']
        results['threshold_comparison'] = run_threshold_comparison(W_lsm, fdr=0.2)

    # Print summary
    print_summary(results)

    # Create visualizations
    logger.info("\nCreating visualizations...")
    visualize_w_statistics(results, args.output_dir)

    return results


if __name__ == '__main__':
    main()
