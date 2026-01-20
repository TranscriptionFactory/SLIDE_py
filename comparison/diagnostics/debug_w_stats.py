#!/usr/bin/env python3
"""
Diagnostic script to compare W statistics between R and Python knockoff implementations.
"""
import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, '/ix/djishnu/Aaron/1_general_use/SLIDE_py/src')
sys.path.insert(0, '/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter')

from loveslide.knockoffs import Knockoffs


def load_data(output_dir, config_path):
    """Load Z matrix and y from a SLIDE output directory."""
    output_dir = Path(output_dir)

    # Load Z matrix from CSV
    z_matrix = pd.read_csv(output_dir / 'z_matrix.csv', index_col=0).values

    # Load Y from config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    y_df = pd.read_csv(config['y_path'], index_col=0)
    y = y_df.values.flatten()

    # If y is categorical (factor), convert to numeric
    if y.dtype == object or config.get('y_factor', False):
        unique_vals = np.unique(y)
        if len(unique_vals) == 2:
            # Binary classification - map to 0/1
            y = (y == unique_vals[1]).astype(float)
        else:
            # Keep as is for regression
            y = y.astype(float)

    return z_matrix, y


def compute_w_stats_python_glmnet(z, y):
    """Compute W stats using Python glmnet-equivalent method."""
    from knockpy.knockoffs import GaussianSampler

    z_scaled = Knockoffs.scale_features(z)

    # Generate knockoffs
    sampler = GaussianSampler(X=z_scaled, method='mvr')
    Xk = sampler.sample_knockoffs()

    # Compute W using glmnet-equivalent
    W = Knockoffs._compute_glmnet_lambdasmax(z_scaled, Xk, y)

    return W, z_scaled, Xk


def compute_w_stats_r(z, y):
    """Compute W stats using R knockoff package."""
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr

    numpy2ri.activate()
    pandas2ri.activate()

    z_scaled = Knockoffs.scale_features(z)

    z_r = robjects.r['as.matrix'](pandas2ri.py2rpy(pd.DataFrame(z_scaled)))
    y_r = robjects.FloatVector(y.flatten())

    knockoff = importr('knockoff')

    # Create knockoffs
    Xk_r = knockoff.create_second_order(z_r)

    # Compute statistics using stat.glmnet_lambdasmax
    stat_func = robjects.r['knockoff::stat.glmnet_lambdasmax']
    W_r = stat_func(z_r, Xk_r, y_r)

    W = np.array(W_r)
    Xk = np.array(Xk_r)

    numpy2ri.deactivate()
    pandas2ri.deactivate()

    return W, z_scaled, Xk


def compute_threshold_comparison(W_py, W_r, fdr=0.1):
    """Compare thresholds computed from W statistics."""
    # Python threshold
    threshold_py = Knockoffs._knockoff_threshold(W_py, fdr, offset=0)
    threshold_py_plus = Knockoffs._knockoff_threshold(W_py, fdr, offset=1)

    # R-style threshold (same formula)
    threshold_r = Knockoffs._knockoff_threshold(W_r, fdr, offset=0)
    threshold_r_plus = Knockoffs._knockoff_threshold(W_r, fdr, offset=1)

    return {
        'py_threshold_offset0': threshold_py,
        'py_threshold_offset1': threshold_py_plus,
        'r_threshold_offset0': threshold_r,
        'r_threshold_offset1': threshold_r_plus,
    }


def main():
    # Use 0.2_0.1 parameters where we had the most matches
    output_dir = '/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/test_output_glmnet/0.2_0.1_out'
    config_path = '/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/test_output_glmnet/params.yaml'

    print("Loading data...")
    z_matrix, y = load_data(output_dir, config_path)
    print(f"Z matrix shape: {z_matrix.shape}")
    print(f"Y shape: {y.shape}")

    print("\n" + "="*70)
    print("Computing W statistics with Python (glmnet-equivalent)...")
    print("="*70)
    W_py, z_scaled, Xk_py = compute_w_stats_python_glmnet(z_matrix, y)

    print(f"\nPython W stats summary:")
    print(f"  Shape: {W_py.shape}")
    print(f"  Min: {W_py.min():.6f}")
    print(f"  Max: {W_py.max():.6f}")
    print(f"  Mean: {W_py.mean():.6f}")
    print(f"  Std: {W_py.std():.6f}")
    print(f"  # positive: {np.sum(W_py > 0)}")
    print(f"  # negative: {np.sum(W_py < 0)}")
    print(f"  # zero: {np.sum(W_py == 0)}")

    # Show top positive W values
    top_idx = np.argsort(W_py)[::-1][:10]
    print(f"\n  Top 10 positive W values:")
    for i, idx in enumerate(top_idx):
        print(f"    Z{idx}: W = {W_py[idx]:.6f}")

    print("\n" + "="*70)
    print("Computing W statistics with R knockoff package...")
    print("="*70)
    try:
        W_r, _, Xk_r = compute_w_stats_r(z_matrix, y)

        print(f"\nR W stats summary:")
        print(f"  Shape: {W_r.shape}")
        print(f"  Min: {W_r.min():.6f}")
        print(f"  Max: {W_r.max():.6f}")
        print(f"  Mean: {W_r.mean():.6f}")
        print(f"  Std: {W_r.std():.6f}")
        print(f"  # positive: {np.sum(W_r > 0)}")
        print(f"  # negative: {np.sum(W_r < 0)}")
        print(f"  # zero: {np.sum(W_r == 0)}")

        # Show top positive W values
        top_idx_r = np.argsort(W_r)[::-1][:10]
        print(f"\n  Top 10 positive W values:")
        for i, idx in enumerate(top_idx_r):
            print(f"    Z{idx}: W = {W_r[idx]:.6f}")

        print("\n" + "="*70)
        print("Comparing W statistics...")
        print("="*70)

        # Note: Knockoffs are random, so W values will differ due to different knockoffs
        # But we can compare the distribution characteristics
        print(f"\nCorrelation between Python and R W stats: {np.corrcoef(W_py, W_r)[0,1]:.4f}")
        print(f"Mean absolute difference: {np.mean(np.abs(W_py - W_r)):.6f}")
        print(f"Max absolute difference: {np.max(np.abs(W_py - W_r)):.6f}")

        # Compare thresholds
        print("\n" + "="*70)
        print("Comparing knockoff thresholds (FDR=0.1)...")
        print("="*70)
        thresholds = compute_threshold_comparison(W_py, W_r, fdr=0.1)
        for k, v in thresholds.items():
            print(f"  {k}: {v:.6f}" if v < np.inf else f"  {k}: inf")

        # How many would be selected?
        print("\n  Selections at FDR=0.1:")
        print(f"    Python (offset=0): {np.sum(W_py >= thresholds['py_threshold_offset0']) if thresholds['py_threshold_offset0'] < np.inf else 0}")
        print(f"    Python (offset=1): {np.sum(W_py >= thresholds['py_threshold_offset1']) if thresholds['py_threshold_offset1'] < np.inf else 0}")
        print(f"    R (offset=0): {np.sum(W_r >= thresholds['r_threshold_offset0']) if thresholds['r_threshold_offset0'] < np.inf else 0}")
        print(f"    R (offset=1): {np.sum(W_r >= thresholds['r_threshold_offset1']) if thresholds['r_threshold_offset1'] < np.inf else 0}")

    except Exception as e:
        print(f"Error with R computation: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("Now testing with same knockoffs (to isolate statistic computation)...")
    print("="*70)

    # Generate knockoffs once and use for both
    from knockpy.knockoffs import GaussianSampler
    sampler = GaussianSampler(X=z_scaled, method='mvr')
    Xk_shared = sampler.sample_knockoffs()

    # Python W
    W_py_shared = Knockoffs._compute_glmnet_lambdasmax(z_scaled, Xk_shared, y)

    # R W with same knockoffs
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()
        pandas2ri.activate()

        z_r = robjects.r['as.matrix'](pandas2ri.py2rpy(pd.DataFrame(z_scaled)))
        Xk_r = robjects.r['as.matrix'](pandas2ri.py2rpy(pd.DataFrame(Xk_shared)))
        y_r = robjects.FloatVector(y.flatten())

        knockoff = importr('knockoff')
        stat_func = robjects.r['knockoff::stat.glmnet_lambdasmax']
        W_r_shared = np.array(stat_func(z_r, Xk_r, y_r))

        numpy2ri.deactivate()
        pandas2ri.deactivate()

        print(f"\nWith SAME knockoffs:")
        print(f"  Correlation: {np.corrcoef(W_py_shared, W_r_shared)[0,1]:.4f}")
        print(f"  Mean abs diff: {np.mean(np.abs(W_py_shared - W_r_shared)):.6f}")
        print(f"  Max abs diff: {np.max(np.abs(W_py_shared - W_r_shared)):.6f}")

        # Show comparison for top features
        print(f"\n  Per-feature comparison (top 10 by R W value):")
        top_r = np.argsort(W_r_shared)[::-1][:10]
        print(f"  {'Z_idx':>6} {'W_py':>12} {'W_r':>12} {'diff':>12}")
        for idx in top_r:
            print(f"  {idx:>6} {W_py_shared[idx]:>12.6f} {W_r_shared[idx]:>12.6f} {W_py_shared[idx] - W_r_shared[idx]:>12.6f}")

        # Check thresholds
        print("\n  Thresholds (FDR=0.1) with same knockoffs:")
        t_py = Knockoffs._knockoff_threshold(W_py_shared, 0.1, offset=0)
        t_r = Knockoffs._knockoff_threshold(W_r_shared, 0.1, offset=0)
        print(f"    Python threshold: {t_py:.6f}" if t_py < np.inf else "    Python threshold: inf")
        print(f"    R threshold: {t_r:.6f}" if t_r < np.inf else "    R threshold: inf")

        sel_py = np.sum(W_py_shared >= t_py) if t_py < np.inf else 0
        sel_r = np.sum(W_r_shared >= t_r) if t_r < np.inf else 0
        print(f"    Python selections: {sel_py}")
        print(f"    R selections: {sel_r}")

    except Exception as e:
        print(f"Error comparing with shared knockoffs: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
