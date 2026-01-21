#!/usr/bin/env python3
"""
Compare W-statistics across all three knockoff backends:
1. R native (knockoff package via rpy2)
2. knockoff-filter (Python, custom glmnet_lambdasmax)
3. knockpy (Python, various fstat options)

This script diagnoses divergence by comparing:
- Knockoff matrices (X_tilde)
- W-statistics
- Thresholds
- Selected variables

Usage:
    python compare_w_statistics_all.py \
        --z-path /path/to/z_matrix.csv \
        --y-path /path/to/y.csv \
        --output-dir ./w_stat_comparison \
        --fdr 0.1

Example with SSc data:
    python compare_w_statistics_all.py \
        --z-path comparison/output_comparison/20260119_225506_rstyle/Py_pyLOVE_pyKO/delta_0.1_lambda_0.5_out/z_matrix.csv \
        --y-path comparison/output_comparison/20260119_225506_rstyle/Py_pyLOVE_pyKO/delta_0.1_lambda_0.5_out/y_vector.csv \
        --output-dir comparison/diagnostics/output/w_stat_comparison
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project source to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Add knockoff-filter to path
KNOCKOFF_FILTER_PATH = "/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter"
if KNOCKOFF_FILTER_PATH not in sys.path:
    sys.path.insert(0, KNOCKOFF_FILTER_PATH)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Backend 1: R native
# =============================================================================

def compute_w_stats_r(X, y, fdr=0.1, method='sdp', seed=42):
    """
    Compute W-statistics using R's knockoff package.

    Returns dict with:
        - W: W statistics
        - threshold: knockoff threshold
        - selected: selected variable indices
        - knockoffs: knockoff matrix
    """
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.packages import importr
    except (ImportError, Exception) as e:
        logger.warning(f"rpy2 not available or failed to initialize: {e}")
        return None

    try:
        numpy2ri.activate()
        pandas2ri.activate()

        # Set seed in R
        robjects.r(f'set.seed({seed})')

        knockoff_r = importr('knockoff')

        # Convert to R objects
        X_r = robjects.r['as.matrix'](pandas2ri.py2rpy(pd.DataFrame(X)))
        y_r = robjects.FloatVector(y.flatten())

        # Run knockoff filter with diagnostics
        # Use create.second_order for knockoff construction
        result = knockoff_r.knockoff_filter(
            X=X_r,
            y=y_r,
            knockoffs=knockoff_r.create_second_order,
            statistic=knockoff_r.stat_glmnet_lambdasmax,
            offset=0,  # Original knockoff
            fdr=fdr
        )

        # Extract results
        selected = np.array(result.rx2('selected')) - 1  # Convert to 0-based

        # Get W statistics (stored in result$statistic)
        W = np.array(result.rx2('statistic'))

        # Get threshold
        threshold = float(np.array(result.rx2('threshold'))[0])

        # Get knockoffs matrix if available
        knockoffs = None
        try:
            knockoffs = np.array(result.rx2('Xk'))
        except Exception:
            pass

        numpy2ri.deactivate()
        pandas2ri.deactivate()

        return {
            'W': W,
            'threshold': threshold,
            'selected': selected,
            'knockoffs': knockoffs,
            'backend': 'R_native'
        }
    except Exception as e:
        logger.warning(f"R knockoff computation failed: {e}")
        try:
            numpy2ri.deactivate()
            pandas2ri.deactivate()
        except Exception:
            pass
        return None


# =============================================================================
# Backend 2: knockoff-filter (Python)
# =============================================================================

def compute_w_stats_knockoff_filter(X, y, fdr=0.1, method='sdp', offset=0, seed=42):
    """
    Compute W-statistics using knockoff-filter Python package.

    Uses the ACTUAL knockoff-filter implementation:
    - create_gaussian() for knockoff generation (uses DSDP solver)
    - stat_glmnet_lambdasmax() for W-statistics (uses vendored Fortran glmnet)
    - knockoff_threshold() for FDR threshold

    This should match R's knockoff package behavior.
    """
    np.random.seed(seed)

    try:
        from knockoff.stats import stat_glmnet_lambdasmax
        from knockoff.stats.glmnet import HAS_GLMNET, _lasso_max_lambda_glmnet
        from knockoff.solve import create_solve_sdp, create_solve_asdp, create_solve_equi
        from knockoff.create import create_gaussian
        from knockoff.filter import knockoff_threshold
        from knockoff.utils import is_posdef
        import knockoff.stats.glmnet as glmnet_module
    except ImportError as e:
        logger.warning(f"knockoff-filter not available: {e}")
        return None

    # Check which version of knockoff-filter is being used
    import inspect
    sig = inspect.signature(_lasso_max_lambda_glmnet)
    std_default = sig.parameters['standardize'].default
    logger.info(f"knockoff-filter: vendored glmnet available = {HAS_GLMNET}")
    logger.info(f"knockoff-filter: glmnet module path = {glmnet_module.__file__}")
    logger.info(f"knockoff-filter: standardize default = {std_default}")

    n, p = X.shape

    # Compute covariance
    Sigma = np.cov(X, rowvar=False)
    if Sigma.ndim == 0:
        Sigma = np.array([[Sigma]])

    # Ensure positive definite
    if not is_posdef(Sigma):
        min_eig = np.min(np.linalg.eigvalsh(Sigma))
        if min_eig < 1e-10:
            Sigma = Sigma + (1e-10 - min_eig) * np.eye(p)

    # Compute S matrix based on method (returns 1D array - diagonal elements)
    logger.info(f"Computing S matrix using method={method}")
    try:
        if method == 'sdp':
            diag_s = create_solve_sdp(Sigma)
        elif method == 'asdp':
            diag_s = create_solve_asdp(Sigma)
        else:  # equi
            diag_s = create_solve_equi(Sigma)
    except Exception as e:
        logger.warning(f"SDP method {method} failed: {e}, falling back to equi")
        diag_s = create_solve_equi(Sigma)

    # Generate knockoffs using knockoff-filter's create_gaussian
    # This uses the same algorithm as R's knockoff package
    mu = np.mean(X, axis=0)
    Xk = create_gaussian(X, mu, Sigma, method=method, diag_s=diag_s)

    # Compute W statistics using glmnet_lambdasmax
    # Uses vendored Fortran glmnet if available, else sklearn fallback
    W = stat_glmnet_lambdasmax(X, Xk, y.flatten())

    # Compute threshold using knockoff-filter's implementation
    threshold = knockoff_threshold(W, fdr=fdr, offset=offset)

    # Select variables
    if threshold < np.inf:
        selected = np.where(W >= threshold)[0]
    else:
        selected = np.array([], dtype=int)

    return {
        'W': W,
        'threshold': threshold,
        'selected': selected,
        'knockoffs': Xk,
        'diag_s': diag_s,
        'backend': 'knockoff_filter',
        'has_vendored_glmnet': HAS_GLMNET
    }


# =============================================================================
# Backend 2b: knockoff-filter with use_sklearn=True (R-compatible)
# =============================================================================

def compute_w_stats_knockoff_filter_sklearn(X, y, fdr=0.1, method='sdp', offset=0, seed=42):
    """
    Compute W-statistics using knockoff-filter with use_sklearn=True.

    This forces sklearn's lasso_path instead of vendored Fortran glmnet,
    which produces W-statistics with positive correlation to R's knockoff package.
    """
    np.random.seed(seed)

    try:
        from knockoff.stats import stat_glmnet_lambdasmax
        from knockoff.stats.glmnet import HAS_GLMNET
        from knockoff.solve import create_solve_sdp, create_solve_asdp, create_solve_equi
        from knockoff.create import create_gaussian
        from knockoff.filter import knockoff_threshold
        from knockoff.utils import is_posdef
    except ImportError as e:
        logger.warning(f"knockoff-filter not available: {e}")
        return None

    logger.info(f"knockoff-filter (sklearn): forcing use_sklearn=True")

    n, p = X.shape

    # Compute covariance
    Sigma = np.cov(X, rowvar=False)
    if Sigma.ndim == 0:
        Sigma = np.array([[Sigma]])

    # Ensure positive definite
    if not is_posdef(Sigma):
        min_eig = np.min(np.linalg.eigvalsh(Sigma))
        if min_eig < 1e-10:
            Sigma = Sigma + (1e-10 - min_eig) * np.eye(p)

    # Compute S matrix based on method
    logger.info(f"Computing S matrix using method={method}")
    try:
        if method == 'sdp':
            diag_s = create_solve_sdp(Sigma)
        elif method == 'asdp':
            diag_s = create_solve_asdp(Sigma)
        else:  # equi
            diag_s = create_solve_equi(Sigma)
    except Exception as e:
        logger.warning(f"SDP method {method} failed: {e}, falling back to equi")
        diag_s = create_solve_equi(Sigma)

    # Generate knockoffs using knockoff-filter's create_gaussian
    mu = np.mean(X, axis=0)
    Xk = create_gaussian(X, mu, Sigma, method=method, diag_s=diag_s)

    # Compute W statistics with use_sklearn=True for R-compatibility
    W = stat_glmnet_lambdasmax(X, Xk, y.flatten(), use_sklearn=True)

    # Compute threshold
    threshold = knockoff_threshold(W, fdr=fdr, offset=offset)

    # Select variables
    if threshold < np.inf:
        selected = np.where(W >= threshold)[0]
    else:
        selected = np.array([], dtype=int)

    return {
        'W': W,
        'threshold': threshold,
        'selected': selected,
        'knockoffs': Xk,
        'diag_s': diag_s,
        'backend': 'knockoff_filter_sklearn',
        'has_vendored_glmnet': HAS_GLMNET,
        'use_sklearn': True
    }


# =============================================================================
# Backend 3: knockpy
# =============================================================================

def compute_w_stats_knockpy(X, y, fdr=0.1, method='mvr', fstat='lsm', offset=0, seed=42):
    """
    Compute W-statistics using knockpy package.

    Parameters:
        fstat: 'lsm' (LARS path), 'lasso', 'lcd', 'ols'
        method: 'mvr', 'sdp', 'equicorrelated'
    """
    np.random.seed(seed)

    try:
        from knockpy import KnockoffFilter
        from knockpy.knockoffs import GaussianSampler
    except ImportError:
        logger.warning("knockpy not available, skipping")
        return None

    # Map method names
    method_map = {'equi': 'equicorrelated', 'asdp': 'sdp'}
    kp_method = method_map.get(method, method)

    kfilter = KnockoffFilter(
        ksampler='gaussian',
        fstat=fstat,
        knockoff_kwargs={'method': kp_method}
    )

    rejections = kfilter.forward(
        X=X,
        y=y.flatten(),
        fdr=fdr,
        shrinkage=None
    )

    selected = np.where(rejections)[0]

    return {
        'W': kfilter.W,
        'threshold': kfilter.threshold,
        'selected': selected,
        'knockoffs': kfilter.Xk if hasattr(kfilter, 'Xk') else None,
        'backend': f'knockpy_{fstat}'
    }


# =============================================================================
# Custom glmnet_lambdasmax (matches SLIDE implementation)
# =============================================================================

def compute_w_stats_custom_glmnet(X, y, fdr=0.1, method='sdp', offset=0, seed=42):
    """
    Compute W-statistics using the custom _compute_glmnet_lambdasmax from knockoffs.py.

    This is what SLIDE actually uses when running with knockpy backend + glmnet fstat.
    """
    from sklearn.linear_model import lasso_path
    np.random.seed(seed)

    try:
        from knockpy.knockoffs import GaussianSampler
    except ImportError:
        logger.warning("knockpy not available for knockoff generation")
        return None

    n, p = X.shape

    # Map method names
    method_map = {'equi': 'equicorrelated', 'asdp': 'sdp'}
    kp_method = method_map.get(method, method)

    # Create knockoffs using knockpy's sampler
    mu = np.mean(X, axis=0)
    Sigma = np.cov(X, rowvar=False)
    if Sigma.ndim == 0:
        Sigma = np.array([[Sigma]])

    sampler = GaussianSampler(X=X, mu=mu, Sigma=Sigma, method=kp_method)
    Xk = sampler.sample_knockoffs()
    S = sampler.S

    # Custom glmnet_lambdasmax implementation (from knockoffs.py)
    y_flat = y.flatten()
    nlambda, eps = 500, 0.0005

    # Random swap for symmetry
    swap = np.random.binomial(1, 0.5, size=p)
    X_swap = X * (1 - swap) + Xk * swap
    Xk_swap = X * swap + Xk * (1 - swap)

    X_full = np.hstack([X_swap, Xk_swap])

    # Lambda grid
    lambda_max = np.max(np.abs(X_full.T @ y_flat)) / n
    lambda_min = lambda_max * eps
    lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), nlambda)

    # Lasso path
    try:
        _, coef_path, _ = lasso_path(X_full, y_flat, alphas=lambdas, max_iter=10000)
    except Exception:
        _, coef_path, _ = lasso_path(X_full, y_flat, n_alphas=nlambda, max_iter=10000)
        lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), coef_path.shape[1])

    # Entry times
    Z = np.zeros(p)
    Z_k = np.zeros(p)

    for j in range(p):
        nonzero = np.where(np.abs(coef_path[j, :]) > 1e-10)[0]
        Z[j] = lambdas[nonzero[0]] * n if len(nonzero) > 0 else 0

        nonzero_k = np.where(np.abs(coef_path[p + j, :]) > 1e-10)[0]
        Z_k[j] = lambdas[nonzero_k[0]] * n if len(nonzero_k) > 0 else 0

    # W statistics
    W = np.maximum(Z, Z_k) * np.sign(Z - Z_k)
    W = W * (1 - 2 * swap)

    # Threshold - include 0 in candidates to match R's knockoff.threshold
    W_abs = np.abs(W)
    candidates = np.sort(np.concatenate([[0], W_abs]))

    threshold = np.inf
    for t in candidates:
        numerator = offset + np.sum(W <= -t)
        denominator = max(1, np.sum(W >= t))
        if numerator / denominator <= fdr:
            threshold = t
            break

    # Select
    if threshold < np.inf:
        selected = np.where(W >= threshold)[0]
    else:
        selected = np.array([], dtype=int)

    return {
        'W': W,
        'threshold': threshold,
        'selected': selected,
        'knockoffs': Xk,
        'S': S,
        'backend': 'custom_glmnet'
    }


# =============================================================================
# Backend 5: knockpy knockoffs + knockoff-filter's Fortran glmnet
# =============================================================================

def compute_w_stats_knockpy_fortran_glmnet(X, y, fdr=0.1, method='sdp', offset=0, seed=42):
    """
    Compute W-statistics using:
    - knockpy's GaussianSampler for knockoff generation
    - knockoff-filter's stat_glmnet_lambdasmax (uses vendored Fortran glmnet)

    This hybrid should give good R compatibility since:
    - knockpy uses proper SDP/MVR methods
    - knockoff-filter's glmnet is the actual Fortran glmnet (like R)
    """
    np.random.seed(seed)

    try:
        from knockpy.knockoffs import GaussianSampler
    except ImportError:
        logger.warning("knockpy not available for knockoff generation")
        return None

    try:
        from knockoff.stats import stat_glmnet_lambdasmax
        from knockoff.stats.glmnet import HAS_GLMNET
        from knockoff.filter import knockoff_threshold
    except ImportError as e:
        logger.warning(f"knockoff-filter not available: {e}")
        return None

    logger.info(f"knockpy_fortran_glmnet: vendored Fortran glmnet available = {HAS_GLMNET}")

    n, p = X.shape

    # Map method names for knockpy
    method_map = {'equi': 'equicorrelated', 'asdp': 'sdp'}
    kp_method = method_map.get(method, method)

    # Create knockoffs using knockpy's sampler
    mu = np.mean(X, axis=0)
    Sigma = np.cov(X, rowvar=False)
    if Sigma.ndim == 0:
        Sigma = np.array([[Sigma]])

    sampler = GaussianSampler(X=X, mu=mu, Sigma=Sigma, method=kp_method)
    Xk = sampler.sample_knockoffs()
    S = sampler.S

    # Compute W statistics using knockoff-filter's stat_glmnet_lambdasmax
    # This uses the vendored Fortran glmnet when available
    W = stat_glmnet_lambdasmax(X, Xk, y.flatten())

    # Compute threshold using knockoff-filter's implementation
    threshold = knockoff_threshold(W, fdr=fdr, offset=offset)

    # Select variables
    if threshold < np.inf:
        selected = np.where(W >= threshold)[0]
    else:
        selected = np.array([], dtype=int)

    return {
        'W': W,
        'threshold': threshold,
        'selected': selected,
        'knockoffs': Xk,
        'S': S,
        'backend': 'knockpy_fortran_glmnet',
        'has_vendored_glmnet': HAS_GLMNET
    }


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_w_statistics(results: dict, output_dir: Path):
    """Compare W statistics across backends."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to backends that succeeded
    valid = {k: v for k, v in results.items() if v is not None and 'W' in v}

    if len(valid) < 2:
        logger.warning("Need at least 2 backends for comparison")
        return

    backends = list(valid.keys())
    n_backends = len(backends)

    # Get W statistics
    W_dict = {k: v['W'] for k, v in valid.items()}

    # 1. Correlation matrix
    p = len(list(W_dict.values())[0])
    corr_matrix = np.zeros((n_backends, n_backends))

    for i, b1 in enumerate(backends):
        for j, b2 in enumerate(backends):
            corr_matrix[i, j] = np.corrcoef(W_dict[b1], W_dict[b2])[0, 1]

    # 2. Summary statistics
    summary = {}
    for name, res in valid.items():
        W = res['W']
        summary[name] = {
            'n_positive': int(np.sum(W > 0)),
            'n_negative': int(np.sum(W < 0)),
            'n_zero': int(np.sum(W == 0)),
            'min': float(np.min(W)),
            'max': float(np.max(W)),
            'mean': float(np.mean(W)),
            'std': float(np.std(W)),
            'threshold': float(res['threshold']) if res['threshold'] < np.inf else 'inf',
            'n_selected': len(res['selected']),
            'selected': res['selected'].tolist() if len(res['selected']) > 0 else []
        }

    # 3. Pairwise differences
    pairwise = {}
    for i, b1 in enumerate(backends):
        for j, b2 in enumerate(backends):
            if i < j:
                diff = W_dict[b1] - W_dict[b2]
                key = f"{b1}_vs_{b2}"
                pairwise[key] = {
                    'correlation': float(corr_matrix[i, j]),
                    'max_abs_diff': float(np.max(np.abs(diff))),
                    'mean_abs_diff': float(np.mean(np.abs(diff))),
                    'rmse': float(np.sqrt(np.mean(diff ** 2)))
                }

    # Save results
    results_dict = {
        'summary': summary,
        'correlation_matrix': {
            'backends': backends,
            'values': corr_matrix.tolist()
        },
        'pairwise_differences': pairwise
    }

    with open(output_dir / 'w_statistics_comparison.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Save raw W statistics
    W_df = pd.DataFrame(W_dict)
    W_df.to_csv(output_dir / 'w_statistics_all.csv', index=True)

    return results_dict


def plot_w_statistics(results: dict, output_dir: Path):
    """Create visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    valid = {k: v for k, v in results.items() if v is not None and 'W' in v}
    backends = list(valid.keys())
    n = len(backends)

    if n < 2:
        return

    # Figure 1: W-statistic histograms
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, valid.items()):
        W = res['W']
        ax.hist(W, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='W=0')
        if res['threshold'] < np.inf:
            ax.axvline(x=res['threshold'], color='green', linestyle='--',
                      linewidth=2, label=f"τ={res['threshold']:.2f}")
        ax.set_title(f"{name}\n(n_sel={len(res['selected'])})")
        ax.set_xlabel('W statistic')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'w_histograms.png', dpi=150)
    plt.close()

    # Figure 2: Pairwise scatter plots
    if n >= 2:
        fig, axes = plt.subplots(n-1, n-1, figsize=(4 * (n-1), 4 * (n-1)), squeeze=False)

        for i in range(n - 1):
            for j in range(i + 1, n):
                ax = axes[i, j-1]
                W1 = valid[backends[i]]['W']
                W2 = valid[backends[j]]['W']

                ax.scatter(W1, W2, alpha=0.5, s=10)

                # Add y=x line
                lims = [min(W1.min(), W2.min()), max(W1.max(), W2.max())]
                ax.plot(lims, lims, 'r--', linewidth=1, label='y=x')

                corr = np.corrcoef(W1, W2)[0, 1]
                ax.set_title(f"r={corr:.4f}")
                ax.set_xlabel(backends[i])
                ax.set_ylabel(backends[j])

        # Hide unused subplots
        for i in range(n - 1):
            for j in range(i):
                axes[i, j].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / 'w_pairwise_scatter.png', dpi=150)
        plt.close()

    # Figure 3: Selection agreement heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    agreement_matrix = np.zeros((n, n))
    for i, b1 in enumerate(backends):
        for j, b2 in enumerate(backends):
            s1 = set(valid[b1]['selected'])
            s2 = set(valid[b2]['selected'])
            union = len(s1 | s2)
            if union > 0:
                agreement_matrix[i, j] = len(s1 & s2) / union
            else:
                agreement_matrix[i, j] = 1.0

    im = ax.imshow(agreement_matrix, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(backends, rotation=45, ha='right')
    ax.set_yticklabels(backends)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{agreement_matrix[i, j]:.2f}", ha='center', va='center')

    ax.set_title('Selection Agreement (Jaccard Index)')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / 'selection_agreement.png', dpi=150)
    plt.close()

    logger.info(f"Plots saved to {output_dir}")


def print_summary(results: dict, comparison: dict):
    """Print summary to console."""
    print("\n" + "=" * 80)
    print("W-STATISTIC COMPARISON: knockpy vs knockoff-filter vs R")
    print("=" * 80)

    # Per-backend summary
    print("\n--- Per-Backend Summary ---")
    for name, stats in comparison['summary'].items():
        print(f"\n{name}:")
        print(f"  W range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Positive/Negative/Zero: {stats['n_positive']}/{stats['n_negative']}/{stats['n_zero']}")
        print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Threshold: {stats['threshold']}")
        print(f"  Selected: {stats['n_selected']} variables")
        if stats['n_selected'] > 0 and stats['n_selected'] <= 20:
            print(f"    Indices: {stats['selected']}")

    # Correlation matrix
    print("\n--- W-Statistic Correlation Matrix ---")
    backends = comparison['correlation_matrix']['backends']
    corr = np.array(comparison['correlation_matrix']['values'])

    header = "            " + "  ".join(f"{b:>12}" for b in backends)
    print(header)
    for i, b in enumerate(backends):
        row = f"{b:>12}" + "  ".join(f"{corr[i, j]:>12.4f}" for j in range(len(backends)))
        print(row)

    # Pairwise differences
    print("\n--- Pairwise Differences ---")
    for key, diff in comparison['pairwise_differences'].items():
        print(f"\n{key}:")
        print(f"  Correlation: {diff['correlation']:.4f}")
        print(f"  Max |diff|: {diff['max_abs_diff']:.4f}")
        print(f"  Mean |diff|: {diff['mean_abs_diff']:.4f}")
        print(f"  RMSE: {diff['rmse']:.4f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare W-statistics across knockoff backends")
    parser.add_argument('--z-path', required=True, help='Path to Z matrix CSV (samples x features)')
    parser.add_argument('--y-path', required=True, help='Path to Y vector CSV')
    parser.add_argument('--output-dir', default='./w_stat_comparison', help='Output directory')
    parser.add_argument('--fdr', type=float, default=0.1, help='FDR threshold')
    parser.add_argument('--method', default='sdp', choices=['sdp', 'asdp', 'equi', 'mvr'],
                        help='Knockoff construction method')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--standardize', action='store_true', help='Standardize X before knockoffs')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {args.z_path}")
    Z_df = pd.read_csv(args.z_path, index_col=0)
    X = Z_df.values.astype(np.float64)

    logger.info(f"Loading response from {args.y_path}")
    y_df = pd.read_csv(args.y_path, index_col=0)
    y = y_df.values.flatten().astype(np.float64)

    logger.info(f"Data: X shape={X.shape}, y shape={y.shape}")

    # Standardize if requested
    # Use R-compatible standardization (ddof=1) to match R's scale() function
    if args.standardize:
        logger.info("Standardizing features using R-compatible scale (ddof=1)...")
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0, ddof=1)  # R uses ddof=1 (sample std)
        stds = np.where(stds == 0, 1.0, stds)  # Handle constant columns
        X = (X - means) / stds

    # Run all backends
    results = {}

    # 1. R native
    logger.info("\n=== Backend 1: R native ===")
    results['R_native'] = compute_w_stats_r(X, y, fdr=args.fdr, method=args.method, seed=args.seed)

    # 2. knockoff-filter (Fortran glmnet)
    logger.info("\n=== Backend 2: knockoff-filter (Fortran glmnet) ===")
    results['knockoff_filter'] = compute_w_stats_knockoff_filter(
        X, y, fdr=args.fdr, method=args.method, offset=0, seed=args.seed
    )

    # 2b. knockoff-filter with use_sklearn=True (R-compatible)
    logger.info("\n=== Backend 2b: knockoff-filter (sklearn, R-compatible) ===")
    results['knockoff_filter_sklearn'] = compute_w_stats_knockoff_filter_sklearn(
        X, y, fdr=args.fdr, method=args.method, offset=0, seed=args.seed
    )

    # 3. knockpy with lsm (LARS path) - uses sdp if available
    logger.info("\n=== Backend 3: knockpy (lsm) ===")
    results['knockpy_lsm'] = compute_w_stats_knockpy(
        X, y, fdr=args.fdr, method=args.method,  # knockpy supports sdp
        fstat='lsm', offset=0, seed=args.seed
    )

    # 4. knockpy with lasso
    logger.info("\n=== Backend 4: knockpy (lasso) ===")
    results['knockpy_lasso'] = compute_w_stats_knockpy(
        X, y, fdr=args.fdr, method=args.method,  # knockpy supports sdp
        fstat='lasso', offset=0, seed=args.seed
    )

    # 5. Custom glmnet (what SLIDE's _compute_glmnet_lambdasmax uses)
    logger.info("\n=== Backend 5: custom glmnet (SLIDE implementation) ===")
    results['custom_glmnet'] = compute_w_stats_custom_glmnet(
        X, y, fdr=args.fdr, method=args.method,  # knockpy sampler supports sdp
        offset=0, seed=args.seed
    )

    # 6. knockpy knockoffs + knockoff-filter's Fortran glmnet
    logger.info("\n=== Backend 6: knockpy + Fortran glmnet ===")
    results['knockpy_fortran_glmnet'] = compute_w_stats_knockpy_fortran_glmnet(
        X, y, fdr=args.fdr, method=args.method,
        offset=0, seed=args.seed
    )

    # Compare
    logger.info("\n=== Comparing W-statistics ===")
    comparison = compare_w_statistics(results, output_dir)

    # Plot
    logger.info("\n=== Creating visualizations ===")
    plot_w_statistics(results, output_dir)

    # Print summary
    if comparison:
        print_summary(results, comparison)

    logger.info(f"\nResults saved to {output_dir}")
    return results, comparison


if __name__ == '__main__':
    main()
