#!/usr/bin/env python3
"""
Comprehensive Knockoff Implementation Comparison (v2 - Optimized)

Two-phase approach:
  Phase 1: Pre-compute LOVE (Z matrices) for all parameter sets
  Phase 2: Run knockoffs on pre-computed Z matrices

This avoids redundant LOVE computation across backends.

Usage:
    # Phase 1: Pre-compute LOVE
    python run_knockoff_comparison_v2.py --config config.yaml --output-dir out/ --phase love --love-backend python
    python run_knockoff_comparison_v2.py --config config.yaml --output-dir out/ --phase love --love-backend r

    # Phase 2: Run knockoffs (after Phase 1 completes)
    python run_knockoff_comparison_v2.py --config config.yaml --output-dir out/ --phase knockoff --backend R_native
    python run_knockoff_comparison_v2.py --config config.yaml --output-dir out/ --phase knockoff --backend knockoff_filter

    # Phase 3: Aggregate results
    python run_knockoff_comparison_v2.py --config config.yaml --output-dir out/ --phase aggregate
"""

import argparse
import json
import logging
import sys
import warnings
import itertools
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Add knockoff-filter to path
KNOCKOFF_FILTER_PATH = "/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter"
if KNOCKOFF_FILTER_PATH not in sys.path:
    sys.path.insert(0, KNOCKOFF_FILTER_PATH)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Backend to LOVE mapping
BACKEND_LOVE_MAP = {
    'R_native': 'r',
    'knockoff_filter': 'python',
    'knockoff_filter_sklearn': 'python',
    'knockpy_lsm': 'python',
    'knockpy_lasso': 'python',
    'custom_glmnet': 'python',
}

ALL_BACKENDS = list(BACKEND_LOVE_MAP.keys())


# =============================================================================
# Data Loading
# =============================================================================

def load_data(x_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load X and Y data from CSV files."""
    logger.info(f"Loading X from {x_path}")
    X_df = pd.read_csv(x_path, index_col=0)
    X = X_df.values.astype(np.float64)

    logger.info(f"Loading Y from {y_path}")
    Y_df = pd.read_csv(y_path, index_col=0)

    if Y_df.shape[1] == 1:
        y = Y_df.iloc[:, 0].values
    else:
        for col in ['MRSS', 'y', 'Y', 'response', 'outcome']:
            if col in Y_df.columns:
                y = Y_df[col].values
                break
        else:
            y = Y_df.iloc[:, 0].values

    y = y.astype(np.float64)
    logger.info(f"Data loaded: X shape={X.shape}, y shape={y.shape}")
    return X, y, X_df


# =============================================================================
# Phase 1: LOVE Pre-computation
# =============================================================================

def run_love_r(X: np.ndarray, delta: float, lam: float) -> np.ndarray:
    """Run R LOVE implementation.

    Parameters match Python call_love convention:
    - delta maps to R LOVE's 'mu' (thresholding loading matrix)
    - lam maps to R LOVE's 'lbd' (precision estimation)
    """
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.packages import importr

    numpy2ri.activate()
    pandas2ri.activate()

    love_r = importr('LOVE')
    X_r = robjects.r['as.matrix'](X)
    result = love_r.LOVE(X=X_r, mu=delta, lbd=lam)
    C = np.array(result.rx2('C'))

    numpy2ri.deactivate()
    pandas2ri.deactivate()

    return C


def run_love_python(X: np.ndarray, delta: float, lam: float) -> np.ndarray:
    """Run Python LOVE implementation."""
    from loveslide.love import call_love
    result = call_love(X, lbd=lam, mu=delta)
    return result['C']


def precompute_love(config_path: str, output_dir: str, love_backend: str, quick: bool = False):
    """Pre-compute LOVE Z matrices for all parameter sets."""
    logger.info(f"Phase 1: Pre-computing LOVE (backend={love_backend})")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(output_dir)
    z_dir = output_dir / "z_matrices" / f"{love_backend}_love"
    z_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, _ = load_data(config['x_path'], config['y_path'])

    # Get parameter grid
    deltas = config.get('delta', [0.1])
    lambdas = config.get('lambda', [0.5])

    if isinstance(deltas, (int, float)):
        deltas = [deltas]
    if isinstance(lambdas, (int, float)):
        lambdas = [lambdas]

    if quick:
        deltas = deltas[:1]
        lambdas = lambdas[:1]

    logger.info(f"Computing Z matrices for {len(deltas)} x {len(lambdas)} = {len(deltas)*len(lambdas)} parameter sets")

    # Save y vector
    np.save(z_dir / "y.npy", y)

    # Compute Z for each parameter set
    for delta, lam in itertools.product(deltas, lambdas):
        param_str = f"d{delta}_l{lam}"
        z_file = z_dir / f"Z_{param_str}.npy"

        if z_file.exists():
            logger.info(f"  {param_str}: Already exists, skipping")
            continue

        logger.info(f"  {param_str}: Computing LOVE...")
        start = datetime.now()

        try:
            if love_backend == 'r':
                Z = run_love_r(X, delta, lam)
            else:
                Z = run_love_python(X, delta, lam)

            np.save(z_file, Z)
            elapsed = (datetime.now() - start).total_seconds()
            logger.info(f"  {param_str}: Done in {elapsed:.1f}s, Z shape={Z.shape}")

        except Exception as e:
            logger.error(f"  {param_str}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    # Save metadata
    metadata = {
        'love_backend': love_backend,
        'timestamp': datetime.now().isoformat(),
        'deltas': deltas,
        'lambdas': lambdas,
        'x_shape': list(X.shape),
        'config_path': str(config_path)
    }
    with open(z_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"LOVE pre-computation complete. Z matrices saved to {z_dir}")


# =============================================================================
# Phase 2: Knockoff Computation
# =============================================================================

def compute_knockoffs_r(Z: np.ndarray, y: np.ndarray, fdr: float = 0.1, seed: int = 42) -> Optional[Dict]:
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

        selected = np.array(result.rx2('selected')) - 1
        W = np.array(result.rx2('statistic'))
        threshold = float(np.array(result.rx2('threshold'))[0])

        numpy2ri.deactivate()
        pandas2ri.deactivate()

        return {'W': W, 'threshold': threshold, 'selected': selected, 'backend': 'R_native'}

    except Exception as e:
        logger.error(f"R knockoff failed: {e}")
        return None


def compute_knockoffs_knockoff_filter(Z: np.ndarray, y: np.ndarray, fdr: float = 0.1,
                                      method: str = 'sdp', use_sklearn: bool = False,
                                      seed: int = 42) -> Optional[Dict]:
    """knockoff-filter Python implementation."""
    np.random.seed(seed)

    try:
        from knockoff.stats import stat_glmnet_lambdasmax
        from knockoff.solve import create_solve_sdp, create_solve_equi
        from knockoff.create import create_gaussian
        from knockoff.filter import knockoff_threshold
        from knockoff.utils import is_posdef
    except ImportError as e:
        logger.error(f"knockoff-filter not available: {e}")
        return None

    n, p = Z.shape
    Sigma = np.cov(Z, rowvar=False)
    if Sigma.ndim == 0:
        Sigma = np.array([[Sigma]])

    if not is_posdef(Sigma):
        min_eig = np.min(np.linalg.eigvalsh(Sigma))
        if min_eig < 1e-10:
            Sigma = Sigma + (1e-10 - min_eig) * np.eye(p)

    try:
        diag_s = create_solve_sdp(Sigma)
    except Exception:
        diag_s = create_solve_equi(Sigma)

    mu = np.mean(Z, axis=0)
    Zk = create_gaussian(Z, mu, Sigma, method=method, diag_s=diag_s)
    W = stat_glmnet_lambdasmax(Z, Zk, y.flatten(), use_sklearn=use_sklearn)
    threshold = knockoff_threshold(W, fdr=fdr, offset=0)

    if threshold < np.inf:
        selected = np.where(W >= threshold)[0]
    else:
        selected = np.array([], dtype=int)

    backend_name = 'knockoff_filter_sklearn' if use_sklearn else 'knockoff_filter'
    return {'W': W, 'threshold': threshold, 'selected': selected, 'backend': backend_name}


def compute_knockoffs_knockpy(Z: np.ndarray, y: np.ndarray, fdr: float = 0.1,
                              method: str = 'sdp', fstat: str = 'lsm', seed: int = 42) -> Optional[Dict]:
    """knockpy implementation."""
    np.random.seed(seed)

    try:
        from knockpy import KnockoffFilter
    except ImportError:
        logger.error("knockpy not available")
        return None

    method_map = {'equi': 'equicorrelated', 'asdp': 'sdp'}
    kp_method = method_map.get(method, method)

    kfilter = KnockoffFilter(
        ksampler='gaussian',
        fstat=fstat,
        knockoff_kwargs={'method': kp_method}
    )

    rejections = kfilter.forward(X=Z, y=y.flatten(), fdr=fdr, shrinkage=None)
    selected = np.where(rejections)[0]

    return {'W': kfilter.W, 'threshold': kfilter.threshold, 'selected': selected, 'backend': f'knockpy_{fstat}'}


def compute_knockoffs_custom_glmnet(Z: np.ndarray, y: np.ndarray, fdr: float = 0.1,
                                    method: str = 'sdp', seed: int = 42) -> Optional[Dict]:
    """SLIDE's custom glmnet implementation."""
    from sklearn.linear_model import lasso_path
    np.random.seed(seed)

    try:
        from knockpy.knockoffs import GaussianSampler
    except ImportError:
        logger.error("knockpy not available")
        return None

    n, p = Z.shape
    method_map = {'equi': 'equicorrelated', 'asdp': 'sdp'}
    kp_method = method_map.get(method, method)

    mu = np.mean(Z, axis=0)
    Sigma = np.cov(Z, rowvar=False)
    if Sigma.ndim == 0:
        Sigma = np.array([[Sigma]])

    sampler = GaussianSampler(X=Z, mu=mu, Sigma=Sigma, method=kp_method)
    Zk = sampler.sample_knockoffs()

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


def run_knockoff_backend(config_path: str, output_dir: str, backend: str,
                         quick: bool = False, seed: int = 42):
    """Run knockoff computation for a single backend using pre-computed Z matrices."""
    logger.info(f"Phase 2: Running knockoffs for backend={backend}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(output_dir)
    love_backend = BACKEND_LOVE_MAP[backend]
    z_dir = output_dir / "z_matrices" / f"{love_backend}_love"

    if not z_dir.exists():
        logger.error(f"Z matrices not found at {z_dir}. Run Phase 1 first!")
        return

    # Load metadata
    with open(z_dir / "metadata.json") as f:
        metadata = json.load(f)

    deltas = metadata['deltas']
    lambdas = metadata['lambdas']
    fdr = config.get('fdr', 0.1)

    if quick:
        deltas = deltas[:1]
        lambdas = lambdas[:1]

    # Load y
    y = np.load(z_dir / "y.npy")

    logger.info(f"Running {backend} on {len(deltas)*len(lambdas)} parameter sets")

    results = []

    for delta, lam in itertools.product(deltas, lambdas):
        param_str = f"d{delta}_l{lam}"
        z_file = z_dir / f"Z_{param_str}.npy"

        if not z_file.exists():
            logger.warning(f"  {param_str}: Z matrix not found, skipping")
            continue

        logger.info(f"  {param_str}: Loading Z and running knockoffs...")
        Z = np.load(z_file)

        try:
            if backend == 'R_native':
                result = compute_knockoffs_r(Z, y, fdr=fdr, seed=seed)
            elif backend == 'knockoff_filter':
                result = compute_knockoffs_knockoff_filter(Z, y, fdr=fdr, use_sklearn=False, seed=seed)
            elif backend == 'knockoff_filter_sklearn':
                result = compute_knockoffs_knockoff_filter(Z, y, fdr=fdr, use_sklearn=True, seed=seed)
            elif backend == 'knockpy_lsm':
                result = compute_knockoffs_knockpy(Z, y, fdr=fdr, fstat='lsm', seed=seed)
            elif backend == 'knockpy_lasso':
                result = compute_knockoffs_knockpy(Z, y, fdr=fdr, fstat='lasso', seed=seed)
            elif backend == 'custom_glmnet':
                result = compute_knockoffs_custom_glmnet(Z, y, fdr=fdr, seed=seed)
            else:
                raise ValueError(f"Unknown backend: {backend}")

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
                    'result': result_clean
                })
                logger.info(f"    Selected: {len(result['selected'])} variables")

        except Exception as e:
            logger.error(f"  {param_str}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    # Save results
    backend_dir = output_dir / backend
    backend_dir.mkdir(parents=True, exist_ok=True)

    output_file = backend_dir / f"backend_{backend}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'backend': backend,
            'love_backend': love_backend,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)

    # Mark complete
    (output_dir / f".{backend}_complete").touch()

    logger.info(f"Results saved to {output_file}")


# =============================================================================
# Phase 3: Aggregation
# =============================================================================

def aggregate_results(output_dir: str):
    """Aggregate results from all backends."""
    output_dir = Path(output_dir)
    logger.info(f"Phase 3: Aggregating results from {output_dir}")

    # Find all backend results
    all_results = {}

    for backend in ALL_BACKENDS:
        result_file = output_dir / backend / f"backend_{backend}.json"
        if result_file.exists():
            with open(result_file) as f:
                all_results[backend] = json.load(f)
            logger.info(f"  Loaded {backend}")
        else:
            logger.warning(f"  {backend}: Not found")

    if not all_results:
        logger.error("No results found!")
        return

    # Reorganize by parameter set
    param_comparison = {}

    for backend, data in all_results.items():
        for r in data['results']:
            param_key = f"d{r['params']['delta']}_l{r['params']['lambda']}"

            if param_key not in param_comparison:
                param_comparison[param_key] = {
                    'params': r['params'],
                    'backends': {}
                }

            param_comparison[param_key]['backends'][backend] = r['result']

    # Compute comparison metrics
    comparison_results = []

    for param_key, pc in param_comparison.items():
        logger.info(f"  Computing metrics for {param_key}")

        backends = list(pc['backends'].keys())
        W_dict = {}

        for b in backends:
            W_dict[b] = np.array(pc['backends'][b]['W'])

        # Correlation matrix
        n_backends = len(backends)
        corr_matrix = np.zeros((n_backends, n_backends))

        with np.errstate(divide='ignore', invalid='ignore'):
            for i, b1 in enumerate(backends):
                for j, b2 in enumerate(backends):
                    corr_matrix[i, j] = np.corrcoef(W_dict[b1], W_dict[b2])[0, 1]

        # R correlation
        r_corr = {}
        if 'R_native' in backends:
            r_idx = backends.index('R_native')
            for i, b in enumerate(backends):
                if b != 'R_native':
                    c = corr_matrix[r_idx, i]
                    r_corr[b] = float(c) if not np.isnan(c) else None

        # Selection agreement with R
        r_agreement = {}
        if 'R_native' in pc['backends']:
            r_selected = set(pc['backends']['R_native']['selected'])
            for b in backends:
                if b == 'R_native':
                    continue
                py_selected = set(pc['backends'][b]['selected'])
                union = len(r_selected | py_selected)
                intersection = len(r_selected & py_selected)
                jaccard = intersection / union if union > 0 else 1.0
                r_agreement[b] = {
                    'jaccard': jaccard,
                    'exact_match': r_selected == py_selected,
                    'r_selected': len(r_selected),
                    'py_selected': len(py_selected)
                }

        comparison_results.append({
            'params': pc['params'],
            'backends': list(pc['backends'].keys()),
            'r_correlation': r_corr,
            'r_agreement': r_agreement,
            'correlation_matrix': {
                'backends': backends,
                'values': corr_matrix.tolist()
            }
        })

    # Aggregate across params
    avg_r_corr = {}
    avg_jaccard = {}
    exact_match_count = {}

    for cr in comparison_results:
        for b, corr in cr['r_correlation'].items():
            if b not in avg_r_corr:
                avg_r_corr[b] = []
            if corr is not None:
                avg_r_corr[b].append(corr)

        for b, agreement in cr['r_agreement'].items():
            if b not in avg_jaccard:
                avg_jaccard[b] = []
                exact_match_count[b] = {'matches': 0, 'total': 0}
            avg_jaccard[b].append(agreement['jaccard'])
            exact_match_count[b]['total'] += 1
            if agreement['exact_match']:
                exact_match_count[b]['matches'] += 1

    aggregated = {
        'avg_r_correlation': {k: np.mean(v) if v else None for k, v in avg_r_corr.items()},
        'avg_jaccard': {k: np.mean(v) if v else None for k, v in avg_jaccard.items()},
        'exact_match_rate': {k: v['matches']/v['total'] if v['total'] > 0 else None
                            for k, v in exact_match_count.items()}
    }

    # Save
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'backends_found': list(all_results.keys()),
        'n_parameter_sets': len(comparison_results),
        'comparison_results': comparison_results,
        'aggregated': aggregated
    }

    with open(output_dir / 'full_comparison.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)

    # Print report
    print("\n" + "="*70)
    print("KNOCKOFF COMPARISON RESULTS")
    print("="*70)
    print(f"\nBackends: {', '.join(all_results.keys())}")
    print(f"Parameter sets: {len(comparison_results)}")

    print("\nAverage Correlation with R:")
    for b, corr in aggregated['avg_r_correlation'].items():
        print(f"  {b:30s}: {corr:.4f}" if corr else f"  {b:30s}: N/A")

    print("\nAverage Jaccard Agreement with R:")
    for b, j in aggregated['avg_jaccard'].items():
        print(f"  {b:30s}: {j:.4f}" if j else f"  {b:30s}: N/A")

    print("\nExact Match Rate with R:")
    for b, rate in aggregated['exact_match_rate'].items():
        print(f"  {b:30s}: {rate:.1%}" if rate else f"  {b:30s}: N/A")

    print("\n" + "="*70)

    logger.info(f"Full results saved to {output_dir / 'full_comparison.json'}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Knockoff Comparison (v2 - Optimized)')
    parser.add_argument('--config', '-c', required=True, help='YAML config file')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory')
    parser.add_argument('--phase', required=True, choices=['love', 'knockoff', 'aggregate'],
                        help='Phase to run: love, knockoff, or aggregate')
    parser.add_argument('--love-backend', choices=['r', 'python'],
                        help='LOVE backend (for phase=love)')
    parser.add_argument('--backend', '-b', choices=ALL_BACKENDS,
                        help='Knockoff backend (for phase=knockoff)')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    if args.phase == 'love':
        if not args.love_backend:
            parser.error("--love-backend required for phase=love")
        precompute_love(args.config, args.output_dir, args.love_backend, args.quick)

    elif args.phase == 'knockoff':
        if not args.backend:
            parser.error("--backend required for phase=knockoff")
        run_knockoff_backend(args.config, args.output_dir, args.backend, args.quick, args.seed)

    elif args.phase == 'aggregate':
        aggregate_results(args.output_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())
