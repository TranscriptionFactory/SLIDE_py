#!/usr/bin/env python3
"""
Comprehensive Knockoff Implementation Comparison

End-to-end comparison of knockoff filtering across R and Python implementations.
Tests knockoff generation, W-statistic computation, threshold selection, and
variable selection across multiple parameter sets.

Backends tested:
    1. R_native: R's knockoff package (gold standard)
    2. knockoff_filter: Python knockoff-filter with Fortran glmnet
    3. knockoff_filter_sklearn: knockoff-filter with sklearn fallback
    4. knockpy: knockpy package with various fstat options
    5. custom_glmnet: SLIDE's custom sklearn lasso_path implementation

Usage:
    # Interactive (single process)
    python run_knockoff_comparison.py --config comparison_config_binary.yaml

    # SLURM submission
    sbatch run_knockoff_comparison.sh comparison_config_binary.yaml

    # Quick test (single parameter set)
    python run_knockoff_comparison.py --config comparison_config_binary.yaml --quick

Example SLURM submission:
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=knockoff_cmp
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/knockoff_cmp_%j.out

module load gcc/12.2.0
module load python/ondemand-jupyter-python3.11
module load r/4.4.0

export PYTHONPATH="/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter:$PYTHONPATH"

/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python \
    run_knockoff_comparison.py \
    --config comparison_config_binary.yaml \
    --output-dir output_comparison/knockoff_test_$(date +%Y%m%d_%H%M%S)
EOF



sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=knockoff_cmp
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-5
#SBATCH --output=logs/knockoff_cmp_%A_%a.out
#SBATCH --error=logs/knockoff_cmp_%A_%a.err

cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
mkdir -p logs

module load gcc/12.2.0
module load python/ondemand-jupyter-python3.11
module load r/4.4.0

export PYTHONPATH="/ix/djishnu/Aaron/1_general_use/knockoff-filter/k
nockoff-filter:$PYTHONPATH"

BACKENDS=("R_native" "knockoff_filter" "knockoff_filter_sklearn"
"knockpy_lsm" "knockpy_lasso" "custom_glmnet")
BACKEND="${BACKENDS[$SLURM_ARRAY_TASK_ID]}"

# Shared parent dir + backend subfolder
OUTPUT_BASE="output_comparison/knockoff_cmp_${SLURM_ARRAY_JOB_ID}"
OUTPUT_DIR="${OUTPUT_BASE}/${BACKEND}"

mkdir -p "$OUTPUT_DIR"

/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python \
    run_knockoff_comparison.py \
    --config comparison_config_binary.yaml \
    --output-dir "$OUTPUT_DIR" \
    --backend "$BACKEND"

touch "${OUTPUT_BASE}/.task${SLURM_ARRAY_TASK_ID}_complete"
EOF

  After all tasks complete, aggregate results:
  # Check job status first
  squeue -u $USER

  # Once all complete, run aggregation (replace JOB_ID with your array
   job ID)
  JOB_ID=<your_job_id>
  module load gcc/12.2.0 python/ondemand-jupyter-python3.11 r/4.4.0
  export PYTHONPATH="/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter:$PYTHONPATH"
  cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
  /ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python \
      run_knockoff_comparison.py \
      --config comparison_config_binary.yaml \
      --output-dir output_comparison/knockoff_cmp_${JOB_ID} \
      --aggregate


"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import itertools

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

    # Handle Y - get the response column
    if Y_df.shape[1] == 1:
        y = Y_df.iloc[:, 0].values
    else:
        # Try common column names
        for col in ['MRSS', 'y', 'Y', 'response', 'outcome']:
            if col in Y_df.columns:
                y = Y_df[col].values
                break
        else:
            y = Y_df.iloc[:, 0].values

    y = y.astype(np.float64)

    logger.info(f"Data loaded: X shape={X.shape}, y shape={y.shape}")
    return X, y, X_df


def run_love_to_get_z(X: np.ndarray, y: np.ndarray, config: dict,
                      love_backend: str = 'python') -> Tuple[np.ndarray, np.ndarray]:
    """
    Run LOVE to get latent factor matrix Z.

    Returns:
        Z: Latent factor matrix (samples x latent factors)
        y: Response vector (potentially transformed)
    """
    delta = config.get('delta', 0.1)
    lam = config.get('lambda', 0.5)

    logger.info(f"Running LOVE (backend={love_backend}, delta={delta}, lambda={lam})")

    if love_backend == 'r':
        # Use R LOVE
        try:
            import rpy2.robjects as robjects
            from rpy2.robjects import numpy2ri, pandas2ri
            from rpy2.robjects.packages import importr

            numpy2ri.activate()
            pandas2ri.activate()

            love_r = importr('LOVE')

            X_r = robjects.r['as.matrix'](X)
            result = love_r.CovEst(X=X_r, delta=delta, lam=lam)

            # Extract Z matrix
            Z = np.array(result.rx2('Z'))

            numpy2ri.deactivate()
            pandas2ri.deactivate()

            logger.info(f"R LOVE completed: Z shape={Z.shape}")
            return Z, y

        except Exception as e:
            logger.warning(f"R LOVE failed: {e}, falling back to Python")
            love_backend = 'python'

    if love_backend == 'python':
        from loveslide.love import call_love

        result = call_love(X, lbd=lam, mu=delta)
        Z = result['Z']

        logger.info(f"Python LOVE completed: Z shape={Z.shape}")
        return Z, y

    raise ValueError(f"Unknown LOVE backend: {love_backend}")


# =============================================================================
# Knockoff Backends
# =============================================================================

def compute_knockoffs_r(Z: np.ndarray, y: np.ndarray, fdr: float = 0.1,
                        seed: int = 42) -> Optional[Dict]:
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

        selected = np.array(result.rx2('selected')) - 1  # 0-based
        W = np.array(result.rx2('statistic'))
        threshold = float(np.array(result.rx2('threshold'))[0])

        try:
            Xk = np.array(result.rx2('Xk'))
        except Exception:
            Xk = None

        numpy2ri.deactivate()
        pandas2ri.deactivate()

        return {
            'W': W,
            'threshold': threshold,
            'selected': selected,
            'knockoffs': Xk,
            'backend': 'R_native'
        }

    except Exception as e:
        logger.warning(f"R knockoff failed: {e}")
        return None


def compute_knockoffs_knockoff_filter(Z: np.ndarray, y: np.ndarray, fdr: float = 0.1,
                                      method: str = 'sdp', use_sklearn: bool = False,
                                      seed: int = 42) -> Optional[Dict]:
    """knockoff-filter Python implementation."""
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

    n, p = Z.shape

    # Compute covariance
    Sigma = np.cov(Z, rowvar=False)
    if Sigma.ndim == 0:
        Sigma = np.array([[Sigma]])

    # Ensure positive definite
    if not is_posdef(Sigma):
        min_eig = np.min(np.linalg.eigvalsh(Sigma))
        if min_eig < 1e-10:
            Sigma = Sigma + (1e-10 - min_eig) * np.eye(p)

    # Compute S matrix
    try:
        if method == 'sdp':
            diag_s = create_solve_sdp(Sigma)
        elif method == 'asdp':
            diag_s = create_solve_asdp(Sigma)
        else:
            diag_s = create_solve_equi(Sigma)
    except Exception as e:
        logger.warning(f"SDP method {method} failed: {e}, using equi")
        diag_s = create_solve_equi(Sigma)

    # Generate knockoffs
    mu = np.mean(Z, axis=0)
    Zk = create_gaussian(Z, mu, Sigma, method=method, diag_s=diag_s)

    # Compute W statistics
    W = stat_glmnet_lambdasmax(Z, Zk, y.flatten(), use_sklearn=use_sklearn)

    # Compute threshold
    threshold = knockoff_threshold(W, fdr=fdr, offset=0)

    # Select variables
    if threshold < np.inf:
        selected = np.where(W >= threshold)[0]
    else:
        selected = np.array([], dtype=int)

    backend_name = 'knockoff_filter_sklearn' if use_sklearn else 'knockoff_filter'

    return {
        'W': W,
        'threshold': threshold,
        'selected': selected,
        'knockoffs': Zk,
        'backend': backend_name,
        'has_glmnet': HAS_GLMNET
    }


def compute_knockoffs_knockpy(Z: np.ndarray, y: np.ndarray, fdr: float = 0.1,
                              method: str = 'sdp', fstat: str = 'lsm',
                              seed: int = 42) -> Optional[Dict]:
    """knockpy implementation."""
    np.random.seed(seed)

    try:
        from knockpy import KnockoffFilter
    except ImportError:
        logger.warning("knockpy not available")
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
        X=Z, y=y.flatten(),
        fdr=fdr, shrinkage=None
    )

    selected = np.where(rejections)[0]

    return {
        'W': kfilter.W,
        'threshold': kfilter.threshold,
        'selected': selected,
        'knockoffs': getattr(kfilter, 'Xk', None),
        'backend': f'knockpy_{fstat}'
    }


def compute_knockoffs_custom_glmnet(Z: np.ndarray, y: np.ndarray, fdr: float = 0.1,
                                    method: str = 'sdp', seed: int = 42) -> Optional[Dict]:
    """SLIDE's custom glmnet implementation (knockpy sampler + sklearn lasso_path)."""
    from sklearn.linear_model import lasso_path
    np.random.seed(seed)

    try:
        from knockpy.knockoffs import GaussianSampler
    except ImportError:
        logger.warning("knockpy not available for knockoff generation")
        return None

    n, p = Z.shape

    # Map method names
    method_map = {'equi': 'equicorrelated', 'asdp': 'sdp'}
    kp_method = method_map.get(method, method)

    # Create knockoffs
    mu = np.mean(Z, axis=0)
    Sigma = np.cov(Z, rowvar=False)
    if Sigma.ndim == 0:
        Sigma = np.array([[Sigma]])

    sampler = GaussianSampler(X=Z, mu=mu, Sigma=Sigma, method=kp_method)
    Zk = sampler.sample_knockoffs()

    # Custom glmnet implementation
    y_flat = y.flatten()
    nlambda, eps = 500, 0.0005

    # Random swap
    swap = np.random.binomial(1, 0.5, size=p)
    Z_swap = Z * (1 - swap) + Zk * swap
    Zk_swap = Z * swap + Zk * (1 - swap)

    Z_full = np.hstack([Z_swap, Zk_swap])

    # Lambda grid
    lambda_max = np.max(np.abs(Z_full.T @ y_flat)) / n
    lambda_min = lambda_max * eps
    lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), nlambda)

    # Lasso path
    try:
        _, coef_path, _ = lasso_path(Z_full, y_flat, alphas=lambdas, max_iter=10000)
    except Exception:
        _, coef_path, _ = lasso_path(Z_full, y_flat, n_alphas=nlambda, max_iter=10000)
        lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), coef_path.shape[1])

    # Entry times
    Z_entry = np.zeros(p)
    Zk_entry = np.zeros(p)

    for j in range(p):
        nonzero = np.where(np.abs(coef_path[j, :]) > 1e-10)[0]
        Z_entry[j] = lambdas[nonzero[0]] * n if len(nonzero) > 0 else 0

        nonzero_k = np.where(np.abs(coef_path[p + j, :]) > 1e-10)[0]
        Zk_entry[j] = lambdas[nonzero_k[0]] * n if len(nonzero_k) > 0 else 0

    # W statistics
    W = np.maximum(Z_entry, Zk_entry) * np.sign(Z_entry - Zk_entry)
    W = W * (1 - 2 * swap)

    # Threshold (include 0 to match R)
    W_abs = np.abs(W)
    candidates = np.sort(np.concatenate([[0], W_abs]))

    threshold = np.inf
    for t in candidates:
        numerator = np.sum(W <= -t)
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
        'knockoffs': Zk,
        'backend': 'custom_glmnet'
    }


# =============================================================================
# Comparison Functions
# =============================================================================

def run_all_backends(Z: np.ndarray, y: np.ndarray, fdr: float = 0.1,
                     method: str = 'sdp', seed: int = 42) -> Dict:
    """Run all knockoff backends and collect results."""
    results = {}

    # 1. R native
    logger.info("Running R native knockoff...")
    results['R_native'] = compute_knockoffs_r(Z, y, fdr=fdr, seed=seed)

    # 2. knockoff-filter (Fortran glmnet)
    logger.info("Running knockoff-filter (Fortran glmnet)...")
    results['knockoff_filter'] = compute_knockoffs_knockoff_filter(
        Z, y, fdr=fdr, method=method, use_sklearn=False, seed=seed
    )

    # 3. knockoff-filter (sklearn)
    logger.info("Running knockoff-filter (sklearn)...")
    results['knockoff_filter_sklearn'] = compute_knockoffs_knockoff_filter(
        Z, y, fdr=fdr, method=method, use_sklearn=True, seed=seed
    )

    # 4. knockpy (lsm)
    logger.info("Running knockpy (lsm)...")
    results['knockpy_lsm'] = compute_knockoffs_knockpy(
        Z, y, fdr=fdr, method=method, fstat='lsm', seed=seed
    )

    # 5. knockpy (lasso)
    logger.info("Running knockpy (lasso)...")
    results['knockpy_lasso'] = compute_knockoffs_knockpy(
        Z, y, fdr=fdr, method=method, fstat='lasso', seed=seed
    )

    # 6. custom_glmnet (SLIDE implementation)
    logger.info("Running custom_glmnet...")
    results['custom_glmnet'] = compute_knockoffs_custom_glmnet(
        Z, y, fdr=fdr, method=method, seed=seed
    )

    return results


def compute_comparison_metrics(results: Dict) -> Dict:
    """Compute comparison metrics between backends."""
    valid = {k: v for k, v in results.items() if v is not None and 'W' in v}

    if len(valid) < 2:
        return {'error': 'Insufficient backends for comparison'}

    backends = list(valid.keys())
    n_backends = len(backends)

    # W statistics
    W_dict = {k: v['W'] for k, v in valid.items()}

    # Correlation matrix
    # Suppress divide-by-zero warnings when a backend has constant W (std=0)
    corr_matrix = np.zeros((n_backends, n_backends))
    with np.errstate(divide='ignore', invalid='ignore'):
        for i, b1 in enumerate(backends):
            for j, b2 in enumerate(backends):
                corr_matrix[i, j] = np.corrcoef(W_dict[b1], W_dict[b2])[0, 1]

    # Summary per backend
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

    # Selection agreement with R
    r_selected = set(valid.get('R_native', {}).get('selected', []))
    agreement = {}
    for name, res in valid.items():
        if name == 'R_native':
            continue
        py_selected = set(res['selected'])
        union = len(r_selected | py_selected)
        intersection = len(r_selected & py_selected)
        jaccard = intersection / union if union > 0 else 1.0
        agreement[name] = {
            'jaccard': jaccard,
            'r_selected': len(r_selected),
            'py_selected': len(py_selected),
            'overlap': intersection,
            'exact_match': r_selected == py_selected
        }

    # Correlation with R
    r_corr = {}
    if 'R_native' in backends:
        r_idx = backends.index('R_native')
        for i, name in enumerate(backends):
            if name != 'R_native':
                r_corr[name] = float(corr_matrix[r_idx, i]) if not np.isnan(corr_matrix[r_idx, i]) else None

    return {
        'summary': summary,
        'correlation_matrix': {
            'backends': backends,
            'values': corr_matrix.tolist()
        },
        'r_correlation': r_corr,
        'r_agreement': agreement
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def run_comparison_for_params(X: np.ndarray, y: np.ndarray, config: dict,
                              delta: float, lam: float, fdr: float,
                              love_backend: str = 'python',
                              knockoff_method: str = 'sdp',
                              seed: int = 42) -> Dict:
    """Run full comparison for one parameter set."""
    param_config = {**config, 'delta': delta, 'lambda': lam}

    logger.info(f"\n{'='*60}")
    logger.info(f"Parameters: delta={delta}, lambda={lam}, fdr={fdr}")
    logger.info(f"{'='*60}")

    # Run LOVE to get Z
    Z, y_out = run_love_to_get_z(X, y, param_config, love_backend=love_backend)

    # Run all knockoff backends
    results = run_all_backends(Z, y_out, fdr=fdr, method=knockoff_method, seed=seed)

    # Compute metrics
    metrics = compute_comparison_metrics(results)

    return {
        'params': {'delta': delta, 'lambda': lam, 'fdr': fdr},
        'z_shape': Z.shape,
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'knockoffs'}
                    for k, v in results.items() if v is not None},
        'metrics': metrics
    }


def run_full_comparison(config_path: str, output_dir: str,
                        quick: bool = False, seed: int = 42) -> Dict:
    """Run full comparison across all parameter combinations."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, X_df = load_data(config['x_path'], config['y_path'])

    # Get parameter grid
    deltas = config.get('delta', [0.1])
    lambdas = config.get('lambda', [0.5])
    fdrs = [config.get('fdr', 0.1)]

    if isinstance(deltas, (int, float)):
        deltas = [deltas]
    if isinstance(lambdas, (int, float)):
        lambdas = [lambdas]

    # Quick mode: single parameter set
    if quick:
        deltas = deltas[:1]
        lambdas = lambdas[:1]

    logger.info(f"Running comparison with {len(deltas)} deltas x {len(lambdas)} lambdas x {len(fdrs)} fdrs")
    logger.info(f"Total parameter combinations: {len(deltas) * len(lambdas) * len(fdrs)}")

    all_results = []

    for delta, lam, fdr in itertools.product(deltas, lambdas, fdrs):
        try:
            result = run_comparison_for_params(
                X, y, config, delta, lam, fdr,
                love_backend='python',
                knockoff_method='sdp',
                seed=seed
            )
            all_results.append(result)

            # Save intermediate result
            param_str = f"d{delta}_l{lam}_fdr{fdr}"
            with open(output_dir / f"result_{param_str}.json", 'w') as f:
                json.dump(result, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed for delta={delta}, lambda={lam}, fdr={fdr}: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate results
    aggregated = aggregate_results(all_results)

    # Save full results
    full_results = {
        'config_path': str(config_path),
        'output_dir': str(output_dir),
        'timestamp': datetime.now().isoformat(),
        'data_shape': {'n_samples': X.shape[0], 'n_features': X.shape[1]},
        'parameters_tested': {
            'deltas': deltas,
            'lambdas': lambdas,
            'fdrs': fdrs
        },
        'results': all_results,
        'aggregated': aggregated
    }

    with open(output_dir / 'full_comparison.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)

    # Generate report
    generate_report(full_results, output_dir)

    return full_results


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate results across parameter combinations."""
    if not results:
        return {}

    # Collect R correlations
    r_corrs = {}
    r_agreements = {}

    for r in results:
        metrics = r.get('metrics', {})
        for backend, corr in metrics.get('r_correlation', {}).items():
            if backend not in r_corrs:
                r_corrs[backend] = []
            if corr is not None:
                r_corrs[backend].append(corr)

        for backend, agreement in metrics.get('r_agreement', {}).items():
            if backend not in r_agreements:
                r_agreements[backend] = {'exact_matches': 0, 'total': 0, 'jaccards': []}
            r_agreements[backend]['total'] += 1
            if agreement.get('exact_match'):
                r_agreements[backend]['exact_matches'] += 1
            r_agreements[backend]['jaccards'].append(agreement.get('jaccard', 0))

    # Compute averages
    avg_r_corr = {k: np.mean(v) if v else None for k, v in r_corrs.items()}
    avg_jaccard = {k: np.mean(v['jaccards']) if v['jaccards'] else None
                   for k, v in r_agreements.items()}
    exact_match_rate = {k: v['exact_matches'] / v['total'] if v['total'] > 0 else None
                        for k, v in r_agreements.items()}

    return {
        'avg_r_correlation': avg_r_corr,
        'avg_jaccard': avg_jaccard,
        'exact_match_rate': exact_match_rate,
        'n_parameter_sets': len(results)
    }


def generate_report(results: Dict, output_dir: Path):
    """Generate human-readable report."""
    report_lines = []

    report_lines.append("=" * 70)
    report_lines.append("KNOCKOFF IMPLEMENTATION COMPARISON REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Timestamp: {results['timestamp']}")
    report_lines.append(f"Config: {results['config_path']}")
    report_lines.append(f"Data: {results['data_shape']['n_samples']} samples x {results['data_shape']['n_features']} features")
    report_lines.append("")

    # Parameters tested
    params = results['parameters_tested']
    report_lines.append("Parameters tested:")
    report_lines.append(f"  Deltas: {params['deltas']}")
    report_lines.append(f"  Lambdas: {params['lambdas']}")
    report_lines.append(f"  FDRs: {params['fdrs']}")
    report_lines.append(f"  Total combinations: {len(results['results'])}")
    report_lines.append("")

    # Aggregated results
    agg = results.get('aggregated', {})

    report_lines.append("-" * 70)
    report_lines.append("AGGREGATED RESULTS (across all parameter sets)")
    report_lines.append("-" * 70)

    report_lines.append("\nAverage Correlation with R:")
    for backend, corr in agg.get('avg_r_correlation', {}).items():
        corr_str = f"{corr:.4f}" if corr is not None else "N/A"
        report_lines.append(f"  {backend:30s}: {corr_str}")

    report_lines.append("\nAverage Jaccard Agreement with R:")
    for backend, jaccard in agg.get('avg_jaccard', {}).items():
        j_str = f"{jaccard:.4f}" if jaccard is not None else "N/A"
        report_lines.append(f"  {backend:30s}: {j_str}")

    report_lines.append("\nExact Match Rate with R:")
    for backend, rate in agg.get('exact_match_rate', {}).items():
        r_str = f"{rate:.1%}" if rate is not None else "N/A"
        report_lines.append(f"  {backend:30s}: {r_str}")

    # Per-parameter results
    report_lines.append("")
    report_lines.append("-" * 70)
    report_lines.append("PER-PARAMETER RESULTS")
    report_lines.append("-" * 70)

    for r in results['results']:
        params = r['params']
        report_lines.append(f"\n  delta={params['delta']}, lambda={params['lambda']}, fdr={params['fdr']}")
        report_lines.append(f"  Z shape: {r['z_shape']}")

        # R correlation for this param set
        r_corr = r.get('metrics', {}).get('r_correlation', {})
        if r_corr:
            report_lines.append("  R correlation:")
            for backend, corr in r_corr.items():
                corr_str = f"{corr:.4f}" if corr is not None else "N/A"
                report_lines.append(f"    {backend:28s}: {corr_str}")

        # Selections
        summary = r.get('metrics', {}).get('summary', {})
        if summary:
            report_lines.append("  Selections:")
            for backend, s in summary.items():
                sel_str = str(s['selected'][:10]) + "..." if len(s['selected']) > 10 else str(s['selected'])
                report_lines.append(f"    {backend:28s}: {s['n_selected']} vars {sel_str}")

    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)

    # Print to console
    print(report_text)

    # Save to file
    with open(output_dir / 'comparison_report.txt', 'w') as f:
        f.write(report_text)

    logger.info(f"Report saved to {output_dir / 'comparison_report.txt'}")


# =============================================================================
# Single Backend Mode (for array jobs)
# =============================================================================

BACKEND_FUNCTIONS = {
    'R_native': compute_knockoffs_r,
    'knockoff_filter': lambda Z, y, fdr, method, seed: compute_knockoffs_knockoff_filter(
        Z, y, fdr=fdr, method=method, use_sklearn=False, seed=seed),
    'knockoff_filter_sklearn': lambda Z, y, fdr, method, seed: compute_knockoffs_knockoff_filter(
        Z, y, fdr=fdr, method=method, use_sklearn=True, seed=seed),
    'knockpy_lsm': lambda Z, y, fdr, method, seed: compute_knockoffs_knockpy(
        Z, y, fdr=fdr, method=method, fstat='lsm', seed=seed),
    'knockpy_lasso': lambda Z, y, fdr, method, seed: compute_knockoffs_knockpy(
        Z, y, fdr=fdr, method=method, fstat='lasso', seed=seed),
    'custom_glmnet': compute_knockoffs_custom_glmnet,
}

ALL_BACKENDS = list(BACKEND_FUNCTIONS.keys())


def run_single_backend(config_path: str, output_dir: str, backend: str,
                       quick: bool = False, seed: int = 42):
    """Run knockoff comparison for a single backend (for array jobs)."""
    import itertools

    logger.info(f"Running single backend: {backend}")

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, X_df = load_data(config['x_path'], config['y_path'])

    # Get parameter grid
    deltas = config.get('delta', [0.1])
    lambdas = config.get('lambda', [0.5])
    fdrs = [config.get('fdr', 0.1)]

    if isinstance(deltas, (int, float)):
        deltas = [deltas]
    if isinstance(lambdas, (int, float)):
        lambdas = [lambdas]

    if quick:
        deltas = deltas[:1]
        lambdas = lambdas[:1]

    logger.info(f"Parameter combinations: {len(deltas)} x {len(lambdas)} x {len(fdrs)}")

    backend_results = []

    for delta, lam, fdr in itertools.product(deltas, lambdas, fdrs):
        param_str = f"d{delta}_l{lam}_fdr{fdr}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Parameters: {param_str}")
        logger.info(f"{'='*60}")

        try:
            # Run LOVE to get Z
            param_config = {**config, 'delta': delta, 'lambda': lam}
            Z, y_out = run_love_to_get_z(X, y, param_config, love_backend='python')

            # Run single backend
            logger.info(f"Running {backend}...")

            if backend == 'R_native':
                result = compute_knockoffs_r(Z, y_out, fdr=fdr, seed=seed)
            elif backend == 'knockoff_filter':
                result = compute_knockoffs_knockoff_filter(
                    Z, y_out, fdr=fdr, method='sdp', use_sklearn=False, seed=seed)
            elif backend == 'knockoff_filter_sklearn':
                result = compute_knockoffs_knockoff_filter(
                    Z, y_out, fdr=fdr, method='sdp', use_sklearn=True, seed=seed)
            elif backend == 'knockpy_lsm':
                result = compute_knockoffs_knockpy(
                    Z, y_out, fdr=fdr, method='sdp', fstat='lsm', seed=seed)
            elif backend == 'knockpy_lasso':
                result = compute_knockoffs_knockpy(
                    Z, y_out, fdr=fdr, method='sdp', fstat='lasso', seed=seed)
            elif backend == 'custom_glmnet':
                result = compute_knockoffs_custom_glmnet(
                    Z, y_out, fdr=fdr, method='sdp', seed=seed)
            else:
                raise ValueError(f"Unknown backend: {backend}")

            if result is not None:
                # Remove knockoffs from result (too large to serialize)
                result_clean = {k: v for k, v in result.items() if k != 'knockoffs'}
                # Convert numpy arrays
                if 'W' in result_clean:
                    result_clean['W'] = result_clean['W'].tolist()
                if 'selected' in result_clean:
                    result_clean['selected'] = result_clean['selected'].tolist()

                backend_results.append({
                    'params': {'delta': delta, 'lambda': lam, 'fdr': fdr},
                    'z_shape': list(Z.shape),
                    'result': result_clean
                })

                logger.info(f"  Selected: {len(result['selected'])} variables")
                logger.info(f"  Threshold: {result['threshold']}")

        except Exception as e:
            logger.error(f"Failed for {param_str}: {e}")
            import traceback
            traceback.print_exc()

    # Save backend results
    output_file = output_dir / f"backend_{backend}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'backend': backend,
            'timestamp': datetime.now().isoformat(),
            'config_path': str(config_path),
            'results': backend_results
        }, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_file}")


def aggregate_backend_results(output_dir: str):
    """Aggregate results from individual backend runs (array job completion)."""
    output_dir = Path(output_dir)

    logger.info(f"Aggregating results from {output_dir}")

    # Find all backend result files
    backend_files = list(output_dir.glob("backend_*.json"))

    if not backend_files:
        logger.error("No backend result files found!")
        return

    logger.info(f"Found {len(backend_files)} backend files")

    # Load all results
    all_backend_data = {}
    for bf in backend_files:
        with open(bf) as f:
            data = json.load(f)
            backend_name = data['backend']
            all_backend_data[backend_name] = data

    # Reorganize by parameter set
    param_results = {}

    for backend_name, data in all_backend_data.items():
        for r in data['results']:
            param_key = f"d{r['params']['delta']}_l{r['params']['lambda']}_fdr{r['params']['fdr']}"

            if param_key not in param_results:
                param_results[param_key] = {
                    'params': r['params'],
                    'z_shape': r['z_shape'],
                    'backends': {}
                }

            param_results[param_key]['backends'][backend_name] = r['result']

    # Compute comparison metrics for each parameter set
    final_results = []

    for param_key, pr in param_results.items():
        logger.info(f"\nComputing metrics for {param_key}")

        # Build results dict for compute_comparison_metrics
        results_for_metrics = {}
        for backend_name, result in pr['backends'].items():
            if result:
                results_for_metrics[backend_name] = {
                    'W': np.array(result['W']),
                    'threshold': result['threshold'] if result['threshold'] != 'inf' else np.inf,
                    'selected': np.array(result['selected']),
                    'backend': backend_name
                }

        metrics = compute_comparison_metrics(results_for_metrics)

        final_results.append({
            'params': pr['params'],
            'z_shape': pr['z_shape'],
            'results': pr['backends'],
            'metrics': metrics
        })

    # Aggregate across all parameter sets
    aggregated = aggregate_results(final_results)

    # Build full results
    full_results = {
        'output_dir': str(output_dir),
        'timestamp': datetime.now().isoformat(),
        'backends_found': list(all_backend_data.keys()),
        'parameters_tested': {
            'n_combinations': len(param_results)
        },
        'results': final_results,
        'aggregated': aggregated
    }

    # Save
    with open(output_dir / 'full_comparison.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)

    # Generate report
    generate_report(full_results, output_dir)

    logger.info(f"\nAggregated results saved to {output_dir / 'full_comparison.json'}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive knockoff implementation comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--config', '-c', required=True,
                        help='Path to YAML config file')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory (default: auto-generated)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (single parameter set)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--backend', '-b', default=None,
                        choices=['R_native', 'knockoff_filter', 'knockoff_filter_sklearn',
                                 'knockpy_lsm', 'knockpy_lasso', 'custom_glmnet', 'all'],
                        help='Run only specific backend (for array jobs). Default: all')
    parser.add_argument('--aggregate', action='store_true',
                        help='Aggregate results from individual backend runs')

    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f"output_comparison/knockoff_cmp_{timestamp}"

    # Handle aggregate mode (combine results from array job)
    if args.aggregate:
        aggregate_backend_results(args.output_dir)
        return 0

    # Handle single backend mode (for array jobs)
    if args.backend and args.backend != 'all':
        run_single_backend(
            config_path=args.config,
            output_dir=args.output_dir,
            backend=args.backend,
            quick=args.quick,
            seed=args.seed
        )
        return 0

    # Run all backends
    results = run_full_comparison(
        config_path=args.config,
        output_dir=args.output_dir,
        quick=args.quick,
        seed=args.seed
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
