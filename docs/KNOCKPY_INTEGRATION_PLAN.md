# knockpy Integration Plan

## Overview

Replace the current `knockoff-filter` Python package with `knockpy` for the Python knockoff backend. knockpy uses the same DSDP solver as R (via `scikit-dsdp`), which should eliminate the SDP failures we're seeing.

## Why knockpy?

| Feature | Current (knockoff-filter) | knockpy |
|---------|---------------------------|---------|
| SDP Solver | CVXPY (generic) | scikit-dsdp (same as R) |
| SDP Reliability | Fails on difficult matrices | Robust (matches R) |
| Methods | equi, sdp, asdp | equi, sdp, mvr, mmi, ci, maxent |
| Default | asdp | mvr (often more powerful than sdp) |
| Maintenance | Limited | Active development |

## Installation

```bash
# Activate environment
conda activate /ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env

# Install knockpy
pip install knockpy

# Install scikit-dsdp for DSDP solver (critical for matching R behavior)
pip install scikit-dsdp
```

## Code Changes

### 1. Update `src/loveslide/knockoffs.py`

Add a new method `filter_knockoffs_iterative_knockpy`:

```python
@staticmethod
def filter_knockoffs_iterative_knockpy(z, y, fdr=0.1, niter=1, spec=0.2,
                                       method='mvr', shrink=False, **kwargs):
    """Run knockoff filter using knockpy package.

    Parameters
    ----------
    z : np.ndarray
        Feature matrix.
    y : np.ndarray
        Response vector.
    fdr : float
        Target false discovery rate.
    niter : int
        Number of knockoff iterations.
    spec : float
        Proportion threshold for selection frequency.
    method : str
        Knockoff construction method:
        - 'mvr': Minimum Variance-based Reconstructability (default, often best)
        - 'sdp': Semidefinite programming (uses DSDP if scikit-dsdp installed)
        - 'equicorrelated': Always works, lower power
        - 'maxent': Maximum entropy
        - 'mmi': Minimize mutual information
    shrink : bool
        Whether to use Ledoit-Wolf covariance shrinkage.
    **kwargs
        Additional keyword arguments (ignored).

    Returns
    -------
    np.ndarray
        Indices of selected variables.
    """
    from knockpy import KnockoffFilter

    # Map method names for compatibility
    method_map = {
        'asdp': 'sdp',  # knockpy doesn't have asdp, sdp uses DSDP
        'equi': 'equicorrelated',
    }
    kp_method = method_map.get(method, method)

    # Configure shrinkage
    shrinkage = 'ledoitwolf' if shrink else None

    # Create knockoff filter
    kfilter = KnockoffFilter(
        ksampler='gaussian',
        fstat='lasso',
        knockoff_kwargs={'method': kp_method}
    )

    results = []
    for _ in range(niter):
        # Run knockoff filter
        rejections = kfilter.forward(
            X=z,
            y=y.flatten(),
            fdr=fdr,
            shrinkage=shrinkage
        )
        # rejections is a boolean array
        selected = np.where(rejections)[0]
        if len(selected) > 0:
            results.extend(selected.tolist())

    if len(results) == 0:
        return np.array([], dtype=int)

    results = np.array(results)
    idx, counts = np.unique(results, return_counts=True)
    sig_idxs = idx[np.where(counts >= spec * niter)]

    return sig_idxs
```

### 2. Update `filter_knockoffs_iterative` dispatcher

Modify the main dispatcher to support `knockpy` as a backend option:

```python
@staticmethod
def filter_knockoffs_iterative(z, y, fdr=0.1, niter=1, spec=0.2, n_workers=1,
                               backend='r', method='asdp', shrink=False):
    """
    Run knockoff filter to find significant variables.

    Parameters
    ----------
    backend : str
        Which knockoff implementation: 'r' (default), 'python', or 'knockpy'.
    method : str
        Knockoff construction method:
        - For 'python' backend: 'asdp' (default), 'sdp', or 'equi'
        - For 'knockpy' backend: 'mvr' (default), 'sdp', 'equicorrelated', 'maxent', 'mmi'
    ...
    """
    if backend == 'knockpy':
        return Knockoffs.filter_knockoffs_iterative_knockpy(
            z, y, fdr=fdr, niter=niter, spec=spec, method=method, shrink=shrink)
    elif backend == 'python':
        return Knockoffs.filter_knockoffs_iterative_python(
            z, y, fdr=fdr, niter=niter, spec=spec, method=method, shrink=shrink)
    else:
        return Knockoffs.filter_knockoffs_iterative_r(
            z, y, fdr=fdr, niter=niter, spec=spec)
```

### 3. Update `select_short_freq` method

Update the `select_short_freq` method signature and docstring to document the new backend option (no code changes needed since it passes `backend` through to `filter_knockoffs_iterative`).

### 4. Update `comparison/run_slide_py.py`

Update the CLI argument parser:

```python
parser.add_argument('--knockoff-backend', dest='knockoff_backend',
                    choices=['python', 'r', 'knockpy'],
                    default='r',
                    help='Knockoff implementation: r (default), python, or knockpy')
parser.add_argument('--knockoff-method', dest='knockoff_method',
                    choices=['asdp', 'sdp', 'equi', 'mvr', 'equicorrelated', 'maxent', 'mmi'],
                    default='mvr',
                    help='Knockoff construction method. For knockpy: mvr (default), sdp, '
                         'equicorrelated, maxent, mmi. For python: asdp, sdp, equi.')
```

### 5. Update `comparison/run_comparison.sh`

Replace the Python knockoff tasks to use knockpy:

```bash
elif [ "$TASK_ID" -eq 2 ]; then
    # Task 2: Python (R LOVE + knockpy Knockoffs)
    echo "Running Python SLIDE (R LOVE + knockpy Knockoffs)"
    TASK_OUT="${OUT_PATH}/Py_rLOVE_knockpy"
    mkdir -p "$TASK_OUT"

    # Use knockpy with mvr method (matches R's DSDP solver)
    "$PYTHON_ENV" run_slide_py.py "$YAML_CONFIG" "$TASK_OUT" \
        --love-backend r --knockoff-backend knockpy --knockoff-method mvr

    touch "${OUT_PATH}/.task2_complete"

elif [ "$TASK_ID" -eq 4 ]; then
    # Task 4: Python (Python LOVE + knockpy Knockoffs)
    echo "Running Python SLIDE (Python LOVE + knockpy Knockoffs)"
    TASK_OUT="${OUT_PATH}/Py_pyLOVE_knockpy"
    mkdir -p "$TASK_OUT"

    # Use knockpy with mvr method
    "$PYTHON_ENV" run_slide_py.py "$YAML_CONFIG" "$TASK_OUT" \
        --love-backend python --knockoff-backend knockpy --knockoff-method mvr

    touch "${OUT_PATH}/.task4_complete"
```

Update header comments:
```bash
# Array tasks:
#   0 = R SLIDE (native R package - R LOVE + R knockoffs)
#   1 = Python SLIDE (R LOVE + R Knockoffs)
#   2 = Python SLIDE (R LOVE + knockpy Knockoffs, MVR method)
#   3 = Python SLIDE (Py LOVE + R Knockoffs)
#   4 = Python SLIDE (Py LOVE + knockpy Knockoffs, MVR method)
```

## knockpy Method Recommendations

| Method | Description | When to Use |
|--------|-------------|-------------|
| `mvr` | Minimum Variance Reconstructability | **Default choice** - often more powerful than SDP |
| `sdp` | Semidefinite Programming | When you want to match R exactly (uses DSDP) |
| `equicorrelated` | Equal correlations | Fallback if others fail |
| `maxent` | Maximum entropy | Alternative to MVR |

## Testing

After implementation, test with:

```bash
# Quick test - single parameter combination
cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
python run_slide_py.py comparison_config_binary.yaml ./test_knockpy \
    --love-backend python --knockoff-backend knockpy --knockoff-method mvr
```

## Verification

Check that scikit-dsdp is being used:

```python
import knockpy
# If scikit-dsdp is installed, knockpy will automatically use it for SDP
# You can verify by checking for warnings about CVXPY during SDP solving
```

## Rollback

If knockpy causes issues, the equicorrelated method fix is already in place as a fallback:
```bash
--knockoff-backend python --knockoff-method equi
```

## References

- [knockpy documentation](https://amspector100.github.io/knockpy/)
- [knockpy GitHub](https://github.com/amspector100/knockpy)
- [scikit-dsdp](https://pypi.org/project/scikit-dsdp/)
- [R knockoff package](https://cran.r-project.org/web/packages/knockoff/)
