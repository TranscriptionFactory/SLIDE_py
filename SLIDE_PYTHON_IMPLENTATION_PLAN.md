# Plan: Package Python SLIDE with All Dependencies

## Summary

Create a new branch to package Python SLIDE (`loveslide`) with all Python dependencies for LOVE and knockoffs. Make R integration optional.

## Repository Structure

### Current State
```
SLIDE_py/                           # Main repository
├── src/loveslide/                  # Python SLIDE package
│   ├── love.py                     # Python LOVE wrapper
│   ├── knockoffs.py                # Python knockoffs wrapper
│   ├── slide.py                    # Main SLIDE logic
│   └── Py_LOVE/love/               # Pure Python LOVE implementation (vendored)
└── pyproject.toml                  # Package config

knockoff-filter/knockoff-filter/    # Separate repository (forked)
├── knockoff/                       # Python knockoff implementation
│   ├── create.py                   # Knockoff generation
│   ├── solve.py                    # SDP solvers
│   ├── filter.py                   # Knockoff filter
│   └── stats/                      # W-statistics
└── pyproject.toml
```

### Target State
```
SLIDE_py/
├── src/loveslide/
│   ├── __init__.py
│   ├── love.py                     # Python LOVE wrapper
│   ├── knockoffs.py                # Python knockoffs wrapper (updated imports)
│   ├── slide.py                    # Main SLIDE logic
│   ├── Py_LOVE/love/               # Pure Python LOVE implementation (vendored)
│   ├── knockoff/                   # BUNDLED knockoff-filter package
│   │   ├── __init__.py             # With source attribution
│   │   ├── create.py               # Knockoff generation
│   │   ├── solve.py                # SDP solvers (ASDP, SDP, equi)
│   │   ├── filter.py               # Knockoff filter
│   │   ├── utils.py                # Utilities
│   │   ├── stats/                  # W-statistics
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # swap_columns, etc.
│   │   │   ├── glmnet.py           # lasso-based stats
│   │   │   └── lasso.py
│   │   ├── pydsdp/                 # SDP solver Python interface (imports from pydsdp_ext)
│   │   └── _vendor/glmnet/         # Vendored glmnet (already included)
│   └── pydsdp_ext/                 # BUNDLED scikit-dsdp (C extension)
│       ├── __init__.py             # With source attribution
│       ├── setup.py                # For rebuilding C extension
│       ├── dsdp/                   # DSDP source code
│       └── *.so                    # Compiled C extension (platform-specific)
├── pyproject.toml                  # Updated: bundled deps, rpy2 optional
└── comparison/                     # Testing/comparison tools
```

## Required Python Dependencies

### Core Dependencies (Required)
| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.20.0 | Array operations |
| scipy | >=1.7.0 | Linear algebra, optimization |
| scikit-learn | >=1.0.0 | Lasso path, preprocessing |
| pandas | >=2.0.0 | Data handling |
| joblib | >=0.14.1 | Parallel processing |

### Bundled Dependencies (Copy into repo)
| Source | Destination | Purpose | Notes |
|--------|-------------|---------|-------|
| `knockoff-filter/knockoff/` | `src/loveslide/knockoff/` | Knockoff generation, W-stats | Includes vendored glmnet |
| `scikit-dsdp/` | `src/loveslide/pydsdp_ext/` | SDP solver (DSDP) | Optional, requires compilation |

### SDP Solver Strategy
**Primary**: cvxpy with SCS solver (pure Python, no compilation)
**Fallback**: scikit-dsdp/DSDP if available (exact R match)

The solver will try scikit-dsdp first for exact R compatibility, then fall back to cvxpy if not available.

### Optional Dependencies
| Package | Version | Purpose | Install Extra |
|---------|---------|---------|---------------|
| rpy2 | >=3.5.0 | R integration | `[r]` |

### NOT Needed
| Package | Reason |
|---------|--------|
| python-glmnet | Already vendored in knockoff-filter (`_vendor/glmnet/`) |
| knockpy | Different algorithms, not R-compatible |

## Implementation Plan

### Phase 1: Create Branch
```bash
cd /ix/djishnu/Aaron/1_general_use/SLIDE_py
git checkout -b feat/python-packaging
```

### Phase 2: Copy Dependencies into Repository

#### 2a. Copy knockoff-filter

```bash
# Copy knockoff package
cp -r /ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter/knockoff \
    /ix/djishnu/Aaron/1_general_use/SLIDE_py/src/loveslide/
```

Add attribution header to `src/loveslide/knockoff/__init__.py`:
```python
"""
Knockoff Filter for Controlled Variable Selection.

Source: https://github.com/TranscriptionFactory/knockoff-filter
        (forked from https://github.com/msesia/knockofffilter)

Original authors: Rina Foygel Barber, Emmanuel Candes, Lucas Janson,
                  Evan Patterson, Matteo Sesia

License: GPL-3.0
"""
```

#### 2b. Copy scikit-dsdp (DSDP SDP solver)

```bash
# Copy scikit-dsdp for SDP solving
cp -r /ix/djishnu/Aaron/1_general_use/scikit-dsdp \
    /ix/djishnu/Aaron/1_general_use/SLIDE_py/src/loveslide/
mv /ix/djishnu/Aaron/1_general_use/SLIDE_py/src/loveslide/scikit-dsdp \
    /ix/djishnu/Aaron/1_general_use/SLIDE_py/src/loveslide/pydsdp_ext
```

Add attribution header to `src/loveslide/pydsdp_ext/__init__.py`:
```python
"""
DSDP5 SDP Solver - Python interface.

Source: https://github.com/sburer/scikit-dsdp

Note: The C extension (.so file) is platform-specific and may need
      to be recompiled for different Python versions/platforms.
      See setup.py for build instructions.

License: GPL
"""
```

**Important**: The `.so` file is compiled for Python 3.9 on Linux x86_64.
For other platforms/versions, run: `python setup.py build_ext --inplace`

#### 2c. Update knockoff SDP solver with cvxpy fallback

Modify `src/loveslide/knockoff/solve.py` to add cvxpy fallback:

```python
def create_solve_sdp(Sigma, gaptol=1e-6, maxit=1000, verbose=False, **kwargs):
    """SDP optimization with DSDP primary, cvxpy fallback."""

    # Try DSDP first (exact R match)
    try:
        from .pydsdp import dsdp
        return _solve_sdp_dsdp(Sigma, gaptol, maxit, verbose)
    except ImportError:
        pass

    # Fallback to cvxpy (pure Python, no compilation needed)
    try:
        import cvxpy as cp
        return _solve_sdp_cvxpy(Sigma, gaptol, maxit, verbose)
    except ImportError:
        raise ImportError(
            "No SDP solver available. Install either:\n"
            "  - scikit-dsdp (for exact R match)\n"
            "  - cvxpy (pure Python fallback)"
        )

def _solve_sdp_cvxpy(Sigma, gaptol, maxit, verbose):
    """Solve knockoff SDP using cvxpy."""
    import cvxpy as cp

    G = cov2cor(Sigma)
    p = G.shape[0]

    s = cp.Variable(p)
    constraints = [s >= 0, s <= 1, 2 * G - cp.diag(s) >> 0]
    prob = cp.Problem(cp.Maximize(cp.sum(s)), constraints)
    prob.solve(solver=cp.SCS, eps=gaptol, max_iters=maxit, verbose=verbose)

    s_val = np.clip(s.value, 0, 1)

    # Apply same feasibility adjustment as DSDP version
    s_eps = 1e-8
    while s_eps <= 0.1:
        if is_posdef(2 * G - np.diag(s_val * (1 - s_eps)), tol=1e-9):
            break
        s_eps *= 10

    return s_val * (1 - s_eps) * np.diag(Sigma)
```

Also update pydsdp import for bundled version:
```python
# In pydsdp/dsdp5.py, change:
from pydsdp.pydsdp5 import pyreadsdpa

# To:
try:
    from ...pydsdp_ext.pydsdp5 import pyreadsdpa  # Bundled
except ImportError:
    from pydsdp.pydsdp5 import pyreadsdpa  # System package
```

#### 2d. Update knockoffs.py imports

```python
# Change from:
from knockoff.stats import stat_glmnet_lambdasmax
from knockoff.create import create_gaussian

# To:
from .knockoff.stats import stat_glmnet_lambdasmax
from .knockoff.create import create_gaussian
```

### Phase 3: Update pyproject.toml

```toml
[project]
name = "loveslide"
version = "0.1.0"
description = "Python implementation of SLIDE (LOVE + Knockoffs)"
requires-python = ">=3.9"

dependencies = [
    # Core - always required
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "pandas>=2.0.0",
    "cvxpy>=1.3.0",
    "joblib>=0.14.1",
    # knockoff-filter is bundled in src/loveslide/knockoff/
]

[project.optional-dependencies]
r = [
    "rpy2>=3.5.0",
]
full = [
    "rpy2>=3.5.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]
```

Note: knockoff-filter is bundled directly in `src/loveslide/knockoff/` - no external dependency needed.

### Phase 4: Update Knockoff Integration

Update `src/loveslide/knockoffs.py` to:
1. Remove internal seed setting (mirror R)
2. Add `use_r_rng` option
3. Remove knockpy references

```python
def compute_knockoffs(Z, y, fdr=0.1, method='asdp',
                      seed=None, use_r_rng=False, **kwargs):
    """
    Compute knockoff filter for variable selection.

    Parameters
    ----------
    seed : int, optional
        Random seed. If None, uses current RNG state (like R).
    use_r_rng : bool, default=False
        Use R's RNG for random steps (requires rpy2).
    """
    if seed is not None:
        np.random.seed(seed)

    if use_r_rng:
        # Requires optional rpy2 dependency
        try:
            import rpy2.robjects as robjects
            # Use R for random number generation
        except ImportError:
            raise ImportError("rpy2 required for use_r_rng=True. Install with: pip install loveslide[r]")
    ...
```

### Phase 5: Documentation

Append to `comparison/KNOCKOFF_COMPARISON.md`:

```markdown
## RNG and Reproducibility Analysis

### Sources of Randomness

Both R and Python use **identical algorithms** but different random number generators.

| Step | R Implementation | Python Implementation | Impact |
|------|------------------|----------------------|--------|
| ASDP perturbation | `matrix(rnorm(p*p), p) * 1e-6` | `np.random.randn(p, p) * 1e-6` | Block assignments (p>500) |
| Knockoff sampling | `matrix(rnorm(n*p), n) %*% chol(Sigma_k)` | `np.random.randn(n, p) @ L.T` | **Major** - W magnitudes |
| Column swap | `rbinom(p, 1, 0.5)` | `np.random.binomial(1, 0.5, p)` | W signs |

### Numerical Thresholds (All Identical)

| Parameter | R Value | Python Value |
|-----------|---------|--------------|
| `gaptol` | 1e-6 | 1e-6 |
| `maxit` | 1000 | 1000 |
| `s_eps` range | 1e-8 to 0.1 | 1e-8 to 0.1 |
| `max.size` | 500 | 500 |

### Seed Handling

R does NOT set seeds internally. Python mirrors this behavior.

### Using R's RNG (Maximum Compatibility)

```python
# Pure Python (default)
result = compute_knockoffs(Z, y, seed=42)

# R RNG for maximum compatibility (requires rpy2)
result = compute_knockoffs(Z, y, seed=42, use_r_rng=True)
```
```

### Phase 6: Clean Up

1. Remove knockpy imports from comparison scripts
2. Remove knockpy backends from `run_knockoffs_on_precomputed.py`
3. Update `BACKENDS` dict to only include:
   - `R_native` (requires rpy2)
   - `R_knockoffs_py_sklearn` (requires rpy2)
   - `knockoff_filter_sklearn` (pure Python)
   - `knockoff_filter_sklearn_r_rng` (requires rpy2)

## Files to Modify/Create

| File | Changes |
|------|---------|
| `src/loveslide/knockoff/` | **NEW** - Copy from knockoff-filter repo |
| `src/loveslide/knockoff/__init__.py` | Add source attribution comment |
| `src/loveslide/knockoff/solve.py` | Add cvxpy fallback for SDP solver |
| `src/loveslide/knockoff/pydsdp/dsdp5.py` | Update import to try bundled then system pydsdp |
| `src/loveslide/pydsdp_ext/` | **NEW** - Copy from scikit-dsdp repo (optional) |
| `src/loveslide/pydsdp_ext/__init__.py` | Add source attribution comment |
| `pyproject.toml` | Update deps: cvxpy required, rpy2 optional, remove external knockoff |
| `src/loveslide/knockoffs.py` | Update imports to `.knockoff`, add `use_r_rng`, mirror R seed handling |
| `comparison/KNOCKOFF_COMPARISON.md` | Append RNG documentation |
| `comparison/run_knockoffs_on_precomputed.py` | Remove knockpy, update seed handling, update imports |

## Verification

1. Install package in clean environment:
   ```bash
   pip install -e .           # Core only
   pip install -e ".[r]"      # With R support
   ```

2. Test pure Python knockoffs:
   ```python
   from loveslide import compute_knockoffs
   result = compute_knockoffs(Z, y, seed=42)
   ```

3. Test R RNG option (if rpy2 installed):
   ```python
   result = compute_knockoffs(Z, y, seed=42, use_r_rng=True)
   ```

4. Run comparison to verify W-correlation with R_native

## Summary of Backends

| Backend | Knockoff Gen | W-Statistic | Requires rpy2 |
|---------|--------------|-------------|---------------|
| `R_native` | R | R glmnet | Yes |
| `R_knockoffs_py_sklearn` | R | sklearn | Yes |
| `knockoff_filter_sklearn` | Python | sklearn | No |
| `knockoff_filter_sklearn_r_rng` | Python (R RNG) | sklearn | Yes |
