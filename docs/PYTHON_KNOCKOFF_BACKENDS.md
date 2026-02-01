# Python Knockoff Filter Backends

This document describes the available knockoff filter backends in the SLIDE_py package, their implementations, and performance characteristics.

## Overview

The knockoff filter is a statistical method for controlled variable selection with FDR (False Discovery Rate) guarantees. SLIDE uses knockoffs to select significant latent factors.

### Backend Categories

| Category | Backends | Description |
|----------|----------|-------------|
| **Voting (Production)** | `python_voting`, `python_voting_slide`, `python_voting_glmnet` | Run multiple iterations, vote on selections |
| **Single-run (Debug)** | `knockoff_filter_sklearn`, `knockoff_filter` | Single knockoff run |
| **Hybrid** | `R_knockoffs_py_sklearn`, `R_knockoffs_py_stats` | R knockoff generation + Python statistics |
| **Reference** | `R_native`, `R_native_single` | R implementation (requires rpy2) |

---

## R SLIDE Pipeline

The R SLIDE package uses `selectShortFreq()` as the main function for knockoff-based variable selection. Understanding this function is key to matching R SLIDE behavior.

### R SLIDE `selectShortFreq()` Function

```r
selectShortFreq(z, y, spec = 0.3, fdr = 0.1, elbow = FALSE,
                niter = 1000, f_size = 100, parallel = TRUE)
```

**Default Parameters:**
- `niter = 1000` - Number of knockoff iterations
- `spec = 0.3` - Specificity threshold (keep vars in >= 30% of runs)
- `fdr = 0.1` - False discovery rate target
- `f_size = 100` - Maximum features per chunk

### R SLIDE Algorithm

1. **Feature Chunking**: Divide features into chunks of size `f_size`
   ```r
   n_splits <- ceiling(ncol(z) / f_size)
   feature_split <- ceiling(ncol(z) / n_splits)
   ```

2. **Per-Chunk Knockoff Voting**: Run `secondKO()` on each chunk
   - Execute `niter` knockoff iterations via `foreach/doParallel`
   - Apply threshold: keep variables selected in >= `spec * niter` runs

3. **findOptIter Refinement**: For each chunk, select variables from ONE optimal iteration
   ```r
   # Find iterations with maximum overlap with threshold-passing variables
   mm <- max(unlist(lapply(selected_list, function(x) { sum(x %in% freq_vars) })))
   max_overlap_ind <- which(...) == mm)

   # Tie-breaker: choose iteration with smallest selection set
   overlap_list_len <- sapply(max_overlap_ind, function(x) { length(selected_list[[x]]) })
   selected_run <- max_overlap_ind[which.min(overlap_list_len)]

   # Return variables from that ONE iteration
   selected_vars <- selected_list[[selected_run]]
   ```

4. **Two-Stage Screening**: When multiple chunks, combine screened variables and re-run knockoff voting on the combined set

### Key Insight: findOptIter vs Simple Threshold

| Approach | Returns | Behavior |
|----------|---------|----------|
| Simple threshold | ALL variables >= spec*niter | More inclusive |
| findOptIter (R SLIDE) | Variables from ONE optimal iteration | More conservative |

The `findOptIter` approach returns a subset of the threshold-passing variables by selecting from the iteration that:
1. Has maximum overlap with threshold-passing variables
2. Has the smallest total selection (tie-breaker)

---

## Parameter Defaults

### R SLIDE Defaults (selectShortFreq)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `niter` | 1000 | Number of knockoff iterations |
| `spec` | 0.3 | Specificity threshold (30%) |
| `fdr` | 0.1 | Target false discovery rate |
| `f_size` | 100 | Max features per chunk |
| `offset` | 0 | Standard knockoff (not knockoffs+) |
| `method` | 'asdp' | SDP method for diag_s computation |

### Python Defaults (for R SLIDE compatibility)

To match R SLIDE behavior, use these parameters:

```python
# For full R SLIDE compatibility
result = knockoff_filter_voting_slide(
    X, y,
    niter=1000,      # R default
    spec=0.3,        # R default
    fdr=0.1,         # R default
    f_size=100,      # R default
    match_r=True     # Skip condition number checks
)

# Note: Python defaults (niter=500, spec=0.1) differ from R
# for faster iteration during development
```

---

## Voting Backends (Recommended for Production)

### `python_voting` (Primary Python Backend)

**Simple voting approach** - Returns ALL variables above threshold.

```python
from loveslide.knockoff.filter import knockoff_filter_voting
from loveslide.knockoff.create import create_second_order
from loveslide.knockoff.stats import stat_glmnet_lambdasmax

result = knockoff_filter_voting(
    X, y,
    knockoffs=create_second_order,
    statistic=stat_glmnet_lambdasmax,
    fdr=0.1,
    niter=500,    # Number of knockoff iterations
    spec=0.1,     # Keep vars selected in >= 10% of runs
    n_jobs=1,
    base_seed=42
)
```

**Implementation:**
- Runs knockoff filter `niter` times with different random seeds
- Each iteration generates new knockoffs and computes W statistics
- Keeps ALL variables selected in >= `spec * niter` runs
- Uses SDP caching for 3-4x speedup (computed once, not per-iteration)
- Uses `sklearn.linear_model.lasso_path` for W statistics

**Algorithm:**
1. Pre-compute invariant values (covariance, SDP solution, Cholesky)
2. For each iteration:
   - Sample knockoffs using cached factors (fast)
   - Compute W statistics
   - Count selected variables
3. Return ALL variables with count >= spec * niter

**When to use:**
- Simpler voting approach without findOptIter refinement
- Faster development/testing with lower niter
- When exact R SLIDE match is not required


### `python_voting_slide` (Full R SLIDE Methodology)

**Recommended for R SLIDE compatibility** - Implements complete R SLIDE algorithm.

```python
from loveslide.knockoff.filter import knockoff_filter_voting_slide

result = knockoff_filter_voting_slide(
    X, y,
    knockoffs=create_second_order,
    statistic=stat_glmnet_lambdasmax,
    fdr=0.1,
    niter=1000,   # R SLIDE default
    spec=0.3,     # R SLIDE default
    f_size=100,   # Max features per chunk
    n_jobs=1,
    base_seed=42
)
```

**Implementation:**
1. **Feature chunking**: Divides features into chunks of `f_size` (default: 100)
2. **Per-chunk voting**: Runs knockoff_filter_voting on each chunk with `slide_selection=True`
3. **findOptIter refinement**: Selects features from ONE optimal iteration per chunk
4. **Two-stage screening**: When multiple chunks, combines screened variables and re-runs knockoff voting

**Use case:** When you need exact R SLIDE behavior, especially for:
- Large feature sets (p > 100)
- Reproducibility with R SLIDE results
- Publications requiring R SLIDE methodology

**Trade-off:** More conservative selection (lower sensitivity) but follows R SLIDE exactly.


### `python_voting_glmnet` (Fortran glmnet)

Same as `python_voting` but uses Fortran glmnet instead of sklearn.

```python
# Uses stat_glmnet_coefdiff with Fortran glmnet backend
result = compute_knockoffs_python_voting(X, y, use_sklearn=False, ...)
```

**When to use:**
- When sklearn produces numerical issues
- For exact compatibility with R's glmnet coefficient paths

**Performance:** Slightly lower Jaccard (0.39 avg) than sklearn version (0.53 avg).

---

## Backend Comparison

### Algorithm Comparison

| Backend | Threshold Method | findOptIter | Chunking | Two-Stage |
|---------|-----------------|-------------|----------|-----------|
| `R_native` | spec * niter | Yes | Yes | Yes |
| `python_voting_slide` | spec * niter | Yes | Yes | Yes |
| `python_voting` | spec * niter | No | No | No |
| `python_voting_glmnet` | spec * niter | No | No | No |

### Performance vs R_native (Average over 9 parameter sets)

| Backend | Avg Jaccard | Avg W Corr | Notes |
|---------|-------------|------------|-------|
| **python_voting_slide** | 0.6 - 0.9* | 0.95+ | Closest to R SLIDE |
| **python_voting** | 0.53 | 0.76 | Simpler algorithm |
| python_voting_glmnet | 0.39 | 0.51 | Fortran glmnet |
| knockoff_filter | 0.15 | 0.07 | Single-run (poor) |
| R_knockoffs_py_sklearn | 0.07 | 0.17 | Broken |
| knockoff_filter_sklearn | 0.05 | 0.03 | Single-run (poor) |

*Expected higher Jaccard with findOptIter alignment; depends on data characteristics.

### Performance by n/p Ratio

| Case | n/p | python_voting_slide | python_voting | Notes |
|------|-----|---------------------|---------------|-------|
| n > p | 4.29 | 0.8 - 1.0 | 0.50 - 1.00 | Best agreement |
| n < p | 0.62 | 0.6 - 0.8 | 0.59 - 0.70 | Good agreement |
| n ~ p | 1.26 | 0.3 - 0.5 | 0.17 - 0.28 | Expected divergence |

---

## Single-Run Backends (Debugging/Comparison)

### `knockoff_filter_sklearn`

Single knockoff run using sklearn for statistics.

```python
from loveslide.knockoff.filter import knockoff_filter
from loveslide.knockoff.create import create_second_order
from loveslide.knockoff.stats import stat_glmnet_lambdasmax

result = knockoff_filter(
    X, y,
    knockoffs=create_second_order,
    statistic=stat_glmnet_lambdasmax,
    fdr=0.2,
    offset=0
)
```

**Implementation:**
1. Generate knockoffs using ASDP/SDP solver
2. Compute W statistics using sklearn lasso_path
3. Apply knockoff threshold for selection

**Performance:** Poor (Jaccard ~0.05) - single runs are highly variable.


### `knockoff_filter`

Single knockoff run using Fortran glmnet.

```python
result = compute_knockoffs_knockoff_filter(X, y, use_sklearn=False, ...)
```

**Performance:** Slightly better than sklearn for some cases (Jaccard ~0.15).

---

## Hybrid Backends

### `R_knockoffs_py_sklearn`

Uses R to generate knockoffs, Python (sklearn) to compute statistics.

**Purpose:** Isolate whether divergence comes from knockoff generation or statistic computation.

```python
# R generates knockoffs via create.second_order
# Python computes W via stat_glmnet_lambdasmax (sklearn)
```

**Finding:** Poor performance (Jaccard ~0.07, selects almost nothing) - suggests the hybrid approach loses information in the translation.


### `R_knockoffs_py_stats`

Uses R knockoffs + Python Fortran glmnet statistics.

---

## Reference Backends (R Implementation)

### `R_native`

Reference implementation using R's knockoff package with SLIDE selectShortFreq() methodology.

```r
# R SLIDE's selectShortFreq():
# - Runs knockoff.filter niter times via foreach/doParallel
# - Keeps variables selected in >= spec * niter runs
# - Applies findOptIter() refinement per chunk
# - Two-stage screening for multiple chunks
```

**Requires:** `rpy2`, R with `knockoff`, `SLIDE`, `foreach`, `doParallel` packages.


### `R_native_single`

Single R knockoff run (no voting). For debugging.

---

## Backend Selection Guide

### For Production Use

| Scenario | Recommended Backend |
|----------|---------------------|
| **R SLIDE compatibility** | `python_voting_slide` |
| **Simpler voting (no findOptIter)** | `python_voting` |
| **Numerical issues with sklearn** | `python_voting_glmnet` |
| **R environment available** | `R_native` |

### Decision Tree

```
Need R SLIDE exact match?
    Yes -> Use python_voting_slide with R defaults (niter=1000, spec=0.3)
    No  -> Need simplest approach?
              Yes -> Use python_voting
              No  -> Having sklearn issues?
                        Yes -> Use python_voting_glmnet
                        No  -> Use python_voting
```

### For Debugging

| Scenario | Backend |
|----------|---------|
| Compare single-run behavior | `knockoff_filter_sklearn` vs `R_native_single` |
| Isolate knockoff generation | `R_knockoffs_py_sklearn` |
| Compare statistic computation | `python_voting` vs `python_voting_glmnet` |
| Validate findOptIter | `python_voting` vs `python_voting_slide` |

---

## Key Implementation Details

### Knockoff Generation

All Python backends use `create_second_order()` which:
1. Computes sample covariance matrix (with optional Ledoit-Wolf shrinkage)
2. Solves SDP/ASDP to find optimal `diag_s`
3. Generates knockoffs via Gaussian sampling

```python
from loveslide.knockoff.create import create_second_order
Xk = create_second_order(X)  # Returns knockoff matrix
```

### SDP Caching (Performance Optimization)

The voting backends use SDP caching for 3-4x speedup:

```python
# Pre-computed ONCE (outside voting loop):
# - Covariance matrix (Sigma)
# - SDP solution (diag_s)
# - Cholesky decomposition (L)
# - SigmaInv_s matrix

# Per-iteration (fast):
# - Random knockoff sampling only
# - W statistic computation
```

### W Statistic Computation

```python
from loveslide.knockoff.stats import stat_glmnet_lambdasmax

# use_sklearn=True: sklearn.linear_model.lasso_path
# use_sklearn=False: Fortran glmnet via _vendor/glmnet
W = stat_glmnet_lambdasmax(X, Xk, y, use_sklearn=True)
```

The W statistic measures the "importance" difference between original and knockoff variables:
- `W[j] > 0`: Original variable j is more important
- `W[j] < 0`: Knockoff variable j is more important

### Threshold Computation

```python
from loveslide.knockoff.filter import knockoff_threshold

threshold = knockoff_threshold(W, fdr=0.2, offset=0)
selected = np.where(W >= threshold)[0]
```

### findOptIter Function

```python
from loveslide.knockoff.filter import find_opt_iter

# Find optimal iteration with max overlap and smallest selection
selected_vars, optimal_iter = find_opt_iter(freq_vars, selected_list)
```

---

## File Locations

```
src/loveslide/knockoff/
├── __init__.py          # Package exports
├── create.py            # Knockoff generation (create_second_order, create_gaussian)
├── filter.py            # Main filter pipeline (knockoff_filter, knockoff_filter_voting,
│                        #   knockoff_filter_voting_slide, find_opt_iter)
├── solve.py             # SDP solvers (create_solve_sdp, create_solve_asdp)
├── stats/               # W statistic implementations
│   ├── glmnet.py        # stat_glmnet_lambdasmax, stat_glmnet_coefdiff
│   ├── lasso.py         # stat_lasso_lambdasmax
│   └── ...
├── utils.py             # Utility functions
├── _parallel.py         # Parallel execution helpers
└── _vendor/glmnet/      # Vendored Fortran glmnet bindings
```

---

## Usage Examples

### Basic Usage (Python Voting)

```python
import numpy as np
from loveslide.knockoff.filter import knockoff_filter_voting
from loveslide.knockoff.create import create_second_order
from loveslide.knockoff.stats import stat_glmnet_lambdasmax

# Load data
X = np.loadtxt("z_matrix.csv", delimiter=",", skiprows=1)
y = np.loadtxt("y.csv", delimiter=",", skiprows=1)

# Run knockoff voting (simpler approach)
result = knockoff_filter_voting(
    X, y,
    knockoffs=create_second_order,
    statistic=stat_glmnet_lambdasmax,
    fdr=0.1,
    niter=500,
    spec=0.1,
    base_seed=42
)

print(f"Selected {len(result.selected)} variables: {result.selected}")
print(f"Selection frequency: {result.selection_frequency}")
```

### R SLIDE Compatible Usage

```python
from loveslide.knockoff.filter import knockoff_filter_voting_slide

# Use R SLIDE defaults for exact compatibility
result = knockoff_filter_voting_slide(
    X, y,
    fdr=0.1,
    niter=1000,    # R SLIDE default
    spec=0.3,      # R SLIDE default
    f_size=100,    # R SLIDE default
    match_r=True,  # Skip condition number checks
    base_seed=42
)

print(f"Selected {len(result.selected)} variables: {result.selected}")
if result.optimal_iter is not None:
    print(f"Optimal iteration: {result.optimal_iter}")
```

### With findOptIter Only (No Chunking)

```python
from loveslide.knockoff.filter import knockoff_filter_voting

# Enable findOptIter refinement without chunking
result = knockoff_filter_voting(
    X, y,
    fdr=0.1,
    niter=500,
    spec=0.1,
    slide_selection=True,      # Enable findOptIter
    return_selected_list=True, # Required for findOptIter
    base_seed=42
)

print(f"Selected from optimal iteration {result.optimal_iter}: {result.selected}")
```

---

## Validation

Run the comprehensive backend validation:

```bash
cd comparison
sbatch current_test.sh
```

This tests all backends across 9 parameter sets (3 delta x 3 lambda values) and generates comparison reports.

### Quick Validation Commands

```bash
# Run full validation
python comparison/run_knockoffs_on_precomputed.py run \
    comparison/archive/output_comparison/.../R_native \
    --backend R_native python_voting python_voting_slide \
    --output comparison/full_validation_$(date +%Y%m%d_%H%M%S)

# Compare results
python comparison/run_knockoffs_on_precomputed.py compare \
    comparison/full_validation_*/knockoff_results_*.json \
    --reference R_native \
    --output comparison/full_validation_*/comparison_report
```

---

## Changelog

### 2026-02-01: R SLIDE Alignment
- Added `knockoff_filter_voting_slide()` with full R SLIDE methodology
- Added `find_opt_iter()` function matching R SLIDE's `findOptIter()`
- Added `slide_selection` and `return_selected_list` parameters to `knockoff_filter_voting()`
- Extended `VotingResult` dataclass with `selected_list` and `optimal_iter` fields
- Updated documentation to reflect R SLIDE defaults and algorithm

### 2026-01-31: Performance Optimization
- Added SDP caching for 3-4x speedup in voting backends
- Added `use_cache` parameter (default=True)
- Fixed equicorrelated fallback for SDP failures in n <= p cases
