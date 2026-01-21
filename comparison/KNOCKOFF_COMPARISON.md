# Knockoff Backend Comparison Framework

This document describes the framework for comparing knockoff filter implementations between R and Python.

## Overview

The knockoff filter procedure has two main components:
1. **Knockoff Generation** - Creating knockoff variables that mimic the correlation structure
2. **W-Statistic Computation** - Computing feature importance statistics using lasso/glmnet

Different implementations can vary in either component, leading to different selections. This framework isolates these differences to understand where R and Python diverge.

## Available Backends

### Primary Backends (Recommended)

| Backend | Knockoff Gen | W-Statistic | Expected R Correlation | Use Case |
|---------|-------------|-------------|------------------------|----------|
| `R_native` | R (`create.second_order`) | R glmnet | 1.0 (reference) | Baseline |
| `R_knockoffs_py_sklearn` | R | sklearn `lasso_path` | ~0.65 | Isolate glmnet difference |
| `knockoff_filter_sklearn` | Python (ASDP) | sklearn `lasso_path` | ~0.35 | Pure Python |

### Secondary Backends

| Backend | Knockoff Gen | W-Statistic | Notes |
|---------|-------------|-------------|-------|
| `R_knockoffs_py_stats` | R | Fortran glmnet | May have sign correlation issues |
| `knockoff_filter` | Python | Fortran glmnet | May have sign correlation issues |
| `knockpy_lasso` | Python | knockpy lasso | Alternative Python package |
| `knockpy_lsm` | Python | LSM (marginal) | Different statistic entirely |
| `custom_glmnet` | Python | sklearn custom | SLIDE's implementation |

## Key Findings

### Sources of Divergence

1. **Knockoff Generation (~0.30 correlation loss)**
   - R and Python use different random number generators
   - Even with same seed, different random sequences are produced
   - ASDP solver finds slightly different solutions (same objective value)

2. **Glmnet Implementation (~0.35 correlation loss)**
   - R's glmnet vs sklearn's `lasso_path`
   - Different tie-breaking behavior
   - Different convergence criteria

### Statistical Equivalence

Despite numerical differences, the Python implementation is **statistically equivalent** to R:
- Same ASDP algorithm and objective
- Same s-vector sum (knockoff "power")
- Same W-statistic formula (signed max lambda)
- Valid FDR control guarantees

## Usage

### Running Comparisons

```bash
cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison

# Run primary backends on pre-computed LOVE results
sbatch --array=0-2 submit_knockoffs_precomputed.sh \
    /path/to/love_results/R_native

# Or run a single backend
sbatch submit_knockoffs_precomputed.sh \
    /path/to/love_results/R_native \
    R_knockoffs_py_sklearn
```

### Comparing Results

```bash
# Load required modules
module load gcc/12.2.0 python/ondemand-jupyter-python3.11 r/4.4.0
export PYTHONPATH="/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter:$PYTHONPATH"

# Run comparison
python run_knockoffs_on_precomputed.py compare \
    output_knockoffs/*/R_native/knockoff_results_R_native.json \
    output_knockoffs/*/R_knockoffs_py_sklearn/knockoff_results_R_knockoffs_py_sklearn.json \
    output_knockoffs/*/knockoff_filter_sklearn/knockoff_results_knockoff_filter_sklearn.json \
    -o comparison_report.txt
```

### Quick Test (Single Parameter Set)

```bash
python run_knockoffs_on_precomputed.py run \
    --love-dir /path/to/love_results/R_native \
    --backend R_knockoffs_py_sklearn \
    --output-dir output_knockoffs/test \
    --params "0.1_0.1"
```

## Interpreting Results

### Comparison Report Metrics

1. **W Correlation**: Pearson correlation of W-statistics between backends
   - >0.8: Excellent agreement
   - 0.5-0.8: Good agreement (expected for R vs Python)
   - <0.5: Significant divergence

2. **Jaccard Index**: Selection overlap
   - 1.0: Identical selections
   - 0.5-1.0: Substantial overlap
   - <0.5: Different selections (but may still be valid)

3. **Selection Counts**: Number of variables selected
   - Should be similar across backends
   - Large differences suggest threshold/FDR computation issues

### Expected Results

| Comparison | W Corr | Jaccard | Notes |
|------------|--------|---------|-------|
| R_native vs R_knockoffs_py_sklearn | ~0.65 | ~0.5-0.8 | Glmnet difference only |
| R_knockoffs_py_sklearn vs knockoff_filter_sklearn | ~0.55 | ~0.4-0.7 | Knockoff gen difference |
| R_native vs knockoff_filter_sklearn | ~0.35 | ~0.3-0.5 | Total difference |

## Recommendations

### For Production Use

1. **If R compatibility is critical**: Use `R_knockoffs_py_sklearn`
   - Uses R's knockoff generation for consistency
   - Python statistics for speed/integration

2. **If pure Python is preferred**: Use `knockoff_filter_sklearn`
   - Statistically equivalent to R
   - Faster, no R dependency
   - Accept moderate correlation with R results

3. **For validation**: Run both and compare
   - If selections overlap substantially, either is valid
   - If selections differ, investigate specific variables

### For Comparison Studies

Run all three primary backends and report:
1. Which variables are selected by all backends (high confidence)
2. Which variables are selected by some backends (moderate confidence)
3. W-statistic rankings (more stable than binary selections)

## File Structure

```
comparison/
├── run_knockoffs_on_precomputed.py   # Main script
├── submit_knockoffs_precomputed.sh   # SLURM submission
├── r_rng.py                          # R RNG replication (experimental)
├── KNOCKOFF_COMPARISON.md            # This documentation
└── output_knockoffs/                 # Results directory
    └── <dataset>/
        └── <love_backend>/
            ├── R_native/
            │   └── knockoff_results_R_native.json
            ├── R_knockoffs_py_sklearn/
            │   └── knockoff_results_R_knockoffs_py_sklearn.json
            └── knockoff_filter_sklearn/
                └── knockoff_results_knockoff_filter_sklearn.json
```

## Technical Details

### Knockoff Generation Algorithm

Both R and Python use the same algorithm:
```
1. Estimate covariance: Σ = cov(Z)
2. Solve SDP for s-vector: maximize sum(s) subject to 2Σ - diag(s) ≥ 0
3. Compute knockoff covariance: Σ_k = 2*diag(s) - diag(s) @ inv(Σ) @ diag(s)
4. Sample knockoffs: Z_k = μ_k + N(0,I) @ chol(Σ_k)
```

Step 4 is the only random component. Different RNGs produce different knockoffs.

### W-Statistic Computation

```
1. Randomly swap Z and Z_k columns (coin flip per variable)
2. Run lasso on [Z_swap, Z_k_swap] vs y
3. Record lambda at which each variable enters
4. W_j = max(λ_j, λ_{j+p}) * sign(λ_j - λ_{j+p})
5. Correct for swapping
```

The random swapping (step 1) and lasso path (step 2) can differ between implementations.

## Changelog

- 2026-01-21: Initial framework with R/Python comparison backends
- 2026-01-21: Added hybrid `R_knockoffs_py_sklearn` backend
- 2026-01-21: Documented statistical equivalence findings
