# SLIDE R vs Python Comparison

This directory contains scripts to compare the R and Python implementations of SLIDE end-to-end.

## Overview

The comparison framework tests three main stages of the SLIDE pipeline:

1. **Latent Factor Estimation (LOVE/getLatentFactors)**: Estimates K, A, C, Gamma, and pure variable indices
2. **Z Matrix Calculation**: Computes latent factor scores for each sample
3. **SLIDE Knockoffs**: Identifies significant marginal and interaction latent factors

## Prerequisites

### R Requirements
- R 4.0+
- Packages: `foreach`, `doParallel`, `MASS`

Install R packages if needed:
```r
install.packages(c("foreach", "doParallel", "MASS"))
```

### Python Requirements
- Python 3.8+
- loveslide package (this repository)

Install from the repository root:
```bash
pip install -e .
```

Or install dependencies directly:
```bash
pip install numpy scipy pandas networkx tqdm
```

## Running the Comparison

### Quick Start (Recommended)

The easiest way is to use the unified `run_comparison.sh` script:

```bash
cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison

# Option 1: Edit defaults in run_comparison.sh, then run
sbatch run_comparison.sh

# Option 2: Use a YAML config file
cp comparison_config.yaml my_config.yaml
# Edit my_config.yaml with your paths/parameters
sbatch run_comparison.sh my_config.yaml

# Option 3: Set environment variables
X_FILE=/path/to/X.csv Y_FILE=/path/to/Y.csv TAG=my_test sbatch run_comparison.sh
```

This will:
1. Run R SLIDE with your parameters
2. Run Python SLIDE with identical parameters
3. Generate a detailed comparison report

### Using the YAML Config

Create a config file from the template:

```yaml
# my_config.yaml
x_path: /path/to/X.csv
y_path: /path/to/Y.csv
tag: my_experiment
delta: 0.1
lambda: 0.5
spec: 0.1
fdr: 0.1
niter: 500
```

Then run:
```bash
sbatch run_comparison.sh my_config.yaml
```

### Manual Step-by-Step (Alternative)

If you prefer to run each step separately:

#### Step 1: Prepare Test Data

Your data should be in CSV format:
- **X matrix**: Features as columns, samples as rows (with row names)
- **Y vector**: Single column, samples as rows (with row names)

#### Step 2: Run R Implementation

```bash
Rscript run_slide_R.R /path/to/X.csv /path/to/Y.csv my_experiment \
    --delta 0.1 \
    --lambda 0.5 \
    --spec 0.1 \
    --fdr 0.1 \
    --niter 500
```

#### Step 3: Run Python Implementation

```bash
python run_slide_py.py /path/to/X.csv /path/to/Y.csv my_experiment \
    --delta 0.1 \
    --lambda 0.5 \
    --spec 0.1 \
    --fdr 0.1 \
    --niter 500
```

#### Step 4: Generate Comparison Report

```bash
python compare_outputs.py my_experiment --tolerance 0.01 --detailed \
    --output-file outputs/my_experiment/comparison_report.txt
```

## Output Files

After running both scripts, `outputs/<tag>/` will contain:

### From R (`*` suffix):
| File | Description |
|------|-------------|
| `<tag>_A.csv` | Loading/membership matrix (p × K) |
| `<tag>_C.csv` | Latent factor covariance matrix (K × K) |
| `<tag>_Gamma.csv` | Error variance vector (p × 1) |
| `<tag>_I.csv` | Pure variable indices |
| `<tag>_Z.csv` | Latent factor scores (n × K) |
| `<tag>_params.csv` | K, opt_delta, opt_lambda |
| `<tag>_marginal_LFs.csv` | Significant marginal LFs |
| `<tag>_interactions.csv` | Interaction pairs (p1, p2) |
| `<tag>_AllLatentFactors.rds` | Full R result object |

### From Python (`*_py` suffix):
| File | Description |
|------|-------------|
| `<tag>_A_py.csv` | Loading/membership matrix |
| `<tag>_C_py.csv` | Latent factor covariance matrix |
| `<tag>_Gamma_py.csv` | Error variance vector |
| `<tag>_I_py.csv` | Pure variable indices (1-indexed for comparison) |
| `<tag>_Z_py.csv` | Latent factor scores |
| `<tag>_params_py.csv` | K, opt_delta |
| `<tag>_marginal_LFs_py.csv` | Significant marginal LFs |
| `<tag>_interactions_py.csv` | Interaction pairs |

## Comparison Tests

| Test | Metric | Notes |
|------|--------|-------|
| K (# of LFs) | Exact match | Critical |
| A matrix | Element-wise (tol=0.1) | Sign/permutation may differ |
| C matrix | Element-wise (tol=0.1) | |
| Gamma | Element-wise (tol=0.1) | |
| Pure variables | Jaccard similarity | Index differences expected |
| Z matrix | Element-wise (tol=0.1) | |
| Marginal LFs | Set comparison | May differ due to knockoff randomness |
| Interactions | Set comparison | May differ due to knockoff randomness |

## Expected Differences

Due to implementation differences, some variations are expected:

1. **Random Number Generators**: R and Python have different RNG implementations
2. **Numerical Precision**: Floating-point differences in matrix operations
3. **Knockoff Randomness**: The knockoff procedure is inherently stochastic
4. **LP Solver Differences**: Different LP solvers may produce slightly different results

### Acceptable Tolerances

- **K**: Should match exactly (critical)
- **Matrices (A, C, Z)**: Correlation > 0.95, max diff < 0.1
- **Gamma**: max diff < 0.1
- **Pure variables**: Jaccard > 0.8
- **Marginals/Interactions**: Jaccard > 0.6 (highly variable due to randomness)

## Troubleshooting

### R script fails to find source files

Make sure the SLIDE R package is available at:
```
/ix/djishnu/Aaron/1_general_use/SLIDE/R/
```

### Python import errors

Install the loveslide package:
```bash
cd /ix/djishnu/Aaron/1_general_use/SLIDE_py
pip install -e .
```

### Large differences in results

1. Check that both use the same input data
2. Verify parameter values match
3. Try running with `--niter 1000` for more stable knockoff results
4. Check if data meets SLIDE requirements (n > p)

## Example Workflow

```bash
# Navigate to comparison directory
cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison

# Run with example data (SSc dataset)
DATA_DIR="/ix/djishnu/Aaron/1_general_use/SLIDE_py/tests/data"

# Run R
Rscript run_slide_R.R \
    $DATA_DIR/test_X.csv \
    $DATA_DIR/test_Y.csv \
    test_comparison \
    --delta 0.1 \
    --lambda 0.5

# Run Python and compare
python run_slide_py.py \
    $DATA_DIR/test_X.csv \
    $DATA_DIR/test_Y.csv \
    test_comparison \
    --delta 0.1 \
    --lambda 0.5

# Generate detailed report
python compare_outputs.py test_comparison --detailed --output-file test_report.txt
```

## References

- SLIDE R package: `/ix/djishnu/Aaron/1_general_use/SLIDE/`
- LOVE comparison: `/ix/djishnu/Aaron/1_general_use/LOVE/comparison/`
- Original paper: [Essential Regression](https://arxiv.org/abs/2004.07955)
