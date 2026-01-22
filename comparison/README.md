# SLIDE R vs Python Comparison

Scripts for comparing R and Python SLIDE implementations.

## Quick Start

```bash
cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison

# Run full R vs Python comparison
sbatch run_comparison.sh comparison_config.yaml

# Or run knockoff backend comparison on pre-computed LOVE results
sbatch submit_knockoffs_precomputed.sh /path/to/love_results
```

## Scripts

### Comparison Runners

| Script | Description |
|--------|-------------|
| `run_comparison.sh` | SLURM job: runs R and Python SLIDE, then compares |
| `run_knockoff_comparison.py` | Two-phase knockoff comparison (LOVE → knockoffs) |
| `run_knockoffs_on_precomputed.py` | Run knockoffs on existing Z matrices |

### Individual Runners

| Script | Description |
|--------|-------------|
| `run_slide_py.py` | Run Python SLIDE with CLI args |
| `run_slide_R.R` | Run R SLIDE with CLI args |
| `extract_r_performance.R` | Extract R performance metrics |

### Analysis

| Script | Description |
|--------|-------------|
| `compare_outputs.py` | Numerical validation (element-wise) |
| `compare_full.py` | Full pipeline comparison |
| `compare_latent_factors.py` | Semantic latent factor matching |

## Configuration

Edit `comparison_config.yaml`:

```yaml
x_path: /path/to/X.csv
y_path: /path/to/Y.csv
tag: my_experiment
delta: 0.1
lambda: 0.5
spec: 0.1
fdr: 0.1
niter: 500
```

## Archive

Historical outputs and analysis documents are in `archive/`.
