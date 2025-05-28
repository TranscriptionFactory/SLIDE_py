# SLIDE_py

A bunch of python wrappers for R code

## Overview

SLIDE combines the LOVE (Latent Model-Based Clustering for Biological Discovery) clustering algorithm with knockoff-based statistical inference to identify significant standalone and interacting latent factors. 

## Quick Start

### Basic Usage
##### From command line 
Use the full path if you are not calling slide.py from the same directory

```bash
python slide.py \
    --x_path /path/to/your/features.csv \
    --y_path /path/to/your/labels.csv \
    --out_path /path/to/output/directory
```

##### In a notebook
```python
import sys
sys.path.append('src/SLIDE')

from slide import OptimizeSLIDE

# Configure input parameters
input_params = {
    'x_path': '/path/to/your/features.csv',
    'y_path': '/path/to/your/labels.csv',
    'fdr': 0.1,
    'thresh_fdr': 0.1,
    'spec': 0.2,
    'y_factor': True,
    'niter': 500,
    'SLIDE_top_feats': 20,
    'rep_CV': 50,
    'pure_homo': True,
    'delta': [0.01],
    'lambda': [0.5, 0.1],
    'out_path': '/path/to/output/directory'
}

# Initialize and run SLIDE
slider = OptimizeSLIDE(input_params)
slider.run_pipeline(verbose=True, n_workers=1)
```

## Pipeline Overview

The `run_pipeline()` has three main parts:

### Stage 1: Latent Factor Discovery
- **LOVE Algorithm**: Runs the overlapping clustering algorithm to identify latent factors
- **Output**: Generates the latent factors (z_matrix) representing underlying data structure

### Stage 2: Statistical Inference with SLIDE
- **2a) Standalone Factor Analysis**: Uses knockoffs to identify statistically significant standalone latent factors
- **2b) Interaction Analysis**: Applies knockoffs to discover significant interacting latent factor pairs
- **Feature Selection**: Controls false discovery rate (FDR) while maintaining statistical power

### Stage 3: Visualization
- **Control Plots**: Generates diagnostic plots to assess model performance and statistical validity
- **Latent Factor Genes**: For each latent factor, plots the top features with loadings > abs(0.05)

## Parameter Configuration

| Parameter | Type | Description | Default/Example |
|-----------|------|-------------|-----------------|
| `x_path` | str | Path to feature matrix CSV file | Required |
| `y_path` | str | Path to response labels CSV file | Required |
| `fdr` | float | False discovery rate threshold (Knockoffs) | 0.1 |
| `thresh_fdr` | float | FDR threshold for feature selection (LOVE) | 0.1 |
| `spec` | float | minimum % times an LF found to be significant in order to be included | 0.2 |
| `y_factor` | bool | Treat response as factor variable | True |
| `niter` | int | Number of iterations | 500 |
| `SLIDE_top_feats` | int | Number of top features to display | 20 |
| `pure_homo` | bool | Use homogeneous loadings for pure variables | True |
| `delta` | list | Regularization parameter(s) | [0.5, 0.1] |
| `lambda` | list | Penalty parameter(s) | [0.1] |
| `out_path` | str | Output directory path | Required |

### Advanced Configuration

- **`pure_homo=True`**: Forces pure variable loadings to be 1 (recommended)
- **`pure_homo=False`**: Relaxes the pure variable loading constraint being 1 without losing any guarantees. However, it is difficult to find the right delta parameter
- **`n_workers`**: Controls parallelization (1 for sequential processing), but CURRENTLY NOTHING IS PARALLELIZED
- **`verbose`**: Enables detailed progress reporting (just a bunch of print statements)

## Project Structure

```
SLIDE_py/
├── src/
│   ├── SLIDE/              # Core SLIDE implementation
│   │   ├── slide.py        # Main Python interface
│   │   └── ...            # Supporting R functions
│   └── LOVE-master/        # Original LOVE algorithm
│       ├── ...            # Original LOVE code (do not use)
│       ├── ...            # pure_homo LOVE code (use carefully)
|   └── LOVE-SLIDE/        # SLIDE implementation of LOVE
```

## Implementation Details

### LOVE Algorithm Integration
- **Primary Implementation**: Located in `src/SLIDE/get_Latent_Factors.R`
- **Alternative Version**: Available in `LOVE-master` when `pure_homo=False`
- **Note**: The original LOVE code in `LOVE-master` may yield different results than the SLIDE implementation and is provided for reference


## To-do list

These files
- **Yaml conversion**: Since people already have pipelines set up, it would be convenient to have a function to read yamls into dictionaries
- **Other y_factor**: Currently only binary y is accomodated. 
- **Parallelization**: Knockoffs can be made much faster. Please see `select_short_freq` in `src/SLIDE/knockoffs.py`. I was trying to use concurrent futures/ pqdm but I couldn't figure out the errors and gave up. 
- **Correlation networks**: I think networkx can make similar graph-like figures, but I'm not familiar with making them

