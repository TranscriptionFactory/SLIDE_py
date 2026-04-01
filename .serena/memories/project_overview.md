# SLIDE_py Project Overview

## Purpose
**loveslide** is a Python wrapper for the SLIDE framework (Statistical Latent Inference for Discovery and Explanation) that combines:
- **LOVE**: A latent factor discovery algorithm using model-based overlapping clustering
- **Knockoffs**: For statistically rigorous identification of significant standalone and interacting latent factors

The package wraps R implementations into a user-friendly Python interface for machine learning pipelines and bioinformatics workflows.

## Key Features
- Latent factor discovery in high-dimensional data
- Statistical inference with FDR control
- R integration via rpy2 for core statistical computations
- Python orchestration layer for ML workflow integration
- Modular, extensible architecture
- Both CLI and programmatic interfaces

## Main Pipeline Stages
1. **Latent Factor Discovery**: LOVE algorithm identifies overlapping latent factors
2. **Statistical Inference**: Knockoffs identify significant factors with FDR control  
3. **Visualization**: Diagnostic plots and feature importance visualization

## Target Users
- Bioinformatics researchers
- Machine learning practitioners working with high-dimensional data
- Data scientists needing rigorous statistical inference