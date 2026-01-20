#!/bin/bash
#SBATCH --job-name=w_stat_compare
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/w_stat_compare_%j.out
#SBATCH --error=logs/w_stat_compare_%j.err

# W-Statistic Comparison: knockpy vs knockoff-filter vs R native
#
# This script compares W-statistics across all three knockoff backends
# to identify the source of divergence in LF selection.

set -e

# Setup environment
module load gcc/12.2.0
module load r/4.4.0
module load python/ondemand-jupyter-python3.11
source activate loveslide_env

# Set R library path for rpy2
export R_HOME=/software/rhel9/manual/install/r/4.4.0/lib64/R
export LD_LIBRARY_PATH="$R_HOME/lib:$LD_LIBRARY_PATH"

cd /ix/djishnu/Aaron/1_general_use/SLIDE_py

# Add knockoff-filter to path
export PYTHONPATH="/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter:$PYTHONPATH"

# Create output directory
OUTDIR="comparison/diagnostics/output/w_stat_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
mkdir -p logs

echo "Output directory: $OUTDIR"

# Use z_matrix from a specific comparison run (delta=0.1, lambda=0.5)
Z_PATH="/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/output_comparison/20260119_225506_rstyle/0.1_0.5_out/z_matrix.csv"
Y_PATH="/ix/djishnu/Aaron/0_for_others/Crystal/SLIDE/Scleroderma_Control/Scleroderma_Control_y.csv"

echo "Z matrix: $Z_PATH"
echo "Y vector: $Y_PATH"

# Run comparison
# Use sdp method to match R's knockoff package
python -u comparison/diagnostics/compare_w_statistics_all.py \
    --z-path "$Z_PATH" \
    --y-path "$Y_PATH" \
    --output-dir "$OUTDIR" \
    --fdr 0.1 \
    --method sdp \
    --seed 42 \
    --standardize

echo "Results saved to: $OUTDIR"
