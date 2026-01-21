#!/bin/bash
#SBATCH --job-name=knockoff_pre
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/knockoff_pre_%A_%a.out
#SBATCH --error=logs/knockoff_pre_%A_%a.err

# =============================================================================
# Run knockoffs on pre-computed LOVE Z matrices
# =============================================================================
# Usage:
#   sbatch submit_knockoffs_precomputed.sh <love_dir> [backend]
#
# Example:
#   sbatch submit_knockoffs_precomputed.sh \
#     /ix/djishnu/Aaron/.../R_native \
#     R_native
#
# For multiple backends, use array mode:
#   sbatch --array=0-5 submit_knockoffs_precomputed.sh <love_dir>
#   sbatch --array=0-5 submit_knockoffs_precomputed.sh /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/output_comparison/2026-01-17_04-55-31/SSc_binary_comparison/R_native
#
#
# after

# module load gcc/12.2.0
# module load python/ondemand-jupyter-python3.11
# module load r/4.4.0
# source activate loveslide_env
# python run_knockoffs_on_precomputed.py compare \
#       output_knockoffs/SSc_binary_comparison/R_native/*/knockoff_results_*.json \
#       -o output_knockoffs/SSc_binary_comparison/R_native/knockoff_comparison.txt


# =============================================================================

set -e

cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
mkdir -p logs

# Configuration
LOVE_DIR="${1:?Error: Please provide LOVE directory as first argument}"
BACKEND="${2:-}"  # Optional: specific backend, or use array

# Available backends (indexed for array jobs)
# =============================================================================
# PRIMARY (recommended for comparison) - indices 0-2:
#   0: R_native              - Pure R baseline (reference)
#   1: R_knockoffs_py_sklearn - R knockoffs + sklearn stats (~0.65 corr with R)
#   2: knockoff_filter_sklearn - Pure Python (~0.35 corr, statistically equivalent)
#
# SECONDARY (detailed analysis) - indices 3-5:
#   3: R_knockoffs_py_stats  - R knockoffs + Fortran glmnet
#   4: knockoff_filter       - Python knockoffs + Fortran glmnet
#   5: knockpy_lasso         - knockpy with lasso statistic
#
# Usage:
#   sbatch --array=0-2 submit_knockoffs_precomputed.sh <love_dir>  # Primary only
#   sbatch --array=0-5 submit_knockoffs_precomputed.sh <love_dir>  # All
# =============================================================================
BACKENDS=(
    "R_native"                  # 0: Pure R (baseline reference)
    "R_knockoffs_py_sklearn"    # 1: R knockoffs + sklearn (best hybrid)
    "knockoff_filter_sklearn"   # 2: Pure Python (recommended)
    "R_knockoffs_py_stats"      # 3: R knockoffs + Fortran glmnet
    "knockoff_filter"           # 4: Python knockoffs + Fortran glmnet
    "knockpy_lasso"             # 5: knockpy with lasso statistic
)

# Determine backend
if [ -n "$BACKEND" ]; then
    # Use specified backend
    SELECTED_BACKEND="$BACKEND"
elif [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Array job mode
    SELECTED_BACKEND="${BACKENDS[$SLURM_ARRAY_TASK_ID]}"
else
    # Default to R_native
    SELECTED_BACKEND="R_native"
fi

# Create output directory
LOVE_BASENAME=$(basename "$LOVE_DIR")
LOVE_PARENT=$(basename "$(dirname "$LOVE_DIR")")
OUTPUT_DIR="output_knockoffs/${LOVE_PARENT}/${LOVE_BASENAME}/${SELECTED_BACKEND}"

echo "=============================================================="
echo "Knockoff on Pre-computed LOVE"
echo "=============================================================="
echo "LOVE dir: $LOVE_DIR"
echo "Backend:  $SELECTED_BACKEND"
echo "Output:   $OUTPUT_DIR"
echo "Time:     $(date)"
echo "=============================================================="

# Load modules
module load gcc/12.2.0
module load python/ondemand-jupyter-python3.11
module load r/4.4.0

export PYTHONPATH="/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter:$PYTHONPATH"
export PYTHONPATH="/ix/djishnu/Aaron/1_general_use/SLIDE_py/src:$PYTHONPATH"

# Run
/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python \
    run_knockoffs_on_precomputed.py run \
    --love-dir "$LOVE_DIR" \
    --backend "$SELECTED_BACKEND" \
    --output-dir "$OUTPUT_DIR" \
    --seed 42

echo ""
echo "=============================================================="
echo "Done: $(date)"
echo "=============================================================="
