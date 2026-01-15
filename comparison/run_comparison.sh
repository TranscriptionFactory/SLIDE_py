#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH --job-name=slide_compare
#SBATCH --mail-user=aar126@pitt.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=75G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=comparison_%A_%a.out
#SBATCH --error=comparison_%A_%a.err

# =============================================================================
# SLIDE R vs Python Comparison Script (Array Job Version)
# =============================================================================
# Runs R and Python as independent array tasks so one failure doesn't kill both.
#
# Array tasks:
#   0 = R implementation
#   1 = Python implementation
#
# Usage:
#   sbatch run_comparison.sh <yaml_config>
#   sbatch run_comparison.sh comparison_config.yaml
#
# The YAML config should contain:
#   x_path, y_path, out_path, delta, lambda, spec, fdr, etc.
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR=/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
cd "$SCRIPT_DIR"

# Get YAML config from argument
YAML_CONFIG="${1:-comparison_config.yaml}"

if [ ! -f "$YAML_CONFIG" ]; then
    echo "ERROR: YAML config not found: $YAML_CONFIG"
    exit 1
fi

# Parse output paths from YAML for R and Python
# Using simple grep/sed to extract values
OUT_PATH=$(grep "^out_path:" "$YAML_CONFIG" | sed 's/out_path: *//' | tr -d '"'"'" | xargs)
OUT_PATH="${OUT_PATH:-$SCRIPT_DIR/outputs}"

# Create base output directory
mkdir -p "$OUT_PATH"

# Load modules
module load gcc/12.2.0
module load r/4.4.0
module load python/ondemand-jupyter-python3.11 2>/dev/null || true

# Python environment
PYTHON_ENV="${PYTHON_ENV:-/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python}"

# -----------------------------------------------------------------------------
# Print configuration
# -----------------------------------------------------------------------------
echo "=============================================================="
echo "SLIDE R vs Python Comparison"
echo "=============================================================="
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo ""
echo "YAML config: $YAML_CONFIG"
echo "Output path: $OUT_PATH"
echo "=============================================================="

# -----------------------------------------------------------------------------
# Run based on array task ID
# -----------------------------------------------------------------------------
START_TIME=$(date +%s)

if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ]; then
    # =========================================================================
    # Task 0: R Implementation
    # =========================================================================
    echo ""
    echo "=============================================================="
    echo "Running R SLIDE Implementation"
    echo "=============================================================="

    R_OUT="${OUT_PATH}/R_outputs"
    mkdir -p "$R_OUT"

    Rscript run_slide_R.R "$YAML_CONFIG" "$R_OUT"

    # Mark R as complete
    touch "${OUT_PATH}/.r_complete"
    echo "R outputs saved to: $R_OUT"

elif [ "${SLURM_ARRAY_TASK_ID:-1}" -eq 1 ]; then
    # =========================================================================
    # Task 1: Python Implementation
    # =========================================================================
    echo ""
    echo "=============================================================="
    echo "Running Python SLIDE Implementation"
    echo "=============================================================="

    PY_OUT="${OUT_PATH}/Py_outputs"
    mkdir -p "$PY_OUT"

    "$PYTHON_ENV" run_slide_py.py "$YAML_CONFIG" "$PY_OUT"

    # Mark Python as complete
    touch "${OUT_PATH}/.py_complete"
    echo "Python outputs saved to: $PY_OUT"

fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================================="
echo "Task ${SLURM_ARRAY_TASK_ID:-0} completed in ${ELAPSED} seconds"
echo "=============================================================="
