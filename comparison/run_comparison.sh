#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH --job-name=slide_compare
#SBATCH --mail-user=aar126@pitt.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=75G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --output=logs/comparison_%A_%a.out
#SBATCH --error=logs/comparison_%A_%a.err

# =============================================================================
# SLIDE 3-Way Comparison Script (Array Job Version)
# =============================================================================
# Compares LOVE implementations across R and Python backends.
#
# Array tasks:
#   0 = R SLIDE (native R package - R LOVE + R knockoffs)
#   1 = Python SLIDE with R LOVE backend
#   2 = Python SLIDE with Python LOVE backend
#
# Usage:
#   sbatch run_comparison.sh <yaml_config>
#   sbatch run_comparison.sh comparison_config.yaml
#
# The YAML config should contain:
#   x_path, y_path, out_path, delta, lambda, spec, fdr, etc.
# =============================================================================

mkdir -p logs/
set -e  # Exit on error

SCRIPT_DIR=/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
cd "$SCRIPT_DIR"

# Get YAML config from argument
YAML_CONFIG="${1:-comparison_config.yaml}"

if [ ! -f "$YAML_CONFIG" ]; then
    echo "ERROR: YAML config not found: $YAML_CONFIG"
    exit 1
fi

# Parse output paths from YAML
OUT_PATH=$(grep "^out_path:" "$YAML_CONFIG" | sed 's/out_path: *//' | tr -d '"'"'" | xargs)
OUT_PATH="${OUT_PATH:-$SCRIPT_DIR/outputs}"

# Create base output directory
mkdir -p "$OUT_PATH"

# Load modules
module load gcc/12.2.0
module load python/ondemand-jupyter-python3.11
module load r/4.4.0  # Needed for all tasks (R native, and rpy2 for Python)

# Python environment
PYTHON_ENV="${PYTHON_ENV:-/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python}"

# -----------------------------------------------------------------------------
# Print configuration
# -----------------------------------------------------------------------------
TASK_NAMES=("R_native" "Py_R_LOVE" "Py_Py_LOVE")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "=============================================================="
echo "SLIDE 3-Way Comparison"
echo "=============================================================="
echo "Task ID: ${TASK_ID} (${TASK_NAMES[$TASK_ID]})"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo ""
echo "YAML config: $YAML_CONFIG"
echo "Output path: $OUT_PATH"
echo "=============================================================="

# -----------------------------------------------------------------------------
# Run based on array task ID
# -----------------------------------------------------------------------------
START_TIME=$(date +%s)

if [ "$TASK_ID" -eq 0 ]; then
    # =========================================================================
    # Task 0: R SLIDE (Native R package)
    # =========================================================================
    echo ""
    echo "=============================================================="
    echo "Running R SLIDE (Native R Implementation)"
    echo "=============================================================="

    R_OUT="${OUT_PATH}/R_native"
    mkdir -p "$R_OUT"

    Rscript run_slide_R.R "$YAML_CONFIG" "$R_OUT"

    touch "${OUT_PATH}/.r_native_complete"
    echo "R native outputs saved to: $R_OUT"

elif [ "$TASK_ID" -eq 1 ]; then
    # =========================================================================
    # Task 1: Python SLIDE with R LOVE backend
    # =========================================================================
    echo ""
    echo "=============================================================="
    echo "Running Python SLIDE with R LOVE Backend"
    echo "=============================================================="

    PY_R_OUT="${OUT_PATH}/Py_R_LOVE"
    mkdir -p "$PY_R_OUT"

    "$PYTHON_ENV" run_slide_py.py "$YAML_CONFIG" "$PY_R_OUT" --love-backend r

    touch "${OUT_PATH}/.py_r_love_complete"
    echo "Python (R LOVE) outputs saved to: $PY_R_OUT"

elif [ "$TASK_ID" -eq 2 ]; then
    # =========================================================================
    # Task 2: Python SLIDE with Python LOVE backend
    # =========================================================================
    echo ""
    echo "=============================================================="
    echo "Running Python SLIDE with Python LOVE Backend"
    echo "=============================================================="

    PY_PY_OUT="${OUT_PATH}/Py_Py_LOVE"
    mkdir -p "$PY_PY_OUT"

    "$PYTHON_ENV" run_slide_py.py "$YAML_CONFIG" "$PY_PY_OUT" --love-backend python

    touch "${OUT_PATH}/.py_py_love_complete"
    echo "Python (Python LOVE) outputs saved to: $PY_PY_OUT"

fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================================="
echo "Task ${TASK_ID} (${TASK_NAMES[$TASK_ID]}) completed in ${ELAPSED} seconds"
echo "=============================================================="
