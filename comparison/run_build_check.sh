#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH --job-name=slide_build_check
#SBATCH --mail-user=aar126@pitt.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-3
#SBATCH --output=logs/build_check_%A_%a.out
#SBATCH --error=logs/build_check_%A_%a.err

# =============================================================================
# SLIDE Build Check - 4-Way Knockoff Backend Validation
# =============================================================================
# Automated build validation comparing Python SLIDE knockoff backends against
# R native baseline.
#
# Array tasks:
#   0 = R SLIDE (native R package - R LOVE + R knockoffs) - BASELINE
#   1 = Python SLIDE (Python LOVE + R Knockoffs)
#   2 = Python SLIDE (Python LOVE + knockoff-filter with glmnet)
#   3 = Python SLIDE (Python LOVE + knockoff-filter with sklearn)
#
# Usage:
#   sbatch run_build_check.sh [yaml_config] [output_path]
#   sbatch run_build_check.sh                                    # uses build_check_config.yaml
#   sbatch run_build_check.sh build_check_config.yaml           # explicit config
#   sbatch run_build_check.sh build_check_config.yaml /my/path  # override output
#
# Output structure:
#   <output_path>/
#     <timestamp>/
#       R_native/           - Baseline R implementation
#       Py_pyLOVE_rKO/      - Python with R knockoffs
#       Py_pyLOVE_kf_glmnet/ - Python with knockoff-filter (glmnet)
#       Py_pyLOVE_kf_sklearn/ - Python with knockoff-filter (sklearn)
#       .task*_complete     - Task completion markers
# =============================================================================

mkdir -p logs/
set -e  # Exit on error

SCRIPT_DIR=/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
cd "$SCRIPT_DIR"

# Get YAML config from argument
YAML_CONFIG="${1:-build_check_config.yaml}"
OUT_PATH_OVERRIDE="${2:-}"  # Optional output path override

if [ ! -f "$YAML_CONFIG" ]; then
    echo "ERROR: YAML config not found: $YAML_CONFIG"
    exit 1
fi

# Use override if provided, otherwise parse from YAML
if [ -n "$OUT_PATH_OVERRIDE" ]; then
    OUT_PATH="$OUT_PATH_OVERRIDE"
else
    OUT_PATH=$(grep "^out_path:" "$YAML_CONFIG" | sed 's/out_path: *//' | tr -d '"'"'" | xargs)
    OUT_PATH="${OUT_PATH:-$SCRIPT_DIR/build_check_outputs}"  # fallback
fi

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_PATH="${OUT_PATH}/${TIMESTAMP}"
mkdir -p "$OUT_PATH"

# Load modules
module load gcc/12.2.0
module load python/ondemand-jupyter-python3.11
module load r/4.4.0  # Needed for all tasks

# Python environment
PYTHON_ENV="${PYTHON_ENV:-/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python}"

# -----------------------------------------------------------------------------
# Task configuration
# -----------------------------------------------------------------------------
TASK_NAMES=("R_native" "Py_pyLOVE_rKO" "Py_pyLOVE_kf_glmnet" "Py_pyLOVE_kf_sklearn")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "=============================================================="
echo "SLIDE Build Check - Knockoff Backend Validation"
echo "=============================================================="
echo "Task ID: ${TASK_ID} (${TASK_NAMES[$TASK_ID]})"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo ""
echo "YAML config: $YAML_CONFIG"
echo "Output path: $OUT_PATH"
echo "Timestamp: $TIMESTAMP"
echo "=============================================================="

# -----------------------------------------------------------------------------
# Run based on array task ID
# -----------------------------------------------------------------------------
START_TIME=$(date +%s)

if [ "$TASK_ID" -eq 0 ]; then
    # =========================================================================
    # Task 0: R SLIDE (Native R package - BASELINE)
    # =========================================================================
    echo ""
    echo "=============================================================="
    echo "Running R SLIDE (Native R Implementation - BASELINE)"
    echo "=============================================================="

    TASK_OUT="${OUT_PATH}/R_native"
    mkdir -p "$TASK_OUT"

    Rscript run_slide_R.R "$YAML_CONFIG" "$TASK_OUT"

    touch "${OUT_PATH}/.task0_complete"

elif [ "$TASK_ID" -eq 1 ]; then
    # =========================================================================
    # Task 1: Python (Python LOVE + R Knockoffs)
    # =========================================================================
    echo ""
    echo "=============================================================="
    echo "Running Python SLIDE (Python LOVE + R Knockoffs)"
    echo "=============================================================="

    TASK_OUT="${OUT_PATH}/Py_pyLOVE_rKO"
    mkdir -p "$TASK_OUT"

    "$PYTHON_ENV" run_slide_py.py "$YAML_CONFIG" "$TASK_OUT" \
        --love-backend python \
        --knockoff-backend r

    touch "${OUT_PATH}/.task1_complete"

elif [ "$TASK_ID" -eq 2 ]; then
    # =========================================================================
    # Task 2: Python (Python LOVE + knockoff-filter with glmnet)
    # =========================================================================
    echo ""
    echo "=============================================================="
    echo "Running Python SLIDE (Python LOVE + knockoff-filter glmnet)"
    echo "=============================================================="

    TASK_OUT="${OUT_PATH}/Py_pyLOVE_kf_glmnet"
    mkdir -p "$TASK_OUT"

    # Use knockoff-filter (python backend) with glmnet feature statistic
    "$PYTHON_ENV" run_slide_py.py "$YAML_CONFIG" "$TASK_OUT" \
        --love-backend python \
        --knockoff-backend python \
        --knockoff-method asdp \
        --fstat glmnet_lambdasmax

    touch "${OUT_PATH}/.task2_complete"

elif [ "$TASK_ID" -eq 3 ]; then
    # =========================================================================
    # Task 3: Python (Python LOVE + knockoff-filter with sklearn)
    # =========================================================================
    echo ""
    echo "=============================================================="
    echo "Running Python SLIDE (Python LOVE + knockoff-filter sklearn)"
    echo "=============================================================="

    TASK_OUT="${OUT_PATH}/Py_pyLOVE_kf_sklearn"
    mkdir -p "$TASK_OUT"

    # Use knockoff-filter (python backend) with sklearn lasso
    "$PYTHON_ENV" run_slide_py.py "$YAML_CONFIG" "$TASK_OUT" \
        --love-backend python \
        --knockoff-backend python \
        --knockoff-method asdp \
        --fstat lasso_lambdasmax

    touch "${OUT_PATH}/.task3_complete"

fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================================="
echo "Task ${TASK_ID} (${TASK_NAMES[$TASK_ID]}) completed in ${ELAPSED} seconds"
echo "=============================================================="
