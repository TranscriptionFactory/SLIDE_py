#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH --job-name=slide_compare
#SBATCH --mail-user=aar126@pitt.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-4
#SBATCH --output=logs/comparison_%A_%a.out
#SBATCH --error=logs/comparison_%A_%a.err

# =============================================================================
# SLIDE 5-Way Comparison Script (Array Job Version)
# =============================================================================
# Compares LOVE and Knockoff implementations across R and Python backends.
#
# Array tasks:
#   0 = R SLIDE (native R package - R LOVE + R knockoffs)
#   1 = Python SLIDE (R LOVE + R Knockoffs)
#   2 = Python SLIDE (R LOVE + knockpy Knockoffs, MVR method)
#   3 = Python SLIDE (Py LOVE + R Knockoffs)
#   4 = Python SLIDE (Py LOVE + knockpy Knockoffs, MVR method)
#
# Usage:
#   sbatch run_comparison.sh <yaml_config> [output_path]
#   sbatch run_comparison.sh comparison_config.yaml
#   sbatch run_comparison.sh comparison_config.yaml /path/to/output  # override out_path

# sbatch <<'EOF'
# #!/bin/bash
# #SBATCH -t 00:05:00
# #SBATCH --job-name=submit_compare
# #SBATCH --mem=1G
# #SBATCH --cpus-per-task=1
# #SBATCH --output=submit_comparison_%j.out

# cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison

# # Generate timestamp once
# TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# # Use it in the output directory
# OUTPUT_DIR="/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/output_comparison/${TIMESTAMP}"
# CONTINUOUS_OUT="${OUTPUT_DIR}/SSc_continuous_comparison"
# BINARY_OUT="${OUTPUT_DIR}/SSc_binary_comparison"

# # Create the base directory
# mkdir -p "$OUTPUT_DIR"

# # Submit array jobs (pass output path as second argument)
# JOB1=$(sbatch --parsable run_comparison.sh comparison_config_continuous.yaml "$CONTINUOUS_OUT")
# JOB2=$(sbatch --parsable run_comparison.sh comparison_config_binary.yaml "$BINARY_OUT")

# echo "Submitted continuous comparison: $JOB1"
# echo "Submitted binary comparison: $JOB2"

# # Queue reports with dependencies
# sbatch --dependency=afterany:$JOB1 --output="$CONTINUOUS_OUT/report_submission.log" run_report.sh "$CONTINUOUS_OUT"
# sbatch --dependency=afterany:$JOB2 --output="$BINARY_OUT/report_submission.log" run_report.sh "$BINARY_OUT"

# echo "Reports queued with dependencies"
# EOF
# =============================================================================
# rm -r logs/*
mkdir -p logs/
set -e  # Exit on error

SCRIPT_DIR=/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
cd "$SCRIPT_DIR"

# Get YAML config from argument
YAML_CONFIG="${1:-comparison_config.yaml}"
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
    OUT_PATH="${OUT_PATH:-$SCRIPT_DIR/outputs}"  # fallback if not in YAML
fi

# Create base output directory
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
TASK_NAMES=("R_native" "Py_rLOVE_rKO" "Py_rLOVE_knockpy" "Py_pyLOVE_rKO" "Py_pyLOVE_knockpy")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "=============================================================="
echo "SLIDE 5-Way Comparison"
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

    TASK_OUT="${OUT_PATH}/R_native"
    mkdir -p "$TASK_OUT"

    Rscript run_slide_R.R "$YAML_CONFIG" "$TASK_OUT"

    touch "${OUT_PATH}/.task0_complete"

elif [ "$TASK_ID" -eq 1 ]; then
    # =========================================================================
    # Task 1: Python (R LOVE + R Knockoffs)
    # =========================================================================
    echo ""
    echo "=============================================================="
    echo "Running Python SLIDE (R LOVE + R Knockoffs)"
    echo "=============================================================="

    TASK_OUT="${OUT_PATH}/Py_rLOVE_rKO"
    mkdir -p "$TASK_OUT"

    "$PYTHON_ENV" run_slide_py.py "$YAML_CONFIG" "$TASK_OUT" --love-backend r --knockoff-backend r

    touch "${OUT_PATH}/.task1_complete"

elif [ "$TASK_ID" -eq 2 ]; then
    # =========================================================================
    # Task 2: Python (R LOVE + knockpy Knockoffs)
    # =========================================================================
    echo ""
    echo "=============================================================="
    echo "Running Python SLIDE (R LOVE + knockpy Knockoffs)"
    echo "=============================================================="

    TASK_OUT="${OUT_PATH}/Py_rLOVE_knockpy"
    mkdir -p "$TASK_OUT"

    # Use knockpy with mvr method (matches R's DSDP solver)
    "$PYTHON_ENV" run_slide_py.py "$YAML_CONFIG" "$TASK_OUT" --love-backend r --knockoff-backend knockpy --knockoff-method mvr

    touch "${OUT_PATH}/.task2_complete"

elif [ "$TASK_ID" -eq 3 ]; then
    # =========================================================================
    # Task 3: Python (Python LOVE + R Knockoffs)
    # =========================================================================
    echo ""
    echo "=============================================================="
    echo "Running Python SLIDE (Python LOVE + R Knockoffs)"
    echo "=============================================================="

    TASK_OUT="${OUT_PATH}/Py_pyLOVE_rKO"
    mkdir -p "$TASK_OUT"

    "$PYTHON_ENV" run_slide_py.py "$YAML_CONFIG" "$TASK_OUT" --love-backend python --knockoff-backend r

    touch "${OUT_PATH}/.task3_complete"

elif [ "$TASK_ID" -eq 4 ]; then
    # =========================================================================
    # Task 4: Python (Python LOVE + knockpy Knockoffs)
    # =========================================================================
    echo ""
    echo "=============================================================="
    echo "Running Python SLIDE (Python LOVE + knockpy Knockoffs)"
    echo "=============================================================="

    TASK_OUT="${OUT_PATH}/Py_pyLOVE_knockpy"
    mkdir -p "$TASK_OUT"

    # Use knockpy with mvr method (matches R's DSDP solver)
    "$PYTHON_ENV" run_slide_py.py "$YAML_CONFIG" "$TASK_OUT" --love-backend python --knockoff-backend knockpy --knockoff-method mvr

    touch "${OUT_PATH}/.task4_complete"

fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================================="
echo "Task ${TASK_ID} (${TASK_NAMES[$TASK_ID]}) completed in ${ELAPSED} seconds"
echo "=============================================================="
