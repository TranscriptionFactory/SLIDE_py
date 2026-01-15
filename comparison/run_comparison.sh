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
#   sbatch run_comparison.sh                    # Use defaults below
#   sbatch run_comparison.sh config.yaml        # Load from YAML config
#
# After both complete, run the comparison report manually:
#   python compare_outputs.py <tag> --detailed
#
# Or use run_compare_report.sh (submitted automatically as dependency)
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# SHARED PARAMETERS - Edit these or use a config YAML
# -----------------------------------------------------------------------------
# Data paths (REQUIRED - must point to actual files)
X_FILE="${X_FILE:-/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/X.csv}"
Y_FILE="${Y_FILE:-/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/Y.csv}"

# Run tag (output subdirectory name)
TAG="${TAG:-comparison_run}"

# SLIDE parameters (same for R and Python)
DELTA="${DELTA:-0.1}"
LAMBDA="${LAMBDA:-0.5}"
SPEC="${SPEC:-0.1}"
FDR="${FDR:-0.1}"
NITER="${NITER:-500}"
THRESH_FDR="${THRESH_FDR:-0.2}"

# Output directory (use absolute path)
OUTPUT_DIR="${OUTPUT_DIR:-/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/outputs}"

# Python environment
PYTHON_ENV="${PYTHON_ENV:-/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python}"

# -----------------------------------------------------------------------------
# Load config from YAML if provided
# -----------------------------------------------------------------------------
CONFIG_FILE="${1:-}"
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    echo "Loading config from: $CONFIG_FILE"
    # Parse YAML (simple key: value format)
    while IFS=': ' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        # Strip inline comments (everything after #)
        value="${value%%#*}"
        # Remove quotes and whitespace
        value=$(echo "$value" | tr -d '"' | tr -d "'" | xargs)
        # Skip if value is empty after stripping
        [[ -z "$value" ]] && continue
        case "$key" in
            x_path|X_FILE)     X_FILE="$value" ;;
            y_path|Y_FILE)     Y_FILE="$value" ;;
            tag|TAG)           TAG="$value" ;;
            delta|DELTA)       DELTA="$value" ;;
            lambda|LAMBDA)     LAMBDA="$value" ;;
            spec|SPEC)         SPEC="$value" ;;
            fdr|FDR)           FDR="$value" ;;
            niter|NITER)       NITER="$value" ;;
            thresh_fdr)        THRESH_FDR="$value" ;;
            out_path)          OUTPUT_DIR="$value" ;;
            python_env)        PYTHON_ENV="$value" ;;
        esac
    done < "$CONFIG_FILE"
fi

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load modules
module load gcc/12.2.0
module load r/4.4.0
module load python/ondemand-jupyter-python3.11 2>/dev/null || true

# Create output directory
mkdir -p "${OUTPUT_DIR}/${TAG}"

# -----------------------------------------------------------------------------
# Print configuration
# -----------------------------------------------------------------------------
echo "=============================================================="
echo "SLIDE R vs Python Comparison"
echo "=============================================================="
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo ""
echo "X file:      $X_FILE"
echo "Y file:      $Y_FILE"
echo "Tag:         $TAG"
echo "Output dir:  ${OUTPUT_DIR}/${TAG}"
echo ""
echo "Parameters:"
echo "  delta:     $DELTA"
echo "  lambda:    $LAMBDA"
echo "  spec:      $SPEC"
echo "  fdr:       $FDR"
echo "  niter:     $NITER"
echo "  thresh_fdr: $THRESH_FDR"
echo "=============================================================="

# Validate input files exist
if [ ! -f "$X_FILE" ]; then
    echo "ERROR: X file not found: $X_FILE"
    exit 1
fi
if [ ! -f "$Y_FILE" ]; then
    echo "ERROR: Y file not found: $Y_FILE"
    exit 1
fi

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

    Rscript run_slide_R.R \
        "$X_FILE" \
        "$Y_FILE" \
        "$TAG" \
        --delta "$DELTA" \
        --lambda "$LAMBDA" \
        --spec "$SPEC" \
        --fdr "$FDR" \
        --niter "$NITER" \
        --thresh-fdr "$THRESH_FDR"

    # Mark R as complete
    touch "${OUTPUT_DIR}/${TAG}/.r_complete"
    echo "R outputs saved. Completion marker: ${OUTPUT_DIR}/${TAG}/.r_complete"

elif [ "${SLURM_ARRAY_TASK_ID:-1}" -eq 1 ]; then
    # =========================================================================
    # Task 1: Python Implementation
    # =========================================================================
    echo ""
    echo "=============================================================="
    echo "Running Python SLIDE Implementation"
    echo "=============================================================="

    "$PYTHON_ENV" run_slide_py.py \
        "$X_FILE" \
        "$Y_FILE" \
        "$TAG" \
        --delta "$DELTA" \
        --lambda "$LAMBDA" \
        --spec "$SPEC" \
        --fdr "$FDR" \
        --niter "$NITER" \
        --thresh-fdr "$THRESH_FDR" \
        --generate-only

    # Mark Python as complete
    touch "${OUTPUT_DIR}/${TAG}/.py_complete"
    echo "Python outputs saved. Completion marker: ${OUTPUT_DIR}/${TAG}/.py_complete"

fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================================="
echo "Task ${SLURM_ARRAY_TASK_ID:-0} completed in ${ELAPSED} seconds"
echo "=============================================================="
