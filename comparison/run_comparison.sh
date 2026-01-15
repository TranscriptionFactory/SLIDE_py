#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH --job-name=slide_compare
#SBATCH --mail-user=aar126@pitt.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=10
#SBATCH --output=comparison_%j.out
#SBATCH --error=comparison_%j.err

# =============================================================================
# SLIDE R vs Python Comparison Script
# =============================================================================
# This script runs both R and Python SLIDE implementations on the same data
# with identical parameters for direct comparison.
#
# Usage:
#   sbatch run_comparison.sh                    # Use defaults below
#   sbatch run_comparison.sh config.yaml        # Load from YAML config
#   bash run_comparison.sh                      # Run interactively
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# SHARED PARAMETERS - Edit these or use a config YAML
# -----------------------------------------------------------------------------
# Data paths (REQUIRED - must point to actual files)
X_FILE="${X_FILE:-/ix/djishnu/Aaron/0_for_others/Shailja/20260109_data/X.csv}"
Y_FILE="${Y_FILE:-/ix/djishnu/Aaron/0_for_others/Shailja/20260109_data/Y.csv}"

# Run tag (output subdirectory name)
TAG="${TAG:-comparison_run}"

# SLIDE parameters (same for R and Python)
DELTA="${DELTA:-0.1}"
LAMBDA="${LAMBDA:-0.5}"
SPEC="${SPEC:-0.1}"
FDR="${FDR:-0.1}"
NITER="${NITER:-500}"
THRESH_FDR="${THRESH_FDR:-0.2}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname $0)/outputs}"

# Python environment
PYTHON_ENV="${PYTHON_ENV:-/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python}"

# -----------------------------------------------------------------------------
# Load config from YAML if provided
# -----------------------------------------------------------------------------
if [ -n "$1" ] && [ -f "$1" ]; then
    echo "Loading config from: $1"
    # Parse YAML (simple key: value format)
    while IFS=': ' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        # Remove quotes and whitespace
        value=$(echo "$value" | tr -d '"' | tr -d "'" | xargs)
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
    done < "$1"
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
# Step 1: Run R Implementation
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "Step 1: Running R SLIDE"
echo "=============================================================="

R_START=$(date +%s)

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

R_END=$(date +%s)
echo "R completed in $((R_END - R_START)) seconds"

# -----------------------------------------------------------------------------
# Step 2: Run Python Implementation
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "Step 2: Running Python SLIDE"
echo "=============================================================="

PY_START=$(date +%s)

"$PYTHON_ENV" run_slide_py.py \
    "$X_FILE" \
    "$Y_FILE" \
    "$TAG" \
    --delta "$DELTA" \
    --lambda "$LAMBDA" \
    --spec "$SPEC" \
    --fdr "$FDR" \
    --niter "$NITER" \
    --thresh-fdr "$THRESH_FDR"

PY_END=$(date +%s)
echo "Python completed in $((PY_END - PY_START)) seconds"

# -----------------------------------------------------------------------------
# Step 3: Generate Comparison Report
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "Step 3: Generating Comparison Report"
echo "=============================================================="

"$PYTHON_ENV" compare_outputs.py \
    "$TAG" \
    --detailed \
    --output-file "${OUTPUT_DIR}/${TAG}/comparison_report.txt"

echo ""
echo "=============================================================="
echo "Comparison Complete!"
echo "=============================================================="
echo "Outputs saved to: ${OUTPUT_DIR}/${TAG}/"
echo "Report: ${OUTPUT_DIR}/${TAG}/comparison_report.txt"
echo ""
echo "R time:     $((R_END - R_START)) seconds"
echo "Python time: $((PY_END - PY_START)) seconds"
echo "=============================================================="
