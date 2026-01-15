#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH --job-name=slide_report
#SBATCH --mail-user=aar126@pitt.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=comparison_report_%j.out
#SBATCH --error=comparison_report_%j.err

# =============================================================================
# SLIDE Comparison Report Generator
# =============================================================================
# Generates comparison report after R and Python jobs complete.
#
# Usage:
#   # Manual (after array jobs finish):
#   sbatch run_report.sh <tag>
#
#   # With dependency (automatic):
#   JOB_ID=$(sbatch --parsable run_comparison.sh config.yaml)
#   sbatch --dependency=afterok:${JOB_ID} run_report.sh <tag>
#
#   # Or use the wrapper script:
#   ./submit_comparison.sh config.yaml
# =============================================================================

set -e

# Get tag from argument or use default
TAG="${1:-comparison_run}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname $0)/outputs}"
PYTHON_ENV="${PYTHON_ENV:-/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load modules
module load gcc/12.2.0
module load python/ondemand-jupyter-python3.11 2>/dev/null || true

echo "=============================================================="
echo "SLIDE Comparison Report"
echo "=============================================================="
echo "Tag: $TAG"
echo "Output dir: ${OUTPUT_DIR}/${TAG}"
echo "=============================================================="

# Check completion markers
R_COMPLETE="${OUTPUT_DIR}/${TAG}/.r_complete"
PY_COMPLETE="${OUTPUT_DIR}/${TAG}/.py_complete"

if [ -f "$R_COMPLETE" ]; then
    echo "R completed successfully"
else
    echo "WARNING: R did not complete (marker not found: $R_COMPLETE)"
fi

if [ -f "$PY_COMPLETE" ]; then
    echo "Python completed successfully"
else
    echo "WARNING: Python did not complete (marker not found: $PY_COMPLETE)"
fi

# Generate comparison report
echo ""
echo "Generating comparison report..."

"$PYTHON_ENV" compare_outputs.py \
    "$TAG" \
    --detailed \
    --output-file "${OUTPUT_DIR}/${TAG}/comparison_report.txt"

echo ""
echo "=============================================================="
echo "Report saved to: ${OUTPUT_DIR}/${TAG}/comparison_report.txt"
echo "=============================================================="

# Print summary
if [ -f "${OUTPUT_DIR}/${TAG}/comparison_report.txt" ]; then
    echo ""
    echo "Report Summary:"
    echo "---------------"
    tail -20 "${OUTPUT_DIR}/${TAG}/comparison_report.txt"
fi
