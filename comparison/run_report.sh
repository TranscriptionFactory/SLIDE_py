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
#   sbatch run_report.sh <out_path>
#
# The out_path should contain:
#   R_outputs/   - R SLIDE outputs
#   Py_outputs/  - Python SLIDE outputs
# =============================================================================

set -e

OUT_PATH="${1:-/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/outputs/SSc_comparison}"

SCRIPT_DIR=/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
cd "$SCRIPT_DIR"

# Load modules
module load gcc/12.2.0
module load python/ondemand-jupyter-python3.11 2>/dev/null || true

PYTHON_ENV="${PYTHON_ENV:-/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python}"

echo "=============================================================="
echo "SLIDE Comparison Report"
echo "=============================================================="
echo "Output path: $OUT_PATH"
echo "=============================================================="

# Check completion markers
R_COMPLETE="${OUT_PATH}/.r_complete"
PY_COMPLETE="${OUT_PATH}/.py_complete"

R_OUT="${OUT_PATH}/R_outputs"
PY_OUT="${OUT_PATH}/Py_outputs"

echo ""
echo "Checking completion status:"

if [ -f "$R_COMPLETE" ]; then
    echo "  R:      COMPLETED"
    if [ -d "$R_OUT" ]; then
        echo "          Outputs in: $R_OUT"
        ls -la "$R_OUT"/*.csv 2>/dev/null | head -5 || echo "          (no CSV files found)"
    fi
else
    echo "  R:      NOT COMPLETED (marker not found)"
fi

echo ""

if [ -f "$PY_COMPLETE" ]; then
    echo "  Python: COMPLETED"
    if [ -d "$PY_OUT" ]; then
        echo "          Outputs in: $PY_OUT"
        ls -la "$PY_OUT"/*.csv 2>/dev/null | head -5 || echo "          (no CSV files found)"
    fi
else
    echo "  Python: NOT COMPLETED (marker not found)"
fi

echo ""
echo "=============================================================="

# If both completed, we can potentially run a comparison
# For now, just list what's in each directory

if [ -d "$R_OUT" ] && [ -d "$PY_OUT" ]; then
    echo ""
    echo "Both R and Python outputs exist."
    echo ""
    echo "R output subdirectories:"
    ls -d "$R_OUT"/*_out 2>/dev/null || echo "  (none found)"
    echo ""
    echo "Python output subdirectories:"
    ls -d "$PY_OUT"/*_out 2>/dev/null || echo "  (none found)"
    echo ""
    echo "Manual comparison can be done by examining:"
    echo "  R:      $R_OUT/<delta>_<lambda>_out/"
    echo "  Python: $PY_OUT/<delta>_<lambda>_out/"
fi

echo ""
echo "=============================================================="
echo "Report complete"
echo "=============================================================="
