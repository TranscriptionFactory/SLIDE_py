#!/bin/bash
# =============================================================================
# SLIDE Comparison Submission Wrapper
# =============================================================================
# Submits the array job and automatically queues the comparison report
# to run after both R and Python complete.
#
# Usage:
#   ./submit_comparison.sh                           # Use default config
#   ./submit_comparison.sh my_config.yaml            # Use custom config
#
# This will:
#   1. Submit run_comparison.sh as array job (tasks 0=R, 1=Python)
#   2. Submit run_report.sh with dependency on array job
# =============================================================================

set -e

SCRIPT_DIR=/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
cd "$SCRIPT_DIR"

YAML_CONFIG="${1:-comparison_config.yaml}"

if [ ! -f "$YAML_CONFIG" ]; then
    echo "ERROR: YAML config not found: $YAML_CONFIG"
    exit 1
fi

# Parse output path from YAML
OUT_PATH=$(grep "^out_path:" "$YAML_CONFIG" | sed 's/out_path: *//' | tr -d '"'"'" | xargs)
OUT_PATH="${OUT_PATH:-$SCRIPT_DIR/outputs}"

echo "=============================================================="
echo "Submitting SLIDE R vs Python Comparison"
echo "=============================================================="
echo "Config: $YAML_CONFIG"
echo "Output: $OUT_PATH"
echo ""

# Submit array job
ARRAY_JOB_ID=$(sbatch --parsable run_comparison.sh "$YAML_CONFIG")

echo "Submitted array job: $ARRAY_JOB_ID"
echo "  Task 0: R implementation  -> ${OUT_PATH}/R_outputs/"
echo "  Task 1: Python implementation -> ${OUT_PATH}/Py_outputs/"

# Submit report job with dependency (runs after ALL array tasks complete)
REPORT_JOB_ID=$(sbatch --parsable --dependency=afterany:${ARRAY_JOB_ID} run_report.sh "$OUT_PATH")

echo ""
echo "Submitted report job: $REPORT_JOB_ID"
echo "  Dependency: afterany:${ARRAY_JOB_ID}"
echo "  (Will run after both R and Python finish)"

echo ""
echo "=============================================================="
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "After completion, find outputs in:"
echo "  ${OUT_PATH}/R_outputs/"
echo "  ${OUT_PATH}/Py_outputs/"
echo "=============================================================="
