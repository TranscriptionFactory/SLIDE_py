#!/bin/bash
# =============================================================================
# SLIDE Comparison Submission Wrapper
# =============================================================================
# Submits the array job and automatically queues the comparison report
# to run after both R and Python complete.
#
# Usage:
#   ./submit_comparison.sh                      # Use defaults
#   ./submit_comparison.sh config.yaml          # Use YAML config
#   ./submit_comparison.sh config.yaml my_tag   # Override tag
#
# This will:
#   1. Submit run_comparison.sh as array job (tasks 0=R, 1=Python)
#   2. Submit run_report.sh with dependency on array job
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_FILE="${1:-}"
TAG_OVERRIDE="${2:-}"

# Parse tag from config if provided
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    TAG=$(grep -E "^tag:" "$CONFIG_FILE" | cut -d: -f2 | tr -d ' "'"'" | xargs)
fi

# Use override if provided
if [ -n "$TAG_OVERRIDE" ]; then
    TAG="$TAG_OVERRIDE"
fi

# Default tag
TAG="${TAG:-comparison_run}"

echo "=============================================================="
echo "Submitting SLIDE R vs Python Comparison"
echo "=============================================================="
echo "Config: ${CONFIG_FILE:-defaults}"
echo "Tag: $TAG"
echo ""

# Submit array job
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    ARRAY_JOB_ID=$(sbatch --parsable run_comparison.sh "$CONFIG_FILE")
else
    ARRAY_JOB_ID=$(sbatch --parsable run_comparison.sh)
fi

echo "Submitted array job: $ARRAY_JOB_ID"
echo "  Task 0: R implementation"
echo "  Task 1: Python implementation"

# Submit report job with dependency (runs after ALL array tasks complete)
REPORT_JOB_ID=$(sbatch --parsable --dependency=afterany:${ARRAY_JOB_ID} run_report.sh "$TAG")

echo ""
echo "Submitted report job: $REPORT_JOB_ID"
echo "  Dependency: afterany:${ARRAY_JOB_ID}"
echo "  (Will run after both R and Python finish, regardless of success/failure)"

echo ""
echo "=============================================================="
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "After completion, find outputs in:"
echo "  outputs/${TAG}/"
echo "=============================================================="
