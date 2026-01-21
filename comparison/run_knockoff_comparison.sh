#!/bin/bash
#SBATCH --job-name=knockoff_cmp
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-5
#SBATCH --output=logs/knockoff_cmp_%A_%a.out
#SBATCH --error=logs/knockoff_cmp_%A_%a.err

# =============================================================================
# Comprehensive Knockoff Implementation Comparison (Array Job Version)
# =============================================================================
#
# Runs each knockoff backend as a separate array task:
#   0 = R_native (R knockoff package)
#   1 = knockoff_filter (Fortran glmnet)
#   2 = knockoff_filter_sklearn (sklearn fallback)
#   3 = knockpy_lsm (knockpy with LSM fstat)
#   4 = knockpy_lasso (knockpy with lasso fstat)
#   5 = custom_glmnet (SLIDE's implementation)
#
# After all tasks complete, run aggregation manually:
#   python run_knockoff_comparison.py --config <config> --output-dir <dir> --aggregate
#
# Usage:
#   sbatch run_knockoff_comparison.sh                              # Default config
#   sbatch run_knockoff_comparison.sh comparison_config_binary.yaml
#   sbatch run_knockoff_comparison.sh config.yaml /custom/output/path
#
# Quick test (single parameter set):
#   sbatch --export=QUICK=1 run_knockoff_comparison.sh config.yaml
#
# =============================================================================

set -e

SCRIPT_DIR=/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
cd "$SCRIPT_DIR"

mkdir -p logs

# =============================================================================
# Arguments
# =============================================================================
YAML_CONFIG="${1:-comparison_config_binary.yaml}"
OUTPUT_DIR="${2:-}"

if [ ! -f "$YAML_CONFIG" ]; then
    echo "ERROR: Config file not found: $YAML_CONFIG"
    exit 1
fi

# Generate output directory if not provided (use job ID for consistency across array tasks)
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="output_comparison/knockoff_cmp_${SLURM_ARRAY_JOB_ID}"
fi

# =============================================================================
# Backend Configuration
# =============================================================================
BACKENDS=("R_native" "knockoff_filter" "knockoff_filter_sklearn" "knockpy_lsm" "knockpy_lasso" "custom_glmnet")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
BACKEND="${BACKENDS[$TASK_ID]}"

# =============================================================================
# Environment Setup
# =============================================================================
echo "=============================================================="
echo "Knockoff Implementation Comparison - Array Job"
echo "=============================================================="
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Task ID: ${TASK_ID}"
echo "Backend: ${BACKEND}"
echo "Config: $YAML_CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Quick mode: ${QUICK:-0}"
echo "=============================================================="

module load gcc/12.2.0
module load python/ondemand-jupyter-python3.11
module load r/4.4.0

# Python environment
PYTHON_ENV="${PYTHON_ENV:-/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python}"

# Add knockoff-filter to path (with intercept fix)
export PYTHONPATH="/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter:$PYTHONPATH"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Run Backend
# =============================================================================
START_TIME=$(date +%s)

QUICK_FLAG=""
if [ "${QUICK:-0}" = "1" ]; then
    QUICK_FLAG="--quick"
    echo "Running in QUICK mode (single parameter set)"
fi

echo ""
echo "Starting ${BACKEND} at $(date)"
echo ""

"$PYTHON_ENV" run_knockoff_comparison.py \
    --config "$YAML_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --backend "$BACKEND" \
    $QUICK_FLAG

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================================="
echo "Backend ${BACKEND} completed in ${ELAPSED} seconds"
echo "Results saved to: $OUTPUT_DIR/backend_${BACKEND}.json"
echo "=============================================================="

# Mark task as complete
touch "$OUTPUT_DIR/.task${TASK_ID}_complete"

# Check if all tasks are complete and aggregate
ALL_COMPLETE=true
for i in {0..5}; do
    if [ ! -f "$OUTPUT_DIR/.task${i}_complete" ]; then
        ALL_COMPLETE=false
        break
    fi
done

if [ "$ALL_COMPLETE" = true ]; then
    echo ""
    echo "All tasks complete! Running aggregation..."
    "$PYTHON_ENV" run_knockoff_comparison.py \
        --config "$YAML_CONFIG" \
        --output-dir "$OUTPUT_DIR" \
        --aggregate
    echo "Aggregation complete!"
fi
