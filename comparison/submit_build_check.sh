#!/bin/bash
# =============================================================================
# SLIDE Build Check Submission Wrapper
# =============================================================================
# Submits build check SLURM array job, monitors completion, and runs validation
#
# Usage:
#   bash submit_build_check.sh [config_file] [output_path]
#   bash submit_build_check.sh                              # use defaults
#   bash submit_build_check.sh build_check_config.yaml     # explicit config
#   bash submit_build_check.sh config.yaml /my/output      # override output
#
# This script:
#   1. Submits SLURM array job (4 tasks)
#   2. Monitors job completion
#   3. Runs validation script automatically
#   4. Displays summary and exits with appropriate status code
# =============================================================================

set -e

SCRIPT_DIR=/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
cd "$SCRIPT_DIR"

# Get config and output path from arguments
CONFIG="${1:-build_check_config.yaml}"
OUTPUT_OVERRIDE="${2:-}"

# Check config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# Get base output path from config
if [ -n "$OUTPUT_OVERRIDE" ]; then
    BASE_OUT_PATH="$OUTPUT_OVERRIDE"
else
    BASE_OUT_PATH=$(grep "^out_path:" "$CONFIG" | sed 's/out_path: *//' | tr -d '"'"'" | xargs)
    BASE_OUT_PATH="${BASE_OUT_PATH:-$SCRIPT_DIR/build_check_outputs}"
fi

echo "=============================================================="
echo "SLIDE Build Check Submission"
echo "=============================================================="
echo "Config file: $CONFIG"
echo "Base output: $BASE_OUT_PATH"
echo "=============================================================="
echo ""

# Submit SLURM array job
echo "Submitting SLURM array job (4 tasks)..."
if [ -n "$OUTPUT_OVERRIDE" ]; then
    JOB_ID=$(sbatch --parsable run_build_check.sh "$CONFIG" "$OUTPUT_OVERRIDE")
else
    JOB_ID=$(sbatch --parsable run_build_check.sh "$CONFIG")
fi

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Failed to submit job"
    exit 1
fi

echo "Job submitted: $JOB_ID"
echo ""

# Extract job ID (handle array job format like "12345_0")
JOB_ID_BASE=$(echo "$JOB_ID" | cut -d'_' -f1)

# Wait for job to start
echo "Waiting for job to start..."
sleep 5

# Get the actual output directory with timestamp
# It will be created by the job, so we need to wait and find it
TIMEOUT=300  # 5 minutes timeout for job to create output dir
ELAPSED=0
OUTPUT_DIR=""

while [ $ELAPSED -lt $TIMEOUT ]; do
    # Find most recent directory in base output path
    if [ -d "$BASE_OUT_PATH" ]; then
        LATEST_DIR=$(ls -td "$BASE_OUT_PATH"/*/ 2>/dev/null | head -1)
        if [ -n "$LATEST_DIR" ]; then
            OUTPUT_DIR="$LATEST_DIR"
            break
        fi
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ -z "$OUTPUT_DIR" ]; then
    echo "WARNING: Could not find output directory. Will check after job completes."
    OUTPUT_DIR="$BASE_OUT_PATH/UNKNOWN"
else
    echo "Output directory: $OUTPUT_DIR"
fi

echo ""
echo "=============================================================="
echo "Monitoring job progress..."
echo "=============================================================="
echo "Job ID: $JOB_ID_BASE"
echo "Output: $OUTPUT_DIR"
echo ""
echo "You can monitor progress with:"
echo "  squeue -u \$USER | grep $JOB_ID_BASE"
echo "  tail -f logs/build_check_${JOB_ID_BASE}_*.out"
echo ""
echo "Waiting for job to complete..."
echo ""

# Monitor job completion
while true; do
    # Check if job is still in queue
    JOB_STATUS=$(squeue -j "$JOB_ID_BASE" -h -o "%T" 2>/dev/null | head -1)
    
    if [ -z "$JOB_STATUS" ]; then
        # Job no longer in queue - completed
        echo ""
        echo "Job completed!"
        break
    fi
    
    # Count completed tasks
    COMPLETED_TASKS=$(find "$OUTPUT_DIR" -name ".task*_complete" 2>/dev/null | wc -l)
    echo -ne "\rStatus: $JOB_STATUS | Completed tasks: $COMPLETED_TASKS/4"
    
    sleep 10
done

echo ""
echo ""

# Find the actual output directory if we couldn't before
if [ "$OUTPUT_DIR" == "$BASE_OUT_PATH/UNKNOWN" ] || [ ! -d "$OUTPUT_DIR" ]; then
    echo "Finding output directory..."
    LATEST_DIR=$(ls -td "$BASE_OUT_PATH"/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        OUTPUT_DIR="$LATEST_DIR"
        echo "Found: $OUTPUT_DIR"
    else
        echo "ERROR: Could not find output directory in $BASE_OUT_PATH"
        exit 1
    fi
fi

# Brief pause to ensure all file writes complete
sleep 5

echo "=============================================================="
echo "Running validation..."
echo "=============================================================="
echo ""

# Load Python environment
module load gcc/12.2.0 2>/dev/null || true
module load python/ondemand-jupyter-python3.11 2>/dev/null || true

PYTHON_ENV="${PYTHON_ENV:-/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python}"

# Run validation script
"$PYTHON_ENV" validate_build.py "$OUTPUT_DIR"
VALIDATION_STATUS=$?

echo ""
echo "=============================================================="
echo "Build Check Complete"
echo "=============================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Validation results: $OUTPUT_DIR/validation_results.json"
echo ""

if [ $VALIDATION_STATUS -eq 0 ]; then
    echo "✓ BUILD CHECK PASSED"
    echo ""
    echo "All Python SLIDE knockoff backends validated successfully!"
    exit 0
else
    echo "✗ BUILD CHECK FAILED"
    echo ""
    echo "One or more validation checks failed."
    echo "Review detailed results in: $OUTPUT_DIR/validation_results.json"
    echo "Check logs in: logs/build_check_${JOB_ID_BASE}_*.err"
    exit 1
fi
