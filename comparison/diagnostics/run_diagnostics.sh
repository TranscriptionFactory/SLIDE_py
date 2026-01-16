#!/bin/bash
#
# Run step-by-step diagnostics comparing R and Python LOVE implementations
#
# Usage: ./run_diagnostics.sh <data_file> [--mode hetero|homo] [--fixed-delta VALUE]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_BASE="$SCRIPT_DIR/outputs"

# Default values
MODE="hetero"
FIXED_DELTA=""
DATA_FILE=""
TAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --fixed-delta)
            FIXED_DELTA="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 <data_file> [--mode hetero|homo] [--fixed-delta VALUE] [--tag NAME]"
            echo ""
            echo "Arguments:"
            echo "  data_file      Path to input data CSV (samples as rows)"
            echo "  --mode         hetero or homo (default: hetero)"
            echo "  --fixed-delta  Use a fixed delta value for deterministic comparison"
            echo "  --tag          Name for output directory (default: basename of data file)"
            exit 0
            ;;
        *)
            if [[ -z "$DATA_FILE" ]]; then
                DATA_FILE="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$DATA_FILE" ]]; then
    echo "Error: No data file specified"
    echo "Usage: $0 <data_file> [--mode hetero|homo] [--fixed-delta VALUE]"
    exit 1
fi

if [[ ! -f "$DATA_FILE" ]]; then
    echo "Error: Data file not found: $DATA_FILE"
    exit 1
fi

# Set tag from filename if not provided
if [[ -z "$TAG" ]]; then
    TAG=$(basename "$DATA_FILE" .csv)
fi

# Create output directory
OUTPUT_DIR="$OUTPUT_BASE/${TAG}_${MODE}"
mkdir -p "$OUTPUT_DIR"

echo "=============================================================================="
echo "LOVE DIAGNOSTIC COMPARISON"
echo "=============================================================================="
echo "Data file:   $DATA_FILE"
echo "Mode:        $MODE"
echo "Output dir:  $OUTPUT_DIR"
if [[ -n "$FIXED_DELTA" ]]; then
    echo "Fixed delta: $FIXED_DELTA"
fi
echo "=============================================================================="

# Build R arguments
R_ARGS="$DATA_FILE $OUTPUT_DIR --mode $MODE"
if [[ -n "$FIXED_DELTA" ]]; then
    R_ARGS="$R_ARGS --fixed-delta $FIXED_DELTA"
fi

# Build Python arguments
PY_ARGS="$DATA_FILE $OUTPUT_DIR --mode $MODE"
if [[ -n "$FIXED_DELTA" ]]; then
    PY_ARGS="$PY_ARGS --fixed-delta $FIXED_DELTA"
fi

# Step 1: Run R to save intermediate results
echo ""
echo ">>> Running R LOVE step-by-step..."
echo ""

# Check if R is available
if ! command -v Rscript &> /dev/null; then
    echo "Error: Rscript not found. Please load R module."
    exit 1
fi

Rscript "$SCRIPT_DIR/save_r_intermediate.R" $R_ARGS

# Step 2: Run Python comparison
echo ""
echo ">>> Running Python comparison..."
echo ""

# Activate conda environment if needed
if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
else
    # Try to activate slide_py environment
    if command -v conda &> /dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        if conda env list | grep -q "slide_py"; then
            conda activate slide_py
            echo "Activated conda environment: slide_py"
        fi
    fi
fi

python "$SCRIPT_DIR/compare_step_by_step.py" $PY_ARGS

echo ""
echo "=============================================================================="
echo "DIAGNOSTIC COMPLETE"
echo "=============================================================================="
echo "R intermediate results: $OUTPUT_DIR"
echo ""
echo "To re-run Python comparison only:"
echo "  python $SCRIPT_DIR/compare_step_by_step.py $PY_ARGS"
echo "=============================================================================="
