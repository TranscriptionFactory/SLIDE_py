#!/bin/bash
#
# Run LOVE diagnostics for both Python and R implementations
# and compare the results.
#
# Usage:
#   ./run_diagnostics.sh /path/to/data.csv [delta] [output_dir]
#
# Example:
#   ./run_diagnostics.sh /path/to/SSc_binary_X.csv 0.05 ./diagnostics_output
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Arguments
DATA_PATH="${1:?Error: Please provide path to data CSV}"
DELTA="${2:-0.05}"
OUTPUT_BASE="${3:-$SCRIPT_DIR/output}"
THRESH_FDR="${4:-0.2}"

# Output directories
PY_OUTPUT="$OUTPUT_BASE/python_delta${DELTA}"
R_OUTPUT="$OUTPUT_BASE/r_delta${DELTA}"

echo "=========================================="
echo "LOVE Diagnostics Runner"
echo "=========================================="
echo "Data path: $DATA_PATH"
echo "Delta: $DELTA"
echo "FDR threshold: $THRESH_FDR"
echo "Python output: $PY_OUTPUT"
echo "R output: $R_OUTPUT"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    # Try to activate the SLIDE_py environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate slide_py 2>/dev/null || conda activate base
fi

# Run Python diagnostics
echo ""
echo "=========================================="
echo "Running Python LOVE diagnostics..."
echo "=========================================="
cd "$SCRIPT_DIR/../.."
python -m comparison.diagnostics.love_diagnostics_py \
    --data_path "$DATA_PATH" \
    --delta "$DELTA" \
    --thresh_fdr "$THRESH_FDR" \
    --output_dir "$PY_OUTPUT"

# Run R diagnostics
echo ""
echo "=========================================="
echo "Running R LOVE diagnostics..."
echo "=========================================="
Rscript "$SCRIPT_DIR/love_diagnostics_r.R" \
    --data_path "$DATA_PATH" \
    --delta "$DELTA" \
    --thresh_fdr "$THRESH_FDR" \
    --output_dir "$R_OUTPUT"

# Run comparison
echo ""
echo "=========================================="
echo "Comparing Python and R outputs..."
echo "=========================================="
python "$SCRIPT_DIR/compare_love_diagnostics.py" \
    --py_dir "$PY_OUTPUT" \
    --r_dir "$R_OUTPUT" \
    --output "$OUTPUT_BASE/comparison_report_delta${DELTA}.txt"

echo ""
echo "=========================================="
echo "Done! Results saved to: $OUTPUT_BASE"
echo "=========================================="
echo ""
echo "Key files:"
echo "  - Python outputs: $PY_OUTPUT/"
echo "  - R outputs: $R_OUTPUT/"
echo "  - Comparison report: $OUTPUT_BASE/comparison_report_delta${DELTA}.txt"
