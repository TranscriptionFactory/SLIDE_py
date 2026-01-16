#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH --job-name=slide_report
#SBATCH --mail-user=aar126@pitt.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=comparison_report_%j.out
#SBATCH --error=comparison_report_%j.err

# =============================================================================
# SLIDE 5-Way Comparison Report Generator
# =============================================================================
# Generates comparison report after all jobs complete.
#
# Usage:
#   sbatch run_report.sh [out_path]
#
# The out_path should contain subdirectories for each implementation:
#   R_native/       - R SLIDE (native)
#   Py_rLOVE_rKO/   - Python (R LOVE + R Knockoffs)
#   Py_rLOVE_pyKO/  - Python (R LOVE + Py Knockoffs)
#   Py_pyLOVE_rKO/  - Python (Py LOVE + R Knockoffs)
#   Py_pyLOVE_pyKO/ - Python (Py LOVE + Py Knockoffs)
# =============================================================================

set -e

OUT_PATH="${1:-/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/updated_outputs}"
SCRIPT_DIR=/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
cd "$SCRIPT_DIR"

# Load modules
module load gcc/12.2.0
module load python/ondemand-jupyter-python3.11 2>/dev/null || true

PYTHON_ENV="${PYTHON_ENV:-/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python}"

# Task configuration (must match run_comparison.sh)
TASK_NAMES=("R_native" "Py_rLOVE_rKO" "Py_rLOVE_pyKO" "Py_pyLOVE_rKO" "Py_pyLOVE_pyKO")
TASK_DESCRIPTIONS=(
    "R SLIDE (native R LOVE + R Knockoffs)"
    "Python (R LOVE + R Knockoffs)"
    "Python (R LOVE + Py Knockoffs)"
    "Python (Py LOVE + R Knockoffs)"
    "Python (Py LOVE + Py Knockoffs)"
)

echo "=============================================================="
echo "SLIDE 5-Way Comparison Report"
echo "=============================================================="
echo "Output path: $OUT_PATH"
echo "=============================================================="
echo ""

# -----------------------------------------------------------------------------
# Check completion status for each task
# -----------------------------------------------------------------------------
echo "Completion Status:"
echo "------------------"

COMPLETED=0
for i in "${!TASK_NAMES[@]}"; do
    TASK_NAME="${TASK_NAMES[$i]}"
    TASK_DESC="${TASK_DESCRIPTIONS[$i]}"
    TASK_OUT="${OUT_PATH}/${TASK_NAME}"
    MARKER="${OUT_PATH}/.task${i}_complete"

    if [ -f "$MARKER" ]; then
        STATUS="✓ COMPLETED"
        COMPLETED=$((COMPLETED + 1))
    elif [ -d "$TASK_OUT" ]; then
        STATUS="? PARTIAL (dir exists, no marker)"
    else
        STATUS="✗ NOT STARTED"
    fi

    printf "  [%d] %-20s %s\n" "$i" "$TASK_NAME" "$STATUS"
    printf "      %s\n" "$TASK_DESC"
done

echo ""
echo "Completed: ${COMPLETED}/${#TASK_NAMES[@]}"
echo ""

# -----------------------------------------------------------------------------
# List output subdirectories for each completed task
# -----------------------------------------------------------------------------
echo "=============================================================="
echo "Output Summary"
echo "=============================================================="

for i in "${!TASK_NAMES[@]}"; do
    TASK_NAME="${TASK_NAMES[$i]}"
    TASK_OUT="${OUT_PATH}/${TASK_NAME}"

    if [ -d "$TASK_OUT" ]; then
        echo ""
        echo "[$i] ${TASK_NAME}:"
        echo "    Directory: $TASK_OUT"

        # Count parameter combinations
        NUM_COMBOS=$(ls -d "$TASK_OUT"/*_out 2>/dev/null | wc -l || echo 0)
        echo "    Parameter combinations: $NUM_COMBOS"

        # List subdirectories
        if [ "$NUM_COMBOS" -gt 0 ]; then
            echo "    Subdirectories:"
            ls -d "$TASK_OUT"/*_out 2>/dev/null | while read -r dir; do
                COMBO=$(basename "$dir")
                # Check for key output files (Python: .csv/.txt, R: .rds)
                # A matrix: Python=A.csv, R=AllLatentFactors.rds
                HAS_A=$([ -f "$dir/A.csv" ] || [ -f "$dir/AllLatentFactors.rds" ] && echo "A" || echo "-")
                # Z matrix: Python=z_matrix.csv, R=feature_list_Z*.txt files
                HAS_Z=$([ -f "$dir/z_matrix.csv" ] || ls "$dir"/feature_list_Z*.txt &>/dev/null && echo "Z" || echo "-")
                # Significant LFs: Python=sig_LFs.txt, R=SLIDE_LFs.rds
                HAS_LF=$([ -f "$dir/sig_LFs.txt" ] || [ -f "$dir/SLIDE_LFs.rds" ] && echo "LF" || echo "-")
                printf "      %-20s [%s %s %s]\n" "$COMBO" "$HAS_A" "$HAS_Z" "$HAS_LF"
            done
        fi
    fi
done

# -----------------------------------------------------------------------------
# Compare results across implementations (if multiple completed)
# -----------------------------------------------------------------------------
if [ "$COMPLETED" -ge 2 ]; then
    echo ""
    echo "=============================================================="
    echo "Cross-Implementation Comparison"
    echo "=============================================================="

    # Find common parameter combinations across all completed tasks
    echo ""
    echo "Finding common parameter combinations..."

    # Get first completed task's combinations as reference
    REF_TASK=""
    for i in "${!TASK_NAMES[@]}"; do
        MARKER="${OUT_PATH}/.task${i}_complete"
        if [ -f "$MARKER" ]; then
            REF_TASK="${TASK_NAMES[$i]}"
            break
        fi
    done

    if [ -n "$REF_TASK" ]; then
        REF_OUT="${OUT_PATH}/${REF_TASK}"

        ls -d "$REF_OUT"/*_out 2>/dev/null | while read -r ref_dir; do
            COMBO=$(basename "$ref_dir")
            echo ""
            echo "Parameter: $COMBO"
            echo "  Significant LFs found:"

            for i in "${!TASK_NAMES[@]}"; do
                TASK_NAME="${TASK_NAMES[$i]}"
                TASK_DIR="${OUT_PATH}/${TASK_NAME}/${COMBO}"
                LF_FILE="${TASK_DIR}/sig_LFs.txt"

                if [ -f "$LF_FILE" ]; then
                    # Python output: read from sig_LFs.txt
                    NUM_LFS=$(wc -l < "$LF_FILE" 2>/dev/null || echo 0)
                    LFS=$(cat "$LF_FILE" 2>/dev/null | tr '\n' ' ' | head -c 50)
                    printf "    %-20s: %d LFs [%s]\n" "$TASK_NAME" "$NUM_LFS" "$LFS"
                elif ls "$TASK_DIR"/feature_list_Z*.txt &>/dev/null 2>&1; then
                    # R output: extract LF names from feature_list_Z*.txt filenames
                    LFS=$(ls "$TASK_DIR"/feature_list_Z*.txt 2>/dev/null | xargs -n1 basename | sed 's/feature_list_//' | sed 's/\.txt//' | tr '\n' ' ')
                    NUM_LFS=$(echo "$LFS" | wc -w)
                    printf "    %-20s: %d LFs [%s]\n" "$TASK_NAME" "$NUM_LFS" "$LFS"
                else
                    printf "    %-20s: (no results)\n" "$TASK_NAME"
                fi
            done
        done
    fi
fi

echo ""
echo "=============================================================="
echo "Report complete"
echo "=============================================================="
