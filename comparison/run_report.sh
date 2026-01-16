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

# OUT_PATH="/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/output_comparison/2026-01-16_13-27-33/SSc_continuous_comparison"
# bash /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/run_report.sh $OUT_PATH >> $OUT_PATH/comparison_report_intermediate.txt
# =============================================================================

set -e

OUT_PATH="${1:-/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/updated_outputs}"
SCRIPT_DIR=/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
cd "$SCRIPT_DIR"

# Load modules
module load gcc/12.2.0
module load python/ondemand-jupyter-python3.11 2>/dev/null || true
module load r/4.4.0 2>/dev/null || true

PYTHON_ENV="${PYTHON_ENV:-/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python}"

# Task configuration (must match run_comparison.sh)
TASK_NAMES=("R_native" "Py_rLOVE_rKO" "Py_rLOVE_knockpy" "Py_pyLOVE_rKO" "Py_pyLOVE_knockpy")
TASK_DESCRIPTIONS=(
    "R SLIDE (native R LOVE + R Knockoffs)"
    "Python (R LOVE + R Knockoffs)"
    "Python (R LOVE + knockpy Knockoffs)"
    "Python (Py LOVE + R Knockoffs)"
    "Python (Py LOVE + knockpy Knockoffs)"
)

# Extract performance metrics from R output folder
# Generates performance_metrics.csv if not present
extract_r_metrics() {
    local R_DIR="$1"
    local PERF_CSV="${R_DIR}/performance_metrics.csv"

    # Generate CSV if not present (requires SLIDE_LFs.rds)
    if [ ! -f "$PERF_CSV" ] && [ -f "${R_DIR}/SLIDE_LFs.rds" ]; then
        Rscript "${SCRIPT_DIR}/extract_r_performance.R" "$R_DIR" 2>/dev/null
    fi

    # Return CSV path if exists
    if [ -f "$PERF_CSV" ]; then
        echo "$PERF_CSV"
    fi
}

# Read metric from performance_metrics.csv
# Usage: read_r_metric <csv_path> <column_name>
read_r_metric() {
    local CSV="$1"
    local COL="$2"
    if [ -f "$CSV" ]; then
        # Get column index (handle quoted headers), then extract value (skip header)
        awk -F',' -v col="$COL" 'NR==1 {for(i=1;i<=NF;i++) {gsub(/"/,"",$i); if($i==col) c=i}} NR==2 && c {print $c}' "$CSV"
    fi
}

# Find equivalent parameter directory handling R vs Python naming differences
# R uses: 0.05_1_out, Python uses: 0.05_1.0_out
# Usage: find_param_dir <base_path> <combo_name>
# Returns: actual directory path if found, empty otherwise
find_param_dir() {
    local BASE="$1"
    local COMBO="$2"
    local DIR="${BASE}/${COMBO}"

    # Try exact match first
    if [ -d "$DIR" ]; then
        echo "$DIR"
        return
    fi

    # Try converting integer to decimal: 0.05_1_out -> 0.05_1.0_out
    local ALT_COMBO=$(echo "$COMBO" | sed -E 's/_([0-9]+)_out$/_\1.0_out/')
    DIR="${BASE}/${ALT_COMBO}"
    if [ -d "$DIR" ]; then
        echo "$DIR"
        return
    fi

    # Try converting decimal to integer: 0.05_1.0_out -> 0.05_1_out
    ALT_COMBO=$(echo "$COMBO" | sed -E 's/_([0-9]+)\.0_out$/_\1_out/')
    DIR="${BASE}/${ALT_COMBO}"
    if [ -d "$DIR" ]; then
        echo "$DIR"
        return
    fi
}

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
            echo "  Performance metrics:"

            for i in "${!TASK_NAMES[@]}"; do
                TASK_NAME="${TASK_NAMES[$i]}"
                TASK_BASE="${OUT_PATH}/${TASK_NAME}"
                TASK_DIR=$(find_param_dir "$TASK_BASE" "$COMBO")

                # Skip if no matching directory found
                if [ -z "$TASK_DIR" ]; then
                    printf "    %-20s: (no results)\n" "$TASK_NAME"
                    continue
                fi

                if [ "$TASK_NAME" = "R_native" ]; then
                    # R output: extract from RDS via performance_metrics.csv
                    PERF_CSV=$(extract_r_metrics "$TASK_DIR")
                    if [ -n "$PERF_CSV" ]; then
                        TRUE_SCORE=$(read_r_metric "$PERF_CSV" "true_score")
                        NUM_MARG=$(read_r_metric "$PERF_CSV" "num_marginals")
                        NUM_INT=$(read_r_metric "$PERF_CSV" "num_interactors")
                        PARTIAL=$(read_r_metric "$PERF_CSV" "partial_random")
                        FULL=$(read_r_metric "$PERF_CSV" "full_random")
                        # Format output, handling NA values
                        [[ "$TRUE_SCORE" == "NA" || -z "$TRUE_SCORE" ]] && TRUE_SCORE="-"
                        [[ "$PARTIAL" == "NA" || -z "$PARTIAL" ]] && PARTIAL="-"
                        [[ "$FULL" == "NA" || -z "$FULL" ]] && FULL="-"
                        [[ -z "$NUM_MARG" ]] && NUM_MARG="-"
                        [[ -z "$NUM_INT" ]] && NUM_INT="-"
                        if [[ "$TRUE_SCORE" == "-" ]]; then
                            printf "    %-20s: AUC=%s (P=%s F=%s) M=%s I=%s\n" \
                                   "$TASK_NAME" "$TRUE_SCORE" "$PARTIAL" "$FULL" "$NUM_MARG" "$NUM_INT"
                        else
                            printf "    %-20s: AUC=%.3f (P=%.3f F=%.3f) M=%s I=%s\n" \
                                   "$TASK_NAME" "$TRUE_SCORE" "$PARTIAL" "$FULL" "$NUM_MARG" "$NUM_INT"
                        fi
                    else
                        printf "    %-20s: (no metrics extracted)\n" "$TASK_NAME"
                    fi
                else
                    # Python output: read from scores.txt and sig_LFs.txt
                    SCORES_FILE="${TASK_DIR}/scores.txt"
                    LF_FILE="${TASK_DIR}/sig_LFs.txt"
                    if [ -f "$SCORES_FILE" ]; then
                        TRUE_SCORE=$(grep -oP 'True Scores:\s*\K-?[\d.]+' "$SCORES_FILE" || echo "")
                        PARTIAL=$(grep -oP 'Partial Random:\s*\K-?[\d.]+' "$SCORES_FILE" || echo "")
                        FULL=$(grep -oP 'Full Random:\s*\K-?[\d.]+' "$SCORES_FILE" || echo "")
                        NUM_MARG=$(grep -oP 'Number of marginals:\s*\K\d+' "$SCORES_FILE" || echo "-")
                        NUM_INT=$(grep -oP 'Number of interactions:\s*\K\d+' "$SCORES_FILE" || echo "-")
                        if [[ -n "$TRUE_SCORE" && -n "$PARTIAL" && -n "$FULL" ]]; then
                            printf "    %-20s: AUC=%.3f (P=%.3f F=%.3f) M=%s I=%s\n" \
                                   "$TASK_NAME" "$TRUE_SCORE" "$PARTIAL" "$FULL" "$NUM_MARG" "$NUM_INT"
                        else
                            printf "    %-20s: (incomplete scores)\n" "$TASK_NAME"
                        fi
                    elif [ -f "$LF_FILE" ]; then
                        NUM_LFS=$(wc -l < "$LF_FILE")
                        printf "    %-20s: %d LFs (no scores)\n" "$TASK_NAME" "$NUM_LFS"
                    else
                        printf "    %-20s: (no results)\n" "$TASK_NAME"
                    fi
                fi
            done
        done
    fi
fi

# -----------------------------------------------------------------------------
# Latent Factor Content Comparison (if multiple completed)
# -----------------------------------------------------------------------------
if [ "$COMPLETED" -ge 2 ]; then
    echo ""
    echo "=============================================================="
    echo "Latent Factor Content Comparison"
    echo "=============================================================="
    echo ""
    echo "Running compare_latent_factors.py for detailed LF comparison..."
    echo ""

    # Run the Python comparison script
    if [ -f "${SCRIPT_DIR}/compare_latent_factors.py" ]; then
        $PYTHON_ENV "${SCRIPT_DIR}/compare_latent_factors.py" "$OUT_PATH" --detailed 2>&1 || {
            echo "Warning: compare_latent_factors.py failed or not available"
        }
    else
        echo "Warning: compare_latent_factors.py not found at ${SCRIPT_DIR}"
    fi
fi

echo ""
echo "=============================================================="
echo "Report complete"
echo "=============================================================="
