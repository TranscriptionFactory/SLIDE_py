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
#   bash run_report.sh [out_path]
#
# The out_path should contain subdirectories for each implementation:
#   R_native/       - R SLIDE (native)
#   Py_rLOVE_rKO/   - Python (R LOVE + R Knockoffs)
#   Py_rLOVE_knockpy/ - Python (R LOVE + knockpy Knockoffs)
#   Py_pyLOVE_rKO/  - Python (Py LOVE + R Knockoffs)
#   Py_pyLOVE_knockpy/ - Python (Py LOVE + knockpy Knockoffs)
#
# Example:
#   bash run_report.sh /path/to/output >> report.txt
# =============================================================================

set -e

OUT_PATH="${1:-/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/updated_outputs}"
SCRIPT_DIR=/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
cd "$SCRIPT_DIR"

# Load modules (for R metrics extraction if needed)
module load gcc/12.2.0 2>/dev/null || true
module load python/ondemand-jupyter-python3.11 2>/dev/null || true
module load r/4.4.0 2>/dev/null || true

PYTHON_ENV="${PYTHON_ENV:-/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python}"

# Run the unified Python report generator
$PYTHON_ENV "${SCRIPT_DIR}/compare_latent_factors.py" "$OUT_PATH" --detailed
