#!/bin/bash
#SBATCH --job-name=ssc_R_ground_truth
#SBATCH --time=2-12:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --output=/ix/djishnu/Aaron/1_general_use/SLIDE_py/runs/ssc_multi_param/ssc_R_ground_truth_%j.log
#
# Native R SLIDE ground-truth run on SSc data.
# Compares against Python loveslide job 8039789 with identical parameters.
#
# Usage:
#   sbatch submit_ssc_R.sh                  # auto-timestamped output dir
#   sbatch submit_ssc_R.sh /path/to/outdir  # explicit output dir

set -euo pipefail

module load r/4.4.0

export SLIDE_LOCAL_REPO=/ix/djishnu/Aaron/1_general_use/SLIDE

RUN_DIR="/ix/djishnu/Aaron/1_general_use/SLIDE_py/runs/ssc_multi_param"

echo "=== SSc R ground-truth run ==="
echo "  Job ID:    ${SLURM_JOB_ID:-$$}"
echo "  SLIDE repo: ${SLIDE_LOCAL_REPO}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo ""

if [[ -n "${1:-}" ]]; then
    Rscript "${RUN_DIR}/run_ssc_R.R" "$1" 2>&1
else
    Rscript "${RUN_DIR}/run_ssc_R.R" 2>&1
fi

echo ""
echo "=== R ground-truth run complete ==="
