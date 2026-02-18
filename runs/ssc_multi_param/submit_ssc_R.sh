#!/bin/bash
#SBATCH --job-name=ssc_R_ground_truth
#SBATCH --time=2-12:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --output=ssc_R_ground_truth_%j.log

set -euo pipefail

module load r/4.4.0

export SLIDE_LOCAL_REPO=/ix/djishnu/Aaron/1_general_use/SLIDE

cd "${SLURM_SUBMIT_DIR}"

echo "=== SSc R Ground Truth ==="
echo "  SLIDE repo: ${SLIDE_LOCAL_REPO}"
echo "  Working dir: $(pwd)"
echo "  Job ID: ${SLURM_JOB_ID}"
echo ""

Rscript run_ssc_R.R
