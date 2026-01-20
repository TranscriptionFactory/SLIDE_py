#!/bin/bash
#SBATCH --job-name=debug_glmnet
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/debug_glmnet_%j.out
#SBATCH --error=logs/debug_glmnet_%j.err

set -e

module load gcc/12.2.0
module load r/4.4.0
module load python/ondemand-jupyter-python3.11
source activate loveslide_env

export R_HOME=/software/rhel9/manual/install/r/4.4.0/lib64/R
export LD_LIBRARY_PATH="$R_HOME/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter:$PYTHONPATH"

cd /ix/djishnu/Aaron/1_general_use/SLIDE_py
mkdir -p logs

python -u comparison/diagnostics/debug_knockoff_entry_times.py
