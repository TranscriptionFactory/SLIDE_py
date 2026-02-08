#!/bin/bash
#SBATCH --job-name=loveslide_backend_test
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --output=tests/test_output_%j.log

# Load R and Python modules for rpy2 support
module load python/ondemand-jupyter-python3.11
module load r/4.4.0

eval "$(conda shell.bash hook)"
conda activate loveslide_test

cd /ix/djishnu/Aaron/1_general_use/SLIDE_py
python -m pytest tests/test_pipeline.py -v --tb=short -s 2>&1
