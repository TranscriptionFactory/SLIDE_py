#!/bin/bash
#SBATCH --job-name=slide_glmnet_fix
#SBATCH --output=slide_glmnet_fix_%j.out
#SBATCH --error=slide_glmnet_fix_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --cluster=smp

cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison

echo "============================================================"
echo "Running Python SLIDE with fixed glmnet statistic"
echo "============================================================"

/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python run_slide_py.py \
    comparison_config_binary.yaml \
    ./test_output_glmnet_fixed \
    --knockoff-backend knockpy \
    --fstat glmnet

echo "============================================================"
echo "SLIDE with fixed glmnet completed"
echo "============================================================"
