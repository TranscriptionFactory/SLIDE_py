#!/bin/bash
#SBATCH --job-name=SLIDE_py
#SBATCH --output=test.out
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G                   
#SBATCH --time=2-12:00:00               
#SBATCH --partition=htc


conda init
source ~/.bashrc
conda activate /ix3/djishnu/alw399/envs/rhino
echo $CONDA_DEFAULT_ENV

python test.py