#!/bin/bash
#SBATCH --job-name=knockoff_submit
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=logs/knockoff_submit_%j.out
#SBATCH --error=logs/knockoff_submit_%j.err

# =============================================================================
# Knockoff Comparison v2 - Coordinator Job
# =============================================================================
# This small job submits all three phases with automatic dependencies.
# Usage: sbatch submit_knockoff_v2.sh [config.yaml]
# =============================================================================

set -e

cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison
mkdir -p logs

CONFIG="${1:-comparison_config_binary.yaml}"

echo "=============================================================="
echo "Knockoff Comparison v2 - Coordinator"
echo "=============================================================="
echo "Config: $CONFIG"
echo "Timestamp: $(date)"
echo "=============================================================="

# -----------------------------------------------------------------------------
# Phase 1: LOVE pre-computation (2 tasks: R and Python)
# -----------------------------------------------------------------------------
LOVE_JOB=$(sbatch --parsable << 'LOVE_EOF'
#!/bin/bash
#SBATCH --job-name=knockoff_love
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-1
#SBATCH --output=logs/knockoff_love_%A_%a.out
#SBATCH --error=logs/knockoff_love_%A_%a.err

cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison

module load gcc/12.2.0
module load python/ondemand-jupyter-python3.11
module load r/4.4.0

export PYTHONPATH="/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter:$PYTHONPATH"

LOVE_BACKENDS=("r" "python")
LOVE_BACKEND="${LOVE_BACKENDS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="output_comparison/knockoff_cmp_${SLURM_ARRAY_JOB_ID}"

mkdir -p "$OUTPUT_DIR"

echo "Phase 1: LOVE backend=$LOVE_BACKEND"
echo "Output: $OUTPUT_DIR"

/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python \
    run_knockoff_comparison_v2.py \
    --config comparison_config_binary.yaml \
    --output-dir "$OUTPUT_DIR" \
    --phase love \
    --love-backend "$LOVE_BACKEND"

touch "$OUTPUT_DIR/.love_${LOVE_BACKEND}_complete"
LOVE_EOF
)

echo "Phase 1 (LOVE): Job $LOVE_JOB submitted"

# -----------------------------------------------------------------------------
# Phase 2: Knockoff computation (6 tasks)
# -----------------------------------------------------------------------------
KNOCKOFF_JOB=$(sbatch --parsable --dependency=afterok:${LOVE_JOB} << KNOCKOFF_EOF
#!/bin/bash
#SBATCH --job-name=knockoff_run
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-5
#SBATCH --output=logs/knockoff_run_%A_%a.out
#SBATCH --error=logs/knockoff_run_%A_%a.err

cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison

module load gcc/12.2.0
module load python/ondemand-jupyter-python3.11
module load r/4.4.0

export PYTHONPATH="/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter:\$PYTHONPATH"

BACKENDS=("R_native" "knockoff_filter" "knockoff_filter_sklearn" "knockpy_lsm" "knockpy_lasso" "custom_glmnet")
BACKEND="\${BACKENDS[\$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="output_comparison/knockoff_cmp_${LOVE_JOB}"

echo "Phase 2: Knockoff backend=\$BACKEND"
echo "Output: \$OUTPUT_DIR"

/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python \\
    run_knockoff_comparison_v2.py \\
    --config comparison_config_binary.yaml \\
    --output-dir "\$OUTPUT_DIR" \\
    --phase knockoff \\
    --backend "\$BACKEND"

touch "\$OUTPUT_DIR/.\${BACKEND}_complete"
KNOCKOFF_EOF
)

echo "Phase 2 (Knockoffs): Job $KNOCKOFF_JOB submitted (depends on $LOVE_JOB)"

# -----------------------------------------------------------------------------
# Phase 3: Aggregation
# -----------------------------------------------------------------------------
AGG_JOB=$(sbatch --parsable --dependency=afterok:${KNOCKOFF_JOB} << AGG_EOF
#!/bin/bash
#SBATCH --job-name=knockoff_agg
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/knockoff_agg_%j.out
#SBATCH --error=logs/knockoff_agg_%j.err

cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison

module load gcc/12.2.0
module load python/ondemand-jupyter-python3.11

echo "Phase 3: Aggregating results"

/ix3/djishnu/AaronR/8_build/.conda/envs/loveslide_env/bin/python \\
    run_knockoff_comparison_v2.py \\
    --config comparison_config_binary.yaml \\
    --output-dir "output_comparison/knockoff_cmp_${LOVE_JOB}" \\
    --phase aggregate

echo "Done! Results in: output_comparison/knockoff_cmp_${LOVE_JOB}/full_comparison.json"
AGG_EOF
)

echo "Phase 3 (Aggregate): Job $AGG_JOB submitted (depends on $KNOCKOFF_JOB)"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "All jobs submitted!"
echo "=============================================================="
echo "Phase 1 (LOVE):      $LOVE_JOB (2 tasks: R + Python)"
echo "Phase 2 (Knockoffs): $KNOCKOFF_JOB (6 tasks, depends on $LOVE_JOB)"
echo "Phase 3 (Aggregate): $AGG_JOB (depends on $KNOCKOFF_JOB)"
echo ""
echo "Output directory: output_comparison/knockoff_cmp_${LOVE_JOB}/"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "=============================================================="
