#!/bin/bash
#SBATCH --job-name=loveslide_backend_test
#SBATCH --time=04-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --output=logs/test_output_%j.log

set -euo pipefail

PROJECT_DIR="/ix/djishnu/Aaron/1_general_use/SLIDE_py"
TEST_ENV="${PROJECT_DIR}/tests/test_env"
MAMBA=~/.local/bin/mamba

# ---------------------------------------------------------------------------
# Cleanup: always remove the test env, even on failure
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "=== Cleaning up test environment ==="
    conda deactivate 2>/dev/null || true
    "${MAMBA}" env remove --prefix "${TEST_ENV}" -y 2>/dev/null || true
    rm -rf "${TEST_ENV}" 2>/dev/null || true
    echo "=== Cleanup complete ==="
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module load python/ondemand-jupyter-python3.11
module load r/4.4.0

eval "$(conda shell.bash hook)"

# ---------------------------------------------------------------------------
# Create fresh conda env in tests/
# ---------------------------------------------------------------------------
echo "=== Creating test conda environment at ${TEST_ENV} ==="
"${MAMBA}" create --prefix "${TEST_ENV}" python=3.11 -y
conda activate "${TEST_ENV}"

echo "Python: $(python --version)"
echo "Location: $(which python)"

# ---------------------------------------------------------------------------
# Install loveslide + test/R deps into the fresh env
# ---------------------------------------------------------------------------
echo "=== Installing loveslide ==="
cd "${PROJECT_DIR}"
pip install ".[dev,r]"

echo ""
echo "=== Installed packages ==="
pip list | grep -iE "loveslide|numpy|scipy|scikit|pandas|cvxpy|rpy2|pytest"

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
echo ""
echo "=== Running tests ==="
python -m pytest tests/ -v --tb=short -s 2>&1

# python -m pytest tests/test_pipeline.py -v --tb=short -s 2>&1
