#!/bin/bash
#SBATCH --job-name=loveslide_backend_test
#SBATCH --time=04-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --output=logs/test_output_%j.log

set -euo pipefail

PROJECT_DIR="/ix/djishnu/Aaron/1_general_use/SLIDE_py"
JOB_TAG="${SLURM_JOB_ID:-$$}"
TEST_ENV="${PROJECT_DIR}/tests/conda_envs/test_env_${JOB_TAG}"
PKG_CACHE="${PROJECT_DIR}/tests/conda_envs/.conda_pkgs_${JOB_TAG}"
BUILD_COPY="${PROJECT_DIR}/tests/conda_envs/build_${JOB_TAG}"
MAMBA=~/.local/bin/mamba

# ---------------------------------------------------------------------------
# Cleanup: always remove the test env, even on failure
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "=== Cleaning up test environment ==="
    conda deactivate 2>/dev/null || true
    "${MAMBA}" env remove --prefix "${TEST_ENV}" -y 2>/dev/null || true
    rm -rf "${TEST_ENV}" "${PKG_CACHE}" "${BUILD_COPY}" 2>/dev/null || true
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
mkdir -p "${PKG_CACHE}"
export CONDA_PKGS_DIRS="${PKG_CACHE}"

# Override .condarc so mamba doesn't scan the shared pkgs_dirs
CONDARC_TEMP="${PKG_CACHE}/.condarc"
cat > "${CONDARC_TEMP}" <<'RCEOF'
channels:
  - conda-forge
  - bioconda
  - defaults
auto_activate_base: false
RCEOF
export CONDARC="${CONDARC_TEMP}"
"${MAMBA}" create --prefix "${TEST_ENV}" python=3.11 -y
conda activate "${TEST_ENV}"

echo "Python: $(python --version)"
echo "Location: $(which python)"

# ---------------------------------------------------------------------------
# Install loveslide + test/R deps into the fresh env
# ---------------------------------------------------------------------------
echo "=== Creating per-job build copy at ${BUILD_COPY} ==="
mkdir -p "${BUILD_COPY}"
(cd "${PROJECT_DIR}" && tar cf - src pyproject.toml setup.py MANIFEST.in README.md) \
  | (cd "${BUILD_COPY}" && tar xf -)
rm -rf "${BUILD_COPY}"/src/*.egg-info

echo "=== Installing loveslide from isolated build copy ==="
cd "${BUILD_COPY}"
pip install ".[dev,r,viz]"
cd "${PROJECT_DIR}"

echo ""
echo "=== Installed packages ==="
pip list | grep -iE "loveslide|numpy|scipy|scikit|pandas|cvxpy|rpy2|pytest"

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
echo ""
echo "=== Running tests ==="
python -m pytest tests/ -v --tb=auto -s 2>&1

# python -m pytest tests/test_pipeline.py -v --tb=short -s 2>&1
