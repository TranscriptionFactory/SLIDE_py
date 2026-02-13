#!/bin/bash
#SBATCH --job-name=ssc_multi_param
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --output=/ix/djishnu/Aaron/1_general_use/SLIDE_py/runs/ssc_multi_param/ssc_run_%j.log

set -euo pipefail

PROJECT_DIR="/ix/djishnu/Aaron/1_general_use/SLIDE_py"
RUN_DIR="${PROJECT_DIR}/runs/ssc_multi_param"
JOB_TAG="${SLURM_JOB_ID:-$$}"
TEST_ENV="${RUN_DIR}/conda_envs/env_${JOB_TAG}"
PKG_CACHE="${RUN_DIR}/conda_envs/.conda_pkgs_${JOB_TAG}"
BUILD_COPY="${RUN_DIR}/conda_envs/build_${JOB_TAG}"
MAMBA=~/.local/bin/mamba

# ---------------------------------------------------------------------------
# Cleanup: always remove the build env, even on failure
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "=== Cleaning up conda environment ==="
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

export PYTHONUNBUFFERED=1

eval "$(conda shell.bash hook)"

# ---------------------------------------------------------------------------
# Create fresh conda env
# ---------------------------------------------------------------------------
echo "=== Creating conda environment at ${TEST_ENV} ==="
mkdir -p "${PKG_CACHE}"
export CONDA_PKGS_DIRS="${PKG_CACHE}"

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
# Install loveslide + R deps from current source (includes fixes)
# ---------------------------------------------------------------------------
echo "=== Creating per-job build copy at ${BUILD_COPY} ==="
mkdir -p "${BUILD_COPY}"
(cd "${PROJECT_DIR}" && tar cf - src pyproject.toml setup.py MANIFEST.in README.md) \
  | (cd "${BUILD_COPY}" && tar xf -)
rm -rf "${BUILD_COPY}"/src/*.egg-info

echo "=== Installing loveslide from source ==="
cd "${BUILD_COPY}"
pip install ".[dev,r,viz]"
cd "${PROJECT_DIR}"

echo ""
echo "=== Installed packages ==="
pip list | grep -iE "loveslide|numpy|scipy|scikit|pandas|cvxpy|rpy2"

# ---------------------------------------------------------------------------
# Run the SSc multi-parameter pipeline
# ---------------------------------------------------------------------------
echo ""
echo "=== Running SSc multi-delta/lambda pipeline ==="
echo "  deltas:   [0.01, 0.1]"
echo "  lambdas:  [0.1, 1.0]"
echo "  backends: r_knockoffs, r, python"
echo ""

python "${RUN_DIR}/run_ssc.py" 2>&1

echo ""
echo "=== SSc multi-param run complete ==="
