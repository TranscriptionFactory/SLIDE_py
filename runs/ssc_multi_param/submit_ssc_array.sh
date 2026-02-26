#!/bin/bash
#
# YAML-driven SLURM array submission for SLIDE multi-backend runs.
#
# Usage (from login node):
#   ./submit_ssc_array.sh config.yaml                   # build env, clean up after
#   ./submit_ssc_array.sh config.yaml --keep-env        # build env, keep for reuse
#   ./submit_ssc_array.sh config.yaml /path/to/env      # reuse a pre-built env
#
# The script self-submits: it reads the YAML to determine how many backends
# to run and what SLURM resources to request, then calls sbatch on itself.

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml> [--keep-env | /path/to/existing/env]" >&2
    exit 1
fi

CONFIG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
shift
ENV_ARG="${1:-}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG" >&2
    exit 1
fi

PROJECT_DIR="/ix/djishnu/Aaron/1_general_use/SLIDE_py"
RUN_DIR="${PROJECT_DIR}/runs/ssc_multi_param"
MAMBA=/ihome/djishnu/aar126/.local/bin/mamba

# ── Helper: extract values from YAML via Python ──────────────────────────────
yaml_query() {
    python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${CONFIG}'))
expr = sys.argv[1]
print(eval(expr, {'cfg': cfg}))
" "$1"
}

# ── Self-submit from login node ──────────────────────────────────────────────
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    module load python/ondemand-jupyter-python3.11 2>/dev/null || true

    N_BACKENDS=$(yaml_query "len(cfg['slide']['backends'])")
    JOB_NAME=$(yaml_query "cfg.get('slurm',{}).get('job_name','slide_run')")
    SLURM_TIME=$(yaml_query "cfg.get('slurm',{}).get('time','2-12:00:00')")
    SLURM_MEM=$(yaml_query "cfg.get('slurm',{}).get('mem','50G')")
    SLURM_CPUS=$(yaml_query "cfg.get('slurm',{}).get('cpus_per_task',4)")
    SLURM_CLUSTER=$(yaml_query "cfg.get('slurm',{}).get('cluster','htc')")
    LOG_DIR=$(yaml_query "cfg.get('output',{}).get('base_dir','${RUN_DIR}/output')")

    echo "=== Submitting SLURM array job ==="
    echo "  config:   ${CONFIG}"
    echo "  backends: ${N_BACKENDS}"
    echo "  array:    0-$((N_BACKENDS - 1))"
    echo "  cluster:  ${SLURM_CLUSTER}"
    echo "  time:     ${SLURM_TIME}"
    echo "  mem:      ${SLURM_MEM}"
    echo "  cpus:     ${SLURM_CPUS}"

    mkdir -p "${LOG_DIR}"

    SBATCH_ARGS=(
        --job-name="${JOB_NAME}"
        --time="${SLURM_TIME}"
        --mem="${SLURM_MEM}"
        --cpus-per-task="${SLURM_CPUS}"
        --cluster="${SLURM_CLUSTER}"
        --array="0-$((N_BACKENDS - 1))"
        --output="${LOG_DIR}/${JOB_NAME}_%A_%a.log"
    )

    # Forward the original script + config + optional env arg
    if [[ -n "$ENV_ARG" ]]; then
        sbatch "${SBATCH_ARGS[@]}" "$0" "$CONFIG" "$ENV_ARG"
    else
        sbatch "${SBATCH_ARGS[@]}" "$0" "$CONFIG"
    fi
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Below runs inside SLURM (one task per backend)
# ═══════════════════════════════════════════════════════════════════════════════

BACKEND=$(yaml_query "cfg['slide']['backends'][${SLURM_ARRAY_TASK_ID}]")
N_BACKENDS=$(yaml_query "len(cfg['slide']['backends'])")

# ── Env path logic ────────────────────────────────────────────────────────────
KEEP_ENV=false
SKIP_BUILD=false

if [[ "$ENV_ARG" == "--keep-env" ]]; then
    SHARED_ENV="${RUN_DIR}/conda_envs/env_${SLURM_ARRAY_JOB_ID}"
    KEEP_ENV=true
elif [[ -n "$ENV_ARG" ]]; then
    SHARED_ENV="$ENV_ARG"
    SKIP_BUILD=true
else
    SHARED_ENV="${RUN_DIR}/conda_envs/env_${SLURM_ARRAY_JOB_ID}"
fi

PKG_CACHE="${RUN_DIR}/conda_envs/.conda_pkgs_${SLURM_ARRAY_JOB_ID}"
BUILD_COPY="${RUN_DIR}/conda_envs/build_${SLURM_ARRAY_JOB_ID}"
SENTINEL="${SHARED_ENV}/.ready"

echo "=== Array task ${SLURM_ARRAY_TASK_ID}: backend=${BACKEND} ==="
echo "  config: ${CONFIG}"
echo "  env:    ${SHARED_ENV}  (prebuilt=${SKIP_BUILD})"

# ── Cleanup ───────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    conda deactivate 2>/dev/null || true

    if [[ "$SKIP_BUILD" == true ]]; then
        echo "=== Using prebuilt env — skipping cleanup ==="
        return
    fi

    rm -rf "${PKG_CACHE}" "${BUILD_COPY}" 2>/dev/null || true

    if [[ "$KEEP_ENV" == true ]]; then
        echo "=== --keep-env: preserving conda env at ${SHARED_ENV} ==="
        echo "  Reuse with:  $0 ${CONFIG} ${SHARED_ENV}"
        return
    fi

    touch "${SHARED_ENV}/.done_${SLURM_ARRAY_TASK_ID}" 2>/dev/null || true

    # Check if all tasks are done
    ALL_DONE=true
    for i in $(seq 0 $((N_BACKENDS - 1))); do
        if [[ ! -f "${SHARED_ENV}/.done_${i}" ]]; then
            ALL_DONE=false
            break
        fi
    done

    if [[ "$ALL_DONE" == true ]]; then
        echo "=== All tasks done — cleaning up shared conda environment ==="
        "${MAMBA}" env remove --prefix "${SHARED_ENV}" -y 2>/dev/null || true
        rm -rf "${SHARED_ENV}" 2>/dev/null || true
        echo "=== Cleanup complete ==="
    else
        echo "=== Other tasks still running — skipping env cleanup ==="
    fi
}
trap cleanup EXIT

# ── Module + conda init ───────────────────────────────────────────────────────
module load python/ondemand-jupyter-python3.11
module load r/4.4.0

export PYTHONUNBUFFERED=1
eval "$(conda shell.bash hook)"

# ── Activate / build env ─────────────────────────────────────────────────────
if [[ "$SKIP_BUILD" == true ]]; then
    if [[ ! -x "${SHARED_ENV}/bin/python" ]]; then
        echo "ERROR: ${SHARED_ENV}/bin/python not found or not executable" >&2
        exit 1
    fi
    echo "=== Activating prebuilt environment ==="
    conda activate "${SHARED_ENV}"

elif [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
    echo "=== Task 0: creating shared conda environment at ${SHARED_ENV} ==="
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
    "${MAMBA}" create --prefix "${SHARED_ENV}" python=3.11 -y

    conda activate "${SHARED_ENV}"

    echo "=== Creating build copy at ${BUILD_COPY} ==="
    mkdir -p "${BUILD_COPY}"
    (cd "${PROJECT_DIR}" && tar cf - src pyproject.toml setup.py MANIFEST.in README.md) \
      | (cd "${BUILD_COPY}" && tar xf -)
    rm -rf "${BUILD_COPY}"/src/*.egg-info

    echo "=== Installing loveslide from source ==="
    cd "${BUILD_COPY}"
    pip install ".[dev,r,viz]"
    pip install pyyaml
    cd "${PROJECT_DIR}"

    echo ""
    echo "=== Installed packages ==="
    pip list | grep -iE "loveslide|numpy|scipy|scikit|pandas|cvxpy|rpy2|pyyaml"

    touch "${SENTINEL}"
    echo "=== Environment ready (sentinel written) ==="

else
    echo "=== Task ${SLURM_ARRAY_TASK_ID}: waiting for shared conda environment ==="
    while [[ ! -f "${SENTINEL}" ]]; do
        sleep 10
    done
    echo "=== Sentinel found — activating shared environment ==="

    export CONDA_PKGS_DIRS="${PKG_CACHE}"
    export CONDARC="${PKG_CACHE}/.condarc"
    conda activate "${SHARED_ENV}"
fi

# Verify conda activate
ACTUAL_PYTHON="$(which python)"
if [[ "$ACTUAL_PYTHON" != "${SHARED_ENV}/bin/python" ]]; then
    echo "WARNING: conda activate did not set PATH correctly" >&2
    echo "  expected: ${SHARED_ENV}/bin/python" >&2
    echo "  got:      ${ACTUAL_PYTHON}" >&2
    echo "  Forcing PATH prepend as fallback..." >&2
    export PATH="${SHARED_ENV}/bin:${PATH}"
fi

echo "Python: $(python --version)"
echo "Location: $(which python)"

# ── Run SLIDE ─────────────────────────────────────────────────────────────────
OUT_BASE=$(yaml_query "cfg.get('output',{}).get('base_dir','${RUN_DIR}/output')")
OUTPUT_DIR="${OUT_BASE}/${SLURM_ARRAY_JOB_ID}"

echo ""
echo "=== Running SLIDE pipeline: backend=${BACKEND} ==="
echo "  config: ${CONFIG}"
echo "  output: ${OUTPUT_DIR}/${BACKEND}/"
echo ""

python "${RUN_DIR}/run_ssc.py" \
    --config "${CONFIG}" \
    --backend "${BACKEND}" \
    --out-dir "${OUTPUT_DIR}" \
    2>&1

echo ""
echo "=== ${BACKEND} backend complete ==="
