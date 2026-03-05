#!/bin/bash
#
# Generic YAML-driven SLURM array submission for SLIDE.
# No hardcoded paths — everything comes from the config file.
#
# Usage (from login node or interactive shell):
#   ./submit_slide.sh config.yaml                  # build env, auto-cleanup
#   ./submit_slide.sh config.yaml --keep-env       # build env, keep for reuse
#   ./submit_slide.sh config.yaml /path/to/env     # reuse a pre-built env
#
# The script self-submits: it reads the YAML to determine backends and SLURM
# resources, then calls sbatch on itself. Inside SLURM it runs one backend
# per array task.

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    cat >&2 <<'USAGE'
Usage: submit_slide.sh <config.yaml> [--keep-env | /path/to/existing/env]

  config.yaml   SLIDE run configuration (see example_config.yaml)
  --keep-env    Build a fresh env but keep it after the run
  /path/to/env  Reuse an existing conda environment
USAGE
    exit 1
fi

CONFIG_RAW="$1"; shift
ENV_ARG="${1:-}"

# Resolve config to absolute path
CONFIG="$(cd "$(dirname "$CONFIG_RAW")" && pwd)/$(basename "$CONFIG_RAW")"
CONFIG_DIR="$(dirname "$CONFIG")"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG" >&2
    exit 1
fi

# ── Locate script directory ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PROJECT_DIR is resolved later from YAML (env.project_dir), with git fallback.

# ── YAML helper ───────────────────────────────────────────────────────────────
# Queries the config via Python. Falls back to a default if the key is missing.
yaml_query() {
    python3 -c "
import yaml, sys, os
cfg = yaml.safe_load(open('${CONFIG}'))
config_dir = '${CONFIG_DIR}'
expr = sys.argv[1]
val = eval(expr, {'cfg': cfg, 'os': os, 'config_dir': config_dir})
print(val)
" "$1"
}

# Resolve a path from the YAML: absolute paths pass through, relative paths
# are resolved from the config file's directory.
resolve_path() {
    local raw="$1"
    if [[ "$raw" == /* ]]; then
        echo "$raw"
    else
        echo "${CONFIG_DIR}/${raw}"
    fi
}

# ── Resolve project directory from YAML (with git fallback) ──────────────────
_resolve_project_dir() {
    local from_yaml
    from_yaml=$(yaml_query "cfg.get('env',{}).get('project_dir','')" 2>/dev/null || true)
    if [[ -n "$from_yaml" && "$from_yaml" != "None" && -d "$from_yaml" ]]; then
        echo "$from_yaml"
        return
    fi
    # Fallback: git root from script location
    local git_root
    git_root="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -n "$git_root" ]]; then
        echo "$git_root"
        return
    fi
    echo "ERROR: Could not find loveslide repo. Set env.project_dir in your config." >&2
    return 1
}

# ── Self-submit from login node ──────────────────────────────────────────────
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # Bootstrap: we need python3 + pyyaml to parse the config.
    # If the system python can't import yaml, try loading a python module.
    if ! python3 -c "import yaml" 2>/dev/null; then
        for _try_mod in python/ondemand-jupyter-python3.11 python/3.11 python3; do
            module load "$_try_mod" 2>/dev/null && break || true
        done
        if ! python3 -c "import yaml" 2>/dev/null; then
            echo "ERROR: python3 with pyyaml is required to parse the config." >&2
            echo "  Load a python module first, or install pyyaml." >&2
            exit 1
        fi
    fi

    PROJECT_DIR="$(_resolve_project_dir)"

    N_BACKENDS=$(yaml_query "len(cfg['slide']['backends'])")
    JOB_NAME=$(yaml_query   "cfg.get('slurm',{}).get('job_name','slide_run')")
    SLURM_TIME=$(yaml_query "cfg.get('slurm',{}).get('time','2-12:00:00')")
    SLURM_MEM=$(yaml_query  "cfg.get('slurm',{}).get('mem','50G')")
    SLURM_CPUS=$(yaml_query "cfg.get('slurm',{}).get('cpus_per_task',4)")
    SLURM_CLUSTER=$(yaml_query "cfg.get('slurm',{}).get('cluster','htc')")
    OUT_BASE_RAW=$(yaml_query "cfg.get('output',{}).get('base_dir','./output')")
    LOG_DIR="$(resolve_path "$OUT_BASE_RAW")"

    echo "=== Submitting SLURM array job ==="
    echo "  config:   ${CONFIG}"
    echo "  project:  ${PROJECT_DIR}"
    echo "  backends: ${N_BACKENDS}"
    echo "  array:    0-$((N_BACKENDS - 1))"
    echo "  cluster:  ${SLURM_CLUSTER}"
    echo "  time:     ${SLURM_TIME}  mem: ${SLURM_MEM}  cpus: ${SLURM_CPUS}"
    echo "  logs:     ${LOG_DIR}/"

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

    if [[ -n "$ENV_ARG" ]]; then
        SLIDE_JOB=$(sbatch --parsable "${SBATCH_ARGS[@]}" "${BASH_SOURCE[0]}" "$CONFIG" "$ENV_ARG")
    else
        SLIDE_JOB=$(sbatch --parsable "${SBATCH_ARGS[@]}" "${BASH_SOURCE[0]}" "$CONFIG")
    fi

    # Extract numeric job ID (sbatch --parsable may return "JOBID;CLUSTER")
    SLIDE_JOB_ID="${SLIDE_JOB%%;*}"
    echo "  Submitted array job: ${SLIDE_JOB_ID}"

    # ── Submit comparison job (runs after all array tasks finish) ─────────
    COMPARE_SCRIPT="${SCRIPT_DIR}/compare_outputs.py"
    if [[ -f "$COMPARE_SCRIPT" ]]; then
        COMPARE_JOB=$(sbatch --parsable \
            --job-name="${JOB_NAME}_compare" \
            --dependency="afterok:${SLIDE_JOB_ID}" \
            --time="00:30:00" \
            --mem="8G" \
            --cpus-per-task=1 \
            --cluster="${SLURM_CLUSTER}" \
            --output="${LOG_DIR}/${JOB_NAME}_compare_%j.log" \
            --wrap="module load python/ondemand-jupyter-python3.11 && \
                    python '${COMPARE_SCRIPT}' \
                    --config '${CONFIG}' \
                    --job-id '${SLIDE_JOB_ID}' \
                    --output '${LOG_DIR}/comparison_report_${SLIDE_JOB_ID}.txt'"
        )
        COMPARE_JOB_ID="${COMPARE_JOB%%;*}"
        echo "  Submitted comparison job: ${COMPARE_JOB_ID} (afterok:${SLIDE_JOB_ID})"
        echo "  Report will be at: ${LOG_DIR}/comparison_report_${SLIDE_JOB_ID}.txt"
    fi

    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Below runs inside SLURM (one array task per backend)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Read config values ────────────────────────────────────────────────────────
# We need modules loaded before yaml_query works inside SLURM, so we parse
# the modules list with a minimal regex-based approach that doesn't need pyyaml.
_parse_modules() {
    # Extract modules from YAML without pyyaml — handles "- module_name" lines
    # under the env.modules key. Falls back silently if parsing fails.
    python3 -c "
import re, sys
text = open('${CONFIG}').read()
# Find the modules: block under env:
m = re.search(r'(?:^|\n)env:\s*\n((?:[ \t]+.*\n)*)', text)
if not m:
    sys.exit(0)
env_block = m.group(1)
m2 = re.search(r'modules:\s*\n((?:[ \t]+- .*\n)*)', env_block)
if not m2:
    sys.exit(0)
for line in m2.group(1).strip().split('\n'):
    mod = re.sub(r'^\s*-\s*', '', line).strip()
    if mod:
        print(mod)
" 2>/dev/null || true
}

# Load configured modules
while IFS= read -r mod; do
    [[ -n "$mod" ]] && module load "$mod" 2>/dev/null || true
done < <(_parse_modules)

export PYTHONUNBUFFERED=1

# Activate conda — try multiple paths since module loading may provide it
if ! command -v conda &>/dev/null; then
    # Try common conda locations
    for _conda_path in \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh" \
        "${HOME}/.conda/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"; do
        if [[ -f "$_conda_path" ]]; then
            source "$_conda_path"
            break
        fi
    done
fi
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda not found after loading modules. Check env.modules in config." >&2
    echo "  Tried loading modules from: ${CONFIG}" >&2
    echo "  Ensure a module provides conda (e.g., python/ondemand-jupyter-python3.11)" >&2
    exit 1
fi

PROJECT_DIR="$(_resolve_project_dir)"
BACKEND=$(yaml_query "cfg['slide']['backends'][${SLURM_ARRAY_TASK_ID}]")
N_BACKENDS=$(yaml_query "len(cfg['slide']['backends'])")
OUT_BASE_RAW=$(yaml_query "cfg.get('output',{}).get('base_dir','./output')")
OUT_BASE="$(resolve_path "$OUT_BASE_RAW")"
PY_VERSION=$(yaml_query "cfg.get('env',{}).get('python_version','3.11')")
PIP_EXTRAS=$(yaml_query "cfg.get('env',{}).get('pip_extras','dev,r,viz')")

# ── Detect mamba/conda binary ────────────────────────────────────────────────
MAMBA_CFG=$(yaml_query "cfg.get('env',{}).get('mamba','auto')")
if [[ "$MAMBA_CFG" == "auto" || "$MAMBA_CFG" == "None" || -z "$MAMBA_CFG" ]]; then
    # Auto-detect: try mamba, then conda
    if command -v mamba &>/dev/null; then
        MAMBA="$(command -v mamba)"
    elif [[ -x "${HOME}/.local/bin/mamba" ]]; then
        MAMBA="${HOME}/.local/bin/mamba"
    elif command -v conda &>/dev/null; then
        MAMBA="$(command -v conda)"
    else
        echo "ERROR: Could not find mamba or conda. Set env.mamba in your config." >&2
        exit 1
    fi
else
    MAMBA="$MAMBA_CFG"
fi

# ── Env path logic ────────────────────────────────────────────────────────────
ENV_DIR="${OUT_BASE}/conda_envs"
KEEP_ENV=false
SKIP_BUILD=false

if [[ "$ENV_ARG" == "--keep-env" ]]; then
    SHARED_ENV="${ENV_DIR}/env_${SLURM_ARRAY_JOB_ID}"
    KEEP_ENV=true
elif [[ -n "$ENV_ARG" ]]; then
    SHARED_ENV="$(cd "$(dirname "$ENV_ARG")" && pwd)/$(basename "$ENV_ARG")"
    SKIP_BUILD=true
else
    SHARED_ENV="${ENV_DIR}/env_${SLURM_ARRAY_JOB_ID}"
fi

PKG_CACHE="${ENV_DIR}/.conda_pkgs_${SLURM_ARRAY_JOB_ID}"
BUILD_COPY="${ENV_DIR}/build_${SLURM_ARRAY_JOB_ID}"
SENTINEL="${SHARED_ENV}/.ready"

echo "=== Array task ${SLURM_ARRAY_TASK_ID}: backend=${BACKEND} ==="
echo "  config:  ${CONFIG}"
echo "  project: ${PROJECT_DIR}"
echo "  env:     ${SHARED_ENV}  (prebuilt=${SKIP_BUILD})"

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
        echo "  Reuse with:  ${BASH_SOURCE[0]} ${CONFIG} ${SHARED_ENV}"
        return
    fi

    touch "${SHARED_ENV}/.done_${SLURM_ARRAY_TASK_ID}" 2>/dev/null || true

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

# ── Build or activate env ─────────────────────────────────────────────────────
if [[ "$SKIP_BUILD" == true ]]; then
    if [[ ! -x "${SHARED_ENV}/bin/python" ]]; then
        echo "ERROR: ${SHARED_ENV}/bin/python not found or not executable" >&2
        exit 1
    fi
    echo "=== Activating prebuilt environment ==="
    conda activate "${SHARED_ENV}"

elif [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
    echo "=== Task 0: creating shared conda environment ==="
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
    "${MAMBA}" create --prefix "${SHARED_ENV}" "python=${PY_VERSION}" -y

    conda activate "${SHARED_ENV}"

    # Build + install loveslide from source
    echo "=== Installing loveslide from ${PROJECT_DIR} ==="
    mkdir -p "${BUILD_COPY}"
    (cd "${PROJECT_DIR}" && tar cf - src pyproject.toml setup.py MANIFEST.in README.md) \
      | (cd "${BUILD_COPY}" && tar xf -)
    rm -rf "${BUILD_COPY}"/src/*.egg-info

    cd "${BUILD_COPY}"
    pip install ".[${PIP_EXTRAS}]"
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
    echo "WARNING: conda activate PATH mismatch" >&2
    echo "  expected: ${SHARED_ENV}/bin/python" >&2
    echo "  got:      ${ACTUAL_PYTHON}" >&2
    export PATH="${SHARED_ENV}/bin:${PATH}"
fi

echo "Python: $(python --version)"
echo "Location: $(which python)"

# ── Run SLIDE ─────────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUT_BASE}/${SLURM_ARRAY_JOB_ID}"

echo ""
echo "=== Running SLIDE pipeline: backend=${BACKEND} ==="
echo "  config: ${CONFIG}"
echo "  output: ${OUTPUT_DIR}/${BACKEND}/"
echo ""

python "${SCRIPT_DIR}/run_slide.py" \
    --config "${CONFIG}" \
    --backend "${BACKEND}" \
    --out-dir "${OUTPUT_DIR}" \
    2>&1

echo ""
echo "=== ${BACKEND} backend complete ==="
