# 3-Way SLIDE Comparison Implementation Plan

## Overview

Compare three SLIDE implementations with a parameter grid:

| Task | Implementation | LOVE Backend | Knockoffs |
|------|---------------|--------------|-----------|
| 0 | R SLIDE (native) | R | R |
| 1 | loveslide + R LOVE | R (via rpy2) | R (via rpy2) |
| 2 | loveslide + Py LOVE | Python | R (via rpy2) |

## Changes Required

### 1. Add R LOVE wrapper to loveslide (`src/loveslide/love.py`)

Add `call_love_r()` function that:
- Sources R scripts from `LOVE-SLIDE/` directory
- Calls `getLatentFactors()` via rpy2
- Returns results in same format as Python LOVE

Add `backend` parameter to `call_love()`:
```python
def call_love(X, ..., backend='python', **kwargs):
    if backend == 'r':
        return call_love_r(X, ...)
    else:
        return LOVE(X, ...)  # current Python implementation
```

### 2. Update `OptimizeSLIDE` class (`src/loveslide/slide.py`)

Add `love_backend` parameter to `get_latent_factors()`:
```python
def get_latent_factors(self, ..., love_backend='python', ...):
    love_result = call_love(x, ..., backend=love_backend)
```

Thread this through `run_pipeline()` via `input_params['love_backend']`.

### 3. Update YAML config (`comparison/comparison_config.yaml`)

Add parameter grid:
```yaml
delta:
  - 0.05
  - 0.1
  - 0.2

lambda:
  - 0.1
  - 0.5
  - 1.0
```

### 4. Update comparison script (`comparison/run_comparison.sh`)

Change array size: `#SBATCH --array=0-2`

Add Task 2 for Python LOVE:
```bash
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    # R SLIDE (native)
    ...
elif [ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]; then
    # Python + R LOVE
    "$PYTHON_ENV" run_slide_py.py "$YAML_CONFIG" "$PY_R_OUT" --love-backend r
elif [ "${SLURM_ARRAY_TASK_ID}" -eq 2 ]; then
    # Python + Python LOVE
    "$PYTHON_ENV" run_slide_py.py "$YAML_CONFIG" "$PY_PY_OUT" --love-backend python
fi
```

### 5. Update Python script (`comparison/run_slide_py.py`)

Add `--love-backend` CLI argument that passes to `OptimizeSLIDE`.

## File Changes Summary

| File | Change |
|------|--------|
| `src/loveslide/love.py` | Add `call_love_r()`, add `backend` param |
| `src/loveslide/slide.py` | Add `love_backend` param threading |
| `comparison/comparison_config.yaml` | Expand delta/lambda grid |
| `comparison/run_comparison.sh` | 3-task array, separate outputs |
| `comparison/run_slide_py.py` | Add `--love-backend` CLI arg |

## Output Structure

```
outputs/SSc_comparison/
├── R_outputs/           # Task 0: R SLIDE
│   └── {delta}_{lambda}_out/
├── Py_R_LOVE_outputs/   # Task 1: Python + R LOVE
│   └── {delta}_{lambda}_out/
└── Py_Py_LOVE_outputs/  # Task 2: Python + Python LOVE
    └── {delta}_{lambda}_out/
```

## Permissions Needed

- Run Python scripts with R module loaded
- Execute SLURM jobs
