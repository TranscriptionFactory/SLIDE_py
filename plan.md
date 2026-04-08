# Plan: Knockoff Pipeline R-Matching Fixes

## Changes

### 1. SDP solver: Keep the binary-search feasibility scaling (no change)
The binary-search feasibility scaling in `create_solve_sdp` (solve.py:376-398) and `create_solve_asdp` (solve.py:582-614) ensures `2*Sigma - diag(diag_s) > 0` — this is a **correctness** requirement. R's SDP solver does the same feasibility enforcement internally via Rdsdp's native PSD constraint. Removing it would produce invalid knockoff matrices, not better R concordance. **No code change needed.**

### 2. findOptIter: Make non-default in voting path
**Problem:** `knockoff_filter_voting_slide` always passes `slide_selection=True` to `knockoff_filter_voting`, which forces findOptIter refinement. R's `selectShortFreq` uses findOptIter, but to isolate whether it helps or hurts R concordance, it should be opt-in.

**Files to change:**
- `src/loveslide/knockoff/filter.py`:
  - `knockoff_filter_voting_slide()` (line 967): Add `slide_selection: bool = False` parameter
  - Lines 1102, 1139, 1198: Change `slide_selection=True` → `slide_selection=slide_selection`
  - Line 1089 (single-chunk path): Same change
- `src/loveslide/knockoffs.py`:
  - `select_short_freq_slide()` (line 802): Add `slide_selection: bool = False` parameter, pass through to `knockoff_filter_voting_slide`

This means:
- Default behavior: raw frequency voting (like R's `r` backend)
- `slide_selection=True`: enables findOptIter refinement (still available, just not default)

### 3. W statistic: Change nlambda default to 250
**Files to change:**
- `src/loveslide/knockoff/stats/glmnet.py`:
  - Line 46: `_lasso_max_lambda_glmnet()` — change `nlambda: int = 100` → `nlambda: int = 250`
  - Line 62: Fix docstring to match (`default=250`)
  - Line 224: `_cv_coeffs_glmnet()` — change `nlambda: int = 100` → `nlambda: int = 250`
  - Line 239: Fix docstring to match (`default=250`)
- `src/loveslide/knockoffs.py`:
  - Line 442: `_compute_glmnet_lambdasmax()` — change `nlambda=500` → `nlambda=250`
  - Line 460: Fix docstring to match

## Order of operations
1. Change nlambda defaults (stats/glmnet.py, knockoffs.py)
2. Make findOptIter non-default (filter.py, knockoffs.py)
3. Commit
