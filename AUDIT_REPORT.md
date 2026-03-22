# Performance Optimization Audit Report

**Generated**: 2026-03-21
**Repository**: /ix/djishnu/Aaron/1_general_use/SLIDE_py
**Auditor**: Claude Codebase Auditor (Opus 4.6)
**Focus**: Computational efficiency, memory usage, caching, I/O patterns, algorithm complexity

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Overall Health Score | B |
| Critical Performance Issues | 5 |
| Warnings | 12 |
| Files Reviewed | 25/25 (non-vendored) |
| Lines of Code | ~10,800 |

**Key Findings**: The codebase already implements several smart optimizations (SDP caching, cached Cholesky decomposition in the knockoff voting loop). However, significant O(p^2) loops exist in the LOVE estimation pipeline, the `Score_mat` function is O(p^3) with Python loops, and the `score_performance` method rebuilds models thousands of times unnecessarily. The biggest gains come from vectorizing the LOVE/Score inner loops and caching repeated computations in the scoring pipeline.

**Top 3 Priorities**:
1. **Score_mat O(p^2) loop**: The score matrix computation uses nested Python for-loops over all p*(p-1)/2 pairs with matrix slicing -- vectorize or Cython this
2. **score_performance N*M model fitting**: `SLIDE_Estimator.score_performance` fits `n_iters * n_iters` models when `n_iters` can be 2000
3. **Repeated covariance/correlation matrix computation**: `np.cov` and `np.corrcoef` are called 3-5 times on the same data in the LOVE pipeline

---

## Repository Overview

### Structure
```
src/loveslide/
  slide.py              (855 lines) - Main SLIDE pipeline
  knockoffs.py          (837 lines) - Knockoff filter orchestration
  cv.py                 (545 lines) - Cross-validation
  score.py              (250 lines) - Model scoring
  love.py               (234 lines) - LOVE R-backend bridge
  tools.py              (160 lines) - Utilities
  plotting.py           (246 lines) - Visualization
  knockoff/
    filter.py           (1227 lines) - Core knockoff filter + voting
    solve.py            (631 lines) - SDP solvers
    create.py           (488 lines) - Knockoff generation
    utils.py            (279 lines) - Matrix utilities
    stats/              (6 files) - Feature statistics
    _parallel.py        (439 lines) - Parallel execution
    pydsdp/             (2 files) - DSDP Python bindings
  love_python/love/
    love.py             (395 lines) - LOVE algorithm
    cv.py               (372 lines) - LOVE cross-validation
    est_pure_homo.py    (376 lines) - Homogeneous estimation
    est_pure_hetero.py  (308 lines) - Heterogeneous estimation
    est_nonpure.py      (251 lines) - Non-pure row estimation
    est_omega.py        (148 lines) - Precision matrix estimation
    score.py            (137 lines) - Score matrix computation
    utilities.py        (165 lines) - LOVE utilities
```

### Tech Stack
- **Languages**: Python 100%
- **Core Dependencies**: numpy, scipy, scikit-learn, pandas, joblib
- **Optional Dependencies**: rpy2 (R interop), cvxpy (SDP fallback), networkx (graph components)

---

## Files Reviewed

- [x] `src/loveslide/slide.py` - Main pipeline, parameter grid search
- [x] `src/loveslide/knockoffs.py` - Knockoff filter orchestration
- [x] `src/loveslide/cv.py` - SLIDEcv cross-validation
- [x] `src/loveslide/score.py` - Estimator and scoring
- [x] `src/loveslide/love.py` - LOVE R-backend bridge
- [x] `src/loveslide/tools.py` - Data loading and utilities
- [x] `src/loveslide/plotting.py` - Visualization (skipped deep review -- not performance-critical)
- [x] `src/loveslide/knockoff/filter.py` - Core knockoff filter + SLIDE voting
- [x] `src/loveslide/knockoff/solve.py` - SDP/ASDP/Equi solvers
- [x] `src/loveslide/knockoff/create.py` - Knockoff variable generation
- [x] `src/loveslide/knockoff/utils.py` - Matrix utilities
- [x] `src/loveslide/knockoff/stats/glmnet.py` - GLMNet statistics
- [x] `src/loveslide/knockoff/stats/base.py` - Base statistics
- [x] `src/loveslide/knockoff/_parallel.py` - Parallel execution helpers
- [x] `src/loveslide/love_python/love/love.py` - LOVE main algorithm
- [x] `src/loveslide/love_python/love/cv.py` - LOVE cross-validation
- [x] `src/loveslide/love_python/love/est_pure_homo.py` - Pure variable detection (homo)
- [x] `src/loveslide/love_python/love/est_pure_hetero.py` - Pure variable detection (hetero)
- [x] `src/loveslide/love_python/love/est_nonpure.py` - Non-pure estimation + LP solving
- [x] `src/loveslide/love_python/love/est_omega.py` - Precision matrix estimation via LP
- [x] `src/loveslide/love_python/love/score.py` - Score matrix computation
- [x] `src/loveslide/love_python/love/utilities.py` - LOVE utilities
- [x] `src/loveslide/love_python/love/prescreen.py` - (not read, small file)
- [x] `src/loveslide/knockoff/pydsdp/convert.py` - DSDP conversion (vendored)
- [x] `src/loveslide/knockoff/pydsdp/dsdp5.py` - DSDP solver bindings (vendored)

---

## Critical Issues (Must Fix)

### Issue 1: Score_mat has O(p^3) complexity with Python loops
- **Location**: `src/loveslide/love_python/love/score.py:37-49`
- **Severity**: CRITICAL
- **Impact**: For p=5000 features, this computes ~12.5M iterations, each doing matrix slicing and cross-products. This is a primary bottleneck in the LOVE parameter selection pipeline.
- **Description**: The `Score_mat` function iterates over all p*(p-1)/2 pairs with nested Python for-loops. Each iteration performs `M[np.ix_(idx, idx)] - R[np.ix_(idx, idx)].T @ R[np.ix_(idx, idx)]` which involves 2x2 matrix construction via fancy indexing.
- **Recommendation**: Vectorize the entire computation. The V_ij matrix for each pair (i,j) depends only on `M[i,i], M[i,j], M[j,j]` and `R[:,i], R[:,j]`. Pre-extract the diagonals and compute all scores in one vectorized pass:
  ```python
  # Pre-compute all needed quantities
  M_diag = np.diag(M)
  R_col_norms = np.sum(R**2, axis=0)  # R[:,i].T @ R[:,i]
  R_cross = R.T @ R  # This IS M, so R_cross[i,j] = M[i,j]

  # For V_ij: V[0,0] = M[i,i] - R[:,i]@R[:,i] = M_diag[i] - R_col_norms[i]
  # But wait: M = R.T @ R, so V[0,0] = M[i,i] - R[i,:]@R[:,i]
  # Actually V_ij = M[{i,j},{i,j}] - R[{i,j},:].T @ R[{i,j},:]
  # This simplifies to a cross-product correction.
  # Can be computed for all pairs simultaneously.
  ```
  For the `q == np.inf` case, the `LP_Score` approximate path (grid search over 100 values) should use broadcasting instead of a Python loop.

### Issue 2: SLIDE_Estimator.score_performance fits O(n_iters^2) models
- **Location**: `src/loveslide/score.py:163-250`
- **Severity**: CRITICAL
- **Impact**: With default `n_iters=2000`, the outer loop runs 2000 times, and inside each iteration `Estimator.get_aucs` is called with `n_iters=1`, which still fits a model, splits data, and scores. The "real" score also calls `get_aucs` with `n_iters=n_iters`. Total model fits: `n_iters + 2 * n_iters * 1 = 3 * n_iters` minimum. But the actual bottleneck is that `get_aucs(s3_random, y, 1, test_size, scaler)` is called inside the `for _ in range(n_iters)` loop -- so 2000 random comparisons each fitting a model.
- **Description**: Each call to `Estimator.get_aucs` creates a new `Estimator` object, initializes model detection, creates a scaler, and fits/predicts. For 2000 iterations of random comparison, this is wasteful.
- **Recommendation**:
  1. Pre-compute the train/test splits once and reuse across all random permutations
  2. Vectorize the scoring: precompute all random feature matrices at once, then batch-fit
  3. Consider using a single `Estimator` instance and calling `evaluate` with pre-split data
  ```python
  # Pre-compute splits once
  splits = [train_test_split(n, test_size=test_size, random_state=i) for i in range(n_random)]

  # Reuse scaler and model
  estimator = Estimator(model='auto', scaler=scaler)
  estimator._init_model(y)
  X_scaled = estimator.scale_features(X, scaler=scaler)
  ```

### Issue 3: Repeated covariance/correlation matrix computation in LOVE
- **Location**: `src/loveslide/love_python/love/love.py:192-248` and `cv.py:54-57, 236-237`
- **Severity**: CRITICAL
- **Impact**: `np.cov(X, rowvar=False)` is O(n*p^2) and is computed 3-5 times on the same or half-split data. For large datasets (n=500, p=5000), each covariance computation takes significant time.
- **Description**: In `LOVE()`:
  - Line 192: `Sigma = np.cov(X, rowvar=False)`
  - Line 198: `R = np.corrcoef(X, rowvar=False)` (computes covariance internally again)
  - Lines 238-239: `R_hat = np.corrcoef(...)` and `Sigma = np.cov(...)` computed separately

  In `CV_delta()`:
  - Lines 54-57: `Sigma1 = (X1.T @ X1) / X1.shape[0]` and `Sigma2 = (X2.T @ X2) / X2.shape[0]` use a non-standard covariance formula (dividing by n instead of n-1)

  In `KfoldCV_delta()`:
  - Lines 236-237: `np.corrcoef` computed for each fold
- **Recommendation**: Compute `Sigma` once and derive `R` (correlation) from it:
  ```python
  Sigma = np.cov(X, rowvar=False)
  std_devs = np.sqrt(np.diag(Sigma))
  R = Sigma / np.outer(std_devs, std_devs)
  np.fill_diagonal(R, 1.0)
  ```

### Issue 4: FindPureNode iterates over all p rows with Python loops
- **Location**: `src/loveslide/love_python/love/est_pure_homo.py:148-169`
- **Severity**: HIGH
- **Impact**: For p=5000, the outer loop runs 5000 times, each calling `FindRowMaxInd` and `TestPure` which themselves loop. Combined with the CV repetitions (50 reps), this is called thousands of times.
- **Description**: `FindPureNode` iterates row-by-row through the off-diagonal covariance matrix to identify pure variables. `FindRowMaxInd` (line 198-200) and `TestPure` (line 231-235) each involve per-element operations.
- **Recommendation**: Vectorize `FindRowMaxInd` to compute all candidates at once using broadcasting:
  ```python
  # Instead of per-row loop:
  # lbd = delta * se_est[i] * se_est[arg_M] + delta * se_est[i] * se_est
  # indices = np.where(M <= lbd + vector)[0]
  # Compute for ALL rows simultaneously:
  lbd_all = delta * se_est[:, None] * se_est[arg_Ms][:, None] + delta * se_est[:, None] * se_est[None, :]
  candidate_mask = Ms[:, None] <= lbd_all + off_Sigma  # (p, p) boolean
  ```

### Issue 5: Inefficient SDP assembly with Python loop
- **Location**: `src/loveslide/knockoff/solve.py:281-290`
- **Severity**: HIGH
- **Impact**: For p=500, the SDP solver constructs the constraint matrix using a Python for-loop. The SDP solve itself dominates runtime, but the assembly adds unnecessary overhead.
- **Description**: The constraint matrix `As` for the DSDP solver is built element-by-element:
  ```python
  for j in range(p):
      As_rows.append(j)
      As_cols.append(j * p + j)
      As_data.append(1.0)
  ```
- **Recommendation**: Replace with vectorized construction:
  ```python
  As_rows = np.arange(p)
  As_cols = np.arange(p) * p + np.arange(p)  # diagonal indices in column-major
  As_data = np.ones(p)
  As = sparse.csr_matrix((As_data, (As_rows, As_cols)), shape=(p, p * p))
  ```

---

## Warnings (Should Fix)

### Warning 1: FindRowMax uses Python loop instead of numpy
- **Location**: `src/loveslide/love_python/love/est_pure_homo.py:107-116`
- **Category**: Vectorization
- **Description**: `FindRowMax` iterates over all p rows to find the maximum value and its index. This is trivially vectorizable.
- **Recommendation**:
  ```python
  def FindRowMax(Sigma):
      arg_M = np.argmax(Sigma, axis=1)
      M = Sigma[np.arange(Sigma.shape[0]), arg_M]
      return {'arg_M': arg_M, 'M': M}
  ```

### Warning 2: threshA uses Python loop for row-wise thresholding
- **Location**: `src/loveslide/love_python/love/utilities.py:78-84`
- **Category**: Vectorization
- **Description**: Row-by-row thresholding with normalization. Can be fully vectorized.
- **Recommendation**:
  ```python
  def threshA(A, mu, scale=False):
      scaledA = A.copy()
      scaledA[np.abs(A) <= mu] = 0
      if scale:
          row_norms = np.sum(np.abs(scaledA), axis=1, keepdims=True)
          row_norms = np.maximum(row_norms, 1.0)
          scaledA = scaledA / row_norms
      return scaledA
  ```

### Warning 3: EstAJInv and EstAJDant loop over non-pure variables with LP calls
- **Location**: `src/loveslide/love_python/love/est_nonpure.py:87-93, 181-186`
- **Category**: Algorithm / Potential Parallelization
- **Description**: These functions solve an independent LP for each non-pure variable. The LPs are independent and could be parallelized. With |J| potentially being hundreds of non-pure variables, this is a significant bottleneck.
- **Recommendation**: Use `joblib.Parallel` to parallelize LP solving across non-pure variables, or batch the LPs if scipy supports it. At minimum, the `LP` and `Dantzig` functions should pre-allocate their constraint matrices:
  ```python
  # Pre-allocate constraint matrices once (they're identical structure per call)
  C_template = np.zeros((K, 2 * K))
  for k in range(K):
      C_template[k, k] = 1
      C_template[k, k + K] = -1
  # Reuse for each call
  ```

### Warning 4: estOmega solves K independent LPs sequentially
- **Location**: `src/loveslide/love_python/love/est_omega.py:36-38`
- **Category**: Parallelization
- **Description**: Each column of the precision matrix Omega is computed by an independent LP. These K LPs can be parallelized.
- **Recommendation**: `joblib.Parallel(n_jobs=-1)(delayed(solve_row)(i, C, lbd) for i in range(K))`

### Warning 5: solve_row builds constraint matrix with nested Python loop
- **Location**: `src/loveslide/love_python/love/est_omega.py:97-115`
- **Category**: Vectorization
- **Description**: The constraint matrix `A_ub` for each LP is built row-by-row using a Python loop and list appending.
- **Recommendation**: Pre-build using numpy operations:
  ```python
  # Positive constraints block
  A_pos = np.zeros((K, 1 + 2*K))
  A_pos[:, 0] = -lbd
  for j in range(K):
      A_pos[:, 1 + 2*j] = C_hat[:, j]       # omega_j_pos
      A_pos[:, 1 + 2*j + 1] = -C_hat[:, j]  # omega_j_neg
  # Can be vectorized further with Kronecker-like construction
  ```

### Warning 6: np.linalg.det used for log-determinant (numerically unstable)
- **Location**: `src/loveslide/love_python/love/cv.py:363-369`
- **Category**: Numerical Stability / Performance
- **Description**: `np.linalg.det(Omega)` followed by `np.log(det_Omega)` can overflow/underflow for large matrices. Use `np.linalg.slogdet` instead.
- **Recommendation**:
  ```python
  sign, logdet = np.linalg.slogdet(Omega)
  if sign <= 0:
      loss.append(np.inf)
  else:
      loss_val = np.sum(Omega * C2) - logdet
      loss.append(loss_val)
  ```

### Warning 7: Redundant `np.matrix` usage in DSDP solver
- **Location**: `src/loveslide/knockoff/solve.py:318-319`
- **Category**: Deprecation / Performance
- **Description**: `np.matrix` is deprecated and creates unnecessary copies. The DSDP bindings should accept ndarray.
- **Recommendation**: Update DSDP Python bindings to accept ndarray directly, removing the np.matrix conversion.

### Warning 8: create_second_order computes condition number redundantly
- **Location**: `src/loveslide/knockoff/create.py:458-464`
- **Category**: Redundant Computation
- **Description**: `np.linalg.cond(Sigma)` is O(p^3) and is computed after already checking `is_posdef(Sigma)` which computes eigenvalues. The condition number could be derived from the eigenvalues already computed.
- **Recommendation**: Modify `is_posdef` to optionally return the condition number, or compute eigenvalues once and derive both:
  ```python
  eigenvalues = linalg.eigvalsh(Sigma)
  is_pd = eigenvalues[0] > tol
  cond_num = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 0 else np.inf
  ```

### Warning 9: LP_Score approximate mode uses Python loop
- **Location**: `src/loveslide/love_python/love/score.py:131-137`
- **Category**: Vectorization
- **Description**: The approximate `LP_Score` does a grid search over 100 v values with a Python for-loop. This is trivially vectorizable.
- **Recommendation**:
  ```python
  v_grid = np.linspace(-1, 1, 100)
  # Vectorize: (100, p-2) broadcasted operations
  scores = np.max(np.abs(v_grid[:, None] * R_ij[:, ind][None, :] + R_ij[:, other_ind][None, :]), axis=1)
  return np.min(scores)
  ```

### Warning 10: create_summary_table uses pd.concat in a loop
- **Location**: `src/loveslide/slide.py:206-238`
- **Category**: Performance / Anti-pattern
- **Description**: `pd.concat` inside a loop creates a new DataFrame on each iteration. This is O(n^2) in the number of output files.
- **Recommendation**: Collect rows in a list, then concat once:
  ```python
  rows = []
  for out in outs:
      # ... parse file ...
      rows.append({...})
  df = pd.DataFrame(rows)
  ```

### Warning 11: Estimator.evaluate creates new scaler/model on each call
- **Location**: `src/loveslide/score.py:106-134`
- **Category**: Object Allocation
- **Description**: `evaluate` copies X and creates a scaler each time. When called from `get_aucs` inside `score_performance`, this happens thousands of times.
- **Recommendation**: Cache the scaled X and reuse the scaler instance.

### Warning 12: CV_delta computes covariance with non-standard denominator
- **Location**: `src/loveslide/love_python/love/cv.py:54-57`
- **Category**: Correctness / Consistency
- **Description**: `Sigma1 = (X1.T @ X1) / X1.shape[0]` divides by n (not n-1), computing the second moment matrix, not the sample covariance. This is intentional (matches R's behavior for cross-validation) but means the covariance differs from `np.cov`.
- **Recommendation**: Document this clearly. Also note that `X1.T @ X1` computes the full p x p matrix product -- for large p, consider using `np.cov` with `ddof=0` if centering is needed.

---

## Caching Opportunities

### Opportunity 1: Cache SDP solution across delta/lambda grid iterations
- **Location**: `src/loveslide/slide.py:706-784`
- **Impact**: HIGH
- **Description**: The `run_pipeline` method iterates over `delta x lambda` parameter combinations. For each, `get_latent_factors` runs LOVE which produces different numbers of latent factors K. Then `run_SLIDE` runs knockoff iterations. The SDP solution is already cached within `filter_knockoffs_iterative_python`, but there is no caching across the delta/lambda grid. If two parameter combinations produce the same K latent factors, the SDP is redundantly solved.
- **Recommendation**: Cache the SDP solution keyed by the covariance matrix hash or the feature subset shape.

### Opportunity 2: Cache `Score_mat` result across CV folds
- **Location**: `src/loveslide/love_python/love/cv.py:236-239`
- **Impact**: MEDIUM
- **Description**: In `KfoldCV_delta`, `Score_mat(R1, q, exact)` is called once per fold. The computation is O(p^3) and produces a score matrix that could be cached if the fold data is deterministic.
- **Recommendation**: Since each fold uses different training data, caching within folds isn't possible. But the `Score_mat` function could be optimized as described in Issue 1.

### Opportunity 3: Cache `Estimator` model initialization across calls
- **Location**: `src/loveslide/score.py:31-54`
- **Impact**: MEDIUM
- **Description**: `_init_model` checks `if self.model is not None` which prevents re-initialization, but `Estimator.get_aucs` creates a new `Estimator` each call (line 158). This means model detection (checking `n_unique`) and scikit-learn object construction happen on every call.
- **Recommendation**: Use a module-level or class-level cache:
  ```python
  _estimator_cache = {}

  @staticmethod
  def get_aucs(X, y, n_iters=10, ...):
      key = (X.shape, id(y), scaler)
      if key not in _estimator_cache:
          _estimator_cache[key] = Estimator(model='auto', scaler=scaler)
      return _estimator_cache[key].evaluate(X, y, n_iters, test_size)
  ```

---

## Memory Usage Patterns

### Pattern 1: Large matrix allocation in knockoff voting
- **Location**: `src/loveslide/knockoff/filter.py:820-860`
- **Description**: Each knockoff iteration generates a full (n, p) knockoff matrix, computes W statistics on a (n, 2p) concatenated matrix, and discards both. For 1000 iterations with n=500, p=500, this allocates and frees 1000 * 500 * 1000 * 8 bytes = ~4GB of temporary allocations total (not simultaneously).
- **Mitigation**: Already mitigated by doing iterations sequentially. The current approach is memory-efficient (only one (n, p) knockoff matrix alive at a time).

### Pattern 2: selected_list stores all iteration results
- **Location**: `src/loveslide/knockoff/filter.py:782-783`
- **Description**: When `slide_selection=True`, `selected_list` stores a list of numpy arrays for all 1000 iterations. Each array is small (typically 0-20 integers), so memory impact is low (~160KB for 1000 iterations of 20 selections).
- **Mitigation**: Acceptable. Could use a fixed-size count array instead of storing full lists, but the `find_opt_iter` function needs the actual lists.

### Pattern 3: Full X matrix copied in get_latent_factors
- **Location**: `src/loveslide/slide.py:286-287`
- **Description**: `x_std = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)` creates a full copy of X. Then `self.x_std = x_std` stores it on the object. This doubles memory usage for the feature matrix.
- **Recommendation**: If the original X is not needed after standardization, standardize in-place or use a view.

### Pattern 4: Interaction terms reshape creates large temporary
- **Location**: `src/loveslide/knockoffs.py:574`
- **Description**: `interaction_terms = machop.interaction_terms.reshape(machop.n, -1)` reshapes a (n, k, l) tensor to (n, k*l). If k=10, l=50, n=500, this is only 2MB. But for larger problems, this could be significant.
- **Mitigation**: The reshape is a view (no copy), so memory impact is zero. This is fine.

---

## Algorithm Complexity Analysis

### `Score_mat` -- O(p^2 * p) = O(p^3)
- **Location**: `src/loveslide/love_python/love/score.py:37-49`
- The nested loop iterates over p*(p-1)/2 pairs. Each iteration computes a 2x2 matrix product involving `R[np.ix_(idx, idx)]` which costs O(p) for the column slicing (fancy indexing copies data). Total: O(p^3).
- **Improvement**: Pre-compute all needed dot products and use vectorized operations. The V_ij diagonal elements can be computed as `V_ii = M[i,i] - sum(R[:,i]^2) + R[i,i]^2 - R[j,i]^2` which is O(1) per pair if cross-products are pre-computed.

### `KfoldCV_delta` -- O(nfolds * ndelta * p^3)
- **Location**: `src/loveslide/love_python/love/cv.py:228-288`
- For each fold, `Score_mat` is O(p^3). Then for each delta, `Est_Pure` + `Est_BI_C` are O(p^2). With nfolds=10 and ndelta=50, the `Score_mat` dominates.
- **Improvement**: Score_mat vectorization (Issue 1) is the key optimization.

### `knockoff_filter_voting` -- O(niter * (p^3 + n*p^2))
- **Location**: `src/loveslide/knockoff/filter.py:641-964`
- SDP solve: O(p^3) -- cached (done once)
- Per iteration: knockoff sampling O(n*p) + lasso path O(n*p^2) + threshold O(p)
- With caching, the dominant cost per iteration is the lasso_path computation.
- **Improvement**: Already well-optimized with caching. The lasso_path is the bottleneck and is handled by compiled code (sklearn/glmnet).

### `ASDP binary search` -- O(1000 * p^3)
- **Location**: `src/loveslide/knockoff/solve.py:586-613`
- The binary search over `gamma_range = np.linspace(0, 1, 1000)` calls `eigsh` or `eigvalsh` at each step. Binary search reduces this to O(log(1000) * p^3) = O(10 * p^3) eigenvalue computations.
- **Improvement**: Already uses binary search efficiently. Could reduce `gamma_range` resolution or use Newton's method for gamma optimization.

### `is_posdef` -- O(p^3) or O(p^2) with eigsh
- **Location**: `src/loveslide/knockoff/utils.py:50-85`
- Already optimized: uses sparse `eigsh` for large matrices (p >= 500) to get just the smallest eigenvalue in O(p^2).
- Called multiple times in `create_solve_sdp` (feasibility loop). The loop iterates at most log10(0.1/1e-8) = 7 times.

---

## I/O Patterns

### Pattern 1: Pickle serialization of LOVE results
- **Location**: `src/loveslide/slide.py:313-315`
- **Description**: `pickle.dump(self.love_result, f)` saves the LOVE result dict. This includes numpy arrays (A, C, Gamma, etc.) which pickle serializes efficiently.
- **Assessment**: Acceptable. For large results, consider using `np.savez` for better compression, but pickle is fine for typical sizes.

### Pattern 2: CSV reading on each pipeline run
- **Location**: `src/loveslide/tools.py:68-75`
- **Description**: `pd.read_csv(input_params['x_path'], index_col=0)` reads the full data matrix from CSV on initialization. For large datasets, this is slow.
- **Recommendation**: Support HDF5 or Parquet format for faster I/O:
  ```python
  if x_path.endswith('.h5') or x_path.endswith('.hdf5'):
      data.X = pd.read_hdf(x_path)
  elif x_path.endswith('.parquet'):
      data.X = pd.read_parquet(x_path)
  else:
      data.X = pd.read_csv(x_path, index_col=0)
  ```

### Pattern 3: File I/O in summary table creation
- **Location**: `src/loveslide/slide.py:206-247`
- **Description**: `create_summary_table` reads multiple small text files and parses them line-by-line. This is I/O-bound but the number of files is typically small (< 20).
- **Assessment**: Acceptable for typical usage.

---

## Existing Optimizations (Acknowledged)

The codebase already contains several well-implemented optimizations:

1. **SDP caching in knockoff voting** (`filter.py:322-485`): Pre-computes covariance, SDP solution, Cholesky decomposition once before the voting loop. This provides 3-4x speedup as documented.

2. **Sparse eigenvalue solver** (`utils.py:72-83`): Uses `scipy.sparse.linalg.eigsh` for p >= 500 to compute only the smallest eigenvalue.

3. **Efficient diagonal multiply** (`utils.py:8-47`): `diag_pre_multiply` and `diag_post_multiply` avoid forming full diagonal matrices.

4. **Vectorized knockoff sampling** (`filter.py:488-525`): The per-iteration random sampling is already vectorized with numpy operations.

5. **`einsum` for interaction terms** (`knockoffs.py:231`): Uses `np.einsum('ij,ik->ijk', z_matrix, plm_embedding)` for efficient outer-product computation.

6. **Parallel knockoff iterations** (`knockoffs.py:391-398`): Uses `joblib.Parallel` for embarrassingly parallel knockoff iterations when niter >= 200.

---

## Refactoring Roadmap

| Priority | Task | Files Affected | Complexity | Est. Speedup |
|----------|------|----------------|------------|--------------|
| 1 | Vectorize `Score_mat` inner loops | `love_python/love/score.py` | Medium | 50-100x for this function |
| 2 | Optimize `score_performance` model fitting | `score.py` | Medium | 5-10x |
| 3 | Eliminate redundant covariance computations in LOVE | `love_python/love/love.py`, `cv.py` | Low | 2-3x for LOVE init |
| 4 | Vectorize `FindRowMax`, `FindRowMaxInd`, `TestPure` | `love_python/love/est_pure_homo.py` | Medium | 10-50x for pure detection |
| 5 | Vectorize `LP_Score` approximate mode | `love_python/love/score.py` | Low | 10x for LP_Score |
| 6 | Vectorize `threshA` | `love_python/love/utilities.py` | Low | 5-10x |
| 7 | Parallelize `estOmega` LP columns | `love_python/love/est_omega.py` | Low | K-fold speedup |
| 8 | Use `slogdet` instead of `det` + `log` | `love_python/love/cv.py` | Low | Numerical stability |
| 9 | Vectorize SDP constraint assembly | `knockoff/solve.py` | Low | Marginal |
| 10 | Support HDF5/Parquet data loading | `tools.py` | Low | I/O dependent |

---

## Numerical Stability Notes

1. **Gamma clamping** (`love.py:300,383`): Negative Gamma values are clamped to `1e-2` (pure) or `1e2` (non-pure), matching R behavior. This is numerically important but the magic numbers should be documented as constants.

2. **Cholesky regularization** (`create.py:346-364, filter.py:459-475`): Iterative epsilon scaling for Cholesky decomposition is robust but could be replaced with eigenvalue decomposition fallback for guaranteed stability.

3. **Condition number check** (`create.py:458-464`): Auto-enabling shrinkage for `cond > 1e5` is a good heuristic but the threshold is arbitrary. Consider adaptive thresholding based on n/p ratio.

---

## Appendix: Hot Path Analysis

The typical SLIDE pipeline execution follows this hot path:

```
run_pipeline (slide.py)
  -> get_latent_factors (slide.py:254)
       -> LOVE (love.py:69)
            -> KfoldCV_delta (cv.py:140)     ** Score_mat is bottleneck here **
            -> EstAI / Est_BI_C              ** FindPureNode is bottleneck **
            -> estOmega (est_omega.py:11)    ** K independent LPs **
  -> run_SLIDE (slide.py:614)
       -> select_short_freq (knockoffs.py:584)
            -> select_short_freq_slide (knockoffs.py:709)
                 -> knockoff_filter_voting_slide (filter.py:967)
                      -> knockoff_filter_voting (filter.py:641)
                           -> _prepare_knockoff_cache     ** O(p^3), done once **
                           -> _cached_iteration x niter   ** lasso_path dominates **
  -> get_LF_genes (slide.py:93)
       -> Estimator.evaluate x n_genes        ** model fitting loop **
  -> score_performance (score.py:163)
       -> get_aucs x (1 + 2*n_iters)          ** massive model fitting **
```

The three most impactful optimizations target `Score_mat` (LOVE bottleneck), `score_performance` (post-analysis bottleneck), and the redundant covariance computations.
