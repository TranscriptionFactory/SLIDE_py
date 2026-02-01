# ADR-001: Knockoff Voting Parallelization Strategy

## Status
Proposed

## Context

The `knockoff_filter_voting` function in `/ix/djishnu/Aaron/1_general_use/SLIDE_py/src/loveslide/knockoff/filter.py` runs 500 iterations sequentially by default. Each iteration involves:

1. **Knockoff creation** (`create_second_order`): ~50-200ms
   - Covariance estimation (Ledoit-Wolf or sample)
   - SDP/ASDP optimization for diag_s
   - Cholesky decomposition and sampling

2. **Statistic computation** (`stat_glmnet_coefdiff`): ~100-500ms
   - LASSO path computation via sklearn/glmnet
   - Cross-validation for lambda selection

Total per iteration: ~150-700ms depending on n, p dimensions.
Total for 500 iterations: **75-350 seconds** sequentially.

### Current Parallelization Issues

The existing parallel implementation (lines 451-482) has limitations:

```python
if n_jobs == 1:
    # Sequential execution (safer, works with custom knockoffs/statistic)
    ...
else:
    # Parallel execution (only works with default knockoffs/statistic)
    if knockoffs is not create_second_order or statistic is not None:
        warnings.warn("Falling back to sequential...")
```

**Problems:**
1. Cannot pickle lambdas or custom callables
2. Falls back to sequential for custom statistics
3. No shared state optimization (diag_s recomputed every iteration)
4. ProcessPoolExecutor has high overhead for small tasks

## Decision

Implement a **hybrid parallelization strategy** with three tiers:

### Tier 1: Precompute Shared State

The `diag_s` vector (SDP solution) depends only on the covariance matrix Sigma, which is constant across iterations. Compute it **once** before the parallel loop.

```python
# BEFORE parallel loop:
mu = np.mean(X, axis=0)
Sigma = LedoitWolf().fit(X).covariance_  # or sample cov
diag_s = create_solve_sdp(Sigma)  # Expensive SDP - do ONCE
```

### Tier 2: Use joblib with Memory Mapping

`joblib.Parallel` with `mmap_mode='r'` avoids copying large arrays to workers:

```python
from joblib import Parallel, delayed

def _single_iteration(X, y, mu, Sigma, diag_s, fdr, offset, seed):
    """Single knockoff iteration with precomputed diag_s."""
    np.random.seed(seed)
    Xk = create_gaussian(X, mu, Sigma, diag_s=diag_s)
    W = stat_glmnet_coefdiff(X, Xk, y)
    t = knockoff_threshold(W, fdr=fdr, offset=offset)
    return np.where(W >= t)[0]

# Parallel execution with joblib
results = Parallel(n_jobs=n_jobs, backend='loky', prefer='processes')(
    delayed(_single_iteration)(X, y, mu, Sigma, diag_s, fdr, offset, base_seed + i)
    for i in range(niter)
)
```

### Tier 3: Batch Processing for Large niter

For very large iteration counts, process in batches to control memory:

```python
BATCH_SIZE = 100

for batch_start in range(0, niter, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, niter)
    batch_results = Parallel(n_jobs=n_jobs)(
        delayed(_single_iteration)(..., base_seed + i)
        for i in range(batch_start, batch_end)
    )
    # Aggregate batch results
    for selected in batch_results:
        selection_counts[selected] += 1
```

## Implementation

### Option A: joblib.Parallel (Recommended)

**Pros:**
- Simple API, handles serialization gracefully
- Built-in memory mapping for numpy arrays
- Good error handling and progress reporting
- Works with custom callables via cloudpickle

**Cons:**
- External dependency (but already used by sklearn)

```python
def knockoff_filter_voting_parallel(
    X: np.ndarray,
    y: np.ndarray,
    knockoffs: Optional[Callable] = None,
    statistic: Optional[Callable] = None,
    fdr: float = 0.10,
    offset: int = 0,
    niter: int = 500,
    spec: float = 0.1,
    n_jobs: int = -1,
    base_seed: int = 42,
    verbose: bool = False,
    batch_size: int = 100,
    **kwargs
) -> VotingResult:
    """
    Parallel knockoff voting with precomputed diag_s.

    Key optimization: diag_s (SDP solution) computed ONCE before parallel loop.
    """
    from joblib import Parallel, delayed

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    n, p = X.shape

    # Determine n_jobs
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    n_jobs = max(1, min(n_jobs, niter))

    # ========== PRECOMPUTE SHARED STATE ==========
    # This is the key optimization: compute expensive SDP ONCE

    # Estimate mu and Sigma
    mu = np.mean(X, axis=0)

    # Force Ledoit-Wolf for n <= 1.25*p (matches R behavior)
    if n <= 1.25 * p:
        from sklearn.covariance import LedoitWolf
        Sigma = LedoitWolf().fit(X).covariance_
    else:
        Sigma = np.cov(X, rowvar=False, ddof=1)

    # Compute diag_s ONCE (expensive SDP optimization)
    method = 'asdp' if p > 500 else 'sdp'
    if method == 'sdp':
        diag_s = create_solve_sdp(Sigma)
    else:
        diag_s = create_solve_asdp(Sigma)

    # Handle SDP failure with equicorrelated fallback
    if np.max(diag_s) < 1e-6:
        diag_s = create_solve_equi(Sigma)

    if verbose:
        print(f"Precomputed diag_s (max={np.max(diag_s):.4f})")
        print(f"Running {niter} iterations with {n_jobs} workers...")

    # ========== PARALLEL EXECUTION ==========
    selection_counts = np.zeros(p, dtype=np.int32)

    def _worker(seed):
        """Single iteration worker."""
        np.random.seed(seed)

        # Create knockoffs using precomputed diag_s
        Xk = create_gaussian(X, mu, Sigma, diag_s=diag_s)

        # Compute statistics
        if statistic is None:
            from .stats import stat_glmnet_coefdiff
            W = stat_glmnet_coefdiff(X, Xk, y)
        else:
            W = statistic(X, Xk, y)

        # Apply threshold
        t = knockoff_threshold(W, fdr=fdr, offset=offset)
        return np.where(W >= t)[0].tolist()

    # Process in batches to control memory
    for batch_start in range(0, niter, batch_size):
        batch_end = min(batch_start + batch_size, niter)

        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_worker)(base_seed + i)
            for i in range(batch_start, batch_end)
        )

        # Aggregate results
        for selected in results:
            for idx in selected:
                selection_counts[idx] += 1

        if verbose and batch_end % 100 == 0:
            print(f"  Completed {batch_end}/{niter} iterations")

    # ========== COMPUTE RESULTS ==========
    selection_frequency = selection_counts / niter
    min_selections = int(np.ceil(niter * spec))
    selected = np.sort(np.where(selection_counts >= min_selections)[0])

    return VotingResult(
        selection_counts=selection_counts,
        selection_frequency=selection_frequency,
        selected=selected,
        threshold=spec,
        niter=niter,
        spec=spec,
        min_selections=min_selections
    )
```

### Option B: concurrent.futures.ProcessPoolExecutor

**Pros:**
- Standard library (no external deps)
- Fine-grained control over futures

**Cons:**
- Cannot pickle nested functions easily
- Requires module-level worker functions

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools

# Module-level worker (required for pickling)
def _voting_worker(args):
    """Worker that can be pickled."""
    X, y, mu, Sigma, diag_s, fdr, offset, seed, stat_name = args
    np.random.seed(seed)

    Xk = create_gaussian(X, mu, Sigma, diag_s=diag_s)

    # Get statistic by name (can't pickle functions)
    if stat_name == 'coefdiff':
        from .stats import stat_glmnet_coefdiff
        W = stat_glmnet_coefdiff(X, Xk, y)
    elif stat_name == 'lambdasmax':
        from .stats import stat_glmnet_lambdasmax
        W = stat_glmnet_lambdasmax(X, Xk, y)
    else:
        raise ValueError(f"Unknown statistic: {stat_name}")

    t = knockoff_threshold(W, fdr=fdr, offset=offset)
    return np.where(W >= t)[0].tolist()


def knockoff_filter_voting_futures(X, y, n_jobs=-1, niter=500, ...):
    # Precompute shared state (same as Option A)
    ...

    # Prepare arguments
    args_list = [
        (X, y, mu, Sigma, diag_s, fdr, offset, base_seed + i, 'coefdiff')
        for i in range(niter)
    ]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_voting_worker, args): i
                   for i, args in enumerate(args_list)}

        for future in as_completed(futures):
            selected = future.result()
            for idx in selected:
                selection_counts[idx] += 1
```

### Option C: numba @njit(parallel=True)

**Pros:**
- Fastest for numeric computation
- No process overhead

**Cons:**
- Cannot parallelize sklearn/scipy calls
- Only useful for inner loops (not the main iteration)

**Verdict:** Not suitable for this use case since the bottleneck is sklearn LASSO, not numeric operations.

## Performance Analysis

### Computational Profile (per iteration)

| Component | Time | Parallelizable? |
|-----------|------|-----------------|
| Covariance estimation | ~10ms | Precompute once |
| SDP optimization | ~50-100ms | Precompute once |
| Cholesky decomposition | ~5ms | Yes (per iteration) |
| Knockoff sampling | ~2ms | Yes (per iteration) |
| LASSO path | ~100-400ms | Yes (per iteration) |
| Threshold computation | <1ms | Yes (per iteration) |

### Expected Speedup

**Without precomputation (current):**
- 500 iterations x 150-700ms = 75-350 seconds

**With precomputation + 8 cores:**
- Precompute: ~100ms (one-time)
- Per iteration (no SDP): ~100-400ms
- Parallel: 500/8 = 62.5 iterations per core
- Total: ~100ms + (62.5 x 250ms) = **~16 seconds**

**Speedup: 4-8x** depending on:
- Number of cores
- Data dimensions (n, p)
- Whether SDP or equicorrelated method used

### Memory Considerations

| Approach | Memory Overhead |
|----------|-----------------|
| Sequential | Baseline |
| ProcessPoolExecutor | ~n_jobs x sizeof(X, y, Sigma) |
| joblib (mmap) | Baseline + overhead for mmap |
| Threads | Minimal (but GIL limits parallelism) |

For typical SLIDE data (n=100-1000, p=50-500):
- X: ~4MB per worker (1000 x 500 x 8 bytes)
- With 8 workers: ~32MB additional

## Recommended Implementation

1. **Short-term:** Implement Option A (joblib) in `knockoff_filter_voting`
2. **Add `precompute_diag_s` parameter** for advanced users
3. **Add progress bar** via `tqdm` for verbose mode
4. **Benchmark** on representative SLIDE datasets

### API Changes

```python
def knockoff_filter_voting(
    X: np.ndarray,
    y: np.ndarray,
    knockoffs: Optional[Callable] = None,
    statistic: Optional[Callable] = None,
    fdr: float = 0.10,
    offset: int = 0,
    niter: int = 500,
    spec: float = 0.1,
    n_jobs: int = 1,           # Change default to -1 for auto-parallel
    base_seed: int = 42,
    verbose: bool = False,
    precompute_diag_s: bool = True,  # NEW: enable diag_s precomputation
    batch_size: int = 100,           # NEW: batch size for memory control
    **kwargs
) -> VotingResult:
```

## Consequences

### Positive
- 4-8x speedup on multi-core systems
- Reduced redundant computation (SDP once vs 500 times)
- Better memory efficiency with batching
- Maintains compatibility with custom statistics

### Negative
- joblib dependency (already transitive via sklearn)
- Slightly more complex error handling
- Different random number sequences vs sequential (same results, different order)

### Neutral
- No change to external API (backward compatible)
- Results are statistically equivalent

## References

- joblib documentation: https://joblib.readthedocs.io/
- Python multiprocessing: https://docs.python.org/3/library/multiprocessing.html
- R knockoff package parallelization: Uses `doParallel` + `foreach`
