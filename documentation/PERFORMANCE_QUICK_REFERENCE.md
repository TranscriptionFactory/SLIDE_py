# SLIDE Performance Optimization Quick Reference

## Top 5 Critical Performance Issues

### 1. Gene Evaluation Loop (slide.py:117-118) - HIGH PRIORITY
- **Issue:** Sequential `scorer.evaluate()` calls for each gene
- **Impact:** 10-30 seconds vs 1-3 seconds potential
- **Fix:** Vectorize evaluations using batch processing
- **Expected Speedup:** 10-30x

### 2. Matrix Computation Caching (slide.py:351-367) - HIGH PRIORITY  
- **Issue:** Repeated matrix inversions and multiplications
- **Impact:** 10-50ms per iteration vs <1ms cached
- **Fix:** Cache `C_hat` inverse and matrix decompositions
- **Expected Speedup:** 10-50x for repeated operations

### 3. Knockoff Parallelization - HIGH PRIORITY
- **Issue:** 500 iterations run sequentially  
- **Impact:** 75-350 seconds vs 15-45 seconds parallel
- **Fix:** Implement joblib parallelization with precomputed diag_s
- **Expected Speedup:** 4-8x (number of cores)

### 4. Cross-Validation Loops (cv.py:113-118) - MEDIUM PRIORITY
- **Issue:** Sequential processing of independent replicates
- **Impact:** Embarrassingly parallel workload run sequentially
- **Fix:** Parallelize replicates using joblib
- **Expected Speedup:** 4-8x

### 5. Memory Inefficiencies - MEDIUM PRIORITY
- **Issue:** Large matrix copies, repeated R-Python activations
- **Impact:** 30-50% excessive memory usage
- **Fix:** Use views instead of copies, context manager for R interface
- **Expected Improvement:** 30-50% memory reduction

## Implementation Phases

### Phase 1: Quick Wins (1-2 weeks)
1. Matrix inversion caching
2. R-Python interface optimization  
3. Vectorized correlation calculations

### Phase 2: Core Optimizations (3-4 weeks)
1. Gene evaluation vectorization
2. Cross-validation parallelization
3. Memory-efficient matrix operations

### Phase 3: Advanced (4-6 weeks)
1. Knockoff filter parallelization
2. Advanced caching strategies
3. Comprehensive profiling

## Overall Expected Improvement
- **Computational Performance:** 4.8-7.8x speedup
- **Memory Usage:** 30-50% reduction
- **Time to Results:** 150-695s → 31-89s

## Code Locations for Immediate Attention

| File | Lines | Issue | Priority |
|------|-------|-------|----------|
| `src/loveslide/slide.py` | 117-118 | Gene evaluation loop | HIGH |
| `src/loveslide/slide.py` | 351-367 | Matrix computations | HIGH |  
| `src/loveslide/love.py` | 100-106 | R conversion loop | MEDIUM |
| `src/loveslide/cv.py` | 113-118 | Sequential CV | MEDIUM |

See full analysis: [PERFORMANCE_OPTIMIZATION_ANALYSIS.md](PERFORMANCE_OPTIMIZATION_ANALYSIS.md)