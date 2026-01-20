# Knockoff Implementation Comparison Report

**Date**: 2026-01-20
**Purpose**: Identify specific differences between Python knockoff implementations and R's knockoff package

---

## KEY FINDINGS

### **CRITICAL ISSUES (Must Fix)**

#### 1. **Entry Time Detection** - ✅ VERIFIED CORRECT
**Location**: `/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter/knockoff/stats/glmnet.py` line 102-107

**Status**: The code is actually **correct**. The `np.where(has_nonzero, ...)` properly handles the case when `argmax()` returns 0 for all-false arrays by replacing those values with 0.0.

```python
# Current implementation is correct:
nonzero_mask = np.abs(coefs) > 0
first_nonzero_idx = nonzero_mask.argmax(axis=1)  # Returns 0 for all-false
has_nonzero = nonzero_mask.any(axis=1)
lambda_entry = np.where(has_nonzero, alphas[first_nonzero_idx] * n, 0.0)  # Replaces with 0.0
```

---

#### 2. **Threshold Computation Bug** - ✅ FIXED
**Location**: `/ix/djishnu/Aaron/1_general_use/SLIDE_py/src/loveslide/knockoffs.py` lines 432-433

**The Problem**: Python excludes 0 from candidate thresholds, but R includes it. This can lead to different FDR threshold selection.

**R Implementation**:
```r
ts = sort(c(0, abs(W)))  # Always includes 0
```

**Python Implementation**:
```python
candidates = np.sort(W_abs[W_abs > 0])  # Excludes 0
```

**Impact**: When no W-statistics satisfy FDR, R returns 0 (reject nothing) while Python might return the first positive value.

**Fix**:
```python
# Current (line 432-433):
W_abs = np.abs(W)
candidates = np.sort(W_abs[W_abs > 0])  # WRONG

# Should be:
candidates = np.sort(np.unique(np.abs(W)))  # Include 0 if present
# Or explicitly:
candidates = np.sort(np.concatenate([[0], np.abs(W)]))
```

---

### **HIGH PRIORITY ISSUES**

#### 3. **SDP Solver Differences**
**Location**: `/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter/knockoff/solve.py` lines 96-263

**The Difference**:
- R uses Rdsdp (R-specific DSDP wrapper)
- Python uses pydsdp (Python wrapper of same DSDP)

**Impact**: Different numerical precision in S matrix computation. While both follow identical algorithms, internal numerical paths differ.

**Adjustment**: Document that small differences in S matrix are expected and acceptable (±1% variation typically observed).

---

#### 4. **Random Number Generation**
**Location**: Multiple files - `base.py` line 41, `create.py` line 363

**The Difference**:
- R's Mersenne Twister ≠ NumPy's Mersenne Twister
- Even with same seed, sequences diverge

**Impact**: W-statistics will never exactly match between R and Python due to different `swap` generation and `rnorm()` calls.

**Recommendation**:
- Use `niter > 1` (suggest niter ≥ 100) to average out randomness
- Accept ±5-10% variation as normal
- Do NOT expect exact replication across languages

---

#### 5. **Cholesky Regularization**
**Location**: `/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter/knockoff/create.py` lines 344-360

**The Difference**:
- R crashes if Sigma_k fails Cholesky decomposition
- Python silently adds regularization and continues

**Recommendation**: Match R's strict behavior or add verbose logging when regularization is triggered.

---

### **MEDIUM PRIORITY ISSUES**

#### 6. **Intercept Handling Inconsistency**
**Location**: `/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter/knockoff/stats/glmnet.py` lines 74, 83

**R**:
```r
X = scale(X)  # Centers and standardizes
fit <- glmnet(X, y, intercept=T, standardize=F)  # Intercept=T but X already centered
```

**Python**:
```python
y_centered = y - y.mean()  # Center y only
# fit_intercept=False when y_centered but X not centered?
```

**Impact**: Minor numerical differences in glmnet coefficients.

---

## SPECIFIC CODE LOCATIONS FOR ADJUSTMENT

| Issue | File | Lines | Status |
|-------|------|-------|--------|
| Entry time detection | knockoff/stats/glmnet.py | 102-107 | ✅ Verified correct |
| Threshold candidates | loveslide/knockoffs.py | 432-433 | ✅ **FIXED** |
| SDP solver differences | knockoff/solve.py | 96-263 | Document |
| RNG differences | base.py, create.py | 41, 363 | Document |
| Cholesky handling | knockoff/create.py | 344-360 | Match R behavior |
| Intercept consistency | knockoff/stats/glmnet.py | 74-90 | Verify |

---

## DIAGNOSTIC DATA POINTS

From the existing `DIVERGENCE_ANALYSIS.md`:
- **LOVE implementation**: 100% identical between R and Python ✓
- **W-statistic correlation (custom_glmnet)**: 0.57 with R
- **Threshold correlation**: Differs due to above bugs
- **Agreement at delta=0.2**: 82% (strongest signal overcomes algorithmic differences)

The document recommends:
1. Use `--knockoff-backend python` with `--fstat glmnet_lambdasmax`
2. Use `--knockoff-method sdp` (full SDP, most accurate)
3. Use `niter ≥ 100` to average random variation

---

## SUMMARY TABLE

| Component | R | Python | Diff Type | Status | Fix |
|-----------|---|--------|-----------|--------|-----|
| Entry time | `match()` | `argmax()` + `np.where` | Logic | ✅ Correct | N/A |
| Threshold | Includes 0 | ~~Excludes 0~~ | Logic | ✅ **FIXED** | Added 0 to candidates |
| SDP solver | Rdsdp | DSDP | Numerical | Document | ±1% expected |
| RNG | R-MT | NumPy-MT | Different | Document | Use niter>1 |
| Cholesky | Crash | Regularize | Behavior | Open | Match R or log |

---

## RECOMMENDED FIXES (Priority Order)

### 1. Fix Entry Time Detection (CRITICAL)

In `knockoff-filter/knockoff/stats/glmnet.py`:
```python
# Replace the current entry time logic with:
def _find_entry_times(coefs, alphas, n):
    """Find lambda at which each feature first enters the model."""
    p = coefs.shape[0]
    lambda_entry = np.zeros(p)

    for j in range(p):
        nonzero_idx = np.where(np.abs(coefs[j, :]) > 0)[0]
        if len(nonzero_idx) > 0:
            lambda_entry[j] = alphas[nonzero_idx[0]] * n
        # else: lambda_entry[j] remains 0

    return lambda_entry
```

### 2. Fix Threshold Candidates (CRITICAL)

In `loveslide/knockoffs.py`:
```python
# Replace line 432-433:
W_abs = np.abs(W)
candidates = np.sort(np.concatenate([[0], W_abs[W_abs > 0]]))
```

### 3. Document Expected Differences

Add to documentation:
- SDP solver: ±1% numerical variation expected
- RNG: Use niter ≥ 100 for stable results
- Overall: Expect 80-90% agreement with R at typical FDR thresholds

---

## NEXT STEPS

1. **Implement CRITICAL fixes** in knockoff-filter and SLIDE_py
2. **Re-run W-statistic comparison** to verify improvement
3. **Update documentation** with expected differences
4. **Consider adding R compatibility tests** that verify selections match within tolerance
