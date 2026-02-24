# SSc Multi-Parameter Version Comparison

**Date:** 2026-02-23
**Job Array:** 8053658 (submitted via `submit_ssc_array.sh --keep-env`)
**Data:** SSc UnTx (24 samples, 804 features)
**Parameters:** deltas=[0.01, 0.1], lambdas=[0.1, 1.0], niter=500, fdr=0.1, thresh_fdr=0.2, knockoff_method=asdp

## Output Directories

| Backend | Output Directory | LOVE backend | Knockoff backend |
|---------|-----------------|--------------|------------------|
| Native R (ground truth) | `output_native_R/` | R | R |
| `r` | `output_20260221_184120/r/` | python | r |
| `r_knockoffs` | `output_20260221_184120/r_knockoffs/` | python | r_knockoffs |
| `python` | `output_20260221_184201/python/` | python | python |

## Job Status

| Task | Backend | State | Elapsed | Notes |
|------|---------|-------|---------|-------|
| 0 | `r_knockoffs` | FAILED (exit 7) | 1d 0:44 | Bus error at exit, but all results saved before crash |
| 1 | `r` | RUNNING | 2d 5:06+ | Currently doing SLIDEcv for last param combo (0.1/1.0), rep 1/10 |
| 2 | `python` | COMPLETED | 19:35 | Clean exit |

## Main Run Scores (True Scores)

| delta | lambda | Native R | `r` | `r_knockoffs` | `python` |
|-------|--------|----------|-----|---------------|----------|
| 0.01 | 0.1 | 0.205 | 0.329 | 0.271 | 0.462 |
| 0.01 | 1.0 | 0.175 | 0.451 | 0.710 | 0.672 |
| 0.1 | 0.1 | 0.510 | 0.504 | 0.283 | 0.734 |
| 0.1 | 1.0 | 0.550 | 0.287 | NA (0 sig) | 0.470 |

The `r` backend at delta=0.1/lambda=0.1 (0.504) is closest to native R (0.510). The python backend tends to produce higher True Scores but fewer sig LFs in some combos.

## Significant Latent Factors

0-based indexing for loveslide backends; native R uses 1-based (subtract 1 to compare).

| delta | lambda | Native R (1-based) | `r` | `r_knockoffs` | `python` |
|-------|--------|-------------------|-----|---------------|----------|
| 0.01 | 0.1 | 5 LFs | Z18,Z27,Z45,Z74,Z136 (5) | Z18,Z27,Z45,Z73,Z74,Z136 (6) | Z18,Z27,Z45,Z74,Z136 (5) |
| 0.01 | 1.0 | 5 LFs | Z18,Z27,Z45,Z73,Z136 (5) | Z18,Z20,Z27,Z45,Z136 (5) | Z18,Z27,Z45,Z136 (4) |
| 0.1 | 0.1 | 4 LFs | Z5,Z25,Z44,Z62 (4) | Z5,Z10,Z25,Z44,Z61,Z72,Z75 (7) | Z5,Z44 (2) |
| 0.1 | 1.0 | 2 LFs | Z5,Z24,Z25,Z33,Z44 (5) | 0 (failed) | Z5,Z44 (2) |

- The `r` backend matches native R best for delta=0.01 combos -- the core set {Z18, Z27, Z45, Z136} (= native R's {Z19, Z28, Z46, Z137}) is consistently found across all backends.
- The `python` backend finds fewer sig LFs for delta=0.1, only 2 vs native R's 2-4.
- The `r_knockoffs` backend is noisier -- finds extra LFs in some cases (7 for 0.1/0.1) and completely fails at 0.1/1.0.

## Interactor Counts

| delta | lambda | Native R | `r` | `r_knockoffs` | `python` |
|-------|--------|----------|-----|---------------|----------|
| 0.01 | 0.1 | 12 | 14 | 17 | 11 |
| 0.01 | 1.0 | 11 | 19 | 10 | 9 |
| 0.1 | 0.1 | 4 | 18 | 27 | 4 |
| 0.1 | 1.0 | 4 | 16 | 0 | 5 |

The `python` backend's interactor counts (11, 9, 4, 5) are closest to native R (12, 11, 4, 4). Both `r` and `r_knockoffs` backends tend to find many more interactors.

## SLIDEcv Mean Scores (SLIDE vs null)

| delta | lambda | Backend | Mean SLIDE | Mean SLIDE_y (null) | SLIDE > null? |
|-------|--------|---------|------------|---------------------|---------------|
| 0.01 | 0.1 | `r` | 0.187 | 0.001 | Yes |
| 0.01 | 0.1 | `r_knockoffs` | 0.214 | 0.012 | Yes |
| 0.01 | 0.1 | `python` | 0.228 | 0.001 | Yes |
| 0.01 | 1.0 | `r` | 0.170 | 0.002 | Yes |
| 0.01 | 1.0 | `r_knockoffs` | 0.162 | 0.071 | Yes |
| 0.01 | 1.0 | `python` | 0.153 | 0.003 | Yes |
| 0.1 | 0.1 | `r` | 0.025 | 0.081 | **No** |
| 0.1 | 0.1 | `r_knockoffs` | 0.113 | -0.063 | Yes |
| 0.1 | 0.1 | `python` | 0.159 | -0.027 | Yes |
| 0.1 | 1.0 | `python` | 0.042 | -0.014 | Marginal |

Native R sampleCV_Performance: 0.01/0.1=0.205, 0.01/1=0.175, 0.1/0.1=0.510, 0.1/1=0.550.

Note: `r` backend 0.1/1.0 CV still running; `r_knockoffs` 0.1/1.0 had no sig LFs so CV was skipped.

## Key Issues

1. **`r` backend task 1 still running** -- 2+ days elapsed, only on SLIDEcv for the last param combo (0.1/1.0 rep 1/10). The main SLIDE runs completed; it is the CV phase that is slow.

2. **`r_knockoffs` 0.1/1.0 failure** -- Found 0 significant LFs, so scores are NA. The log shows repeated "SDP knockoffs procedure failed" and "Both SDP and equicorrelated methods failed. Knockoffs will have no power" during the main run.

3. **Bus error on task 0** -- `r_knockoffs` backend crashed with a Bus error (exit code 7) after completing all work. Results were fully written before the crash. Likely an rpy2 cleanup issue.

4. **SLIDEcv for `r` backend at 0.1/0.1 shows SLIDE < null** (0.025 vs 0.081), suggesting the model did not generalize well in CV despite a decent True Score (0.504). This contrasts with native R's strong CV performance (0.510) for the same params.

5. **Python backend** is fastest (19.5 hrs total for all 4 combos + CV) but finds fewer sig LFs in the delta=0.1 regime. Its interactor counts match native R best.

## LOVE Divergence Analysis (2026-02-24)

Traced the divergence between native R and Python LOVE implementations step-by-step using intermediates from `AllLatentFactors.rds` (R) and `love_result.pkl` (Python).

### Comparison script

`runs/ssc_multi_param/compare_outputs.py` — compares z_matrix, A matrix, sig LFs, feature lists, and summary metrics across all 4 implementations. Usage:

```bash
python compare_outputs.py                    # full report
python compare_outputs.py --detailed         # per-column and per-feature details
python compare_outputs.py --param 0.1_1      # single parameter combo
python compare_outputs.py --output report.txt
```

### Python vs R-backend: BIT-IDENTICAL

All LOVE intermediates (A, C, Gamma, pureVec, optDelta) are exactly zero-diff between the `python` and `r` backends. The `r_knockoffs` backend also shares the same LOVE output. Differences between these three backends come entirely from the **knockoff filter** step (post-LOVE).

### Native R vs Python: delta=0.01 (K=172) — IDENTICAL

All 172 z_matrix columns have |r| = 1.000000 (machine precision). Six columns are sign-flipped (expected eigenvector ambiguity). Slope = 1.0 exactly, residuals ~1e-13. The LOVE decomposition is **numerically identical** modulo sign.

### Native R vs Python: delta=0.1 (K=88) — DIVERGENT

Step-by-step comparison of LOVE intermediates:

| Step | Component | Match? | Detail |
|------|-----------|--------|--------|
| 1 | K (# factors) | IDENTICAL | 88 |
| 2 | Pure variables (I) | IDENTICAL | 493/493 same indices |
| 3 | A pure rows | IDENTICAL | 461 agree + 32 sign-flips, 0 column mismatches |
| 4 | **A non-pure rows** | **DIVERGENT** | nonzeros R=1776 vs Py=1817; median col \|r\|=0.997 but min=0.016 |
| 5 | C diagonal | IDENTICAL | max_diff = 8.9e-16 |
| 5 | C off-diagonal | DIVERGENT | max_diff = 1.0, corr = 0.94 |
| 6 | Gamma | DIVERGENT | All 311 non-pure vars differ; worst: R=1.0 vs Py=0.0 |
| 7 | Z matrix | Same subspace | R_z ~ Py_z regression max_resid = 3e-14, but individual columns are mixed (not identity) |

### Causal chain

```
A non-pure estimation (EstY / Dantzig) differs
    → C off-diagonal changes (computed from A)
    → Gamma changes for all 311 non-pure variables (depends on A and C)
    → Z = X @ Gamma⁻¹ @ A @ (A'Gamma⁻¹A + C⁻¹)⁻¹ changes
```

The Gamma divergence is strongly correlated with A divergence (Spearman rho=0.66, p=2e-100).

### Why delta=0.01 matches but delta=0.1 doesn't

- **delta=0.01** (K=172): Non-pure A is 98.2% dense (72,157/73,444 nonzeros). The Dantzig thresholding barely removes anything, so EstY method differences are negligible.
- **delta=0.1** (K=88): Non-pure A is only 6.6% dense (1,817/27,368 nonzeros). Dantzig thresholding is aggressive, amplifying differences in the EstY (sigma_TJ) computation.

### Z-matrix column space

Both R and Python z_matrices span the **exact same 23-dimensional subspace** (rank = n_samples - 1 = 23). This was confirmed by regressing R_z on Py_z: max residual = 3e-14. The individual z columns are linear combinations of each other — different bases for the same space — because the A matrix partitions variables differently across factors.

### Prior fix (commit 127f131)

Changed `EstY()` from least-squares regression to sign-adjusted averaging per cluster (matching R's `estSigmaTJ`). This was the primary source of non-pure A divergence. Remaining differences are likely LP solver numerics within the Dantzig selector (`EstAJDant`).
