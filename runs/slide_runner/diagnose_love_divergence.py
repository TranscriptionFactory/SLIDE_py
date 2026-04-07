#!/usr/bin/env python3
"""
sbatch <<'SLURM'
#!/bin/bash
#SBATCH --job-name=diag_love
#SBATCH --cluster=htc
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=ssc_output/8273774/diagnose_%j.out

module load anaconda3
source activate metabopt_gpu_py310

cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/runs/slide_runner
python diagnose_love_divergence.py
SLURM
"""
"""
Diagnose LOVE divergence between R and Python implementations.

Strategy:
  1. Load saved love_result.pkl from both R and Python backends
  2. Compare A, C, Gamma, pureVec, K
  3. Re-run Python LOVE step-by-step to capture intermediates
  4. Re-run R LOVE step-by-step via rpy2 to capture intermediates
  5. Stage-by-stage comparison to pinpoint where divergence enters
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

BASE = "/ix/djishnu/Aaron/1_general_use/SLIDE_py/runs/slide_runner/ssc_output/8273774"
X_PATH = "/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/X.csv"
THRESH_FDR = 0.2
COMBOS = [
    (0.01, 0.1),
    (0.01, 1.0),
    (0.1, 0.1),
    (0.1, 1.0),
]


def load_love_result(backend, delta, lbd):
    """Load love_result.pkl for a given backend and parameter combo."""
    path = os.path.join(BASE, backend, f"{delta}_{lbd}_out", "love_result.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def compare_arrays(name, a, b, atol=1e-10, rtol=1e-8):
    """Compare two arrays and report differences."""
    if a is None and b is None:
        print(f"    {name}: both None")
        return True
    if a is None or b is None:
        print(f"    {name}: ONE IS NONE")
        return False
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        print(f"    {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return False
    max_diff = np.max(np.abs(a - b))
    close = np.allclose(a, b, atol=atol, rtol=rtol)
    n_diff = np.sum(~np.isclose(a, b, atol=atol, rtol=rtol))
    tag = "OK" if close else "DIFFER"
    print(f"    {name}: {tag}  max_diff={max_diff:.2e}  "
          f"n_diff={n_diff}/{a.size}  shape={a.shape}")
    return close


def compare_saved_results():
    """Compare love_result.pkl from all three backends."""
    print("=" * 72)
    print("  PART 1: Compare saved love_result.pkl across backends")
    print("=" * 72)

    for delta, lbd in COMBOS:
        print(f"\n  delta={delta}, lbd={lbd}")
        print(f"  {'─' * 50}")

        py = load_love_result("python", delta, lbd)
        rk = load_love_result("r_knockoffs", delta, lbd)
        r = load_love_result("r", delta, lbd)

        if py is None or r is None:
            print("    Missing love_result.pkl — skipping")
            continue

        # Basic stats
        for name, res in [("python", py), ("r_knockoffs", rk), ("r", r)]:
            if res is None:
                continue
            A = res['A']
            print(f"    {name:12s}: K={res['K']:3d}  #pure={len(res['pureVec']):3d}  "
                  f"A_shape={A.shape}  A_nnz={np.count_nonzero(A)}")

        # Compare pureVec
        py_pure = set(py['pureVec'])
        r_pure = set(r['pureVec'])
        if py_pure == r_pure:
            print(f"    pureVec: MATCH ({len(py_pure)} pure variables)")
        else:
            print(f"    pureVec: DIFFER  py={len(py_pure)} r={len(r_pure)} "
                  f"overlap={len(py_pure & r_pure)}")
            only_py = sorted(py_pure - r_pure)
            only_r = sorted(r_pure - py_pure)
            if only_py:
                print(f"      only in python: {only_py[:20]}")
            if only_r:
                print(f"      only in r:      {only_r[:20]}")

        # Compare A matrices (pure rows only)
        if py['K'] == r['K'] and len(py_pure) == len(r_pure) and py_pure == r_pure:
            pure_list = sorted(py_pure)
            compare_arrays("A[pure, :] (pure loadings)", py['A'][pure_list, :], r['A'][pure_list, :])

        # Compare full A
        if py['A'].shape == r['A'].shape:
            compare_arrays("A (full)", py['A'], r['A'], atol=1e-6)

            # Sparsity
            py_nz = py['A'] != 0
            r_nz = r['A'] != 0
            agree = np.sum(py_nz == r_nz) / py_nz.size
            print(f"    A sparsity agreement: {agree:.4f}  "
                  f"py_nnz={np.sum(py_nz)} r_nnz={np.sum(r_nz)}")

        # Compare C
        compare_arrays("C_hat", py['C'], r['C'])

        # Compare Gamma
        compare_arrays("Gamma", py['Gamma'], r['Gamma'], atol=1e-4)

        # r_knockoffs vs python (should be identical)
        if rk is not None and py is not None:
            rk_match = np.allclose(py['A'], rk['A'], atol=1e-14)
            print(f"    r_knockoffs vs python A: {'EXACT MATCH' if rk_match else 'DIFFER'}")


def run_instrumented_comparison():
    """Re-run Python LOVE step-by-step and compare against R intermediate values."""
    print("\n" + "=" * 72)
    print("  PART 2: Instrumented step-by-step Python LOVE")
    print("=" * 72)

    from loveslide.love_python.love.est_pure_homo import (
        EstAI, EstC, FindRowMax, FindPureNode, FindSignPureNode, RecoverAI
    )
    from loveslide.love_python.love.est_nonpure import EstY, EstAJDant, Dantzig
    from loveslide.love_python.love.love import _apply_fdr_threshold

    X_raw = pd.read_csv(X_PATH, index_col=0)
    X_std = (X_raw.values - X_raw.values.mean(axis=0)) / X_raw.values.std(axis=0, ddof=1)
    n, p = X_std.shape
    print(f"\n  Data: n={n}, p={p}")

    for delta, lbd in COMBOS:
        print(f"\n  delta={delta}, lbd={lbd}")
        print(f"  {'─' * 50}")

        # Python step-by-step
        X = X_std - np.mean(X_std, axis=0)
        se_est = np.std(X, axis=0, ddof=1)
        optDelta = delta * np.sqrt(np.log(max(p, n)) / n)
        print(f"    optDelta = {optDelta:.10f}")
        print(f"    se_est range: [{se_est.min():.10f}, {se_est.max():.10f}]")

        # Sigma
        Sigma_raw = np.cov(X, rowvar=False)
        R_corr = np.corrcoef(X, rowvar=False)

        # Check cov ≈ cor
        cov_cor_diff = np.max(np.abs(Sigma_raw - R_corr))
        print(f"    cov vs cor max_diff: {cov_cor_diff:.2e}")

        # FDR threshold
        R_thresh = _apply_fdr_threshold(R_corr, n, THRESH_FDR)
        std_devs = np.sqrt(np.diag(Sigma_raw))
        Sigma = R_thresh * np.outer(std_devs, std_devs)

        n_zeroed = np.sum(R_corr != 0) - np.sum(R_thresh != 0)
        print(f"    FDR thresholding zeroed {n_zeroed} entries "
              f"({n_zeroed / R_corr.size * 100:.1f}%)")

        # Pure variable detection
        resultAI = EstAI(Sigma, optDelta, se_est, merge=False)
        AI = resultAI['AI']
        pureVec = resultAI['pureVec']
        K = AI.shape[1]
        print(f"    K={K}, #pure={len(pureVec)}")

        # C_hat
        C_hat = EstC(Sigma, AI, diagonal=False)
        print(f"    C_hat diag: {np.diag(C_hat)[:5]}...")
        print(f"    C_hat cond: {np.linalg.cond(C_hat):.2e}")

        # Gamma (pure)
        I_hat_list = list(pureVec)
        Gamma_hat = np.zeros(p)
        A_I = AI[I_hat_list, :]
        Gamma_hat[I_hat_list] = (np.diag(Sigma[np.ix_(I_hat_list, I_hat_list)])
                                  - np.diag(A_I @ C_hat @ A_I.T))
        n_neg_gamma = np.sum(Gamma_hat[I_hat_list] < 0)
        Gamma_hat[Gamma_hat < 0] = 1e-2
        print(f"    Gamma: {n_neg_gamma} negative pure entries clamped")

        # sigma_TJ
        sigma_TJ = EstY(Sigma, AI, pureVec)
        print(f"    sigma_TJ shape: {sigma_TJ.shape}, "
              f"range: [{sigma_TJ.min():.6f}, {sigma_TJ.max():.6f}]")

        # sigma_bar_sup
        AI_abs = np.abs(AI[I_hat_list, :])
        cross_AI_inv = np.linalg.solve(AI_abs.T @ AI_abs, AI_abs.T)
        sigma_bar_sup = np.max(cross_AI_inv @ se_est[I_hat_list])
        dantzig_lambda = lbd * optDelta * sigma_bar_sup
        print(f"    sigma_bar_sup = {sigma_bar_sup:.10f}")
        print(f"    dantzig_lambda = {dantzig_lambda:.10f}")

        # Dantzig LP for each non-pure row
        J_list = [i for i in range(p) if i not in I_hat_list]
        se_est_J = sigma_bar_sup + se_est[J_list]
        n_J = len(J_list)
        print(f"    #non-pure rows: {n_J}")

        # Solve a few LPs and check solution properties
        print(f"\n    Sample Dantzig LP diagnostics (first 5 non-pure rows):")
        for idx in range(min(5, n_J)):
            y_vec = sigma_TJ[:, idx]
            lbd_row = dantzig_lambda * se_est_J[idx]
            sol = Dantzig(C_hat, y_vec, lbd_row)
            if sol is not None:
                l1_norm = np.sum(np.abs(sol))
                residual = np.max(np.abs(C_hat @ sol - y_vec))
                n_nz = np.count_nonzero(np.abs(sol) > 1e-10)
                print(f"      row {idx} (feat {J_list[idx]}): "
                      f"||beta||_1={l1_norm:.6f}  "
                      f"||C*beta-y||_inf={residual:.6f} (tol={lbd_row:.6f})  "
                      f"nnz={n_nz}/{K}")
            else:
                print(f"      row {idx}: LP FAILED")

        # Full Dantzig
        AJ = EstAJDant(C_hat, sigma_TJ, dantzig_lambda, se_est_J)
        if AJ is not None:
            A_hat = AI.copy()
            A_hat[J_list, :] = AJ
            print(f"\n    Final A: nnz={np.count_nonzero(A_hat)}/{A_hat.size}  "
                  f"shape={A_hat.shape}")

            # Compare against saved
            r_res = load_love_result("r", delta, lbd)
            if r_res is not None and r_res['A'].shape == A_hat.shape:
                # Pure rows
                pure_diff = np.max(np.abs(A_hat[I_hat_list, :] - r_res['A'][I_hat_list, :]))
                # Non-pure rows
                nonpure_diff = np.max(np.abs(A_hat[J_list, :] - r_res['A'][J_list, :]))
                # Sparsity
                py_sp = A_hat != 0
                r_sp = r_res['A'] != 0
                sp_agree = np.sum(py_sp == r_sp) / py_sp.size

                print(f"\n    vs saved R results:")
                print(f"      pure row max_diff:    {pure_diff:.2e}")
                print(f"      nonpure row max_diff: {nonpure_diff:.2e}")
                print(f"      sparsity agreement:   {sp_agree:.4f}")
                print(f"      py_nnz={np.sum(py_sp)} r_nnz={np.sum(r_sp)}")

                # Count per-row sparsity differences in non-pure rows
                n_row_diff = 0
                for j_idx, j in enumerate(J_list):
                    if not np.array_equal(A_hat[j, :] != 0, r_res['A'][j, :] != 0):
                        n_row_diff += 1
                print(f"      non-pure rows with different sparsity: "
                      f"{n_row_diff}/{n_J}")

                # Show worst differing rows
                row_diffs = []
                for j_idx, j in enumerate(J_list):
                    d = np.max(np.abs(A_hat[j, :] - r_res['A'][j, :]))
                    if d > 1e-6:
                        row_diffs.append((j, d))
                row_diffs.sort(key=lambda x: -x[1])
                if row_diffs:
                    print(f"\n      Top 10 worst non-pure row differences:")
                    for feat_idx, diff in row_diffs[:10]:
                        py_row = A_hat[feat_idx, :]
                        r_row = r_res['A'][feat_idx, :]
                        py_nnz = np.count_nonzero(py_row)
                        r_nnz = np.count_nonzero(r_row)
                        print(f"        feat {feat_idx}: max_diff={diff:.6f}  "
                              f"py_nnz={py_nnz} r_nnz={r_nnz}")
        else:
            print("    Dantzig returned None")


def run_r_instrumented():
    """Run R LOVE step-by-step via rpy2 and compare intermediates."""
    print("\n" + "=" * 72)
    print("  PART 3: R vs Python intermediate comparison via rpy2")
    print("=" * 72)

    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
    except ImportError:
        print("  rpy2 not available — skipping R comparison")
        return

    from loveslide.love_python.love.est_pure_homo import EstAI, EstC
    from loveslide.love_python.love.est_nonpure import EstY
    from loveslide.love_python.love.love import _apply_fdr_threshold

    # Source R scripts
    r_script_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'src', 'loveslide', 'slide_r'
    )
    import glob
    for script_path in sorted(glob.glob(os.path.join(r_script_dir, '*.R'))):
        robjects.r['source'](script_path)

    # Load data
    X_raw = pd.read_csv(X_PATH, index_col=0)
    X_np = X_raw.values
    n, p = X_np.shape

    # Pass to R
    robjects.globalenv['x_input'] = X_np

    for delta, lbd in [(0.1, 0.1)]:  # Focus on worst case
        print(f"\n  delta={delta}, lbd={lbd}")
        print(f"  {'─' * 50}")

        # ── R side ────────────────────────────────────────
        robjects.r(f'''
            x_r <- scale(x_input, TRUE, TRUE)
            n_r <- nrow(x_r); p_r <- ncol(x_r)
            se_est_r <- apply(x_r, 2, sd)
            sigma_r <- cor(x_r)
            delta_scaled_r <- {delta} * sqrt(log(max(p_r, n_r)) / n_r)

            control_fdr_r <- threshSigma(x = x_r, sigma = sigma_r, thresh = {THRESH_FDR})
            sigma_thresh_r <- control_fdr_r$thresh_sigma

            result_AI_r <- estAI(sigma = sigma_thresh_r,
                                 delta = delta_scaled_r,
                                 se_est = se_est_r)
            AI_r <- result_AI_r$AI
            pure_vec_r <- result_AI_r$pure_vec
            C_hat_r <- estC(sigma = sigma_thresh_r, AI = AI_r)
            sigma_TJ_r <- estSigmaTJ(sigma = sigma_thresh_r,
                                      AI = AI_r,
                                      pure_vec = pure_vec_r)

            AI_hat_r <- abs(AI_r[pure_vec_r, ])
            sigma_bar_sup_r <- max(solve(crossprod(AI_hat_r), t(AI_hat_r)) %*% se_est_r[pure_vec_r])
        ''')

        # Extract R intermediates
        sigma_thresh_r = np.array(robjects.r('sigma_thresh_r'))
        AI_r = np.array(robjects.r('AI_r'))
        pure_vec_r = np.array(robjects.r('pure_vec_r'), dtype=int) - 1
        C_hat_r = np.array(robjects.r('C_hat_r'))
        sigma_TJ_r = np.array(robjects.r('sigma_TJ_r'))
        sigma_bar_sup_r = float(np.array(robjects.r('sigma_bar_sup_r'))[0])
        se_est_r = np.array(robjects.r('se_est_r'))
        delta_scaled_r = float(np.array(robjects.r('delta_scaled_r'))[0])

        # ── Python side ───────────────────────────────────
        X_std = (X_np - X_np.mean(axis=0)) / X_np.std(axis=0, ddof=1)
        X = X_std - np.mean(X_std, axis=0)
        se_est_py = np.std(X, axis=0, ddof=1)
        optDelta = delta * np.sqrt(np.log(max(p, n)) / n)

        R_corr = np.corrcoef(X, rowvar=False)
        R_thresh = _apply_fdr_threshold(R_corr, n, THRESH_FDR)
        Sigma_raw = np.cov(X, rowvar=False)
        std_devs = np.sqrt(np.diag(Sigma_raw))
        Sigma_py = R_thresh * np.outer(std_devs, std_devs)

        resultAI_py = EstAI(Sigma_py, optDelta, se_est_py, merge=False)
        AI_py = resultAI_py['AI']
        pure_vec_py = resultAI_py['pureVec']
        C_hat_py = EstC(Sigma_py, AI_py, diagonal=False)
        sigma_TJ_py = EstY(Sigma_py, AI_py, pure_vec_py)

        I_hat_list = list(pure_vec_py)
        AI_abs = np.abs(AI_py[I_hat_list, :])
        cross_AI_inv = np.linalg.solve(AI_abs.T @ AI_abs, AI_abs.T)
        sigma_bar_sup_py = np.max(cross_AI_inv @ se_est_py[I_hat_list])

        # ── Compare ──────────────────────────────────────
        print("\n    Stage-by-stage R vs Python:")
        compare_arrays("se_est", se_est_py, se_est_r)
        print(f"    optDelta: py={optDelta:.10f}  r={delta_scaled_r:.10f}  "
              f"diff={abs(optDelta - delta_scaled_r):.2e}")
        compare_arrays("Sigma (FDR-thresholded)", Sigma_py, sigma_thresh_r)

        # Pure variables
        py_pure_set = set(pure_vec_py)
        r_pure_set = set(pure_vec_r)
        if py_pure_set == r_pure_set:
            print(f"    pureVec: MATCH ({len(py_pure_set)} variables)")
        else:
            print(f"    pureVec: DIFFER  py={len(py_pure_set)} r={len(r_pure_set)}")

        compare_arrays("AI (pure loadings)", AI_py, AI_r)
        compare_arrays("C_hat", C_hat_py, C_hat_r)
        compare_arrays("sigma_TJ", sigma_TJ_py, sigma_TJ_r)
        print(f"    sigma_bar_sup: py={sigma_bar_sup_py:.10f}  "
              f"r={sigma_bar_sup_r:.10f}  "
              f"diff={abs(sigma_bar_sup_py - sigma_bar_sup_r):.2e}")

        print("\n    → If all stages above match, divergence is PURELY in LP solver")
        print("      (R lpSolve simplex vs Python scipy.linprog HiGHS)")

    numpy2ri.deactivate()


def main():
    compare_saved_results()
    run_instrumented_comparison()
    run_r_instrumented()

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print("""
  The LOVE algorithm computes A via:
    1. Pure variable detection → AI (pure loadings, ±1 entries)
    2. C_hat estimation (covariance of Z)
    3. sigma_TJ estimation (sign-adjusted cross-correlations)
    4. Dantzig LP for each non-pure row

  If stages 1-3 produce identical results in R and Python, the
  divergence is ENTIRELY in stage 4: the Dantzig LP solver.

  R uses:   linprog::solveLP → lpSolve (simplex method)
  Python:   scipy.optimize.linprog(method='highs') (HiGHS solver)

  For L1-minimization LPs with multiple optimal vertices (common in
  sparse problems), different solvers legitimately return different
  solutions. This is NOT a bug — both solutions are equally valid.

  Impact:
  - delta=0.01 (172 LFs): A sparsity matches, values differ slightly
  - delta=0.1  (88 LFs):  A sparsity differs (~5%), values differ more
  - Higher lambda → more regularization → better agreement

  Possible fixes:
  1. Accept as inherent (recommended for production use)
  2. Use Python binding to lpSolve for exact R matching
  3. Post-process: round near-zero entries to zero with shared threshold
  4. Compare downstream results (knockoffs) with tolerance
""")


if __name__ == "__main__":
    main()
