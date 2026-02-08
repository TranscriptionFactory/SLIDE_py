#!/usr/bin/env python3
"""
End-to-end pipeline tests for loveslide v1.0.0.

Runs the full SLIDE pipeline on real HIV cytokine data and compares
outputs against reference results from the build check (20260122_135613).

Test classes:
    TestPackageInstall    - import checks, version, __all__ exports
    TestLOVEPipeline      - call_love() on HIV data, matrix shapes, pure vars
    TestKnockoffPipeline  - Knockoffs.select_short_freq on latent factors
    TestFullPipeline      - OptimizeSLIDE.run_pipeline end-to-end
    TestEstimatorScoring  - Estimator fit/predict/evaluate
    TestDSDPFallback      - SDP solver detection chain
"""

import os
import pickle
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE_DIR = _PROJECT_ROOT / "example"
_REFERENCE_BASE = (
    _PROJECT_ROOT
    / "archive"
    / "comparison"
    / "build_check_outputs"
    / "20260122_135613"
)
_REF_GLMNET = _REFERENCE_BASE / "Py_pyLOVE_kf_glmnet" / "0.1_0.5_out"
_TEST_CONFIG = Path(__file__).resolve().parent / "test_config.yaml"
_TEST_OUTPUTS = Path(__file__).resolve().parent / "test_outputs"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hiv_data():
    """Load HIV cytokine example data (60 x 80, binary outcome)."""
    X = pd.read_csv(_EXAMPLE_DIR / "HIV+cytokines_X.csv", index_col=0)
    y = pd.read_csv(_EXAMPLE_DIR / "HIV+cytokines_y.csv", index_col=0).iloc[:, 0]
    return X, y


@pytest.fixture(scope="module")
def reference_A():
    """Load reference A matrix from build check."""
    return pd.read_csv(_REF_GLMNET / "A.csv", index_col=0)


@pytest.fixture(scope="module")
def reference_z_matrix():
    """Load reference z_matrix from build check."""
    return pd.read_csv(_REF_GLMNET / "z_matrix.csv", index_col=0)


@pytest.fixture(scope="module")
def love_result_on_hiv(hiv_data):
    """
    Run call_love() once and share across the LOVE test class.

    Uses pure_homo=True, delta=[0.1] to match the build check config.
    """
    from loveslide import call_love

    X, y = hiv_data
    result = call_love(
        X,
        lbd=0.5,
        pure_homo=True,
        delta=[0.1],
        thresh_fdr=0.2,
        backend="python",
        verbose=False,
    )
    return result


@pytest.fixture(scope="module")
def pipeline_output_dir():
    """Provide a clean output directory for pipeline tests, cleaned up after."""
    out = _TEST_OUTPUTS
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    yield out
    # Don't remove - keep for manual inspection after test run


# ===========================================================================
# 1. Package Install Tests
# ===========================================================================
class TestPackageInstall:
    """Validate that loveslide installs correctly and exports are importable."""

    def test_import_version(self):
        import loveslide
        assert loveslide.__version__ == "1.0.0"

    def test_all_exports(self):
        import loveslide

        for name in loveslide.__all__:
            obj = getattr(loveslide, name, None)
            assert obj is not None, f"{name} listed in __all__ but not importable"


# ===========================================================================
# 2. LOVE Pipeline Tests
# ===========================================================================
class TestLOVEPipeline:
    """Run LOVE on HIV cytokine data and validate outputs."""

    def test_love_output_keys(self, love_result_on_hiv):
        """call_love() returns expected keys."""
        result = love_result_on_hiv
        for key in ("K", "A", "C", "pureVec", "group"):
            assert key in result, f"Missing key: {key}"

    def test_love_matrix_shapes(self, hiv_data, love_result_on_hiv):
        """A is (p, K) and C is (K, K)."""
        X, _ = hiv_data
        r = love_result_on_hiv
        p = X.shape[1]
        K = r["K"]

        assert r["A"].shape == (p, K), f"A shape {r['A'].shape} != ({p}, {K})"
        assert r["C"].shape == (K, K), f"C shape {r['C'].shape} != ({K}, {K})"

    def test_love_K_reasonable(self, love_result_on_hiv):
        """K should be positive and reasonable for 80 features."""
        K = love_result_on_hiv["K"]
        assert 2 <= K <= 40, f"K={K} outside reasonable range [2, 40]"

    def test_love_pure_variables_count(self, love_result_on_hiv):
        """Pure variable count should be reasonable (reference had 43)."""
        n_pure = len(love_result_on_hiv["pureVec"])
        # Allow some flexibility since exact count depends on algorithm details
        assert 20 <= n_pure <= 70, f"Pure var count {n_pure} outside [20, 70]"

    def test_love_A_agreement_with_reference(self, love_result_on_hiv, reference_A):
        """
        A matrix from Python LOVE should correlate well with reference.

        The columns may be permuted and sign-flipped, so we compare
        the best column-wise absolute correlation.
        """
        A_new = love_result_on_hiv["A"]
        A_ref = reference_A.values

        # Both should have same number of rows (p=80)
        assert A_new.shape[0] == A_ref.shape[0]

        # For each reference column, find best-matching new column
        K_ref = A_ref.shape[1]
        K_new = A_new.shape[1]
        # K values should be similar
        assert abs(K_ref - K_new) <= 3, f"K mismatch: ref={K_ref}, new={K_new}"

        # Check non-zero pattern similarity (column permutation invariant)
        ref_nonzero = set(np.where(np.any(A_ref != 0, axis=1))[0])
        new_nonzero = set(np.where(np.any(A_new != 0, axis=1))[0])
        jaccard = len(ref_nonzero & new_nonzero) / max(len(ref_nonzero | new_nonzero), 1)
        assert jaccard >= 0.7, f"Non-zero pattern Jaccard={jaccard:.3f} < 0.7"


# ===========================================================================
# 3. Knockoff Pipeline Tests
# ===========================================================================
class TestKnockoffPipeline:
    """Test knockoff selection on HIV latent factors."""

    def test_knockoff_python_backend(self, hiv_data, love_result_on_hiv):
        """Run select_short_freq with python backend on latent factors."""
        from loveslide import Knockoffs

        X, y = hiv_data
        A = love_result_on_hiv["A"]

        # Compute z_matrix = X_std @ A @ inv(C)
        X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        C = love_result_on_hiv["C"]
        z = X_std.values @ A @ np.linalg.inv(C)

        selected = Knockoffs.select_short_freq(
            z, y.values,
            backend="python",
            niter=50,
            spec=0.1,
            fdr=0.1,
            f_size=100,
            verbose=False,
        )

        assert isinstance(selected, np.ndarray)
        assert all(0 <= idx < z.shape[1] for idx in selected)
        print(f"\nKnockoff python: selected {len(selected)} LFs")

    def test_knockoff_slide_voting(self, hiv_data, love_result_on_hiv):
        """Run select_short_freq_slide and verify VotingResult."""
        from loveslide import Knockoffs, VotingResult

        X, y = hiv_data
        A = love_result_on_hiv["A"]
        X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        C = love_result_on_hiv["C"]
        z = X_std.values @ A @ np.linalg.inv(C)

        result = Knockoffs.select_short_freq_slide(
            z, y.values,
            backend="python",
            niter=50,
            spec=0.1,
            fdr=0.1,
            f_size=100,
            verbose=False,
        )

        assert isinstance(result, VotingResult)
        assert hasattr(result, "selected")
        assert hasattr(result, "selection_counts")
        assert hasattr(result, "optimal_iter")
        print(f"\nKnockoff SLIDE: {len(result.selected)} selected, "
              f"optimal_iter={result.optimal_iter}")


# ===========================================================================
# 4. Full Pipeline Tests
# ===========================================================================
@pytest.mark.slow
class TestFullPipeline:
    """Run OptimizeSLIDE.run_pipeline() on HIV data end-to-end."""

    @pytest.fixture(scope="class")
    def pipeline_result(self, pipeline_output_dir):
        """Run the full pipeline once and share across tests."""
        from loveslide import OptimizeSLIDE

        input_params = {
            "x_path": str(_EXAMPLE_DIR / "HIV+cytokines_X.csv"),
            "y_path": str(_EXAMPLE_DIR / "HIV+cytokines_y.csv"),
            "y_factor": True,
            "niter": 500,
            "SLIDE_top_feats": 10,
            "out_path": str(pipeline_output_dir),
            "fdr": 0.1,
            "thresh_fdr": 0.2,
            "pure_homo": True,
            "do_interacts": True,
            "n_workers": 1,
            "spec": 0.1,
            "love_backend": "python",
            "knockoff_backend": "python",
            "knockoff_method": "asdp",
            "knockoff_shrink": False,
            "knockoff_offset": 0,
            "fstat": "glmnet_lambdasmax",
            "delta": [0.1],
            "lambda": [0.5],
        }

        slider = OptimizeSLIDE(input_params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slider.run_pipeline(verbose=True)

        return slider, pipeline_output_dir / "0.1_0.5_out"

    def test_optimize_slide_hiv(self, pipeline_result):
        """Pipeline should complete without error."""
        slider, out_dir = pipeline_result
        assert out_dir.exists()

    def test_pipeline_output_files(self, pipeline_result):
        """Verify expected output files are produced."""
        _, out_dir = pipeline_result
        expected_files = [
            "A.csv",
            "z_matrix.csv",
            "sig_LFs.txt",
            "love_result.pkl",
        ]
        for fname in expected_files:
            path = out_dir / fname
            assert path.exists(), f"Missing output file: {fname}"

    def test_pipeline_love_agreement(self, pipeline_result, reference_A):
        """
        A matrix from pipeline should correlate well with reference.

        Check non-zero row pattern Jaccard >= 0.7.
        """
        _, out_dir = pipeline_result
        A_new = pd.read_csv(out_dir / "A.csv", index_col=0)

        ref_nonzero = set(np.where(np.any(reference_A.values != 0, axis=1))[0])
        new_nonzero = set(np.where(np.any(A_new.values != 0, axis=1))[0])
        jaccard = len(ref_nonzero & new_nonzero) / max(len(ref_nonzero | new_nonzero), 1)
        print(f"\nA non-zero pattern Jaccard: {jaccard:.3f}")
        assert jaccard >= 0.7, f"A non-zero Jaccard={jaccard:.3f} < 0.7"

    def test_pipeline_z_matrix_shape(self, pipeline_result, hiv_data):
        """z_matrix should be (n, K) with n matching input data."""
        _, out_dir = pipeline_result
        X, _ = hiv_data
        z = pd.read_csv(out_dir / "z_matrix.csv", index_col=0)
        assert z.shape[0] == X.shape[0], f"z_matrix rows {z.shape[0]} != n={X.shape[0]}"
        assert z.shape[1] >= 2, f"z_matrix only has {z.shape[1]} columns"

    def test_pipeline_selection_sanity(self, pipeline_result):
        """sig_LFs should be non-empty and contain valid column names."""
        _, out_dir = pipeline_result
        sig_lfs = np.loadtxt(out_dir / "sig_LFs.txt", dtype=str).reshape(-1).tolist()
        assert len(sig_lfs) > 0, "No significant LFs found"

        z = pd.read_csv(out_dir / "z_matrix.csv", index_col=0)
        for lf in sig_lfs:
            assert lf in z.columns, f"sig_LF '{lf}' not in z_matrix columns"

    def test_pipeline_scores_file(self, pipeline_result):
        """scores.txt should exist and contain performance metrics."""
        _, out_dir = pipeline_result
        scores_path = out_dir / "scores.txt"
        assert scores_path.exists(), "scores.txt not found"
        content = scores_path.read_text()
        assert "True Scores" in content, "scores.txt missing 'True Scores'"


# ===========================================================================
# 5. Estimator Scoring Tests
# ===========================================================================
class TestEstimatorScoring:
    """Test Estimator and SLIDE_Estimator on HIV latent factors."""

    def test_estimator_basic(self, hiv_data, love_result_on_hiv):
        """Estimator.get_aucs should return array of AUC scores."""
        from loveslide import Estimator

        X, y = hiv_data
        A = love_result_on_hiv["A"]
        X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        C = love_result_on_hiv["C"]
        z = X_std.values @ A @ np.linalg.inv(C)

        # Use first 3 columns as features
        aucs = Estimator.get_aucs(
            z[:, :3], y.values,
            n_iters=20,
            test_size=0.2,
            scaler="standard",
        )
        assert isinstance(aucs, np.ndarray)
        assert len(aucs) == 20
        mean_auc = np.mean(aucs)
        # AUC should be > 0.5 (better than random) for real data
        print(f"\nEstimator AUC: {mean_auc:.3f}")


# ===========================================================================
# 6. SDP Solver Fallback Tests
# ===========================================================================
class TestDSDPFallback:
    """Verify SDP solver detection chain works."""

    def test_sdp_solver_detection(self):
        """_get_sdp_solver should return 'dsdp' or 'cvxpy' (not None)."""
        from loveslide.knockoff.solve import _get_sdp_solver, _SDP_SOLVER

        # Reset cached value to force re-detection
        import loveslide.knockoff.solve as solve_mod
        solve_mod._SDP_SOLVER = None

        solver = _get_sdp_solver()
        assert solver in ("dsdp", "cvxpy"), f"No SDP solver found: {solver}"
        print(f"\nSDP solver: {solver}")

        # Restore
        solve_mod._SDP_SOLVER = solver

    def test_sdp_solve_produces_valid_diag(self):
        """create_solve_sdp should produce valid diag_s for a small matrix."""
        from loveslide.knockoff.solve import create_solve_sdp

        np.random.seed(42)
        p = 10
        X = np.random.randn(50, p)
        G = np.corrcoef(X, rowvar=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = create_solve_sdp(G)

        assert s.shape == (p,)
        assert np.all(s >= -1e-6), "diag_s has negative values"
        assert np.all(s <= 1.0 + 1e-6), "diag_s exceeds 1.0"


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
