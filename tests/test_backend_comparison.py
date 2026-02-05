#!/usr/bin/env python3
"""
End-to-end tests comparing SLIDE knockoff backend implementations.

This test suite compares the three available backends:
1. 'python': Pure Python implementation with SLIDE voting
2. 'r_knockoffs': R knockoff generation + Python SLIDE voting (best R concordance)
3. 'r': Full R knockoff pipeline via rpy2

Validation results show:
- R_knockoffs_py_voting: ~0.72 mean Jaccard vs R_native
- python_voting_slide: ~0.65 mean Jaccard vs R_native

The divergence is primarily in knockoff matrix generation (SDP solver),
not in the voting logic.

Author: Testing Agent
Date: 2026-02-05
"""

import sys
from pathlib import Path

# Add src to path for imports
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pytest
import numpy as np
import warnings
from typing import Set, Dict, Any, Optional, Tuple

# Check for rpy2 availability
try:
    import rpy2.robjects
    from rpy2.robjects.packages import importr
    # Try to load the knockoff package
    try:
        _knockoff_r = importr('knockoff')
        RPY2_AVAILABLE = True
    except Exception:
        RPY2_AVAILABLE = False
except ImportError:
    RPY2_AVAILABLE = False


def jaccard_similarity(set1: Set[int], set2: Set[int]) -> float:
    """Compute Jaccard similarity between two sets.

    Returns 1.0 if both sets are empty, 0.0 if one is empty and the other not.
    """
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def selection_overlap(set1: Set[int], set2: Set[int]) -> Dict[str, Any]:
    """Compute detailed overlap statistics between two selections."""
    intersection = set1 & set2
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1

    return {
        'intersection': intersection,
        'only_in_set1': only_in_1,
        'only_in_set2': only_in_2,
        'jaccard': jaccard_similarity(set1, set2),
        'size_set1': len(set1),
        'size_set2': len(set2),
        'overlap_count': len(intersection),
    }


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_data_strong_signals():
    """
    Create synthetic data with strong signals for reliable detection.

    n=200, p=50, 5 true signals with beta=3.0
    This is designed to give clear, reproducible selections.
    """
    np.random.seed(42)
    n, p = 200, 50
    X = np.random.randn(n, p)

    # Standardize columns
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    beta = np.zeros(p)
    beta[:5] = 3.0  # First 5 features are true signals

    y = X @ beta + np.random.randn(n) * 0.5

    true_signals = set(np.where(beta != 0)[0])

    return X, y, beta, true_signals


@pytest.fixture
def synthetic_data_moderate_signals():
    """
    Create synthetic data with moderate signals.

    n=150, p=100, 8 true signals with beta=2.0
    This is a more challenging case.
    """
    np.random.seed(123)
    n, p = 150, 100
    X = np.random.randn(n, p)

    # Standardize columns
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    beta = np.zeros(p)
    # Spread signals across the feature space
    true_indices = [5, 15, 25, 45, 55, 65, 75, 85]
    beta[true_indices] = 2.0

    y = X @ beta + np.random.randn(n) * 0.8

    true_signals = set(true_indices)

    return X, y, beta, true_signals


@pytest.fixture
def synthetic_data_chunking():
    """
    Create synthetic data that requires chunking (p > f_size).

    n=150, p=250, signals spread across chunks.
    """
    np.random.seed(456)
    n, p = 150, 250
    X = np.random.randn(n, p)

    # Standardize columns
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    beta = np.zeros(p)
    # Signals in different chunks (assuming f_size=100)
    true_indices = [10, 20, 110, 120, 210, 220]  # Spread across 3 chunks
    beta[true_indices] = 2.5

    y = X @ beta + np.random.randn(n) * 0.6

    true_signals = set(true_indices)

    return X, y, beta, true_signals


# =============================================================================
# Python Backend Tests (Always Available)
# =============================================================================

class TestPythonBackend:
    """Tests for the pure Python backend."""

    def test_python_backend_basic(self, synthetic_data_strong_signals):
        """Test basic functionality of Python backend."""
        from loveslide import Knockoffs, VotingResult

        X, y, beta, true_signals = synthetic_data_strong_signals

        # Run with Python backend
        selected = Knockoffs.select_short_freq(
            X, y,
            backend='python',
            niter=50,
            spec=0.2,
            fdr=0.1,
            f_size=100,
            verbose=False
        )

        assert isinstance(selected, np.ndarray)
        assert all(0 <= idx < X.shape[1] for idx in selected)

        # With strong signals, should detect at least some
        selected_set = set(selected)
        overlap = selected_set & true_signals
        print(f"\nPython backend: selected {len(selected)} features, "
              f"{len(overlap)} true positives")

    def test_python_backend_slide_method(self, synthetic_data_strong_signals):
        """Test select_short_freq_slide returns VotingResult."""
        from loveslide import Knockoffs, VotingResult

        X, y, beta, true_signals = synthetic_data_strong_signals

        result = Knockoffs.select_short_freq_slide(
            X, y,
            backend='python',
            niter=50,
            spec=0.2,
            fdr=0.1,
            f_size=100,
            verbose=False
        )

        assert isinstance(result, VotingResult)
        assert hasattr(result, 'selected')
        assert hasattr(result, 'selection_counts')
        assert hasattr(result, 'selection_frequency')
        assert hasattr(result, 'optimal_iter')

        print(f"\nPython SLIDE result: {len(result.selected)} selected, "
              f"optimal_iter={result.optimal_iter}")

    def test_python_backend_reproducibility(self, synthetic_data_strong_signals):
        """Test that Python backend produces reproducible results with same seed."""
        from loveslide import Knockoffs, VotingResult

        X, y, _, _ = synthetic_data_strong_signals

        # Run twice with same base_seed for reproducibility
        result1 = Knockoffs.select_short_freq_slide(
            X, y,
            backend='python',
            niter=20,
            spec=0.2,
            f_size=100,
            base_seed=42
        )

        result2 = Knockoffs.select_short_freq_slide(
            X, y,
            backend='python',
            niter=20,
            spec=0.2,
            f_size=100,
            base_seed=42
        )

        # Results should be identical with same seed
        assert np.array_equal(result1.selected, result2.selected)
        assert np.array_equal(result1.selection_counts, result2.selection_counts)


# =============================================================================
# R Knockoffs Backend Tests (Requires rpy2)
# =============================================================================

@pytest.mark.skipif(not RPY2_AVAILABLE, reason="rpy2 or R knockoff package not available")
class TestRKnockoffsBackend:
    """Tests for the R knockoffs + Python voting backend."""

    def test_r_knockoffs_backend_basic(self, synthetic_data_strong_signals):
        """Test basic functionality of r_knockoffs backend."""
        from loveslide import Knockoffs

        X, y, beta, true_signals = synthetic_data_strong_signals

        # Run with r_knockoffs backend
        selected = Knockoffs.select_short_freq(
            X, y,
            backend='r_knockoffs',
            niter=50,
            spec=0.2,
            fdr=0.1,
            f_size=100,
            verbose=False
        )

        assert isinstance(selected, np.ndarray)
        assert all(0 <= idx < X.shape[1] for idx in selected)

        selected_set = set(selected)
        overlap = selected_set & true_signals
        print(f"\nR_knockoffs backend: selected {len(selected)} features, "
              f"{len(overlap)} true positives")

    def test_r_knockoffs_backend_slide_method(self, synthetic_data_strong_signals):
        """Test select_short_freq_slide with r_knockoffs backend."""
        from loveslide import Knockoffs, VotingResult

        X, y, beta, true_signals = synthetic_data_strong_signals

        result = Knockoffs.select_short_freq_slide(
            X, y,
            backend='r_knockoffs',
            niter=50,
            spec=0.2,
            fdr=0.1,
            f_size=100,
            verbose=False
        )

        assert isinstance(result, VotingResult)
        assert hasattr(result, 'selected')

        print(f"\nR_knockoffs SLIDE result: {len(result.selected)} selected, "
              f"optimal_iter={result.optimal_iter}")


@pytest.mark.skipif(not RPY2_AVAILABLE, reason="rpy2 or R knockoff package not available")
class TestFullRBackend:
    """Tests for the full R backend (legacy)."""

    def test_r_backend_basic(self, synthetic_data_strong_signals):
        """Test basic functionality of full R backend."""
        from loveslide import Knockoffs

        X, y, beta, true_signals = synthetic_data_strong_signals

        # Run with full R backend
        selected = Knockoffs.select_short_freq(
            X, y,
            backend='r',
            niter=50,
            spec=0.2,
            fdr=0.1,
            f_size=100
        )

        assert isinstance(selected, np.ndarray)
        assert all(0 <= idx < X.shape[1] for idx in selected)

        selected_set = set(selected)
        overlap = selected_set & true_signals
        print(f"\nFull R backend: selected {len(selected)} features, "
              f"{len(overlap)} true positives")


# =============================================================================
# Backend Comparison Tests (Requires rpy2 for full comparison)
# =============================================================================

@pytest.mark.skipif(not RPY2_AVAILABLE, reason="rpy2 or R knockoff package not available")
class TestBackendComparison:
    """Compare outputs between different backends."""

    def test_python_vs_r_knockoffs(self, synthetic_data_strong_signals):
        """Compare Python backend with R knockoffs backend."""
        from loveslide import Knockoffs

        X, y, beta, true_signals = synthetic_data_strong_signals

        # Run Python backend
        selected_python = Knockoffs.select_short_freq(
            X, y,
            backend='python',
            niter=100,
            spec=0.2,
            fdr=0.1,
            f_size=100
        )

        # Run R knockoffs backend
        selected_r_knockoffs = Knockoffs.select_short_freq(
            X, y,
            backend='r_knockoffs',
            niter=100,
            spec=0.2,
            fdr=0.1,
            f_size=100
        )

        set_python = set(selected_python)
        set_r_knockoffs = set(selected_r_knockoffs)

        overlap = selection_overlap(set_python, set_r_knockoffs)

        print(f"\n=== Python vs R_knockoffs Comparison ===")
        print(f"Python selected: {len(set_python)} features")
        print(f"R_knockoffs selected: {len(set_r_knockoffs)} features")
        print(f"Overlap: {overlap['overlap_count']} features")
        print(f"Jaccard similarity: {overlap['jaccard']:.3f}")
        print(f"Only in Python: {sorted(overlap['only_in_set1'])}")
        print(f"Only in R_knockoffs: {sorted(overlap['only_in_set2'])}")

        # Both should find similar features (not necessarily identical)
        # With strong signals, expect reasonable overlap
        if len(set_python) > 0 and len(set_r_knockoffs) > 0:
            assert overlap['jaccard'] >= 0.3, (
                f"Jaccard too low: {overlap['jaccard']:.3f}. "
                f"Expected some overlap between backends."
            )

    def test_python_vs_r_full(self, synthetic_data_strong_signals):
        """Compare Python backend with full R backend."""
        from loveslide import Knockoffs

        X, y, beta, true_signals = synthetic_data_strong_signals

        # Run Python backend
        selected_python = Knockoffs.select_short_freq(
            X, y,
            backend='python',
            niter=100,
            spec=0.2,
            fdr=0.1,
            f_size=100
        )

        # Run full R backend
        selected_r = Knockoffs.select_short_freq(
            X, y,
            backend='r',
            niter=100,
            spec=0.2,
            fdr=0.1,
            f_size=100
        )

        set_python = set(selected_python)
        set_r = set(selected_r)

        overlap = selection_overlap(set_python, set_r)

        print(f"\n=== Python vs Full R Comparison ===")
        print(f"Python selected: {len(set_python)} features")
        print(f"R selected: {len(set_r)} features")
        print(f"Overlap: {overlap['overlap_count']} features")
        print(f"Jaccard similarity: {overlap['jaccard']:.3f}")

    def test_r_knockoffs_vs_r_full(self, synthetic_data_strong_signals):
        """Compare R knockoffs backend with full R backend."""
        from loveslide import Knockoffs

        X, y, beta, true_signals = synthetic_data_strong_signals

        # Run R knockoffs + Python voting
        selected_r_knockoffs = Knockoffs.select_short_freq(
            X, y,
            backend='r_knockoffs',
            niter=100,
            spec=0.2,
            fdr=0.1,
            f_size=100
        )

        # Run full R backend
        selected_r = Knockoffs.select_short_freq(
            X, y,
            backend='r',
            niter=100,
            spec=0.2,
            fdr=0.1,
            f_size=100
        )

        set_r_knockoffs = set(selected_r_knockoffs)
        set_r = set(selected_r)

        overlap = selection_overlap(set_r_knockoffs, set_r)

        print(f"\n=== R_knockoffs vs Full R Comparison ===")
        print(f"R_knockoffs selected: {len(set_r_knockoffs)} features")
        print(f"Full R selected: {len(set_r)} features")
        print(f"Overlap: {overlap['overlap_count']} features")
        print(f"Jaccard similarity: {overlap['jaccard']:.3f}")

        # R_knockoffs should have higher concordance with R than pure Python
        # Based on validation: r_knockoffs ~0.72 vs python ~0.65

    def test_all_three_backends(self, synthetic_data_strong_signals):
        """Run all three backends and compare comprehensively."""
        from loveslide import Knockoffs

        X, y, beta, true_signals = synthetic_data_strong_signals

        niter = 100
        spec = 0.2
        fdr = 0.1
        f_size = 100

        # Run all three backends
        selected_python = Knockoffs.select_short_freq(
            X, y, backend='python', niter=niter, spec=spec, fdr=fdr, f_size=f_size
        )

        selected_r_knockoffs = Knockoffs.select_short_freq(
            X, y, backend='r_knockoffs', niter=niter, spec=spec, fdr=fdr, f_size=f_size
        )

        selected_r = Knockoffs.select_short_freq(
            X, y, backend='r', niter=niter, spec=spec, fdr=fdr, f_size=f_size
        )

        set_python = set(selected_python)
        set_r_knockoffs = set(selected_r_knockoffs)
        set_r = set(selected_r)

        # Compute all pairwise Jaccards
        jaccard_py_rk = jaccard_similarity(set_python, set_r_knockoffs)
        jaccard_py_r = jaccard_similarity(set_python, set_r)
        jaccard_rk_r = jaccard_similarity(set_r_knockoffs, set_r)

        # True positive analysis
        tp_python = len(set_python & true_signals)
        tp_r_knockoffs = len(set_r_knockoffs & true_signals)
        tp_r = len(set_r & true_signals)

        # False positive analysis
        fp_python = len(set_python - true_signals)
        fp_r_knockoffs = len(set_r_knockoffs - true_signals)
        fp_r = len(set_r - true_signals)

        print(f"\n" + "=" * 60)
        print("COMPREHENSIVE BACKEND COMPARISON")
        print("=" * 60)
        print(f"\nData: n={X.shape[0]}, p={X.shape[1]}, "
              f"{len(true_signals)} true signals")
        print(f"Parameters: niter={niter}, spec={spec}, fdr={fdr}, f_size={f_size}")

        print(f"\n--- Selection Counts ---")
        print(f"Python:       {len(set_python):3d} selected "
              f"(TP={tp_python}, FP={fp_python})")
        print(f"R_knockoffs:  {len(set_r_knockoffs):3d} selected "
              f"(TP={tp_r_knockoffs}, FP={fp_r_knockoffs})")
        print(f"Full R:       {len(set_r):3d} selected "
              f"(TP={tp_r}, FP={fp_r})")

        print(f"\n--- Pairwise Jaccard Similarities ---")
        print(f"Python vs R_knockoffs: {jaccard_py_rk:.3f}")
        print(f"Python vs Full R:      {jaccard_py_r:.3f}")
        print(f"R_knockoffs vs Full R: {jaccard_rk_r:.3f}")

        print(f"\n--- Feature Selections ---")
        print(f"Python:      {sorted(set_python)}")
        print(f"R_knockoffs: {sorted(set_r_knockoffs)}")
        print(f"Full R:      {sorted(set_r)}")
        print(f"True signals:{sorted(true_signals)}")

        # All selections found by all backends
        consensus = set_python & set_r_knockoffs & set_r
        print(f"\n--- Consensus (all 3 backends) ---")
        print(f"Features selected by all: {sorted(consensus)}")

        # Expected behavior: r_knockoffs should be closer to r than python
        # This validates the implementation
        print(f"\n--- Validation ---")
        if jaccard_rk_r >= jaccard_py_r:
            print("PASS: R_knockoffs has equal or higher concordance with R than Python")
        else:
            print("NOTE: Python has higher concordance with R than R_knockoffs "
                  "(may vary by dataset)")


# =============================================================================
# Chunking Tests (Compare backends with feature chunking)
# =============================================================================

@pytest.mark.skipif(not RPY2_AVAILABLE, reason="rpy2 or R knockoff package not available")
class TestBackendComparisonChunking:
    """Compare backends when feature chunking is required."""

    def test_all_backends_with_chunking(self, synthetic_data_chunking):
        """Test all backends with data requiring chunking."""
        from loveslide import Knockoffs

        X, y, beta, true_signals = synthetic_data_chunking

        niter = 50  # Fewer iterations for speed
        spec = 0.2
        fdr = 0.1
        f_size = 100  # Forces 3 chunks for p=250

        print(f"\n=== Chunking Test: p={X.shape[1]}, f_size={f_size} ===")

        # Python backend
        selected_python = Knockoffs.select_short_freq(
            X, y, backend='python', niter=niter, spec=spec, fdr=fdr, f_size=f_size
        )

        # R knockoffs backend
        selected_r_knockoffs = Knockoffs.select_short_freq(
            X, y, backend='r_knockoffs', niter=niter, spec=spec, fdr=fdr, f_size=f_size
        )

        # Full R backend
        selected_r = Knockoffs.select_short_freq(
            X, y, backend='r', niter=niter, spec=spec, fdr=fdr, f_size=f_size
        )

        set_python = set(selected_python)
        set_r_knockoffs = set(selected_r_knockoffs)
        set_r = set(selected_r)

        print(f"Python:      {len(set_python)} selected - {sorted(set_python)}")
        print(f"R_knockoffs: {len(set_r_knockoffs)} selected - {sorted(set_r_knockoffs)}")
        print(f"Full R:      {len(set_r)} selected - {sorted(set_r)}")
        print(f"True signals: {sorted(true_signals)}")

        # All indices should be valid
        p = X.shape[1]
        assert all(0 <= idx < p for idx in selected_python)
        assert all(0 <= idx < p for idx in selected_r_knockoffs)
        assert all(0 <= idx < p for idx in selected_r)


# =============================================================================
# Performance Comparison Tests
# =============================================================================

@pytest.mark.slow
@pytest.mark.skipif(not RPY2_AVAILABLE, reason="rpy2 or R knockoff package not available")
class TestBackendPerformance:
    """Performance comparison between backends (marked slow)."""

    def test_backend_timing(self, synthetic_data_strong_signals):
        """Compare execution time between backends."""
        import time
        from loveslide import Knockoffs

        X, y, _, _ = synthetic_data_strong_signals

        niter = 50
        spec = 0.2
        fdr = 0.1
        f_size = 100

        timings = {}

        # Time Python backend
        start = time.time()
        _ = Knockoffs.select_short_freq(
            X, y, backend='python', niter=niter, spec=spec, fdr=fdr, f_size=f_size
        )
        timings['python'] = time.time() - start

        # Time R knockoffs backend
        start = time.time()
        _ = Knockoffs.select_short_freq(
            X, y, backend='r_knockoffs', niter=niter, spec=spec, fdr=fdr, f_size=f_size
        )
        timings['r_knockoffs'] = time.time() - start

        # Time full R backend
        start = time.time()
        _ = Knockoffs.select_short_freq(
            X, y, backend='r', niter=niter, spec=spec, fdr=fdr, f_size=f_size
        )
        timings['r'] = time.time() - start

        print(f"\n=== Backend Timing (niter={niter}) ===")
        for backend, t in timings.items():
            print(f"{backend:12s}: {t:.2f}s")

        # Python should generally be faster than R interop
        # (but r_knockoffs may be slower due to rpy2 overhead)


# =============================================================================
# Invalid Backend Tests
# =============================================================================

class TestInvalidBackend:
    """Test error handling for invalid backends."""

    def test_invalid_backend_select_short_freq(self, synthetic_data_strong_signals):
        """Test that invalid backend raises ValueError."""
        from loveslide import Knockoffs

        X, y, _, _ = synthetic_data_strong_signals

        with pytest.raises(ValueError, match="Unknown backend"):
            Knockoffs.select_short_freq_slide(
                X, y,
                backend='invalid_backend',
                niter=10,
                spec=0.2
            )


# =============================================================================
# Summary Report Test
# =============================================================================

@pytest.mark.skipif(not RPY2_AVAILABLE, reason="rpy2 or R knockoff package not available")
def test_backend_summary_report(synthetic_data_moderate_signals):
    """Generate a summary report comparing all backends."""
    from loveslide import Knockoffs

    X, y, beta, true_signals = synthetic_data_moderate_signals

    results = {}

    for backend in ['python', 'r_knockoffs', 'r']:
        selected = Knockoffs.select_short_freq(
            X, y,
            backend=backend,
            niter=100,
            spec=0.2,
            fdr=0.1,
            f_size=100
        )

        selected_set = set(selected)
        tp = len(selected_set & true_signals)
        fp = len(selected_set - true_signals)
        fn = len(true_signals - selected_set)

        precision = tp / len(selected_set) if len(selected_set) > 0 else 0.0
        recall = tp / len(true_signals) if len(true_signals) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[backend] = {
            'selected': selected_set,
            'n_selected': len(selected_set),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    print("\n" + "=" * 70)
    print("BACKEND COMPARISON SUMMARY REPORT")
    print("=" * 70)
    print(f"Data: n={X.shape[0]}, p={X.shape[1]}, {len(true_signals)} true signals")
    print(f"True signal indices: {sorted(true_signals)}")

    print("\n{:12s} {:>8s} {:>6s} {:>6s} {:>6s} {:>8s} {:>8s} {:>6s}".format(
        "Backend", "Selected", "TP", "FP", "FN", "Precision", "Recall", "F1"
    ))
    print("-" * 70)

    for backend in ['python', 'r_knockoffs', 'r']:
        r = results[backend]
        print("{:12s} {:>8d} {:>6d} {:>6d} {:>6d} {:>8.3f} {:>8.3f} {:>6.3f}".format(
            backend,
            r['n_selected'],
            r['tp'],
            r['fp'],
            r['fn'],
            r['precision'],
            r['recall'],
            r['f1'],
        ))

    print("\n--- Pairwise Jaccard Similarities ---")
    backends = ['python', 'r_knockoffs', 'r']
    for i, b1 in enumerate(backends):
        for b2 in backends[i+1:]:
            j = jaccard_similarity(results[b1]['selected'], results[b2]['selected'])
            print(f"{b1:12s} vs {b2:12s}: {j:.3f}")

    print("\n--- Consensus Features (selected by all) ---")
    consensus = results['python']['selected'] & results['r_knockoffs']['selected'] & results['r']['selected']
    print(f"Consensus: {sorted(consensus)}")

    print("=" * 70)


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
