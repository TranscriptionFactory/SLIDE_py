"""
Test coverage for SLIDE error handling and edge cases.
"""
import pytest
import numpy as np
import tempfile
import pickle
from pathlib import Path

from loveslide import SLIDE, OptimizeSLIDE


class TestSLIDEErrorHandling:
    """Test error handling in SLIDE class."""

    def test_load_love_corrupted_file(self):
        """Test loading corrupted love result file."""
        slide = SLIDE({"fdr": 0.1})

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"corrupted data")
            f.flush()

            with pytest.raises((pickle.PickleError, EOFError)):
                slide.load_love(f.name)

    def test_load_love_nonexistent_file(self):
        """Test loading non-existent love result file."""
        slide = SLIDE({"fdr": 0.1})

        with pytest.raises(FileNotFoundError):
            slide.load_love("/nonexistent/path.pkl")

    def test_load_state_invalid_iteration(self):
        """Test loading state with invalid iteration number."""
        slide = SLIDE({"fdr": 0.1})

        with pytest.raises((ValueError, IndexError)):
            slide.load_state(-1)

    def test_calc_default_fsize_edge_cases(self):
        """Test feature size calculation with edge cases."""
        X = np.random.randn(10, 5)  # Very small dataset
        y = np.random.randn(10)
        slide = SLIDE({}, x=X, y=y)

        # Test with K=0
        with pytest.raises(ValueError):
            slide.calc_default_fsize(0)

        # Test with K greater than features
        result = slide.calc_default_fsize(10)  # K > n_features
        assert isinstance(result, int)
        assert result > 0

    def test_show_params_with_none_data(self):
        """Test show_params when data is None."""
        slide = SLIDE({})
        slide.data = None

        # Should handle gracefully without crashing
        slide.show_params()


class TestOptimizeSLIDEErrorHandling:
    """Test error handling in OptimizeSLIDE class."""

    def test_get_latent_factors_singular_matrix(self):
        """Test latent factor calculation with singular covariance matrix."""
        X = np.ones((50, 20))  # Singular matrix (all rows identical)
        y = np.random.randn(50)
        opt_slide = OptimizeSLIDE({}, x=X, y=y)

        # Should handle singular matrix gracefully
        with pytest.raises((np.linalg.LinAlgError, ValueError)):
            opt_slide.get_latent_factors()

    def test_calc_z_matrix_empty_love_result(self):
        """Test Z matrix calculation with empty love result."""
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        opt_slide = OptimizeSLIDE({}, x=X, y=y)

        empty_love_result = {"pure_nodes": [], "pure_edges": {}}

        with pytest.raises((KeyError, ValueError)):
            opt_slide.calc_z_matrix(empty_love_result)

    def test_find_interaction_lfs_insufficient_data(self):
        """Test interaction finding with insufficient data."""
        X = np.random.randn(10, 5)  # Very small dataset
        y = np.random.randn(10)
        opt_slide = OptimizeSLIDE({}, x=X, y=y)

        z_matrix = np.random.randn(10, 3)

        # Should handle gracefully with small datasets
        result = opt_slide.find_interaction_LFs(z_matrix, n_workers=1)
        assert isinstance(result, dict)

    def test_run_slide_invalid_parameters(self):
        """Test SLIDE pipeline with invalid parameters."""
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        opt_slide = OptimizeSLIDE({
            "fdr": 1.5,  # Invalid FDR > 1
            "f_size": -10  # Invalid negative size
        }, x=X, y=y)

        with pytest.raises(ValueError):
            opt_slide.run_SLIDE()


class TestSLIDEMemoryManagement:
    """Test memory efficiency and large dataset handling."""

    def test_large_dataset_memory_efficiency(self):
        """Test SLIDE with large datasets for memory efficiency."""
        # Mock large dataset
        X = np.random.randn(1000, 100)
        y = np.random.randn(1000)

        slide = SLIDE({"fdr": 0.1, "f_size": 50}, x=X, y=y)

        # Should not consume excessive memory
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        slide.calc_default_fsize(10)

        memory_after = process.memory_info().rss
        memory_increase = (memory_after - memory_before) / (1024 * 1024)  # MB

        # Memory increase should be reasonable (< 100MB for this test)
        assert memory_increase < 100

    def test_state_persistence_integrity(self):
        """Test that saved and loaded states maintain integrity."""
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        slide = SLIDE({"fdr": 0.1}, x=X, y=y)

        # Create dummy state data
        test_scores = {"lf_0": np.random.randn(10)}

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir)
            slide.save_params(outpath, test_scores)

            # Verify files were created
            assert (outpath / "scores.pkl").exists()

            # Test loading
            with open(outpath / "scores.pkl", "rb") as f:
                loaded_scores = pickle.load(f)

            np.testing.assert_array_equal(
                loaded_scores["lf_0"],
                test_scores["lf_0"]
            )