"""
Test SLIDE algorithm robustness and error recovery.
Covers state corruption, numerical instability, and recovery scenarios.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import pickle
import os
from unittest.mock import patch, MagicMock

from loveslide import SLIDE, OptimizeSLIDE


class TestSLIDEStateRecovery:
    """Test SLIDE state persistence and recovery mechanisms."""

    def test_load_corrupted_love_result(self):
        """Test handling of corrupted LOVE result files."""
        params = {"fdr": 0.1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        slide = SLIDE(params, x=X, y=y)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            # Write corrupted pickle data
            f.write(b"corrupted data")
            corrupted_path = f.name

        try:
            slide.load_love(corrupted_path)
            # Should handle corruption gracefully without crashing
            assert not hasattr(slide, 'A') or slide.A is None
        finally:
            os.unlink(corrupted_path)

    def test_load_love_missing_required_fields(self):
        """Test LOVE result missing required fields."""
        params = {"fdr": 0.1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        slide = SLIDE(params, x=X, y=y)

        # Create incomplete LOVE result
        incomplete_result = {"partial_data": True}

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(incomplete_result, f)
            incomplete_path = f.name

        try:
            slide.load_love(incomplete_path)
            # Should handle missing fields gracefully
            assert not hasattr(slide, 'A') or slide.A is None
        finally:
            os.unlink(incomplete_path)

    def test_load_state_partial_files(self):
        """Test loading state with partial/missing files."""
        params = {"fdr": 0.1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        slide = SLIDE(params, x=X, y=y)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only some of the required state files
            A_df = pd.DataFrame(np.random.randn(20, 5))
            A_df.to_csv(os.path.join(tmpdir, "A.csv"))

            # Missing z_matrix.csv and sig files
            slide.load_state(tmpdir)

            # Should handle partial state gracefully
            assert hasattr(slide, 'A')
            assert slide.marginal_idxs == []

    def test_load_state_corrupted_csv(self):
        """Test loading state with corrupted CSV files."""
        params = {"fdr": 0.1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        slide = SLIDE(params, x=X, y=y)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create corrupted CSV
            with open(os.path.join(tmpdir, "A.csv"), 'w') as f:
                f.write("corrupted,csv,data\n1,2,invalid_number\n")

            slide.load_state(tmpdir)
            # Should handle corruption without crashing
            assert slide.marginal_idxs == []


class TestSLIDENumericalStability:
    """Test SLIDE algorithm numerical stability."""

    def test_get_lf_genes_extreme_loadings(self):
        """Test get_LF_genes with extreme loading values."""
        # Create matrix with extreme values
        A = pd.DataFrame({
            'Z1': [1e10, -1e10, 1e-10, -1e-10, 0],
            'Z2': [np.inf, -np.inf, np.nan, 0, 1]
        })
        X = pd.DataFrame(np.random.randn(100, 5))
        y = np.random.randn(100)

        # Should handle extreme values without crashing
        result = SLIDE.get_LF_genes(A, 'Z1', X, y, lf_thresh=0.05)
        assert isinstance(result, dict)
        assert 'positive' in result and 'negative' in result

    def test_get_lf_genes_all_zero_loadings(self):
        """Test get_LF_genes when all loadings are zero."""
        A = pd.DataFrame({
            'Z1': [0, 0, 0, 0, 0]
        })
        X = pd.DataFrame(np.random.randn(100, 5))
        y = np.random.randn(100)

        result = SLIDE.get_LF_genes(A, 'Z1', X, y)
        # Should handle zero loadings appropriately
        assert len(result['positive']) + len(result['negative']) == 0

    def test_calc_z_matrix_singular_matrix(self):
        """Test calc_z_matrix with singular/rank-deficient matrices."""
        params = {"fdr": 0.1}
        X = np.random.randn(50, 20)
        # Create rank-deficient X by making columns linearly dependent
        X[:, 1] = X[:, 0] * 2
        X[:, 2] = X[:, 0] * 3
        y = np.random.randn(50)

        slide = SLIDE(params, x=X, y=y)

        # Create LOVE result with potential numerical issues
        love_result = {
            "A": np.random.randn(20, 5),
            "convergence": True
        }

        try:
            z_matrix = slide.calc_z_matrix(love_result)
            # Should handle numerical instability
            assert z_matrix is not None
            assert not np.any(np.isnan(z_matrix))
        except np.linalg.LinAlgError:
            # Acceptable to raise LinAlgError for singular matrices
            pass


class TestOptimizeSLIDEErrorHandling:
    """Test OptimizeSLIDE error handling and edge cases."""

    def test_find_interaction_lfs_empty_candidates(self):
        """Test find_interaction_LFs with no candidate LFs."""
        params = {"fdr": 0.1, "niter": 2}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        opt_slide = OptimizeSLIDE(params, x=X, y=y)

        # Set up state with no available candidates
        opt_slide.latent_factors = pd.DataFrame(np.random.randn(50, 3),
                                              columns=['Z1', 'Z2', 'Z3'])
        opt_slide.marginal_idxs = [0, 1, 2]  # All LFs already used

        # Should handle empty candidate set gracefully
        interactions = opt_slide.find_interaction_LFs()
        assert isinstance(interactions, list)
        assert len(interactions) == 0

    def test_find_interaction_lfs_numerical_overflow(self):
        """Test find_interaction_LFs with numerical overflow conditions."""
        params = {"fdr": 0.1, "niter": 2}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        opt_slide = OptimizeSLIDE(params, x=X, y=y)

        # Create latent factors with extreme values
        lf_data = np.random.randn(50, 5)
        lf_data[0, 0] = 1e10  # Extreme value
        opt_slide.latent_factors = pd.DataFrame(lf_data,
                                              columns=[f'Z{i}' for i in range(5)])
        opt_slide.marginal_idxs = [0]

        # Should handle numerical issues gracefully
        interactions = opt_slide.find_interaction_LFs()
        assert isinstance(interactions, list)

    @patch('loveslide.slide.Knockoffs')
    def test_optimize_slide_knockoff_failure(self, mock_knockoffs):
        """Test OptimizeSLIDE when knockoff generation fails."""
        params = {"fdr": 0.1, "niter": 2}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        opt_slide = OptimizeSLIDE(params, x=X, y=y)

        # Mock knockoff failure
        mock_knockoffs_instance = MagicMock()
        mock_knockoffs_instance.run.side_effect = RuntimeError("Knockoff generation failed")
        mock_knockoffs.return_value = mock_knockoffs_instance

        # Create minimal required state
        love_result = {"A": np.random.randn(20, 3), "convergence": True}
        opt_slide.love_result = love_result
        opt_slide.A = pd.DataFrame(love_result["A"])
        opt_slide.latent_factors = pd.DataFrame(np.random.randn(50, 3))

        # Should handle knockoff failure gracefully
        with pytest.raises(RuntimeError):
            opt_slide.optimize_SLIDE()


class TestConcurrentAccessHandling:
    """Test handling of concurrent access to SLIDE state."""

    def test_concurrent_state_modification(self):
        """Test behavior when state is modified during processing."""
        params = {"fdr": 0.1}
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        slide = SLIDE(params, x=X, y=y)

        # Simulate concurrent modification by changing data mid-process
        original_X = slide.data.X.copy()

        def modify_during_processing():
            # This could happen in multi-threaded scenarios
            slide.data.X = np.random.randn(50, 20)

        # Test should verify thread safety or appropriate error handling
        # Implementation depends on whether thread safety is required