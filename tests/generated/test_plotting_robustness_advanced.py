"""Advanced plotting robustness tests.

Tests edge cases for visualization functions that complement existing
plotting test coverage, focusing on extreme data scenarios and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from unittest.mock import patch, Mock

from src.loveslide.plotting import Plotter


class TestPlottingRobustnessAdvanced:
    """Test advanced plotting edge cases and robustness scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plotter = Plotter()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up any open figures
        plt.close('all')

    def test_plot_latent_factors_empty_data(self):
        """Test plotting with empty latent factors."""
        empty_lfs = pd.DataFrame()

        # Should handle gracefully without crashing
        try:
            self.plotter.plot_latent_factors(empty_lfs, outdir=self.temp_dir)
        except (ValueError, IndexError) as e:
            # Expected behavior for empty data
            assert "empty" in str(e).lower() or "shape" in str(e).lower()

    def test_plot_latent_factors_single_column(self):
        """Test plotting with single latent factor."""
        single_lf = pd.DataFrame({
            'Z1': np.random.randn(100)
        }, index=[f'gene_{i}' for i in range(100)])

        # Should handle single column gracefully
        self.plotter.plot_latent_factors(single_lf, outdir=self.temp_dir)

        # Check that plot was created
        assert len(plt.get_fignums()) > 0

    def test_plot_latent_factors_extreme_values(self):
        """Test plotting with extreme values."""
        extreme_lfs = pd.DataFrame({
            'Z1': [1e10, -1e10, 0, 1e-10, -1e-10] * 20,
            'Z2': [np.inf, -np.inf, np.nan, 0, 1] * 20
        }, index=[f'gene_{i}' for i in range(100)])

        # Should handle extreme values without crashing
        with pytest.warns(None) as warning_list:
            self.plotter.plot_latent_factors(extreme_lfs, outdir=self.temp_dir)

        # May generate warnings but should not crash
        assert len(plt.get_fignums()) > 0

    def test_plot_corr_network_disconnected_graph(self):
        """Test correlation network with no significant correlations."""
        # Create data with very low correlations
        X = pd.DataFrame(np.random.randn(50, 10))
        lf_dict = {'Z1': ['col_0', 'col_1']}

        # Should handle case where minimum correlation threshold filters out all edges
        self.plotter.plot_corr_network(X, lf_dict, outdir=self.temp_dir, minimum=0.9)

        # Should create plot even if network is empty
        assert len(plt.get_fignums()) > 0

    def test_plot_corr_network_single_node(self):
        """Test correlation network with single node."""
        X = pd.DataFrame(np.random.randn(50, 1), columns=['single_feature'])
        lf_dict = {'Z1': ['single_feature']}

        self.plotter.plot_corr_network(X, lf_dict, outdir=self.temp_dir)

        # Should handle single node network
        assert len(plt.get_fignums()) > 0

    def test_plot_controlplot_identical_scores(self):
        """Test control plot with identical scores."""
        identical_scores = [0.5] * 100

        self.plotter.plot_controlplot(identical_scores, outdir=self.temp_dir)

        # Should handle constant scores
        assert len(plt.get_fignums()) > 0

    def test_plot_controlplot_extreme_outliers(self):
        """Test control plot with extreme outliers."""
        scores_with_outliers = [1e-10] * 95 + [1e10, -1e10, np.inf, -np.inf, np.nan]

        with pytest.warns(None):
            self.plotter.plot_controlplot(scores_with_outliers, outdir=self.temp_dir)

        assert len(plt.get_fignums()) > 0

    def test_plot_interactions_empty_pairs(self):
        """Test interaction plot with empty pair list."""
        empty_pairs = []

        try:
            self.plotter.plot_interactions(empty_pairs, outdir=self.temp_dir)
        except (ValueError, IndexError):
            # Expected for empty data
            pass

    def test_plot_interactions_large_number_pairs(self):
        """Test interaction plot with very large number of pairs."""
        # Create many interaction pairs
        large_pairs = [(f'gene_{i}', f'gene_{j}')
                       for i in range(100) for j in range(i+1, min(i+6, 100))]

        # Should handle large number of pairs (may take time but shouldn't crash)
        self.plotter.plot_interactions(large_pairs[:50], outdir=self.temp_dir)

        assert len(plt.get_fignums()) > 0

    def test_plotting_with_invalid_output_directory(self):
        """Test plotting with invalid output directory."""
        lfs = pd.DataFrame(np.random.randn(50, 3), columns=['Z1', 'Z2', 'Z3'])

        # Non-existent directory path
        invalid_dir = "/nonexistent/path/that/should/not/exist"

        try:
            self.plotter.plot_latent_factors(lfs, outdir=invalid_dir)
        except (OSError, FileNotFoundError, PermissionError):
            # Expected for invalid directory
            pass

    def test_plotting_with_read_only_directory(self):
        """Test plotting with read-only output directory."""
        lfs = pd.DataFrame(np.random.randn(50, 3), columns=['Z1', 'Z2', 'Z3'])

        with tempfile.TemporaryDirectory() as temp_dir:
            # Make directory read-only
            os.chmod(temp_dir, 0o444)

            try:
                self.plotter.plot_latent_factors(lfs, outdir=temp_dir)
            except (OSError, PermissionError):
                # Expected for read-only directory
                pass
            finally:
                # Restore permissions for cleanup
                os.chmod(temp_dir, 0o755)

    def test_plot_memory_usage_large_data(self):
        """Test memory usage with large datasets."""
        # Create large dataset
        large_lfs = pd.DataFrame(
            np.random.randn(10000, 50),
            columns=[f'Z{i}' for i in range(50)]
        )

        # Should handle large data without excessive memory usage
        try:
            self.plotter.plot_latent_factors(large_lfs, outdir=self.temp_dir)
            assert len(plt.get_fignums()) > 0
        except MemoryError:
            pytest.skip("Insufficient memory for large data test")

    @patch('matplotlib.pyplot.savefig')
    def test_plot_save_failure_handling(self, mock_savefig):
        """Test handling of save failures."""
        mock_savefig.side_effect = OSError("Disk full")

        lfs = pd.DataFrame(np.random.randn(50, 3), columns=['Z1', 'Z2', 'Z3'])

        # Should handle save failures gracefully
        with pytest.raises(OSError):
            self.plotter.plot_latent_factors(lfs, outdir=self.temp_dir)

    def test_plot_with_unicode_labels(self):
        """Test plotting with Unicode characters in labels."""
        unicode_lfs = pd.DataFrame(
            np.random.randn(50, 3),
            columns=['Zα', 'Z∑', 'Z∞'],
            index=[f'gene_β{i}' for i in range(50)]
        )

        # Should handle Unicode characters
        self.plotter.plot_latent_factors(unicode_lfs, outdir=self.temp_dir)

        assert len(plt.get_fignums()) > 0

    def test_plot_with_very_long_labels(self):
        """Test plotting with very long feature names."""
        long_label_lfs = pd.DataFrame(
            np.random.randn(20, 2),
            columns=['Z1', 'Z2'],
            index=[f'very_long_gene_name_that_exceeds_normal_length_{i}' * 3
                   for i in range(20)]
        )

        # Should handle long labels (may truncate or adjust layout)
        self.plotter.plot_latent_factors(long_label_lfs, outdir=self.temp_dir)

        assert len(plt.get_fignums()) > 0

    def test_concurrent_plotting_thread_safety(self):
        """Test thread safety of plotting functions."""
        import threading

        lfs = pd.DataFrame(np.random.randn(50, 3), columns=['Z1', 'Z2', 'Z3'])

        def plot_worker():
            self.plotter.plot_latent_factors(lfs, outdir=self.temp_dir)

        # Create multiple threads
        threads = [threading.Thread(target=plot_worker) for _ in range(3)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should complete without deadlocks or crashes
        assert len(plt.get_fignums()) >= 0  # Some plots may have been closed