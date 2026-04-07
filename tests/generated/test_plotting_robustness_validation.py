"""
Comprehensive test coverage for plotting functionality and edge cases.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import seaborn as sns

from loveslide.plotting import Plotter


class TestPlotterInitialization:
    """Test Plotter class initialization."""

    def test_plotter_init(self):
        """Test basic Plotter initialization."""
        plotter = Plotter()
        assert isinstance(plotter, Plotter)

    def test_plotter_static_methods_accessible(self):
        """Test that static plotting methods are accessible."""
        assert hasattr(Plotter, 'plot_latent_factors')
        assert hasattr(Plotter, 'plot_corr_network')
        assert hasattr(Plotter, 'plot_controlplot')
        assert hasattr(Plotter, 'plot_interactions')


class TestPlotLatentFactors:
    """Test latent factor plotting functionality."""

    def test_plot_latent_factors_valid_data(self):
        """Test plotting latent factors with valid data."""
        # Create sample latent factor data
        lfs = {
            'LF_0': np.array([0.8, 0.6, 0.4, 0.2]),
            'LF_1': np.array([0.7, 0.5, 0.3, 0.1])
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Should not raise any errors
            Plotter.plot_latent_factors(lfs, outdir=outdir)

            # Check that plot file was created
            plot_files = list(outdir.glob("*.png")) + list(outdir.glob("*.pdf"))
            assert len(plot_files) > 0

    def test_plot_latent_factors_empty_data(self):
        """Test plotting with empty latent factors."""
        lfs = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Should handle empty data gracefully
            with pytest.warns(UserWarning, match="No latent factors"):
                Plotter.plot_latent_factors(lfs, outdir=outdir)

    def test_plot_latent_factors_single_lf(self):
        """Test plotting with single latent factor."""
        lfs = {
            'LF_0': np.array([0.8, 0.6, 0.4, 0.2, 0.1])
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            Plotter.plot_latent_factors(lfs, outdir=outdir)

            # Should create plot even with single LF
            plot_files = list(outdir.glob("*.png")) + list(outdir.glob("*.pdf"))
            assert len(plot_files) > 0

    def test_plot_latent_factors_extreme_values(self):
        """Test plotting with extreme values."""
        lfs = {
            'LF_0': np.array([1e6, -1e6, 0, 1e-10]),
            'LF_1': np.array([np.inf, -np.inf, np.nan, 1.0])
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Should handle extreme values gracefully
            with pytest.warns(UserWarning):
                Plotter.plot_latent_factors(lfs, outdir=outdir)

    def test_plot_latent_factors_different_lengths(self):
        """Test plotting with latent factors of different lengths."""
        lfs = {
            'LF_0': np.array([0.8, 0.6, 0.4]),
            'LF_1': np.array([0.7, 0.5, 0.3, 0.1, 0.05])  # Different length
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Should handle different lengths gracefully
            with pytest.warns(UserWarning):
                Plotter.plot_latent_factors(lfs, outdir=outdir)

    def test_plot_latent_factors_invalid_outdir(self):
        """Test plotting with invalid output directory."""
        lfs = {
            'LF_0': np.array([0.8, 0.6, 0.4, 0.2])
        }

        # Non-existent directory
        invalid_outdir = Path("/nonexistent/directory")

        with pytest.raises((FileNotFoundError, PermissionError)):
            Plotter.plot_latent_factors(lfs, outdir=invalid_outdir)

    def test_plot_latent_factors_no_outdir(self):
        """Test plotting without specifying output directory."""
        lfs = {
            'LF_0': np.array([0.8, 0.6, 0.4, 0.2])
        }

        # Should work without outdir (display only)
        with patch('matplotlib.pyplot.show'):
            Plotter.plot_latent_factors(lfs, outdir=None)


class TestPlotCorrNetwork:
    """Test correlation network plotting functionality."""

    def test_plot_corr_network_valid_data(self):
        """Test plotting correlation network with valid data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)

        lf_dict = {
            'LF_0': [0, 1, 2],
            'LF_1': [3, 4, 5]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            Plotter.plot_corr_network(X, lf_dict, outdir=outdir)

            # Should create network plot file
            plot_files = list(outdir.glob("*.png")) + list(outdir.glob("*.pdf"))
            assert len(plot_files) > 0

    def test_plot_corr_network_empty_lf_dict(self):
        """Test correlation network with empty latent factor dictionary."""
        X = np.random.randn(100, 10)
        lf_dict = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            with pytest.warns(UserWarning):
                Plotter.plot_corr_network(X, lf_dict, outdir=outdir)

    def test_plot_corr_network_high_minimum_threshold(self):
        """Test correlation network with very high minimum threshold."""
        X = np.random.randn(100, 10)
        lf_dict = {
            'LF_0': [0, 1, 2],
            'LF_1': [3, 4, 5]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # High threshold may result in no edges
            with pytest.warns(UserWarning):
                Plotter.plot_corr_network(X, lf_dict, outdir=outdir, minimum=0.99)

    def test_plot_corr_network_invalid_indices(self):
        """Test correlation network with invalid feature indices."""
        X = np.random.randn(100, 10)
        lf_dict = {
            'LF_0': [0, 1, 2],
            'LF_1': [10, 11, 12]  # Indices out of bounds
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            with pytest.raises(IndexError):
                Plotter.plot_corr_network(X, lf_dict, outdir=outdir)

    def test_plot_corr_network_single_feature_lf(self):
        """Test correlation network with single-feature latent factors."""
        X = np.random.randn(100, 10)
        lf_dict = {
            'LF_0': [0],
            'LF_1': [1],
            'LF_2': [2]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Should handle single-feature LFs
            Plotter.plot_corr_network(X, lf_dict, outdir=outdir)


class TestPlotControlPlot:
    """Test control plot functionality."""

    def test_plot_controlplot_valid_scores(self):
        """Test control plot with valid score data."""
        scores = np.random.randn(100)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            Plotter.plot_controlplot(scores, outdir=outdir)

            plot_files = list(outdir.glob("*.png")) + list(outdir.glob("*.pdf"))
            assert len(plot_files) > 0

    def test_plot_controlplot_empty_scores(self):
        """Test control plot with empty scores."""
        scores = np.array([])

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            with pytest.warns(UserWarning):
                Plotter.plot_controlplot(scores, outdir=outdir)

    def test_plot_controlplot_single_score(self):
        """Test control plot with single score value."""
        scores = np.array([0.5])

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Should handle single value gracefully
            with pytest.warns(UserWarning):
                Plotter.plot_controlplot(scores, outdir=outdir)

    def test_plot_controlplot_extreme_scores(self):
        """Test control plot with extreme score values."""
        scores = np.array([1e10, -1e10, np.inf, -np.inf, np.nan])

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            with pytest.warns(UserWarning):
                Plotter.plot_controlplot(scores, outdir=outdir)

    def test_plot_controlplot_all_same_scores(self):
        """Test control plot with identical score values."""
        scores = np.full(100, 0.5)  # All same value

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Should handle zero variance in scores
            with pytest.warns(UserWarning):
                Plotter.plot_controlplot(scores, outdir=outdir)

    def test_plot_controlplot_custom_title_xlabel(self):
        """Test control plot with custom title and xlabel."""
        scores = np.random.randn(50)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            Plotter.plot_controlplot(
                scores,
                outdir=outdir,
                title="Custom Control Plot",
                xlabel="Custom X Label"
            )


class TestPlotInteractions:
    """Test interaction plotting functionality."""

    def test_plot_interactions_valid_pairs(self):
        """Test plotting interactions with valid pairs."""
        interaction_pairs = [
            ('Feature_1', 'Feature_5'),
            ('Feature_2', 'Feature_8'),
            ('Feature_3', 'Feature_7')
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            Plotter.plot_interactions(interaction_pairs, outdir=outdir)

            plot_files = list(outdir.glob("*.png")) + list(outdir.glob("*.pdf"))
            assert len(plot_files) > 0

    def test_plot_interactions_empty_pairs(self):
        """Test plotting interactions with empty pairs list."""
        interaction_pairs = []

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            with pytest.warns(UserWarning):
                Plotter.plot_interactions(interaction_pairs, outdir=outdir)

    def test_plot_interactions_single_pair(self):
        """Test plotting interactions with single pair."""
        interaction_pairs = [('Feature_1', 'Feature_2')]

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            Plotter.plot_interactions(interaction_pairs, outdir=outdir)

    def test_plot_interactions_many_pairs(self):
        """Test plotting interactions with many pairs."""
        # Generate many interaction pairs
        interaction_pairs = [(f'Feature_{i}', f'Feature_{i+10}') for i in range(50)]

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Should handle many pairs efficiently
            Plotter.plot_interactions(interaction_pairs, outdir=outdir)

    def test_plot_interactions_duplicate_pairs(self):
        """Test plotting interactions with duplicate pairs."""
        interaction_pairs = [
            ('Feature_1', 'Feature_2'),
            ('Feature_2', 'Feature_1'),  # Duplicate (reversed)
            ('Feature_1', 'Feature_2'),  # Exact duplicate
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Should handle duplicates gracefully
            Plotter.plot_interactions(interaction_pairs, outdir=outdir)

    def test_plot_interactions_self_interactions(self):
        """Test plotting interactions with self-interactions."""
        interaction_pairs = [
            ('Feature_1', 'Feature_1'),  # Self-interaction
            ('Feature_2', 'Feature_3')
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Should handle self-interactions
            with pytest.warns(UserWarning):
                Plotter.plot_interactions(interaction_pairs, outdir=outdir)

    def test_plot_interactions_long_feature_names(self):
        """Test plotting interactions with very long feature names."""
        interaction_pairs = [
            ('This_is_a_very_long_feature_name_that_might_cause_layout_issues',
             'Another_extremely_long_feature_name_for_testing_purposes'),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Should handle long names gracefully
            Plotter.plot_interactions(interaction_pairs, outdir=outdir)


class TestPlottingErrorHandling:
    """Test error handling across plotting functions."""

    def test_plotting_with_matplotlib_backend_issues(self):
        """Test plotting when matplotlib has backend issues."""
        lfs = {'LF_0': np.array([0.8, 0.6, 0.4])}

        # Mock matplotlib to raise an error
        with patch('matplotlib.pyplot.savefig', side_effect=PermissionError("Permission denied")):
            with tempfile.TemporaryDirectory() as tmpdir:
                outdir = Path(tmpdir)

                with pytest.raises(PermissionError):
                    Plotter.plot_latent_factors(lfs, outdir=outdir)

    def test_plotting_memory_efficiency(self):
        """Test memory efficiency of plotting functions."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Create large dataset for plotting
        large_lfs = {f'LF_{i}': np.random.randn(1000) for i in range(20)}

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            Plotter.plot_latent_factors(large_lfs, outdir=outdir)

        memory_after = process.memory_info().rss
        memory_increase = (memory_after - memory_before) / (1024 * 1024)  # MB

        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB

    def test_plotting_figure_cleanup(self):
        """Test that matplotlib figures are properly cleaned up."""
        initial_figure_count = len(plt.get_fignums())

        lfs = {'LF_0': np.array([0.8, 0.6, 0.4])}

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Create multiple plots
            for i in range(5):
                Plotter.plot_latent_factors(lfs, outdir=outdir, title=f"Plot {i}")

        final_figure_count = len(plt.get_fignums())

        # Figures should be cleaned up
        assert final_figure_count <= initial_figure_count + 1  # Allow some leeway

    @patch('seaborn.set_theme')
    def test_plotting_with_seaborn_failure(self, mock_set_theme):
        """Test plotting when seaborn operations fail."""
        mock_set_theme.side_effect = ImportError("Seaborn not available")

        lfs = {'LF_0': np.array([0.8, 0.6, 0.4])}

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Should fallback to matplotlib gracefully
            Plotter.plot_latent_factors(lfs, outdir=outdir)


class TestPlottingStyleConsistency:
    """Test plotting style and appearance consistency."""

    def test_plot_style_consistency(self):
        """Test that all plots use consistent styling."""
        # This is more of an integration test for style consistency
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Test data
            lfs = {'LF_0': np.array([0.8, 0.6, 0.4, 0.2])}
            scores = np.random.randn(50)
            X = np.random.randn(100, 10)
            lf_dict = {'LF_0': [0, 1, 2]}
            interactions = [('F1', 'F2')]

            # Create all plot types
            Plotter.plot_latent_factors(lfs, outdir=outdir)
            Plotter.plot_controlplot(scores, outdir=outdir)
            Plotter.plot_corr_network(X, lf_dict, outdir=outdir)
            Plotter.plot_interactions(interactions, outdir=outdir)

            # Verify plots were created
            plot_files = list(outdir.glob("*.png")) + list(outdir.glob("*.pdf"))
            assert len(plot_files) >= 4  # At least one for each plot type

    def test_plot_file_formats(self):
        """Test that plots can be saved in different formats."""
        lfs = {'LF_0': np.array([0.8, 0.6, 0.4])}

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Test different save formats by patching savefig
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                Plotter.plot_latent_factors(lfs, outdir=outdir)

                # Verify savefig was called
                assert mock_savefig.called