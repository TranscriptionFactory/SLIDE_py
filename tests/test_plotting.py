"""
Test coverage for loveslide.plotting visualization functionality.

Major gaps:
- Plot generation and output validation
- Error handling for invalid data inputs
- File saving and directory creation
- Plot styling and formatting
- Edge cases with empty or malformed data
- Integration with different data types
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt

from loveslide.plotting import Plotter


class TestPlotterInit:
    """Test Plotter initialization."""

    def test_plotter_init(self):
        """Test basic Plotter initialization."""
        plotter = Plotter()
        assert isinstance(plotter, Plotter)


class TestLatentFactorPlots:
    """Test latent factor plotting functionality."""

    @pytest.fixture
    def sample_lfs_data(self):
        """Create sample latent factors data for testing."""
        return {
            'LF1': np.random.randn(50),
            'LF2': np.random.randn(50),
            'LF3': np.random.randn(50)
        }

    def test_plot_latent_factors_basic(self, sample_lfs_data):
        """Test basic latent factors plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Plotter.plot_latent_factors(sample_lfs_data, outdir=tmpdir)

            # Check if plot file was created
            plot_files = list(Path(tmpdir).glob("*.png"))
            assert len(plot_files) > 0

    def test_plot_latent_factors_no_outdir(self, sample_lfs_data):
        """Test plotting without output directory (display only)."""
        # Should not crash
        with patch('matplotlib.pyplot.show'):
            Plotter.plot_latent_factors(sample_lfs_data, outdir=None)

    def test_plot_latent_factors_empty_data(self):
        """Test plotting with empty latent factors."""
        empty_lfs = {}

        # Should handle gracefully or raise informative error
        with pytest.raises((ValueError, RuntimeError)):
            Plotter.plot_latent_factors(empty_lfs)

    def test_plot_latent_factors_single_lf(self):
        """Test plotting with single latent factor."""
        single_lf = {'LF1': np.random.randn(50)}

        with tempfile.TemporaryDirectory() as tmpdir:
            Plotter.plot_latent_factors(single_lf, outdir=tmpdir)

    def test_plot_latent_factors_large_data(self):
        """Test plotting with large datasets."""
        large_lfs = {f'LF{i}': np.random.randn(1000) for i in range(20)}

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should handle without memory issues
            Plotter.plot_latent_factors(large_lfs, outdir=tmpdir)

    def test_plot_latent_factors_custom_title(self, sample_lfs_data):
        """Test plotting with custom title."""
        custom_title = "Custom Test Title"

        with tempfile.TemporaryDirectory() as tmpdir:
            Plotter.plot_latent_factors(
                sample_lfs_data,
                outdir=tmpdir,
                title=custom_title
            )


class TestCorrelationNetworkPlots:
    """Test correlation network plotting functionality."""

    @pytest.fixture
    def sample_network_data(self):
        """Create sample data for network plotting."""
        X = np.random.randn(100, 20)
        lf_dict = {
            'LF1': [0, 1, 2, 3],
            'LF2': [4, 5, 6],
            'LF3': [7, 8, 9, 10, 11]
        }
        return X, lf_dict

    def test_plot_corr_network_basic(self, sample_network_data):
        """Test basic correlation network plotting."""
        X, lf_dict = sample_network_data

        with tempfile.TemporaryDirectory() as tmpdir:
            Plotter.plot_corr_network(X, lf_dict, outdir=tmpdir)

    def test_plot_corr_network_custom_threshold(self, sample_network_data):
        """Test network plotting with custom correlation threshold."""
        X, lf_dict = sample_network_data

        with tempfile.TemporaryDirectory() as tmpdir:
            Plotter.plot_corr_network(
                X, lf_dict,
                outdir=tmpdir,
                minimum=0.5
            )

    def test_plot_corr_network_no_connections(self, sample_network_data):
        """Test network plotting when no correlations exceed threshold."""
        X, lf_dict = sample_network_data

        # Very high threshold - no connections should be shown
        with tempfile.TemporaryDirectory() as tmpdir:
            Plotter.plot_corr_network(
                X, lf_dict,
                outdir=tmpdir,
                minimum=0.99
            )

    def test_plot_corr_network_single_lf(self):
        """Test network plotting with single latent factor."""
        X = np.random.randn(50, 10)
        lf_dict = {'LF1': [0, 1, 2]}

        with tempfile.TemporaryDirectory() as tmpdir:
            Plotter.plot_corr_network(X, lf_dict, outdir=tmpdir)

    def test_plot_corr_network_invalid_indices(self):
        """Test error handling for invalid feature indices."""
        X = np.random.randn(50, 10)
        lf_dict = {'LF1': [0, 1, 20]}  # Index 20 doesn't exist

        with pytest.raises((IndexError, ValueError)):
            Plotter.plot_corr_network(X, lf_dict)


class TestControlPlots:
    """Test control plot functionality."""

    @pytest.fixture
    def sample_scores(self):
        """Create sample scores data."""
        return np.random.randn(100)

    def test_plot_controlplot_basic(self, sample_scores):
        """Test basic control plot generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Plotter.plot_controlplot(sample_scores, outdir=tmpdir)

    def test_plot_controlplot_custom_labels(self, sample_scores):
        """Test control plot with custom title and xlabel."""
        custom_title = "Custom Control Plot"
        custom_xlabel = "Custom X Label"

        with tempfile.TemporaryDirectory() as tmpdir:
            Plotter.plot_controlplot(
                sample_scores,
                outdir=tmpdir,
                title=custom_title,
                xlabel=custom_xlabel
            )

    def test_plot_controlplot_empty_scores(self):
        """Test control plot with empty scores array."""
        empty_scores = np.array([])

        with pytest.raises((ValueError, RuntimeError)):
            Plotter.plot_controlplot(empty_scores)

    def test_plot_controlplot_single_score(self):
        """Test control plot with single score value."""
        single_score = np.array([1.5])

        with tempfile.TemporaryDirectory() as tmpdir:
            Plotter.plot_controlplot(single_score, outdir=tmpdir)


class TestInteractionPlots:
    """Test interaction plotting functionality."""

    @pytest.fixture
    def sample_interactions(self):
        """Create sample interaction pairs data."""
        return [
            ('LF1', 'LF2'),
            ('LF2', 'LF3'),
            ('LF1', 'LF3')
        ]

    def test_plot_interactions_basic(self, sample_interactions):
        """Test basic interaction plot generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Plotter.plot_interactions(sample_interactions, outdir=tmpdir)

    def test_plot_interactions_empty_pairs(self):
        """Test interaction plot with empty pairs."""
        empty_pairs = []

        # Should handle gracefully or raise informative error
        with pytest.raises((ValueError, RuntimeError)):
            Plotter.plot_interactions(empty_pairs)

    def test_plot_interactions_single_pair(self):
        """Test interaction plot with single pair."""
        single_pair = [('LF1', 'LF2')]

        with tempfile.TemporaryDirectory() as tmpdir:
            Plotter.plot_interactions(single_pair, outdir=tmpdir)

    def test_plot_interactions_custom_title(self, sample_interactions):
        """Test interaction plot with custom title."""
        custom_title = "Custom Interaction Plot"

        with tempfile.TemporaryDirectory() as tmpdir:
            Plotter.plot_interactions(
                sample_interactions,
                outdir=tmpdir,
                title=custom_title
            )


class TestPlottingEdgeCases:
    """Test edge cases and error handling."""

    def test_plot_invalid_output_directory(self):
        """Test plotting with invalid output directory."""
        # Read-only directory or non-existent parent
        invalid_dir = "/nonexistent/readonly/dir"

        sample_data = {'LF1': np.random.randn(10)}

        # Should raise appropriate error
        with pytest.raises((PermissionError, FileNotFoundError, OSError)):
            Plotter.plot_latent_factors(sample_data, outdir=invalid_dir)

    def test_plot_with_nan_data(self):
        """Test plotting with NaN/infinite values in data."""
        nan_data = {'LF1': np.array([1, 2, np.nan, 4, np.inf])}

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should handle gracefully or raise informative error
            with pytest.raises((ValueError, RuntimeError)):
                Plotter.plot_latent_factors(nan_data, outdir=tmpdir)

    def test_plot_memory_management(self):
        """Test that plots properly clean up matplotlib figures."""
        # Create many plots to test memory management
        initial_figs = len(plt.get_fignums())

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                data = {f'LF{i}': np.random.randn(50)}
                Plotter.plot_latent_factors(data, outdir=tmpdir)

        # Should not accumulate figures in memory
        final_figs = len(plt.get_fignums())
        assert final_figs <= initial_figs + 1  # Allow for some tolerance


class TestPlottingIntegration:
    """Integration tests for plotting functionality."""

    def test_full_plotting_workflow(self):
        """Test complete plotting workflow with all plot types."""
        # Generate synthetic SLIDE results
        X = np.random.randn(100, 50)
        lfs = {f'LF{i}': np.random.randn(100) for i in range(5)}
        lf_dict = {f'LF{i}': list(range(i*10, (i+1)*10)) for i in range(5)}
        scores = np.random.randn(50)
        interactions = [('LF1', 'LF2'), ('LF3', 'LF4')]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate all plot types
            Plotter.plot_latent_factors(lfs, outdir=tmpdir)
            Plotter.plot_corr_network(X, lf_dict, outdir=tmpdir)
            Plotter.plot_controlplot(scores, outdir=tmpdir)
            Plotter.plot_interactions(interactions, outdir=tmpdir)

            # Verify all plots were created
            plot_files = list(Path(tmpdir).glob("*.png"))
            assert len(plot_files) >= 4  # At least one for each plot type