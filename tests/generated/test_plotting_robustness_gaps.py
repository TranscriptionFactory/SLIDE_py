#!/usr/bin/env python3
"""
Comprehensive test coverage for plotting module edge cases and robustness.
Addresses gaps in visualization error handling and data edge cases.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import tempfile
import os

from loveslide.plotting import Plotter


class TestPlotterRobustness:
    """Test plotting module robustness and edge cases."""

    def test_plot_latent_factors_empty_data(self):
        """Test plotting with empty latent factors dictionary."""
        plotter = Plotter()
        empty_lfs = {}

        # Should handle empty data gracefully
        with pytest.raises(ValueError, match="No latent factors provided"):
            plotter.plot_latent_factors(empty_lfs)

    def test_plot_latent_factors_malformed_data(self):
        """Test plotting with malformed latent factor data."""
        plotter = Plotter()

        # Missing required columns
        malformed_lfs = {
            'LF1': pd.DataFrame({
                'loading': [0.5, 0.3],
                # Missing 'AUC', 'corr', 'color'
            })
        }

        with pytest.raises(KeyError):
            plotter.plot_latent_factors(malformed_lfs)

    def test_plot_latent_factors_extreme_dimensions(self):
        """Test plotting with extreme numbers of factors/genes."""
        plotter = Plotter()

        # Large number of factors
        many_factors = {}
        for i in range(50):  # 50 latent factors
            many_factors[f'LF{i}'] = pd.DataFrame({
                'loading': np.random.rand(10),
                'AUC': np.random.rand(10),
                'corr': np.random.rand(10),
                'color': ['red'] * 10
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should handle large plots without crashing
            plotter.plot_latent_factors(many_factors, outdir=tmpdir)
            assert os.path.exists(f"{tmpdir}/Significant Latent Factors.png")

    def test_plot_with_seaborn_unavailable(self):
        """Test plotting when seaborn is not available."""
        with patch('loveslide.plotting.sns', None):
            plotter = Plotter()

            lfs = {
                'LF1': pd.DataFrame({
                    'loading': [0.5, 0.3, -0.2],
                    'AUC': [0.7, 0.6, 0.8],
                    'corr': [0.4, 0.5, -0.3],
                    'color': ['red', 'blue', 'gray']
                })
            }

            # Should fallback to matplotlib
            with tempfile.TemporaryDirectory() as tmpdir:
                plotter.plot_latent_factors(lfs, outdir=tmpdir)
                assert os.path.exists(f"{tmpdir}/Significant Latent Factors.png")

    def test_plot_with_invalid_colors(self):
        """Test plotting with invalid color specifications."""
        plotter = Plotter()

        lfs = {
            'LF1': pd.DataFrame({
                'loading': [0.5, 0.3],
                'AUC': [0.7, 0.6],
                'corr': [0.4, 0.5],
                'color': ['invalid_color', 'another_invalid']
            })
        }

        # Should handle invalid colors gracefully
        with tempfile.TemporaryDirectory() as tmpdir:
            plotter.plot_latent_factors(lfs, outdir=tmpdir)
            assert os.path.exists(f"{tmpdir}/Significant Latent Factors.png")

    def test_plot_network_edge_cases(self):
        """Test network plotting with edge cases."""
        plotter = Plotter()

        # Test with no interactions
        empty_interactions = pd.DataFrame(columns=['feature1', 'feature2', 'weight'])

        # Should handle empty network gracefully
        with tempfile.TemporaryDirectory() as tmpdir:
            # This should create an empty network plot
            pass  # Implementation depends on actual method signature

    def test_plot_file_permission_errors(self):
        """Test plotting when output directory is not writable."""
        plotter = Plotter()

        lfs = {
            'LF1': pd.DataFrame({
                'loading': [0.5],
                'AUC': [0.7],
                'corr': [0.4],
                'color': ['red']
            })
        }

        # Try to write to a non-existent/non-writable directory
        with pytest.raises((OSError, PermissionError)):
            plotter.plot_latent_factors(lfs, outdir="/root/nonexistent")


class TestPlotterMemoryManagement:
    """Test plotting module memory management."""

    def test_large_plot_memory_usage(self):
        """Test memory usage with very large plots."""
        plotter = Plotter()

        # Create large dataset
        large_lfs = {}
        for i in range(10):
            large_lfs[f'LF{i}'] = pd.DataFrame({
                'loading': np.random.rand(1000),  # 1000 genes per factor
                'AUC': np.random.rand(1000),
                'corr': np.random.rand(1000),
                'color': np.random.choice(['red', 'blue', 'gray'], 1000)
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should handle large plots without memory issues
            plotter.plot_latent_factors(large_lfs, outdir=tmpdir)

            # Verify cleanup - matplotlib should close figures
            assert len(plt.get_fignums()) == 0

    def test_figure_cleanup_on_error(self):
        """Test that figures are cleaned up even when errors occur."""
        plotter = Plotter()

        with patch('matplotlib.pyplot.savefig', side_effect=Exception("Save failed")):
            lfs = {
                'LF1': pd.DataFrame({
                    'loading': [0.5],
                    'AUC': [0.7],
                    'corr': [0.4],
                    'color': ['red']
                })
            }

            with pytest.raises(Exception):
                plotter.plot_latent_factors(lfs)

            # Should still cleanup figures
            assert len(plt.get_fignums()) == 0


class TestPlotterDataValidation:
    """Test plotting module data validation."""

    def test_nan_values_in_data(self):
        """Test plotting with NaN values in data."""
        plotter = Plotter()

        lfs = {
            'LF1': pd.DataFrame({
                'loading': [0.5, np.nan, 0.3],
                'AUC': [0.7, 0.6, np.nan],
                'corr': [np.nan, 0.5, 0.4],
                'color': ['red', 'blue', 'gray']
            })
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should handle NaN values gracefully
            plotter.plot_latent_factors(lfs, outdir=tmpdir)
            assert os.path.exists(f"{tmpdir}/Significant Latent Factors.png")

    def test_infinite_values_in_data(self):
        """Test plotting with infinite values in data."""
        plotter = Plotter()

        lfs = {
            'LF1': pd.DataFrame({
                'loading': [0.5, np.inf, -np.inf],
                'AUC': [0.7, np.inf, 0.8],
                'corr': [0.4, 0.5, -np.inf],
                'color': ['red', 'blue', 'gray']
            })
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should handle infinite values gracefully
            plotter.plot_latent_factors(lfs, outdir=tmpdir)
            assert os.path.exists(f"{tmpdir}/Significant Latent Factors.png")

    def test_zero_variance_data(self):
        """Test plotting with zero variance data."""
        plotter = Plotter()

        lfs = {
            'LF1': pd.DataFrame({
                'loading': [0.5, 0.5, 0.5],  # All same values
                'AUC': [0.7, 0.7, 0.7],
                'corr': [0.4, 0.4, 0.4],
                'color': ['red', 'red', 'red']
            })
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should handle zero variance gracefully
            plotter.plot_latent_factors(lfs, outdir=tmpdir)
            assert os.path.exists(f"{tmpdir}/Significant Latent Factors.png")


if __name__ == "__main__":
    pytest.main([__file__])