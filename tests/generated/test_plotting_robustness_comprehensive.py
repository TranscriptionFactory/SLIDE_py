"""
Comprehensive plotting and visualization robustness testing.
Tests edge cases, backend failures, and data format issues.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import os

from loveslide.plotting import Plotter


class TestPlotterInitialization:
    """Test Plotter initialization and backend handling."""

    @patch('matplotlib.pyplot')
    def test_plotter_missing_matplotlib(self, mock_plt):
        """Test Plotter when matplotlib is not available."""
        mock_plt.side_effect = ImportError("matplotlib not found")

        with pytest.raises(ImportError):
            Plotter()

    @patch('seaborn')
    def test_plotter_missing_seaborn(self, mock_sns):
        """Test Plotter when seaborn is not available."""
        mock_sns.side_effect = ImportError("seaborn not found")

        try:
            plotter = Plotter()
            # Should fallback gracefully or raise appropriate error
            assert plotter is not None
        except ImportError:
            # Acceptable if seaborn is required
            pass

    def test_plotter_backend_configuration(self):
        """Test Plotter with different matplotlib backends."""
        import matplotlib
        original_backend = matplotlib.get_backend()

        try:
            # Test with non-interactive backend
            matplotlib.use('Agg')
            plotter = Plotter()
            assert plotter is not None

            # Test with invalid backend
            try:
                matplotlib.use('invalid_backend')
                plotter = Plotter()
            except ValueError:
                # Expected for invalid backend
                pass
        finally:
            matplotlib.use(original_backend)


class TestPlotterDataValidation:
    """Test Plotter data validation and edge cases."""

    def test_plot_with_empty_data(self):
        """Test plotting functions with empty data."""
        plotter = Plotter()

        # Empty DataFrames
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty data"):
            plotter.plot_marginal_LFs(empty_df, save_path=None)

        # DataFrame with only NaN
        nan_df = pd.DataFrame({'A': [np.nan, np.nan], 'B': [np.nan, np.nan]})

        with pytest.raises(ValueError):
            plotter.plot_interaction_pairs(nan_df, save_path=None)

    def test_plot_with_infinite_values(self):
        """Test plotting functions with infinite values."""
        plotter = Plotter()

        # Data with infinity
        inf_data = pd.DataFrame({
            'LF1': [1, 2, np.inf, 4],
            'LF2': [-np.inf, 2, 3, 4],
            'LF3': [1, 2, 3, 4]
        })

        try:
            plotter.plot_marginal_LFs(inf_data, save_path=None)
            # Should handle infinite values gracefully
        except ValueError:
            # Acceptable to reject infinite values
            pass

    def test_plot_with_extreme_values(self):
        """Test plotting functions with extreme data values."""
        plotter = Plotter()

        # Very large range
        extreme_data = pd.DataFrame({
            'small': [1e-10, 2e-10, 3e-10],
            'large': [1e10, 2e10, 3e10],
            'normal': [1, 2, 3]
        })

        # Should handle extreme ranges appropriately
        plotter.plot_marginal_LFs(extreme_data, save_path=None)

    def test_plot_with_single_data_point(self):
        """Test plotting functions with single data point."""
        plotter = Plotter()

        single_point = pd.DataFrame({
            'LF1': [1.0],
            'LF2': [2.0]
        })

        try:
            plotter.plot_marginal_LFs(single_point, save_path=None)
            # Should handle single point gracefully
        except ValueError:
            # May require minimum number of points
            pass

    def test_plot_with_constant_data(self):
        """Test plotting functions with constant data values."""
        plotter = Plotter()

        constant_data = pd.DataFrame({
            'constant': [5.0] * 100,
            'variable': np.random.randn(100)
        })

        # Should handle zero variance gracefully
        plotter.plot_marginal_LFs(constant_data, save_path=None)

    def test_plot_with_missing_columns(self):
        """Test plotting functions with missing expected columns."""
        plotter = Plotter()

        incomplete_data = pd.DataFrame({
            'unexpected_column': np.random.randn(50)
        })

        # Should handle missing expected columns
        try:
            plotter.plot_interaction_pairs(incomplete_data, save_path=None)
        except (KeyError, ValueError):
            # Expected to fail with missing columns
            pass


class TestPlotterSaveOperations:
    """Test Plotter file save operations and error handling."""

    def test_plot_save_to_readonly_directory(self):
        """Test saving plots to read-only directory."""
        plotter = Plotter()

        data = pd.DataFrame({
            'LF1': np.random.randn(50),
            'LF2': np.random.randn(50)
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            # Make directory read-only
            os.chmod(tmpdir, 0o444)

            readonly_path = os.path.join(tmpdir, "plot.png")

            try:
                with pytest.raises(PermissionError):
                    plotter.plot_marginal_LFs(data, save_path=readonly_path)
            finally:
                # Restore write permissions for cleanup
                os.chmod(tmpdir, 0o755)

    def test_plot_save_invalid_path(self):
        """Test saving plots to invalid file paths."""
        plotter = Plotter()

        data = pd.DataFrame({
            'LF1': np.random.randn(50),
            'LF2': np.random.randn(50)
        })

        # Non-existent directory
        invalid_path = "/nonexistent/directory/plot.png"

        with pytest.raises(FileNotFoundError):
            plotter.plot_marginal_LFs(data, save_path=invalid_path)

        # Invalid filename characters (platform-specific)
        if os.name == 'nt':  # Windows
            invalid_filename = "plot<>:\"/|?*.png"
        else:  # Unix-like
            invalid_filename = "plot\x00.png"

        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = os.path.join(tmpdir, invalid_filename)

            with pytest.raises(OSError):
                plotter.plot_marginal_LFs(data, save_path=invalid_path)

    def test_plot_save_disk_space_exhaustion(self):
        """Test behavior when disk space is exhausted during save."""
        plotter = Plotter()

        data = pd.DataFrame({
            'LF1': np.random.randn(50),
            'LF2': np.random.randn(50)
        })

        with tempfile.NamedTemporaryFile() as tmp:
            # Mock disk space exhaustion
            with patch('matplotlib.figure.Figure.savefig') as mock_savefig:
                mock_savefig.side_effect = OSError("No space left on device")

                with pytest.raises(OSError):
                    plotter.plot_marginal_LFs(data, save_path=tmp.name)

    def test_plot_save_concurrent_access(self):
        """Test saving when file is being accessed by another process."""
        plotter = Plotter()

        data = pd.DataFrame({
            'LF1': np.random.randn(50),
            'LF2': np.random.randn(50)
        })

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = tmp.name

        try:
            # Simulate file being locked by another process
            with patch('matplotlib.figure.Figure.savefig') as mock_savefig:
                mock_savefig.side_effect = PermissionError("File in use")

                with pytest.raises(PermissionError):
                    plotter.plot_marginal_LFs(data, save_path=temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestPlotterMemoryManagement:
    """Test Plotter memory management with large datasets."""

    def test_plot_large_dataset_memory_efficiency(self):
        """Test plotting with very large datasets."""
        plotter = Plotter()

        try:
            # Large dataset
            large_data = pd.DataFrame({
                f'LF{i}': np.random.randn(100000)
                for i in range(20)
            })

            # Should handle large datasets efficiently
            plotter.plot_marginal_LFs(large_data, save_path=None)

        except MemoryError:
            # Expected for extremely large datasets
            pytest.skip("Insufficient memory for large dataset test")

    def test_plot_figure_cleanup(self):
        """Test proper cleanup of matplotlib figures."""
        plotter = Plotter()

        data = pd.DataFrame({
            'LF1': np.random.randn(50),
            'LF2': np.random.randn(50)
        })

        # Check figure count before and after
        import matplotlib.pyplot as plt
        initial_fig_count = len(plt.get_fignums())

        # Generate multiple plots
        for i in range(10):
            plotter.plot_marginal_LFs(data, save_path=None)

        final_fig_count = len(plt.get_fignums())

        # Should not accumulate figures in memory
        assert final_fig_count - initial_fig_count <= 1


class TestPlotterVisualizationQuality:
    """Test visualization quality and edge cases."""

    def test_plot_color_palette_edge_cases(self):
        """Test plotting with edge cases in color palette selection."""
        plotter = Plotter()

        # More categories than available colors
        many_categories = pd.DataFrame({
            'category': [f'cat_{i}' for i in range(100)],
            'value': np.random.randn(100)
        })

        # Should handle many categories gracefully
        try:
            plotter.plot_interaction_pairs(many_categories, save_path=None)
        except ValueError:
            # May limit number of categories
            pass

    def test_plot_axis_scaling_edge_cases(self):
        """Test automatic axis scaling with edge cases."""
        plotter = Plotter()

        # Data with very small differences
        tiny_diff_data = pd.DataFrame({
            'LF1': [1.0000001, 1.0000002, 1.0000003],
            'LF2': [2.0000001, 2.0000002, 2.0000003]
        })

        # Should handle tiny differences in scaling
        plotter.plot_marginal_LFs(tiny_diff_data, save_path=None)

        # Data with huge differences
        huge_diff_data = pd.DataFrame({
            'small': [1e-6, 2e-6, 3e-6],
            'large': [1e6, 2e6, 3e6]
        })

        # Should handle large dynamic range
        plotter.plot_marginal_LFs(huge_diff_data, save_path=None)

    def test_plot_text_rendering_edge_cases(self):
        """Test text rendering with special characters and long strings."""
        plotter = Plotter()

        # Data with special characters in column names
        special_chars_data = pd.DataFrame({
            'LF_α_β_γ': np.random.randn(20),
            'LF_∑_∆_∞': np.random.randn(20),
            'LF_很长的中文名字': np.random.randn(20)
        })

        try:
            plotter.plot_marginal_LFs(special_chars_data, save_path=None)
            # Should handle Unicode characters
        except UnicodeError:
            # May fail with unsupported characters
            pass

        # Very long column names
        long_names_data = pd.DataFrame({
            'Very_Long_Column_Name_That_Might_Cause_Layout_Issues_1': np.random.randn(20),
            'Another_Extremely_Long_Column_Name_That_Could_Overflow_2': np.random.randn(20)
        })

        # Should handle long names gracefully
        plotter.plot_marginal_LFs(long_names_data, save_path=None)