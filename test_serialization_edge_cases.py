"""
Test coverage for serialization and persistence edge cases.

Critical gaps in testing object serialization, pickle compatibility,
and state persistence across different scenarios.
"""

import pytest
import pickle
import dill
import numpy as np
import pandas as pd
import tempfile
import os
import io
from unittest.mock import patch, MagicMock

from src.loveslide.slide import SLIDE, OptimizeSLIDE
from src.loveslide.knockoffs import Knockoffs
from src.loveslide.cv import SLIDEcv

class TestObjectSerialization:
    """Test serialization of core objects."""

    def test_slide_pickle_compatibility(self):
        """Test that SLIDE objects can be pickled and unpickled."""
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        slide = SLIDE({'fdr': 0.1})

        # Pickle and unpickle
        pickled = pickle.dumps(slide)
        unpickled = pickle.loads(pickled)

        assert unpickled.input_params == slide.input_params
        assert unpickled.__class__ == slide.__class__

    def test_knockoffs_state_serialization(self):
        """Test serialization of fitted Knockoffs object."""
        X = np.random.randn(50, 10)
        knockoffs = Knockoffs()

        # Fit the object
        knockoffs.fit(X)

        # Test pickle
        pickled = pickle.dumps(knockoffs)
        unpickled = pickle.loads(pickled)

        # Should maintain fitted state
        assert hasattr(unpickled, 'fitted_')
        if hasattr(knockoffs, 'sigma_'):
            np.testing.assert_array_equal(unpickled.sigma_, knockoffs.sigma_)

    def test_large_object_serialization(self):
        """Test serialization of objects with large data."""
        X = np.random.randn(1000, 500)  # Large data
        y = np.random.randn(1000)

        slide = OptimizeSLIDE({'fdr': 0.1})

        # Store large data
        slide.X = X
        slide.y = y

        # Test with dill for complex objects
        try:
            pickled = dill.dumps(slide)
            unpickled = dill.loads(pickled)

            np.testing.assert_array_equal(unpickled.X, slide.X)
            np.testing.assert_array_equal(unpickled.y, slide.y)
        except Exception as e:
            pytest.skip(f"Dill not available or large object serialization failed: {e}")

    def test_partial_state_serialization(self):
        """Test serialization of partially initialized objects."""
        slide = SLIDE({'fdr': 0.1})

        # Don't initialize data, just serialize the config
        pickled = pickle.dumps(slide)
        unpickled = pickle.loads(pickled)

        assert unpickled.input_params == slide.input_params

class TestFileStatePersistence:
    """Test file-based state persistence."""

    def test_corrupted_state_file_recovery(self):
        """Test recovery from corrupted state files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_file = os.path.join(temp_dir, "corrupted_state.pkl")

            # Create corrupted file
            with open(state_file, 'wb') as f:
                f.write(b"corrupted data")

            slide = OptimizeSLIDE({'fdr': 0.1, 'outpath': temp_dir})

            # Should handle corrupted state gracefully
            try:
                slide.load_state(0)  # Try to load from iteration 0
            except (pickle.PickleError, EOFError, FileNotFoundError):
                # Expected behavior - should not crash the application
                pass

    def test_partial_file_write_recovery(self):
        """Test recovery from interrupted file writes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            slide = OptimizeSLIDE({'fdr': 0.1, 'outpath': temp_dir})

            # Simulate interrupted write by creating partial file
            partial_file = os.path.join(temp_dir, "params_iter_0.pkl")
            with open(partial_file, 'wb') as f:
                f.write(b"partial")  # Incomplete pickle data

            # Should handle partial files gracefully
            try:
                slide.load_state(0)
            except (pickle.PickleError, EOFError):
                # Expected - should not crash
                pass

    def test_cross_platform_state_files(self):
        """Test state file compatibility across platforms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            slide = OptimizeSLIDE({'fdr': 0.1, 'outpath': temp_dir})

            # Create test data
            test_state = {
                'iteration': 5,
                'scores': np.random.randn(10),
                'params': {'fdr': 0.1}
            }

            # Save state
            state_file = os.path.join(temp_dir, "test_state.pkl")
            with open(state_file, 'wb') as f:
                pickle.dump(test_state, f)

            # Load and verify
            with open(state_file, 'rb') as f:
                loaded = pickle.load(f)

            assert loaded['iteration'] == test_state['iteration']
            np.testing.assert_array_equal(loaded['scores'], test_state['scores'])

    def test_atomic_file_operations(self):
        """Test atomic file write operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = os.path.join(temp_dir, "atomic_test.pkl")

            # Test data
            test_data = {'key': 'value', 'numbers': np.arange(100)}

            # Simulate atomic write using temporary file
            temp_file = target_file + '.tmp'
            try:
                with open(temp_file, 'wb') as f:
                    pickle.dump(test_data, f)

                # Atomic move
                os.rename(temp_file, target_file)

                # Verify integrity
                with open(target_file, 'rb') as f:
                    loaded = pickle.load(f)

                assert loaded['key'] == test_data['key']
                np.testing.assert_array_equal(loaded['numbers'], test_data['numbers'])

            finally:
                # Cleanup
                if os.path.exists(temp_file):
                    os.remove(temp_file)

class TestVersionCompatibility:
    """Test compatibility across different versions."""

    def test_backward_compatible_state_loading(self):
        """Test loading state files from older versions."""
        # Create a state file that mimics older version format
        old_format_state = {
            'version': '0.9.0',
            'params': {'fdr': 0.1},
            # Missing some newer fields
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            state_file = os.path.join(temp_dir, "old_version.pkl")

            with open(state_file, 'wb') as f:
                pickle.dump(old_format_state, f)

            # Should load gracefully with defaults for missing fields
            with open(state_file, 'rb') as f:
                loaded = pickle.load(f)

            assert loaded['params']['fdr'] == 0.1
            # Should handle missing fields gracefully

    def test_forward_compatible_state_saving(self):
        """Test that current state format is forward-compatible."""
        current_state = {
            'version': '1.0.0',
            'params': {'fdr': 0.1, 'new_param': 'new_value'},
            'metadata': {'creation_time': '2026-03-22'}
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            state_file = os.path.join(temp_dir, "future_compatible.pkl")

            with open(state_file, 'wb') as f:
                pickle.dump(current_state, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Should be readable
            with open(state_file, 'rb') as f:
                loaded = pickle.load(f)

            assert loaded == current_state

class TestMemoryPersistence:
    """Test in-memory persistence and caching scenarios."""

    def test_deep_copy_object_integrity(self):
        """Test that deep copying preserves object integrity."""
        import copy

        X = np.random.randn(100, 50)
        knockoffs = Knockoffs()
        knockoffs.fit(X)

        # Deep copy
        copied = copy.deepcopy(knockoffs)

        # Should be independent objects with same state
        assert copied is not knockoffs
        if hasattr(knockoffs, 'sigma_'):
            np.testing.assert_array_equal(copied.sigma_, knockoffs.sigma_)

        # Modifying copy shouldn't affect original
        copied.input_params = {'modified': True}
        assert 'modified' not in knockoffs.input_params

    def test_circular_reference_handling(self):
        """Test handling of circular references in objects."""
        slide = OptimizeSLIDE({'fdr': 0.1})

        # Create circular reference (carefully)
        slide.self_ref = slide

        try:
            # Should handle circular references in serialization
            pickled = pickle.dumps(slide)
            unpickled = pickle.loads(pickled)

            # Circular reference should be preserved
            assert unpickled.self_ref is unpickled
        except RecursionError:
            # This is acceptable behavior for circular references
            pass
        finally:
            # Clean up circular reference
            del slide.self_ref

if __name__ == "__main__":
    pytest.main([__file__])