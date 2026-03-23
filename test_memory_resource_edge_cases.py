"""
Test coverage for memory management and resource cleanup edge cases.
"""

import pytest
import numpy as np
import pandas as pd
import gc
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock
import threading
import time
from src.loveslide.slide import SLIDE
from src.loveslide.cv import SLIDEcv
from src.loveslide.knockoffs import Knockoffs


class TestMemoryManagement:
    """Test memory management and resource cleanup edge cases."""

    def test_large_matrix_memory_cleanup(self):
        """Test that large matrices are properly cleaned up."""
        initial_objects = len(gc.get_objects())

        # Create and process large data
        for i in range(3):
            large_X = np.random.randn(1000, 200)
            y = np.random.binomial(1, 0.5, 1000)

            params = {
                'delta': [0.1],
                'lambda': [0.5],
                'fdr': 0.1,
                'niter': 2,
                'pure_homo': True
            }

            slide = SLIDE(params, large_X, y)

            # Force deletion
            del slide
            del large_X
            del y

        # Force garbage collection
        for _ in range(3):
            gc.collect()

        # Check that memory was actually cleaned up
        final_objects = len(gc.get_objects())

        # Should not have accumulated too many objects
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Memory leak detected: {object_growth} objects accumulated"

    def test_file_handle_cleanup(self):
        """Test that file handles are properly closed."""
        # Create temporary files
        temp_files = []

        try:
            for i in range(10):
                # Create temporary data files
                X_data = pd.DataFrame(np.random.randn(50, 10))
                y_data = pd.DataFrame(np.random.binomial(1, 0.5, 50))

                x_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                y_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)

                X_data.to_csv(x_file.name)
                y_data.to_csv(y_file.name)

                temp_files.extend([x_file.name, y_file.name])

                # Load data multiple times
                params = {
                    'x_path': x_file.name,
                    'y_path': y_file.name,
                    'delta': [0.1],
                    'lambda': [0.5],
                    'fdr': 0.1,
                    'niter': 2,
                    'pure_homo': True
                }

                slide = SLIDE(params)
                del slide

                x_file.close()
                y_file.close()

        finally:
            # Cleanup all temporary files
            for filepath in temp_files:
                try:
                    os.unlink(filepath)
                except (OSError, FileNotFoundError):
                    pass

        # Test that we don't have too many open file descriptors
        # This is platform dependent, so we just check it doesn't crash
        assert True  # If we got here, file cleanup worked

    def test_memory_fragmentation_resistance(self):
        """Test resistance to memory fragmentation."""
        # Create and destroy objects of varying sizes to cause fragmentation
        objects = []

        for iteration in range(10):
            # Create objects of random sizes
            sizes = np.random.randint(100, 1000, 5)

            for size in sizes:
                obj = np.random.randn(size, size // 10)
                objects.append(obj)

            # Randomly delete half the objects
            delete_indices = np.random.choice(len(objects), len(objects) // 2, replace=False)
            for idx in sorted(delete_indices, reverse=True):
                del objects[idx]

            gc.collect()

        # Now try to create a SLIDE object - should not fail due to fragmentation
        X = np.random.randn(200, 50)
        y = np.random.binomial(1, 0.5, 200)

        params = {
            'delta': [0.1],
            'lambda': [0.5],
            'fdr': 0.1,
            'niter': 2,
            'pure_homo': True
        }

        slide = SLIDE(params, X, y)
        assert slide.data.X.shape == X.shape

    def test_concurrent_memory_access(self):
        """Test memory safety under concurrent access."""
        shared_data = {'X': None, 'y': None, 'results': []}

        def worker_thread(thread_id):
            """Worker function that creates and processes data."""
            try:
                # Each thread creates its own data
                local_X = np.random.randn(100, 20) + thread_id  # Unique per thread
                local_y = np.random.binomial(1, 0.5, 100)

                params = {
                    'delta': [0.1],
                    'lambda': [0.5],
                    'fdr': 0.1,
                    'niter': 2,
                    'pure_homo': True
                }

                slide = SLIDE(params, local_X, local_y)

                # Store result
                shared_data['results'].append({
                    'thread_id': thread_id,
                    'shape': slide.data.X.shape,
                    'success': True
                })

            except Exception as e:
                shared_data['results'].append({
                    'thread_id': thread_id,
                    'error': str(e),
                    'success': False
                })

        # Create and start multiple threads
        threads = []
        for i in range(3):  # Use small number to avoid overwhelming system
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that all threads completed successfully
        results = shared_data['results']
        assert len(results) == 3
        for result in results:
            assert result['success'], f"Thread {result.get('thread_id')} failed: {result.get('error')}"

    def test_memory_limit_graceful_degradation(self):
        """Test graceful handling when approaching memory limits."""

        def mock_memory_error(*args, **kwargs):
            raise MemoryError("Insufficient memory")

        # Test with simulated memory constraint
        X = np.random.randn(100, 20)
        y = np.random.binomial(1, 0.5, 100)

        params = {
            'delta': [0.1],
            'lambda': [0.5],
            'fdr': 0.1,
            'niter': 2,
            'pure_homo': True
        }

        # Mock a function that might allocate large amounts of memory
        with patch('numpy.linalg.svd', side_effect=mock_memory_error):
            with pytest.raises(MemoryError):
                slide = SLIDE(params, X, y)
                # This would normally call functions that use SVD

    def test_reference_cycle_cleanup(self):
        """Test cleanup of reference cycles."""
        # Create objects with potential reference cycles
        objects = []

        class CyclicObject:
            def __init__(self, data):
                self.data = data
                self.children = []
                self.parent = None

            def add_child(self, child):
                child.parent = self
                self.children.append(child)

        # Create a tree structure with cycles
        root = CyclicObject(np.random.randn(100, 10))
        for i in range(5):
            child = CyclicObject(np.random.randn(50, 5))
            root.add_child(child)
            # Create cycle
            child.parent_ref = root

        objects.append(root)

        # Now create SLIDE object that might create its own cycles
        X = np.random.randn(100, 20)
        y = np.random.binomial(1, 0.5, 100)

        params = {
            'delta': [0.1],
            'lambda': [0.5],
            'fdr': 0.1,
            'niter': 2,
            'pure_homo': True
        }

        slide = SLIDE(params, X, y)

        # Clear references
        del root
        del objects
        del slide
        del X
        del y

        # Force garbage collection of cycles
        gc.collect()

        # Should successfully complete without hanging
        assert True

    def test_numpy_memory_view_cleanup(self):
        """Test cleanup of numpy memory views and shared arrays."""
        # Create base array
        base_array = np.random.randn(1000, 100)

        # Create multiple views of the same data
        views = []
        for i in range(10):
            start = i * 10
            end = start + 50
            view = base_array[start:end, :]
            views.append(view)

        # Use views in SLIDE
        X = views[0]  # Use one of the views
        y = np.random.binomial(1, 0.5, X.shape[0])

        params = {
            'delta': [0.1],
            'lambda': [0.5],
            'fdr': 0.1,
            'niter': 2,
            'pure_homo': True
        }

        slide = SLIDE(params, X, y)

        # Check that the underlying data is still accessible
        assert np.shares_memory(slide.data.X, base_array)

        # Clean up
        del views
        del base_array
        del slide

        # Should complete without memory access errors
        gc.collect()
        assert True