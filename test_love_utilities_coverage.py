"""
Test coverage for LOVE utilities module - comprehensive gap analysis.

Missing Coverage:
- recoverGroup function with different matrix configurations
- singleton function edge cases
- threshA function boundary conditions
- partition and extract utility functions
"""
import pytest
import numpy as np
from loveslide.love_python.love.utilities import recoverGroup, singleton, threshA, partition, extract


class TestRecoverGroup:
    """Test recoverGroup function for clustering recovery."""

    def test_recover_group_basic(self):
        """Test basic recoverGroup functionality."""
        A = np.array([[1.0, -0.5], [0.0, 1.0], [-1.0, 0.0]])
        groups = recoverGroup(A)

        assert len(groups) == 2
        assert np.array_equal(groups[0]['pos'], [0])
        assert np.array_equal(groups[0]['neg'], [2])
        assert np.array_equal(groups[1]['pos'], [1])
        assert np.array_equal(groups[1]['neg'], [0])

    def test_recover_group_all_positive(self):
        """Test recoverGroup with all positive loadings."""
        A = np.array([[1.0, 0.5], [0.5, 1.0], [0.3, 0.8]])
        groups = recoverGroup(A)

        for group in groups:
            assert len(group['neg']) == 0
            assert len(group['pos']) > 0

    def test_recover_group_all_negative(self):
        """Test recoverGroup with all negative loadings."""
        A = np.array([[-1.0, -0.5], [-0.5, -1.0], [-0.3, -0.8]])
        groups = recoverGroup(A)

        for group in groups:
            assert len(group['pos']) == 0
            assert len(group['neg']) > 0

    def test_recover_group_sparse_matrix(self):
        """Test recoverGroup with sparse loading matrix."""
        A = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, -1.0]])
        groups = recoverGroup(A)

        assert len(groups[0]['pos']) == 1
        assert len(groups[0]['neg']) == 0
        assert len(groups[1]['pos']) == 0
        assert len(groups[1]['neg']) == 1


class TestSingleton:
    """Test singleton function edge cases."""

    def test_singleton_empty_list(self):
        """Test singleton with empty list."""
        assert singleton([]) == True

    def test_singleton_single_element_lists(self):
        """Test singleton with lists containing single elements."""
        assert singleton([[1], [2, 3]]) == True
        assert singleton([[1]]) == True

    def test_singleton_no_single_elements(self):
        """Test singleton with no single element lists."""
        assert singleton([[1, 2], [3, 4, 5]]) == False

    def test_singleton_mixed_lengths(self):
        """Test singleton with mixed length lists."""
        assert singleton([[1, 2, 3], [4], [5, 6]]) == True


class TestThreshA:
    """Test threshA function boundary conditions."""

    def test_thresh_a_basic(self):
        """Test basic threshA functionality."""
        A = np.array([[0.1, 0.8], [0.9, 0.2], [0.05, 0.95]])
        thresh = 0.5
        result = threshA(A, thresh)

        expected = np.array([[0.0, 0.8], [0.9, 0.0], [0.0, 0.95]])
        np.testing.assert_array_equal(result, expected)

    def test_thresh_a_zero_threshold(self):
        """Test threshA with zero threshold."""
        A = np.array([[0.1, 0.8], [0.9, 0.2]])
        result = threshA(A, 0.0)

        np.testing.assert_array_equal(result, A)

    def test_thresh_a_high_threshold(self):
        """Test threshA with threshold higher than all values."""
        A = np.array([[0.1, 0.8], [0.9, 0.2]])
        result = threshA(A, 1.0)

        np.testing.assert_array_equal(result, np.zeros_like(A))

    def test_thresh_a_negative_values(self):
        """Test threshA with negative values in matrix."""
        A = np.array([[-0.1, 0.8], [0.9, -0.2]])
        result = threshA(A, 0.5, absolute=True)

        expected = np.array([[0.0, 0.8], [0.9, 0.0]])
        np.testing.assert_array_equal(result, expected)


class TestPartitionExtract:
    """Test partition and extract utility functions."""

    def test_partition_balanced(self):
        """Test partition with balanced fold sizes."""
        n = 100
        nfolds = 5
        foldid = partition(n, nfolds)

        assert len(foldid) == n
        assert set(foldid) == set(range(nfolds))

        # Check approximately balanced
        for fold in range(nfolds):
            fold_size = np.sum(foldid == fold)
            assert 15 <= fold_size <= 25  # Allow some variation

    def test_partition_uneven_split(self):
        """Test partition with uneven split."""
        n = 97
        nfolds = 10
        foldid = partition(n, nfolds)

        assert len(foldid) == n
        assert max(foldid) < nfolds

    def test_extract_subset(self):
        """Test extract function for subset extraction."""
        data = np.arange(100)
        indices = [10, 20, 30, 40]

        result = extract(data, indices)
        expected = np.array([10, 20, 30, 40])

        np.testing.assert_array_equal(result, expected)

    def test_extract_matrix_rows(self):
        """Test extract for matrix row extraction."""
        X = np.random.randn(50, 10)
        indices = [5, 15, 25]

        result = extract(X, indices, axis=0)
        expected = X[indices, :]

        np.testing.assert_array_equal(result, expected)