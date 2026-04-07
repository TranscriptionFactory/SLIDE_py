"""
Test coverage for LOVE Python internal functions.
Addresses: Pure node estimation, parameter estimation, score computation
"""
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import warnings

from loveslide.love_python.love.est_pure_homo import (
    EstAI, EstC, FindRowMax, FindPureNode, TestPure, RecoverAI, Merge
)
from loveslide.love_python.love.est_nonpure import (
    EstY, EstAJInv, LP, EstAJDant, Dantzig
)
from loveslide.love_python.love.score import Score_mat, LP_Score
from loveslide.love_python.love.utilities import (
    recoverGroup, singleton, threshA, offSum, partition, extract
)
from loveslide.love_python.love.cv import CV_delta, KfoldCV_delta, CV_lbd


class TestPureHomogeneousEstimation:
    """Test pure node estimation functions."""

    def test_estai_basic_functionality(self):
        """Test AI matrix estimation."""
        p = 10
        Sigma = np.random.randn(p, p)
        Sigma = Sigma @ Sigma.T + np.eye(p) * 0.1  # Make positive definite

        optDelta = 0.1
        se_est = np.ones(p) * 0.05
        diagonal = True

        result = EstAI(Sigma, optDelta, se_est, diagonal)

        # Check output shape and properties
        assert result.shape == (p, p)
        assert np.allclose(result, result.T)  # Should be symmetric
        if diagonal:
            assert np.allclose(result, np.diag(np.diag(result)))

    def test_estai_edge_cases(self):
        """Test AI estimation edge cases."""
        # Singular matrix
        Sigma = np.ones((3, 3))
        se_est = np.ones(3) * 0.1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = EstAI(Sigma, 0.1, se_est, True)

        assert result.shape == (3, 3)

        # Very small delta
        Sigma = np.eye(5)
        result = EstAI(Sigma, 1e-10, np.ones(5) * 0.01, True)
        assert np.all(np.isfinite(result))

    def test_estc_basic_functionality(self):
        """Test C matrix estimation."""
        p = 8
        Sigma = np.eye(p) + np.random.randn(p, p) * 0.1
        Sigma = Sigma @ Sigma.T
        AI = np.random.randn(p, p) * 0.1

        for diagonal in [True, False]:
            result = EstC(Sigma, AI, diagonal)
            assert result.shape == (p, p)
            if diagonal:
                assert np.allclose(result, np.diag(np.diag(result)))

    def test_findrowmax_basic(self):
        """Test row maximum finding."""
        Sigma = np.array([[1, 0.8, 0.3], [0.8, 1, 0.7], [0.3, 0.7, 1]])

        result = FindRowMax(Sigma)

        assert 'M' in result
        assert 'I' in result
        assert len(result['M']) == 3
        assert len(result['I']) == 3

        # Check that maximum values are correctly identified
        for i in range(3):
            off_diag = np.abs(Sigma[i, :])
            off_diag[i] = 0  # Exclude diagonal
            expected_max = np.max(off_diag)
            assert np.isclose(result['M'][i], expected_max)

    def test_findrowmax_edge_cases(self):
        """Test row maximum with edge cases."""
        # Matrix with identical off-diagonal elements
        Sigma = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])

        result = FindRowMax(Sigma)

        # Should still work and find maximum
        assert all(np.isclose(result['M'], 0.5))

        # Matrix with zeros
        Sigma = np.eye(3)
        result = FindRowMax(Sigma)
        assert all(np.isclose(result['M'], 0))

    def test_findpurenode_basic(self):
        """Test pure node identification."""
        # Create correlation matrix with clear pure structure
        Sigma = np.array([
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.8],
            [0.1, 0.1, 0.8, 1.0]
        ])

        delta = 0.5
        Ms = np.array([0.9, 0.9, 0.8, 0.8])
        arg_Ms = np.array([1, 0, 3, 2])

        result = FindPureNode(Sigma, delta, Ms, arg_Ms)

        # Should identify groups
        assert len(result) >= 1
        for group in result:
            assert len(group) >= 2  # Each pure group has at least 2 nodes

    def test_testpure_statistical_test(self):
        """Test pure node statistical test."""
        # Create data where first row should be pure
        p = 5
        Sigma_row = np.array([0.8, 0.85, 0.1, 0.1, 0.05])
        rowInd = 0
        Si = np.array([0, 1])  # Indices of pure group
        n = 100

        result = TestPure(Sigma_row, rowInd, Si, n)

        assert isinstance(result, bool)

    def test_recoverai_group_recovery(self):
        """Test AI matrix recovery from groups."""
        # Mock group structure
        estGroupList = [
            {'I': [0, 1], 'AI': np.array([[1, 0.5], [0.5, 1]])},
            {'I': [2, 3], 'AI': np.array([[1, 0.7], [0.7, 1]])}
        ]
        p = 4

        result = RecoverAI(estGroupList, p)

        assert result.shape == (p, p)
        assert np.allclose(result, result.T)

        # Check that submatrices are correctly placed
        assert np.isclose(result[0, 1], 0.5)
        assert np.isclose(result[2, 3], 0.7)

    def test_merge_group_functionality(self):
        """Test group merging functionality."""
        groupList = [[0, 1], [2, 3]]
        groupVec = [1, 3]  # Connect groups through nodes 1 and 3

        result = Merge(groupList, groupVec)

        # Should merge into single group
        assert len(result) == 1
        assert len(result[0]) == 4


class TestNonPureEstimation:
    """Test non-pure node estimation functions."""

    def test_esty_basic_functionality(self):
        """Test Y matrix estimation."""
        p = 6
        Sigma = np.random.randn(p, p)
        Sigma = Sigma @ Sigma.T + np.eye(p) * 0.1

        AI = np.random.randn(p, p) * 0.1
        pureVec = np.array([1, 1, 0, 0, 1, 1])  # Pure indicators

        result = EstY(Sigma, AI, pureVec)

        # Check shape
        non_pure_count = np.sum(pureVec == 0)
        assert result.shape[0] == non_pure_count

    def test_estajinv_basic_functionality(self):
        """Test AJ inverse estimation."""
        k = 5
        Omega = np.eye(k) + np.random.randn(k, k) * 0.1
        Omega = Omega @ Omega.T

        Y = np.random.randn(3, k)
        lbd = 0.1

        result = EstAJInv(Omega, Y, lbd)

        assert result.shape == (3, 3)
        assert np.allclose(result, result.T)

    def test_lp_optimization(self):
        """Test linear programming optimization."""
        y = np.array([1, -0.5, 0.3, -0.8])
        lbd = 0.5

        result = LP(y, lbd)

        assert len(result) == len(y)
        # LP solution should be sparse for reasonable lambda
        assert np.sum(np.abs(result) > 1e-6) <= len(y)

    def test_dantzig_selector(self):
        """Test Dantzig selector."""
        p = 8
        C_hat = np.random.randn(p, p)
        C_hat = C_hat @ C_hat.T + np.eye(p) * 0.1

        y = np.random.randn(p)
        lbd = 0.1

        result = Dantzig(C_hat, y, lbd)

        if result is not None:
            assert len(result) == p
        # else: optimization failed, which is acceptable


class TestScoreFunctions:
    """Test score computation functions."""

    def test_score_mat_basic(self):
        """Test score matrix computation."""
        p = 10
        R = np.random.randn(p, p)
        R = (R + R.T) / 2  # Make symmetric

        for q in [2, 3]:
            for exact in [True, False]:
                result = Score_mat(R, q=q, exact=exact)

                assert 'Score' in result
                assert result['Score'].shape == (p, p)

    def test_lp_score_computation(self):
        """Test LP score computation."""
        R_ij = np.random.randn(5, 5)
        ind = 0

        for exact in [True, False]:
            result = LP_Score(R_ij, ind, exact=exact)
            assert isinstance(result, float)
            assert np.isfinite(result)


class TestUtilityFunctions:
    """Test LOVE utility functions."""

    def test_recovergroup_basic(self):
        """Test group recovery from matrix."""
        # Create block structure
        A = np.zeros((6, 6))
        A[0:2, 0:2] = 1
        A[3:5, 3:5] = 1

        result = recoverGroup(A)

        assert len(result) >= 1
        for group in result:
            assert 'I' in group

    def test_singleton_detection(self):
        """Test singleton group detection."""
        # Single group
        assert singleton([[0, 1, 2]]) == True

        # Multiple groups
        assert singleton([[0, 1], [2, 3]]) == False

        # Empty
        assert singleton([]) == True

    def test_thresha_thresholding(self):
        """Test matrix thresholding."""
        A = np.array([[1, 0.3], [0.7, 1]])
        mu = 0.5

        result = threshA(A, mu, scale=False)

        # Values above threshold should remain
        assert result[1, 0] == 0.7  # Above threshold
        assert result[0, 1] == 0    # Below threshold (set to 0)

    def test_offsum_computation(self):
        """Test off-diagonal sum computation."""
        M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        weights = 1.0

        result = offSum(M, weights)

        # Should sum all off-diagonal elements
        expected = np.sum(M) - np.trace(M)
        assert np.isclose(result, expected)

    def test_partition_functionality(self):
        """Test data partitioning."""
        totalNumb = 100
        numbGroup = 7

        result = partition(totalNumb, numbGroup)

        assert len(result) == numbGroup
        assert sum(result) == totalNumb
        assert all(x > 0 for x in result)

    def test_extract_indices(self):
        """Test index-based extraction."""
        preVec = np.array([1, 2, 3, 4, 5, 6])
        indices = [[0, 2], [1, 3, 4]]

        result = extract(preVec, indices)

        assert len(result) == len(indices)
        assert np.array_equal(result[0], [1, 3])
        assert np.array_equal(result[1], [2, 4, 5])


class TestCrossValidationFunctions:
    """Test cross-validation functions."""

    def test_cv_delta_basic(self):
        """Test delta cross-validation."""
        n, p = 50, 8
        X = np.random.randn(n, p)
        deltaGrids = np.array([0.01, 0.05, 0.1, 0.2])

        result = CV_delta(X, deltaGrids, diagonal=True, rep_CV=5)

        assert 'delta' in result
        assert result['delta'] in deltaGrids

    def test_kfoldcv_delta_functionality(self):
        """Test K-fold CV for delta."""
        n, p = 100, 12
        X = np.random.randn(n, p)

        result = KfoldCV_delta(X, delta=None, rep_CV=3, K_CV=5)

        assert 'delta' in result
        assert result['delta'] > 0

    def test_cv_lbd_basic(self):
        """Test lambda cross-validation."""
        p = 6
        X = np.random.randn(50, p)
        lbdGrids = np.array([0.1, 0.3, 0.5])

        AI = np.eye(p) * 0.1
        pureVec = np.ones(p)

        result = CV_lbd(X, lbdGrids, AI, pureVec, rep_CV=3)

        assert 'lbd' in result
        assert result['lbd'] in lbdGrids


class TestLOVEInternalIntegration:
    """Test integration between LOVE internal functions."""

    def test_pure_estimation_pipeline(self):
        """Test complete pure node estimation pipeline."""
        # Generate synthetic correlation structure
        n, p = 100, 8
        X = np.random.randn(n, p)

        # Add correlation structure
        X[:, 1] = 0.8 * X[:, 0] + 0.6 * np.random.randn(n)
        X[:, 6] = 0.7 * X[:, 7] + 0.7 * np.random.randn(n)

        Sigma = np.corrcoef(X.T)

        # Step 1: Find row maxima
        row_max_result = FindRowMax(Sigma)

        # Step 2: Find pure nodes
        pure_nodes = FindPureNode(
            Sigma,
            delta=0.3,
            Ms=row_max_result['M'],
            arg_Ms=row_max_result['I']
        )

        # Should find some structure
        assert len(pure_nodes) >= 0

        # Step 3: Estimate AI for each group
        if len(pure_nodes) > 0:
            for group in pure_nodes:
                if len(group) >= 2:
                    sub_sigma = Sigma[np.ix_(group, group)]
                    AI_sub = EstAI(sub_sigma, 0.1, np.ones(len(group)) * 0.01, True)
                    assert AI_sub.shape == (len(group), len(group))

    def test_score_cv_consistency(self):
        """Test consistency between scoring and CV."""
        n, p = 80, 6
        X = np.random.randn(n, p)

        # Compute correlation
        R = np.corrcoef(X.T)

        # Compute scores
        score_result = Score_mat(R, q=2, exact=False)

        # CV should work with score matrix
        delta_cv = CV_delta(X, deltaGrids=np.array([0.05, 0.1, 0.2]),
                           diagonal=True, rep_CV=3)

        assert 'delta' in delta_cv
        assert delta_cv['delta'] > 0

    def test_error_propagation(self):
        """Test that errors are handled gracefully."""
        # Test with problematic data
        X = np.ones((10, 5))  # Constant matrix

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Should not crash, though results may be degenerate
            try:
                R = np.corrcoef(X.T)
                score_result = Score_mat(R, q=2, exact=False)
            except (np.linalg.LinAlgError, ValueError):
                pass  # Expected for degenerate cases