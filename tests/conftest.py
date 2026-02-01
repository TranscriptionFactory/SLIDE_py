"""
Pytest configuration and fixtures for SLIDE_py tests.

This module provides shared fixtures and configuration for the test suite.
"""

import sys
from pathlib import Path

import pytest
import numpy as np


# Add src to path for all tests
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session")
def random_seed():
    """Provide consistent random seed for reproducibility."""
    return 42


@pytest.fixture
def set_random_seed(random_seed):
    """Reset numpy random state before each test."""
    np.random.seed(random_seed)
    yield
    # No cleanup needed


@pytest.fixture
def simple_regression_data():
    """
    Generate simple regression data with known true signals.

    Returns (X, y, beta, true_indices)
    """
    np.random.seed(42)
    n, p = 200, 50

    # Generate features
    X = np.random.randn(n, p)
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    # Create sparse true coefficient vector
    beta = np.zeros(p)
    true_indices = np.array([0, 1, 2, 3, 4])
    beta[true_indices] = 3.0

    # Generate response
    y = X @ beta + np.random.randn(n) * 0.5

    return X, y, beta, true_indices


@pytest.fixture
def correlated_features():
    """
    Generate features with known correlation structure.

    Useful for testing covariance estimation.
    """
    np.random.seed(123)
    n, p = 150, 30

    # Create base features
    base = np.random.randn(n, 10)

    # Create correlated groups
    group1 = base[:, :3] + np.random.randn(n, 3) * 0.3
    group2 = base[:, 3:6] + np.random.randn(n, 3) * 0.3
    group3 = base[:, 6:10] + np.random.randn(n, 4) * 0.3

    # Independent features
    independent = np.random.randn(n, p - 10)

    X = np.column_stack([group1, group2, group3, independent])
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    return X


@pytest.fixture
def ill_conditioned_data():
    """
    Generate ill-conditioned data where n ~ p.

    Tests regularization and shrinkage paths.
    """
    np.random.seed(456)
    n, p = 100, 90

    # Create matrix with specific condition number
    U, _ = np.linalg.qr(np.random.randn(n, min(n, p)))
    V, _ = np.linalg.qr(np.random.randn(p, min(n, p)))

    k = min(n, p)
    singular_values = np.logspace(0, -4, k)

    X = U[:, :k] @ np.diag(singular_values) @ V[:, :k].T
    X += np.random.randn(n, p) * 1e-10

    # Standardize
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    # Create response with signal
    beta = np.zeros(p)
    beta[:3] = 2.0
    y = X @ beta + np.random.randn(n) * 0.5

    return X, y, beta
