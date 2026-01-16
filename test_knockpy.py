#!/usr/bin/env python3
"""Test knockpy integration with strong signal."""
import sys
sys.path.insert(0, '/ix/djishnu/Aaron/1_general_use/SLIDE_py/src')

import numpy as np
from knockpy import KnockoffFilter
from loveslide.knockoffs import Knockoffs

print('knockpy imported successfully')

# Create data with STRONG signal in first 3 features
np.random.seed(42)
n, p = 200, 20
X = np.random.randn(n, p)

# Strong coefficients for first 3 features
beta = np.zeros(p)
beta[0] = 3.0
beta[1] = 2.5
beta[2] = 2.0

y = X @ beta + 0.5 * np.random.randn(n)

print(f'Data shape: X={X.shape}, y={y.shape}')
print(f'True non-zero features: [0, 1, 2]')
print()

# Test 1: Wrapper with offset=0 (original knockoff - more power)
print('=== Wrapper test (niter=1, offset=0) ===')
np.random.seed(42)
selected = Knockoffs.filter_knockoffs_iterative_knockpy(X, y, fdr=0.2, niter=1, spec=0.2, method='mvr', offset=0)
print(f'Selected features: {list(selected)}')
print()

# Test 2: Wrapper with offset=1 (knockoff+ - conservative)
print('=== Wrapper test (niter=1, offset=1) ===')
np.random.seed(42)
selected = Knockoffs.filter_knockoffs_iterative_knockpy(X, y, fdr=0.2, niter=1, spec=0.2, method='mvr', offset=1)
print(f'Selected features: {list(selected)}')
print()

# Test 3: Multiple iterations with offset=0
print('=== Wrapper test (niter=5, offset=0) ===')
np.random.seed(42)
selected = Knockoffs.filter_knockoffs_iterative_knockpy(X, y, fdr=0.2, niter=5, spec=0.2, method='mvr', offset=0)
print(f'Selected features: {list(selected)}')
print()

# Test 4: Via dispatcher with offset=0
print('=== Dispatcher test (backend=knockpy, offset=0) ===')
np.random.seed(42)
selected = Knockoffs.filter_knockoffs_iterative(X, y, fdr=0.2, niter=5, spec=0.2, backend='knockpy', method='mvr', offset=0)
print(f'Selected features: {list(selected)}')
print()

# Test 5: Compare with R backend if available
print('=== R backend comparison ===')
try:
    np.random.seed(42)
    selected_r = Knockoffs.filter_knockoffs_iterative(X, y, fdr=0.2, niter=5, spec=0.2, backend='r')
    print(f'R backend selected: {list(selected_r)}')
except Exception as e:
    print(f'R backend not available: {e}')

print()
print('All tests completed!')
