#!/usr/bin/env python3
"""Check make_selections method for offset parameter."""
import numpy as np
from knockpy import KnockoffFilter
import inspect

# Check make_selections signature
print("=== make_selections signature ===")
sig = inspect.signature(KnockoffFilter.make_selections)
print(f"Signature: {sig}")
print()
for name, param in sig.parameters.items():
    print(f"  {name}: default={param.default if param.default != inspect.Parameter.empty else 'required'}")
print()

# Check docstring
print("=== make_selections docstring ===")
print(KnockoffFilter.make_selections.__doc__)
print()

# Test data
np.random.seed(42)
n, p = 200, 20
X = np.random.randn(n, p)
beta = np.zeros(p)
beta[0] = 3.0
beta[1] = 2.5
beta[2] = 2.0
y = X @ beta + 0.5 * np.random.randn(n)

# Create filter and compute W statistics
kfilter = KnockoffFilter(ksampler='gaussian', fstat='lasso')

# First run forward to get knockoffs and W statistics
_ = kfilter.forward(X=X, y=y, fdr=0.2)
print(f"W statistics: {kfilter.W}")
print(f"Current threshold: {kfilter.threshold}")
print()

# Try calling make_selections with offset=0
print("=== Testing make_selections with offset parameter ===")
try:
    # Try with offset=0
    selections = kfilter.make_selections(kfilter.W, fdr=0.2, offset=0)
    print(f"With offset=0: {np.where(selections)[0]}")
except TypeError as e:
    print(f"offset parameter not supported: {e}")

# Try with different fdr values
print()
print("=== Testing different FDR values (default offset=1) ===")
for fdr in [0.1, 0.2, 0.3, 0.4, 0.5]:
    selections = kfilter.make_selections(kfilter.W, fdr=fdr)
    selected = np.where(selections)[0]
    print(f"FDR={fdr}: selected={list(selected)}")
