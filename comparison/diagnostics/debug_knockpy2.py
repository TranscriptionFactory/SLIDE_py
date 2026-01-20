#!/usr/bin/env python3
"""Debug knockpy offset parameter."""
import numpy as np
from knockpy import KnockoffFilter
import knockpy.knockoff_filter as kf_module
import inspect

# Check if there's an offset parameter anywhere
print("=== Checking KnockoffFilter for offset/knockoff+ options ===")
print()

# Check forward method signature
sig = inspect.signature(KnockoffFilter.forward)
print("forward() parameters:")
for name, param in sig.parameters.items():
    print(f"  {name}: {param.default if param.default != inspect.Parameter.empty else 'required'}")
print()

# Check if there's a make_selections or threshold method
print("KnockoffFilter attributes/methods:")
for attr in dir(KnockoffFilter):
    if not attr.startswith('_'):
        print(f"  {attr}")
print()

# Try to find offset in the source
print("=== Testing with offset parameter if available ===")
np.random.seed(42)
n, p = 200, 20
X = np.random.randn(n, p)
beta = np.zeros(p)
beta[0] = 3.0
beta[1] = 2.5
beta[2] = 2.0
y = X @ beta + 0.5 * np.random.randn(n)

# Check knockpy.knockoffs module for threshold calculation
try:
    from knockpy import utilities
    print("knockpy.utilities functions:")
    for attr in dir(utilities):
        if not attr.startswith('_'):
            print(f"  {attr}")
except ImportError:
    print("No utilities module")
print()

# Try to look at the make_selections method
kfilter = KnockoffFilter(ksampler='gaussian', fstat='lasso')
rejections = kfilter.forward(X=X, y=y, fdr=0.2)

print("After forward(), kfilter attributes:")
print(f"  W: {kfilter.W[:5]}...")
print(f"  threshold: {kfilter.threshold}")

# Check if we can manually compute threshold with offset=0
print()
print("=== Manual threshold calculation ===")
W = kfilter.W

def compute_threshold(W, fdr, offset=1):
    """Compute knockoff threshold."""
    W_sorted = np.sort(np.abs(W))[::-1]
    for t in W_sorted:
        if t <= 0:
            continue
        numerator = offset + np.sum(W <= -t)
        denominator = max(1, np.sum(W >= t))
        if numerator / denominator <= fdr:
            return t
    return np.inf

print(f"Threshold with offset=1 (knockoff+): {compute_threshold(W, 0.2, offset=1)}")
print(f"Threshold with offset=0 (original): {compute_threshold(W, 0.2, offset=0)}")
print()

# What features would be selected with offset=0?
t0 = compute_threshold(W, 0.2, offset=0)
if t0 < np.inf:
    selected = np.where(W >= t0)[0]
    print(f"Features selected with offset=0: {selected}")
else:
    print("Still no features selected with offset=0")
