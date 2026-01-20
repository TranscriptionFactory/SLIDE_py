#!/usr/bin/env python3
"""Debug knockpy to understand why it returns no selections."""
import numpy as np
from knockpy import KnockoffFilter

# Create data with STRONG signal
np.random.seed(42)
n, p = 200, 20
X = np.random.randn(n, p)

beta = np.zeros(p)
beta[0] = 3.0
beta[1] = 2.5
beta[2] = 2.0

y = X @ beta + 0.5 * np.random.randn(n)

print('Signal: beta coefficients [3.0, 2.5, 2.0] for features [0, 1, 2]')
print('Noise std: 0.5')
print()

# Test with different methods and shrinkage options
kfilter = KnockoffFilter(ksampler='gaussian', fstat='lasso')

print('=== Testing knockpy with different settings ===')
print()

# Test 1: Default shrinkage (ledoitwolf)
print('1. Default shrinkage (ledoitwolf):')
rejections = kfilter.forward(X=X, y=y, fdr=0.2)
print('   rejections type:', type(rejections))
print('   rejections:', rejections)
print('   selected:', np.where(rejections)[0])
print()

# Test 2: Check W statistics
print('2. W statistics (feature importance scores):')
print('   W:', kfilter.W)
print('   Top 5 W scores (indices):', np.argsort(kfilter.W)[-5:][::-1])
print('   W for features 0,1,2:', kfilter.W[0], kfilter.W[1], kfilter.W[2])
print()

# Test 3: Check the knockoff threshold
print('3. Knockoff threshold:')
print('   threshold:', kfilter.threshold)
print()

# Test 4: What if we use a looser FDR?
print('4. Try with very loose FDR (0.5):')
kfilter2 = KnockoffFilter(ksampler='gaussian', fstat='lasso')
rejections2 = kfilter2.forward(X=X, y=y, fdr=0.5)
print('   selected:', np.where(rejections2)[0])
print('   W:', kfilter2.W)
print('   threshold:', kfilter2.threshold)
print()

# Test 5: Try with equicorrelated knockoffs
print('5. With equicorrelated knockoffs, FDR=0.2:')
kfilter3 = KnockoffFilter(ksampler='gaussian', fstat='lasso', knockoff_kwargs={'method': 'equicorrelated'})
rejections3 = kfilter3.forward(X=X, y=y, fdr=0.2)
print('   selected:', np.where(rejections3)[0])
print('   W:', kfilter3.W)
print('   threshold:', kfilter3.threshold)
print()

# Test 6: Stronger signal?
print('6. Try with MUCH stronger signal (beta = [10, 8, 6]):')
np.random.seed(42)
beta_strong = np.zeros(p)
beta_strong[0] = 10.0
beta_strong[1] = 8.0
beta_strong[2] = 6.0
y_strong = X @ beta_strong + 0.5 * np.random.randn(n)

kfilter4 = KnockoffFilter(ksampler='gaussian', fstat='lasso')
rejections4 = kfilter4.forward(X=X, y=y_strong, fdr=0.2)
print('   selected:', np.where(rejections4)[0])
print('   W:', kfilter4.W)
print('   threshold:', kfilter4.threshold)
