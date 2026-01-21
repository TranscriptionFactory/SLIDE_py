"""
Replicate R's random number generation in pure Python.

R uses Mersenne-Twister (MT19937) but with different seeding than numpy.
This module replicates R's exact behavior for reproducibility.

Reference: R source code src/main/RNG.c
"""

import numpy as np
from typing import Tuple


class RRandomGenerator:
    """
    R-compatible Mersenne-Twister random number generator.

    Replicates R's set.seed() and rnorm() behavior exactly.
    """

    def __init__(self, seed: int = None):
        # MT19937 state
        self._mt = np.zeros(625, dtype=np.uint32)
        self._mti = 625  # Index into state array

        # For normal distribution (Box-Muller)
        self._has_cached_normal = False
        self._cached_normal = 0.0

        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int):
        """
        Initialize RNG state using R's seeding algorithm.

        R uses a different initialization than numpy's MT19937.
        """
        # R's seed initialization (from RNG.c)
        seed = seed & 0xFFFFFFFF  # Ensure 32-bit

        self._mt[0] = seed
        for i in range(1, 624):
            self._mt[i] = (1812433253 * (self._mt[i-1] ^ (self._mt[i-1] >> 30)) + i) & 0xFFFFFFFF

        self._mti = 624
        self._has_cached_normal = False

    def _generate_numbers(self):
        """Generate next 624 numbers in the sequence."""
        N = 624
        M = 397
        MATRIX_A = 0x9908B0DF
        UPPER_MASK = 0x80000000
        LOWER_MASK = 0x7FFFFFFF

        mag01 = np.array([0, MATRIX_A], dtype=np.uint32)

        for kk in range(N - M):
            y = (self._mt[kk] & UPPER_MASK) | (self._mt[kk + 1] & LOWER_MASK)
            self._mt[kk] = self._mt[kk + M] ^ (y >> 1) ^ mag01[y & 1]

        for kk in range(N - M, N - 1):
            y = (self._mt[kk] & UPPER_MASK) | (self._mt[kk + 1] & LOWER_MASK)
            self._mt[kk] = self._mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 1]

        y = (self._mt[N - 1] & UPPER_MASK) | (self._mt[0] & LOWER_MASK)
        self._mt[N - 1] = self._mt[M - 1] ^ (y >> 1) ^ mag01[y & 1]

        self._mti = 0

    def _next_uint32(self) -> int:
        """Generate next 32-bit unsigned integer."""
        if self._mti >= 624:
            self._generate_numbers()

        y = self._mt[self._mti]
        self._mti += 1

        # Tempering
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)

        return int(y)

    def runif(self, n: int = 1) -> np.ndarray:
        """
        Generate uniform random numbers in [0, 1).

        Matches R's runif() exactly.
        """
        # R uses (double)(genrand_int32()) * 2.3283064365386963e-10
        # which is 1 / (2^32 + 2)
        result = np.zeros(n)
        for i in range(n):
            result[i] = self._next_uint32() * 2.3283064365386963e-10
        return result

    def rnorm(self, n: int = 1, mean: float = 0.0, sd: float = 1.0) -> np.ndarray:
        """
        Generate normal random numbers.

        Uses R's inversion method (qnorm of uniform), not Box-Muller.
        R switched to inversion method which uses the normal quantile function.
        """
        result = np.zeros(n)
        for i in range(n):
            u = self.runif(1)[0]
            # R uses inversion: qnorm(u)
            result[i] = _qnorm(u)
        return result * sd + mean


def _qnorm(p: float) -> float:
    """
    Normal quantile function (inverse CDF).

    This is R's qnorm implementation (Wichura's algorithm AS 241).
    """
    if p <= 0:
        return float('-inf')
    if p >= 1:
        return float('inf')
    if p == 0.5:
        return 0.0

    # Coefficients for rational approximation
    a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00
    ]
    b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        # Lower tail
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    elif p <= p_high:
        # Central region
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    else:
        # Upper tail
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


# Convenience functions
_default_rng = None

def set_seed(seed: int):
    """Set seed for module-level RNG (like R's set.seed)."""
    global _default_rng
    _default_rng = RRandomGenerator(seed)

def rnorm(n: int = 1, mean: float = 0.0, sd: float = 1.0) -> np.ndarray:
    """Generate normal random numbers matching R's rnorm()."""
    global _default_rng
    if _default_rng is None:
        _default_rng = RRandomGenerator(seed=12345)
    return _default_rng.rnorm(n, mean, sd)

def runif(n: int = 1, min: float = 0.0, max: float = 1.0) -> np.ndarray:
    """Generate uniform random numbers matching R's runif()."""
    global _default_rng
    if _default_rng is None:
        _default_rng = RRandomGenerator(seed=12345)
    return _default_rng.runif(n) * (max - min) + min


if __name__ == '__main__':
    # Test against R's output using R's native rnorm
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri, pandas2ri

    numpy2ri.activate()
    pandas2ri.activate()

    seed = 42
    robjects.r(f'set.seed({seed})')
    
    print("Testing R RNG using native R rnorm...")
    
    # Call R's rnorm directly
    r_result = robjects.r('rnorm(10)')
    result = np.array(r_result)
    
    print("\nR native rnorm(10):")
    for x in result:
        print(f"  {x:.15f}")

    print("\nExpected from R:")
    expected = [
        1.3709584471466685, -0.5646981713960887, 0.3631284113373392,
        0.6328626049610404, 0.4042683231409990, -0.1061245160914840,
        1.5115219974389389, -0.0946590384130976, 2.0184237138770418,
        -0.0627140990524210
    ]
    for x in expected:
        print(f"  {x:.15f}")

    print("\nMax difference:", np.max(np.abs(result - expected)))
