#!/usr/bin/env python3
"""Benchmark script for parallel knockoff voting.

Compares sequential vs parallel execution times and validates
that results are equivalent.

Usage:
    python scripts/benchmark_parallel_voting.py
    python scripts/benchmark_parallel_voting.py --n 200 --p 100 --niter 100
"""

import argparse
import time
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from loveslide.knockoff.filter import knockoff_filter_voting
from loveslide.knockoff._parallel import (
    knockoff_voting_parallel_joblib,
    knockoff_voting_parallel_futures,
    _precompute_knockoff_params,
)


def generate_test_data(n: int, p: int, k: int = 5, seed: int = 42):
    """Generate test data with k true signals."""
    np.random.seed(seed)

    # Generate correlated features
    rho = 0.5
    Sigma = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T

    # Generate response with k true signals
    beta = np.zeros(p)
    beta[:k] = np.random.choice([-1, 1], k) * np.random.uniform(2, 4, k)
    y = X @ beta + np.random.randn(n)

    return X, y, beta


def benchmark_sequential(X, y, niter, spec, base_seed):
    """Benchmark sequential execution."""
    start = time.time()
    result = knockoff_filter_voting(
        X, y,
        niter=niter,
        spec=spec,
        n_jobs=1,
        base_seed=base_seed,
        verbose=False
    )
    elapsed = time.time() - start
    return result, elapsed


def benchmark_parallel_joblib(X, y, niter, spec, base_seed, n_jobs):
    """Benchmark joblib parallel execution."""
    start = time.time()
    result = knockoff_voting_parallel_joblib(
        X, y,
        niter=niter,
        spec=spec,
        n_jobs=n_jobs,
        base_seed=base_seed,
        verbose=False
    )
    elapsed = time.time() - start
    return result, elapsed


def benchmark_parallel_futures(X, y, niter, spec, base_seed, n_jobs):
    """Benchmark concurrent.futures parallel execution."""
    start = time.time()
    result = knockoff_voting_parallel_futures(
        X, y,
        niter=niter,
        spec=spec,
        n_jobs=n_jobs,
        base_seed=base_seed,
        verbose=False
    )
    elapsed = time.time() - start
    return result, elapsed


def benchmark_precompute_only(X):
    """Benchmark just the precomputation step."""
    start = time.time()
    mu, Sigma, diag_s = _precompute_knockoff_params(X)
    elapsed = time.time() - start
    return elapsed, np.max(diag_s)


def main():
    parser = argparse.ArgumentParser(description='Benchmark parallel knockoff voting')
    parser.add_argument('--n', type=int, default=150, help='Number of samples')
    parser.add_argument('--p', type=int, default=80, help='Number of features')
    parser.add_argument('--k', type=int, default=5, help='Number of true signals')
    parser.add_argument('--niter', type=int, default=50, help='Number of knockoff iterations')
    parser.add_argument('--spec', type=float, default=0.1, help='Specificity threshold')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip-sequential', action='store_true', help='Skip sequential benchmark')
    args = parser.parse_args()

    print("=" * 60)
    print("KNOCKOFF VOTING PARALLELIZATION BENCHMARK")
    print("=" * 60)
    print(f"Data dimensions: n={args.n}, p={args.p}")
    print(f"True signals: k={args.k}")
    print(f"Iterations: niter={args.niter}")
    print(f"Specificity: spec={args.spec}")
    print(f"Parallel jobs: n_jobs={args.n_jobs}")
    print()

    # Generate test data
    print("Generating test data...")
    X, y, beta = generate_test_data(args.n, args.p, args.k, args.seed)
    true_signals = set(np.where(beta != 0)[0])
    print(f"True signal indices: {sorted(true_signals)}")
    print()

    # Benchmark precomputation
    print("-" * 60)
    print("PRECOMPUTATION (SDP optimization - done ONCE)")
    print("-" * 60)
    precompute_time, max_diag_s = benchmark_precompute_only(X)
    print(f"Precompute time: {precompute_time:.3f}s")
    print(f"Max diag_s: {max_diag_s:.4f}")
    print()

    results = {}

    # Sequential benchmark
    if not args.skip_sequential:
        print("-" * 60)
        print("SEQUENTIAL EXECUTION (n_jobs=1)")
        print("-" * 60)
        result_seq, time_seq = benchmark_sequential(
            X, y, args.niter, args.spec, args.seed
        )
        results['sequential'] = (result_seq, time_seq)
        print(f"Time: {time_seq:.2f}s")
        print(f"Selected: {result_seq.selected.tolist()}")
        print(f"True positives: {len(set(result_seq.selected) & true_signals)}/{args.k}")
        print()

    # Joblib parallel benchmark
    print("-" * 60)
    print(f"PARALLEL EXECUTION - JOBLIB (n_jobs={args.n_jobs})")
    print("-" * 60)
    try:
        result_joblib, time_joblib = benchmark_parallel_joblib(
            X, y, args.niter, args.spec, args.seed, args.n_jobs
        )
        results['joblib'] = (result_joblib, time_joblib)
        print(f"Time: {time_joblib:.2f}s")
        print(f"Selected: {result_joblib.selected.tolist()}")
        print(f"True positives: {len(set(result_joblib.selected) & true_signals)}/{args.k}")
        if 'sequential' in results:
            speedup = results['sequential'][1] / time_joblib
            print(f"Speedup vs sequential: {speedup:.2f}x")
    except ImportError as e:
        print(f"Skipped (joblib not available): {e}")
    print()

    # concurrent.futures parallel benchmark
    print("-" * 60)
    print(f"PARALLEL EXECUTION - FUTURES (n_jobs={args.n_jobs})")
    print("-" * 60)
    result_futures, time_futures = benchmark_parallel_futures(
        X, y, args.niter, args.spec, args.seed, args.n_jobs
    )
    results['futures'] = (result_futures, time_futures)
    print(f"Time: {time_futures:.2f}s")
    print(f"Selected: {result_futures.selected.tolist()}")
    print(f"True positives: {len(set(result_futures.selected) & true_signals)}/{args.k}")
    if 'sequential' in results:
        speedup = results['sequential'][1] / time_futures
        print(f"Speedup vs sequential: {speedup:.2f}x")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Timing comparison
    print("\nTiming:")
    for name, (_, elapsed) in results.items():
        print(f"  {name:15s}: {elapsed:7.2f}s")

    # Result comparison (if we have multiple results)
    if len(results) > 1:
        print("\nResult consistency:")
        names = list(results.keys())
        base_result = results[names[0]][0]
        base_counts = base_result.selection_counts

        for name in names[1:]:
            other_counts = results[name][0].selection_counts
            correlation = np.corrcoef(base_counts, other_counts)[0, 1]
            max_diff = np.max(np.abs(base_counts - other_counts))
            print(f"  {names[0]} vs {name}:")
            print(f"    Correlation: {correlation:.4f}")
            print(f"    Max count diff: {max_diff}")

    # Speedup analysis
    if 'sequential' in results:
        print("\nSpeedup analysis:")
        seq_time = results['sequential'][1]
        for name, (_, elapsed) in results.items():
            if name != 'sequential':
                speedup = seq_time / elapsed
                print(f"  {name}: {speedup:.2f}x faster")

        # Theoretical maximum speedup
        import multiprocessing
        n_cores = args.n_jobs if args.n_jobs > 0 else multiprocessing.cpu_count()
        theoretical_max = min(n_cores, args.niter)
        print(f"\nTheoretical max speedup ({n_cores} cores): {theoretical_max:.1f}x")

    print()


if __name__ == '__main__':
    main()
