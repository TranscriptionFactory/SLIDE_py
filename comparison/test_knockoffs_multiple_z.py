#!/usr/bin/env python3
"""
Test all knockoff backends on multiple Z matrices from SSc_binary dataset.

Usage:
    python test_knockoffs_ssc.py


    cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison && \
LOVE_DIR="/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/archive/output_comparison/2026-01-16_10-41-07/SSc_binary_comparison/R_native" && \
OUTPUT_DIR="knockoff_eval_ssc_binary" && \
mkdir -p $OUTPUT_DIR && \
for backend in R_native R_knockoffs_py_sklearn R_knockoffs_py_stats knockoff_filter_sklearn knockoff_filter custom_glmnet; do
    echo "=== $backend ===" && \
    python run_knockoffs_on_precomputed.py run \
        --love-dir "$LOVE_DIR" \
        --backend $backend \
        --output-dir "$OUTPUT_DIR" \
        --fdr 0.1 2>&1 | grep -E "(param|selected|SUMMARY|INFO.*d[0-9])"
    echo ""
done
"""

import subprocess
import sys
from pathlib import Path

# Configuration
LOVE_DIR = "/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/archive/output_comparison/2026-01-16_10-41-07/SSc_binary_comparison/R_native"
OUTPUT_DIR = "knockoff_eval_ssc_binary"
SCRIPT_PATH = "/ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison/run_knockoffs_on_precomputed.py"

BACKENDS = [
    "R_native",
    "R_knockoffs_py_sklearn",
    "R_knockoffs_py_stats",
    "knockoff_filter_sklearn",
    "knockoff_filter",
    "custom_glmnet",
]

def run_backend(backend: str):
    """Run a single knockoff backend."""
    cmd = [
        sys.executable,
        SCRIPT_PATH,
        "run",
        "--love-dir", LOVE_DIR,
        "--backend", backend,
        "--output-dir", OUTPUT_DIR,
        "--fdr", "0.1",
    ]
    print(f"\n{'='*60}")
    print(f"Running: {backend}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

def compare_all():
    """Compare all backend results."""
    result_files = list(Path(OUTPUT_DIR).glob("knockoff_results_*.json"))
    if not result_files:
        print("No result files found!")
        return
    
    cmd = [
        sys.executable,
        SCRIPT_PATH,
        "compare",
        *[str(f) for f in sorted(result_files)],
        "-o", f"{OUTPUT_DIR}/all_backends_ssc_comparison.txt",
    ]
    print(f"\n{'='*60}")
    print("Comparing all backends")
    print(f"{'='*60}")
    subprocess.run(cmd)

def main():
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Run all backends
    for backend in BACKENDS:
        returncode = run_backend(backend)
        if returncode != 0:
            print(f"Warning: {backend} failed with code {returncode}")
    
    # Compare results
    compare_all()
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()