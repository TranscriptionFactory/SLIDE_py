#!/usr/bin/env python3
"""
Python script to run SLIDE on input data and compare with R outputs.

Usage:
    python run_slide_py.py <yaml_path> [out_path] [--love-backend python|r] [--knockoff-backend python|r]

Arguments:
    yaml_path             Path to YAML config file
    out_path              Optional output path override
    --love-backend        Which LOVE implementation: 'python' (default) or 'r'
    --knockoff-backend    Which knockoff implementation: 'r' (default) or 'python'

Examples:
    python run_slide_py.py config.yaml
    python run_slide_py.py config.yaml /path/to/outputs --love-backend r --knockoff-backend python
"""

import argparse
import os
import yaml
import logging
import time

from loveslide import OptimizeSLIDE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Run Python SLIDE implementation')
    parser.add_argument('yaml_path', help='Path to YAML config file')
    parser.add_argument('out_path', nargs='?', default=None, help='Output path override')
    parser.add_argument('--love-backend', dest='love_backend', choices=['python', 'r'],
                        default='python', help='LOVE implementation: python (default) or r')
    parser.add_argument('--knockoff-backend', dest='knockoff_backend', choices=['python', 'r'],
                        default='r', help='Knockoff implementation: r (default) or python')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load parameters from YAML
    with open(args.yaml_path, 'r') as f:
        params = yaml.safe_load(f)

    # Override out_path if provided as argument
    out_path = args.out_path if args.out_path else params.get('out_path')
    params['out_path'] = out_path

    # Set backends from CLI arguments
    love_backend = args.love_backend
    knockoff_backend = args.knockoff_backend

    print("=" * 60)
    print(f"SLIDE Python Analysis")
    print(f"  LOVE backend: {love_backend}")
    print(f"  Knockoff backend: {knockoff_backend}")
    print("=" * 60)
    print(f"YAML config: {args.yaml_path}")
    print(f"Output path: {out_path}")
    print(f"X path: {params.get('x_path')}")
    print(f"Y path: {params.get('y_path')}")
    print(f"Delta: {params.get('delta')}")
    print(f"Lambda: {params.get('lambda')}")
    print("=" * 60)

    # Create output directory
    os.makedirs(out_path, exist_ok=True)

    # Save params to new yaml in output directory
    new_yaml_path = os.path.join(out_path, "params.yaml")
    with open(new_yaml_path, 'w') as f:
        yaml.dump(params, f)

    # Build input_params from YAML config
    input_params = {
        'x_path': params.get('x_path'),
        'y_path': params.get('y_path'),
        'y_factor': params.get('y_factor', True),
        'niter': params.get('SLIDE_iter', 1000),
        'SLIDE_top_feats': params.get('SLIDE_top_feats', 10),
        'rep_CV': params.get('sampleCV_iter', 500),
        'out_path': out_path,
        'fdr': params.get('fdr', 0.1),
        'thresh_fdr': params.get('thresh_fdr', 0.1),
        'pure_homo': params.get('pure_homo', True),
        'do_interacts': params.get('do_interacts', True),
        'n_workers': params.get('n_workers', 2),
        'spec': params.get('spec', 0.1),
        'love_backend': love_backend,
        'knockoff_backend': knockoff_backend,
        # Handle delta/lambda - can be single value or list
        'delta': params.get('delta') if isinstance(params.get('delta'), list) else [params.get('delta', 0.1)],
        'lambda': params.get('lambda') if isinstance(params.get('lambda'), list) else [params.get('lambda', 0.5)],
    }

    print(f"\nInput params:")
    for k, v in input_params.items():
        print(f"  {k}: {v}")
    print()

    # Run SLIDE
    t_start = time.time()

    try:
        slider = OptimizeSLIDE(input_params)
        slider.run_pipeline(verbose=True)
    except Exception as e:
        logger.error(f"Error running SLIDE: {e}")
        import traceback
        traceback.print_exc()

    t_end = time.time()

    print()
    print("=" * 60)
    print(f"Python SLIDE (LOVE={love_backend}, Knockoff={knockoff_backend}) completed in {t_end - t_start:.2f} seconds")
    print(f"Outputs saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
