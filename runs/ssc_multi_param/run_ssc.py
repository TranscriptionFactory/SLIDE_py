"""
SSc multi-delta/lambda SLIDE run across python, r_knockoffs, and r backends.

delta = [0.01, 0.05, 0.1, 0.2]
lambda = [0.1, 1.0]
backends = python, r_knockoffs, r
"""
import sys
import os
import warnings
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Ensure the installed package is used
from loveslide import OptimizeSLIDE

SSC_X = "/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/X.csv"
SSC_Y = "/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/Y.csv"
OUT_BASE = "/ix/djishnu/Aaron/1_general_use/SLIDE_py/runs/ssc_multi_param/output"

DELTAS = [0.01, 0.05, 0.1, 0.2]
LAMBDAS = [0.1, 1.0]
BACKENDS = ["r_knockoffs", "r"]  # python already completed in job 7948693


def run_backend(backend):
    out_dir = os.path.join(OUT_BASE, backend)
    os.makedirs(out_dir, exist_ok=True)

    input_params = {
        "x_path": SSC_X,
        "y_path": SSC_Y,
        "y_factor": False,
        "niter": 500,
        "SLIDE_top_feats": 10,
        "out_path": out_dir,
        "fdr": 0.1,
        "thresh_fdr": 0.2,
        "pure_homo": True,
        "do_interacts": True,
        "n_workers": 4,
        "spec": 0.1,
        "love_backend": "python",
        "knockoff_backend": backend,
        "knockoff_method": "asdp",
        "knockoff_shrink": False,
        "knockoff_offset": 0,
        "fstat": "glmnet_lambdasmax",
        "delta": DELTAS,
        "lambda": LAMBDAS,
    }

    logger.info(f"=== Starting {backend} backend ===")
    logger.info(f"  deltas={DELTAS}, lambdas={LAMBDAS}")
    logger.info(f"  output: {out_dir}")

    slider = OptimizeSLIDE(input_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slider.run_pipeline(verbose=True)

    logger.info(f"=== Completed {backend} backend ===\n")


def main():
    os.makedirs(OUT_BASE, exist_ok=True)

    for backend in BACKENDS:
        try:
            run_backend(backend)
        except Exception as e:
            logger.error(f"Backend {backend} failed: {e}")
            traceback.print_exc()
            continue

    logger.info("All backends complete.")


if __name__ == "__main__":
    main()
