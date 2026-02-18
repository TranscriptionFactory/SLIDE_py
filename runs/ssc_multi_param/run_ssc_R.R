#!/usr/bin/env Rscript
#
# Native R SLIDE ground-truth run on SSc data.
# Mirrors the Python loveslide run (job 8039789) with identical parameters
# so outputs can be compared directly.
#
# Usage:  Rscript run_ssc_R.R [out_path]

args <- commandArgs(trailingOnly = TRUE)

cat("\n**********************************************************************\n")
cat("****** SSc Ground-Truth: Native R SLIDE ******\n")
cat("**********************************************************************\n\n")

seed_num <- 42
set.seed(seed_num)
cat("Seed:", seed_num, "\n")

t_start <- proc.time()

# ---------------------------------------------------------------------------
# Load SLIDE from local repo (matches lab convention in getHelp.R)
# ---------------------------------------------------------------------------
library(devtools)

slide_repo <- Sys.getenv("SLIDE_LOCAL_REPO",
                         unset = "/ix/djishnu/Aaron/1_general_use/SLIDE")
cat("Loading SLIDE from:", slide_repo, "\n")
devtools::load_all(slide_repo)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ssc_x <- "/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/X.csv"
ssc_y <- "/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/Y.csv"

timestamp_str <- format(Sys.time(), "%Y%m%d_%H%M%S")

if (length(args) >= 1) {
  out_path <- args[1]
} else {
  out_path <- paste0(
    "/ix/djishnu/Aaron/1_general_use/SLIDE_py/runs/ssc_multi_param/",
    "output_R_ground_truth_", timestamp_str
  )
}
dir.create(out_path, recursive = TRUE, showWarnings = FALSE)
cat("Output directory:", out_path, "\n\n")

# ---------------------------------------------------------------------------
# Build input_params (matching Python job 8039789 exactly)
# ---------------------------------------------------------------------------
input_params <- list(
  x_path       = ssc_x,
  y_path       = ssc_y,
  out_path     = out_path,
  y_factor     = FALSE,
  delta        = c(0.01, 0.1),
  lambda       = c(0.1, 1.0),
  spec         = 0.1,
  fdr          = 0.1,
  thresh_fdr   = 0.2,
  eval_type    = "corr",
  SLIDE_iter   = 500,
  SLIDE_top_feats = 10,
  do_interacts = TRUE,
  sampleCV_iter = 500,
  rep_cv       = 50
)

# Write top-level yaml for reproducibility and SLIDEcv consumption
yaml_path <- file.path(out_path, "yaml_params.yaml")
yaml::write_yaml(input_params, yaml_path)
cat("Wrote top-level yaml to:", yaml_path, "\n\n")

# ---------------------------------------------------------------------------
# Run optimizeSLIDE (handles the full delta x lambda grid internally)
# ---------------------------------------------------------------------------
cat("********** Running optimizeSLIDE **********\n")
summary_table <- withCallingHandlers(
  optimizeSLIDE(input_params, sink_file = FALSE, continue_on_error = TRUE)
)
cat("\n********** optimizeSLIDE complete **********\n\n")

# ---------------------------------------------------------------------------
# Run SLIDEcv on each completed run (nrep=10, k=5 to match Python job)
# ---------------------------------------------------------------------------
cat("********** Running SLIDEcv (nrep=10, k=5) **********\n")
tryCatch({
  withCallingHandlers(SLIDEcv(yaml_path, nrep = 10, k = 5))
  cat("\n********** SLIDEcv complete **********\n")
}, error = function(e) {
  cat("\nSLIDEcv failed:", conditionMessage(e), "\n")
})

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
elapsed <- proc.time() - t_start
cat(sprintf("\nTotal wall time: %.1f min (%.0f sec)\n",
            elapsed["elapsed"] / 60, elapsed["elapsed"]))

cat("\n**********************************************************************\n")
cat("****** DONE ******\n")
cat("**********************************************************************\n")
