#!/usr/bin/env Rscript
#
# R script to run SLIDE on input data and save outputs for comparison with Python
#
# Usage: Rscript run_slide_R.R <yaml_path> [out_path]
#
# Arguments:
#   yaml_path       Path to YAML config file
#   out_path        Optional output path override
#
# Examples:
#   Rscript run_slide_R.R config.yaml
#   Rscript run_slide_R.R config.yaml /path/to/outputs
#

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  cat("Usage: Rscript run_slide_R.R <yaml_path> [out_path]\n")
  cat("\nArguments:\n")
  cat("  yaml_path       Path to YAML config file\n")
  cat("  out_path        Optional output path override\n")
  quit(status = 1)
}

# Load libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(SLIDE)
  library(yaml)
  library(doParallel)
})

yaml_path <- args[1]

# Load parameters from YAML
params <- yaml::yaml.load_file(yaml_path)

# Override out_path if provided as argument
out_path <- ifelse(length(args) >= 2, args[2], params$out_path)
params$out_path <- out_path

cat("============================================================\n")
cat("SLIDE R Analysis\n")
cat("============================================================\n")
cat("YAML config:", yaml_path, "\n")
cat("Output path:", out_path, "\n")
cat("X path:", params$x_path, "\n")
cat("Y path:", params$y_path, "\n")
cat("Delta:", paste(params$delta, collapse=", "), "\n")
cat("Lambda:", paste(params$lambda, collapse=", "), "\n")
cat("============================================================\n")

# Create output directory
if (!dir.exists(out_path)) {
  dir.create(out_path, recursive = TRUE)
}

# Save params to new yaml in output directory
new_yaml_path <- paste0(out_path, "/params.yaml")
yaml::write_yaml(params, new_yaml_path)

# Setup parallel backend
n_cores <- min(parallel::detectCores() - 1, 10)
cl <- makeCluster(n_cores)
registerDoParallel(cl)
cat("Using", n_cores, "cores for parallel processing\n\n")

# Run SLIDE
set.seed(123)
t_start <- Sys.time()

tryCatch({
  # Run optimizeSLIDE (this does the full pipeline)
  summary_table <- SLIDE::optimizeSLIDE(params, sink_file = FALSE)

  # Plot correlation networks
  SLIDE::plotCorrelationNetworks(params)

  cat("\nSummary table:\n")
  print(summary_table)

}, error = function(e) {
  cat("ERROR in SLIDE:", conditionMessage(e), "\n")
})

t_end <- Sys.time()

# Cleanup
stopCluster(cl)

cat("\n============================================================\n")
cat("R SLIDE completed in", format(t_end - t_start), "\n")
cat("Outputs saved to:", out_path, "\n")
cat("============================================================\n")
