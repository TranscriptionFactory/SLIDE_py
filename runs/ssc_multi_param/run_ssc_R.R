#!/usr/bin/env Rscript
# Native R SLIDE ground truth for SSc data.
# Matches Python loveslide run (job 8039789) parameters exactly.
# Pattern: SLIDEHelpdesk/getHelp.R

cat("\n**********************************************************************\n")
cat("****** SSc R Ground Truth -- SLIDE native ******\n")
cat("**********************************************************************\n\n")

seed_num <- 42
set.seed(seed_num)
cat("Seed:", seed_num, "\n")

library(devtools)

# --- Load SLIDE from local repo -----------------------------------------------
slide_repo <- Sys.getenv("SLIDE_LOCAL_REPO")
if (slide_repo == "") {
  stop("Set SLIDE_LOCAL_REPO env var (e.g. /ix/djishnu/Aaron/1_general_use/SLIDE)")
}
cat("Loading SLIDE from:", slide_repo, "\n")
devtools::load_all(slide_repo)

# --- Load YAML ----------------------------------------------------------------
yaml_path <- "ssc_ground_truth.yaml"
if (!file.exists(yaml_path)) {
  stop(paste("YAML not found:", yaml_path))
}
input_params <- yaml::yaml.load_file(yaml_path)

# --- Create timestamped output directory --------------------------------------
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
job_id <- Sys.getenv("SLURM_JOB_ID", unset = "local")
out_dir <- paste0("output_R_", timestamp, "_", job_id)
dir.create(out_dir, recursive = TRUE)
input_params$out_path <- out_dir

# Write updated YAML (with out_path set) so SLIDEcv can read it
updated_yaml <- file.path(out_dir, "yaml_parameters.yaml")
yaml::write_yaml(input_params, updated_yaml)

cat("\nOutput directory:", out_dir, "\n")
cat("Parameters:\n")
str(input_params)

# --- Pipeline -----------------------------------------------------------------
cat("\n========== checkDataParams ==========\n")
withCallingHandlers(checkDataParams(input_params))

cat("\n========== optimizeSLIDE ==========\n")
withCallingHandlers(SLIDE::optimizeSLIDE(input_params, sink_file = FALSE))

cat("\n========== plotCorrelationNetworks ==========\n")
withCallingHandlers(SLIDE::plotCorrelationNetworks(input_params))

cat("\n========== SLIDEcv (nrep=10, k=5) ==========\n")
withCallingHandlers(SLIDE::SLIDEcv(updated_yaml, nrep = 10, k = 5))

# --- Summary ------------------------------------------------------------------
cat("\n========== Summary ==========\n")
cat("Output:", out_dir, "\n")

# List all delta/lambda output subdirectories
sub_dirs <- list.dirs(out_dir, recursive = FALSE)
for (d in sub_dirs) {
  summary_file <- file.path(d, "summary_table.csv")
  if (file.exists(summary_file)) {
    cat("\n---", basename(d), "---\n")
    summary_df <- read.csv(summary_file)
    print(summary_df)
  }
}

cat("\n**********************************************************************\n")
cat("****** SSc R Ground Truth -- COMPLETE ******\n")
cat("**********************************************************************\n")
